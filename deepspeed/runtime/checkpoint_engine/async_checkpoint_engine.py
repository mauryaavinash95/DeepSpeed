# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.utils import logger, log_dist, instrument_w_nvtx
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine
import time
from threading import Thread, Condition, Lock
from concurrent.futures import ThreadPoolExecutor
import copy
from collections import deque 
import concurrent.futures

class AsyncCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params, r):
        super().__init__(config_params, r)
        self.checkpoint_in_progress = False
        self.rank = r
        in_progress_lock = Lock()
        self.in_progress_cv = Condition(lock=in_progress_lock)
        self.snapshot_futures = deque()
        self.persist_futures = deque()
        self.snapshot_executor = ThreadPoolExecutor(max_workers=1)
        self.snapshots = deque()
        self.persist_executor = ThreadPoolExecutor(max_workers=1)

    def create(self, tag):
        log_dist(f"[AsyncTorch][Rank {self.rank}] Checkpoint {tag} is about to be saved!", ranks=[0])

    @instrument_w_nvtx
    def _to_cpu(self, ele, snapshot):
        try:
            if torch.is_tensor(ele) and ele.device.type=='cuda':
                t = time.time()
                snapshot = ele.cpu()
                logger.info(f"[AsyncTorch][Rank {self.rank}] Tensor copy took {time.time()-t} of size {snapshot.numel()*snapshot.element_size()}")
            # elif isinstance(ele, dict) and not isinstance(ele, OrderedDict):
            elif isinstance(ele, dict):
                snapshot = {}
                for (k, v) in ele.items():
                    snapshot[k] = None
                    snapshot[k] = self._to_cpu(v, snapshot[k])
            elif isinstance(ele, list):
                snapshot = [None for _ in range(len(ele))]
                for (idx, v) in enumerate(ele):
                    snapshot[idx] = self._to_cpu(v, snapshot[idx])
            else:
                log_dist(f"[AsyncTorch][Rank {self.rank}] Got in parse dict of type {type(ele)}: {ele}")
                snapshot = copy.deepcopy(ele)
            return snapshot
        except Exception as exc:
            logger.info(f"[AsyncTorch][Rank {self.rank}] From _to_cpu, generated exception: {exc}")

    def _background_persist(self):
        # logger.info(f"[AsyncTorch] starting _background_persist with {len(self.snapshots)} objects")
        while len(self.snapshots) > 0:
            (state_dict, path) = self.snapshots.popleft()
            logger.info(f"[AsyncTorch][Rank {self.rank}] In _background_persist for path {path}")
            t = time.time()
            torch.save(state_dict, path)
            logger.info(f"[AsyncTorch][Rank {self.rank}] Time to complete background persist {time.time()-t} for path {path}")

    
    def _background_snapshot(self, state_dict, path):
        t = time.time()
        with self.in_progress_cv:
            self.checkpoint_in_progress = True
            self.in_progress_cv.notify_all()
        new_state_dict = {}
        new_state_dict = self._to_cpu(state_dict, new_state_dict)
        logger.info(f"[AsyncTorch][Rank {self.rank}] Time to copy to CPU in background snapshot {time.time()-t} for path {path}")
        # torch.save(state_dict, path)
        self.snapshots.append((new_state_dict, path))
        with self.in_progress_cv:
            self.checkpoint_in_progress = False
            self.in_progress_cv.notify_all()
        logger.info(f"[AsyncTorch][Rank {self.rank}] Time to complete background snapshot {time.time()-t} for path {path}")

    @instrument_w_nvtx
    def save(self, state_dict, path: str):
        # logger.info(f"[AsyncTorch] Saving {path}...")
        t = time.time()
        with self.in_progress_cv:
            while self.checkpoint_in_progress:
                self.in_progress_cv.wait()
        logger.info(f"[AsyncTorch][Rank {self.rank}] Prev completion waiting time {time.time()-t} for incoming {path}...")
        
        # logger.info(f"[AsyncTorch] To CPU snapshot time {time.time()-t} for incoming {path}...")
        # torch.save(state_dict, path)
        ts = time.time()
        f = self.snapshot_executor.submit(self._background_snapshot, state_dict, path)        
        self.snapshot_futures.append(f)
        # logger.info(f"[AsyncTorch] Time to submit to background save {time.time()-ts}")
        logger.info(f"[AsyncTorch][Rank {self.rank}] Saved {path}. in time {time.time()-t}")
        return None

    def load(self, path: str, map_location=None):
        logger.info(f"[AsyncTorch][Rank {self.rank}] Loading checkpoint from {path}...")
        partition = torch.load(path, map_location=map_location)
        logger.info(f"[AsyncTorch][Rank {self.rank}] Loaded checkpoint from {path}.")
        return partition

    def commit(self, tag):
        t = time.time()
        f = self.persist_executor.submit(self._background_persist) 
        self.persist_futures.append(f)
        logger.info(f"[AsyncTorch][Rank {self.rank}] Checkpoint commit {time.time()-t}: {tag}, persist futures {len(self.persist_futures)}")
        return True

    def wait(self, prev_version = -1):
        t = time.time()
        concurrent.futures.wait(self.snapshot_futures)
        self.snapshot_futures.clear() 
        logger.info(f"[AsyncTorch][Rank {self.rank}] Wait time in checkpointing engine {time.time()-t}")   
        return True
    
    def shutdown(self):
        t = time.time()
        self.wait()
        concurrent.futures.wait(self.persist_futures)
        self.persist_futures.clear()    
        self.snapshot_executor.shutdown(True)
        self.persist_executor.shutdown(True)
        logger.info(f"[AsyncTorch][Rank {self.rank}] Shutdown time {time.time()-t}") 
        return True