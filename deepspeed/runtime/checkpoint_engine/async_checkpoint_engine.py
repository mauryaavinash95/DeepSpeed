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

class AsyncCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params=None):
        super().__init__(config_params)
        self.checkpoint_in_progress = False
        in_progress_lock = Lock()
        self.in_progress_cv = Condition(lock=in_progress_lock)
        self.futures = None
        self.executor = ThreadPoolExecutor(max_workers=1)

    def create(self, tag):
        log_dist(f"[AsyncTorch] Checkpoint {tag} is about to be saved!", ranks=[0])

    @instrument_w_nvtx
    def _to_cpu(self, ele, snapshot):
        try:
            if torch.is_tensor(ele) and ele.device.type=='cuda':
                snapshot = ele.cpu()
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
                log_dist(f"[AsyncTorch] Got in parse dict of type {type(ele)}: {ele}")
                snapshot = copy.deepcopy(ele)
            return snapshot
        except Exception as exc:
            logger.info(f"[AsyncTorch] From _to_cpu, generated exception: {exc}")

    def _background_save(self, state_dict, path):
        t = time.time()
        with self.in_progress_cv:
            self.checkpoint_in_progress = True
            self.in_progress_cv.notify_all()
        torch.save(state_dict, path)
        with self.in_progress_cv:
            self.checkpoint_in_progress = False
            self.in_progress_cv.notify_all()
        logger.info(f"[AsyncTorch] Time to complete background save {time.time()-t} for path {path}")

    @instrument_w_nvtx
    def save(self, state_dict, path: str):
        logger.info(f"[AsyncTorch] Saving {path}...")
        t = time.time()
        while self.checkpoint_in_progress:
            self.in_progress_cv.wait()
        logger.info(f"[AsyncTorch] Prev completion waiting time {time.time()-t} for incoming {path}...")
        new_state_dict = {}
        new_state_dict = self._to_cpu(state_dict, new_state_dict)
        logger.info(f"[AsyncTorch] To CPU snapshot time {time.time()-t} for incoming {path}...")
        # torch.save(state_dict, path)
        ts = time.time()
        self.executor.submit(self._background_save, new_state_dict, path)        
        logger.info(f"[AsyncTorch] Time to submit to background save {time.time()-ts}")
        logger.info(f"[AsyncTorch] Saved {path}. in time {time.time()-t}")
        return None

    def load(self, path: str, map_location=None):
        logger.info(f"[AsyncTorch] Loading checkpoint from {path}...")
        partition = torch.load(path, map_location=map_location)
        logger.info(f"[AsyncTorch] Loaded checkpoint from {path}.")
        return partition

    def commit(self, tag):
        logger.info(f"[AsyncTorch] Checkpoint {tag} is ready now!")
        return True

    def wait(self, prev_version):
        return True
    
    def shutdown(self):
        return True