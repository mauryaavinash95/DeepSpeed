# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.utils import logger, log_dist, instrument_w_nvtx
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine
import time
from torchsnapshot import StateDict
from torchsnapshot.snapshot import Snapshot
from collections import deque 
import sys
import logging
from deepspeed.utils import groups

class TSNAsyncCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params, r):
        t = time.time()
        super().__init__(config_params, r)
        self.rank = r
        self.prev_sn = deque()
        logger = logging.getLogger("torchsnapshot.scheduler")
        logger.setLevel(logging.DEBUG)
        print(f"[TSNAsyncCheckpointEngine] Init took {time.time()-t}")

    def create(self, tag):
        log_dist(f"[TSNAsyncCheckpointEngine][Rank {self.rank}] Checkpoint {tag} is about to be saved!", ranks=[0])


    @instrument_w_nvtx
    def _to_statedict(self, ele, snapshot):
        try:
            if isinstance(ele, dict):
                snapshot = {}
                for (k, v) in ele.items():
                    snapshot[k] = None
                    snapshot[k] = self._to_statedict(v, snapshot[k])
                snapshot = StateDict(snapshot)
            elif isinstance(ele, list):
                snapshot = [None for _ in range(len(ele))]
                for (idx, v) in enumerate(ele):
                    snapshot[idx] = self._to_statedict(v, snapshot[idx])
            else:
                snapshot = ele
            return snapshot
        except Exception as exc:
            logger.info(f"[TSNAsyncCheckpointEngine] From _to_statedict, generated exception: {exc}")


    @instrument_w_nvtx
    def save(self, state_dict, path: str):
        logger.info(f"[TSNAsyncCheckpointEngine][Rank {self.rank}] Starting ckpt {path} at {time.time_ns()}")
        t = time.time()
        try:
            p = Snapshot.async_take(path=path, 
                                    app_state={"objects": StateDict(ckpt=state_dict)},
                                    replicated=[]
                                    )
            self.prev_sn.append((path, p))
            # p.wait()
            # Snapshot.take(path=path, app_state={"objects": StateDict(ckpt=state_dict)}, replicated=[])
            # logger.info(f"[TSNAsyncCheckpointEngine][Rank {self.rank}] Saved {path}. in time {time.time()-t} started at {time.time_ns()}")
        except Exception as e:
            print(f"TSNAsyncCheckpointEngine][Rank {self.rank}] Async checkpoint failed with error: {e}")
            sys.exit(-1)
        # import pdb; pdb.set_trace()
        return None

    def load(self, path: str, map_location=None):
        snapshot = Snapshot(path=path)
        partition={"objects": StateDict(ckpt={})}
        snapshot.restore(app_state=partition)
        return partition["objects"]["ckpt"]

    def commit(self, tag):
        logger.info(f"[TSNAsyncCheckpointEngine] Checkpoint {tag} is ready now!")
        return True

    def wait(self, prev_version = -1):
        return True
    
    def shutdown(self):
        t = time.time()
        while len(self.prev_sn) > 0:
            try:
                inner_t = time.time()
                (path, p) = self.prev_sn.popleft()
                while not p.done():
                    logger.info(f"[TSNAsyncCheckpointEngine] Waiting for {path}.")
                    p.wait()
            except Exception as e:
                print(f"TSNAsyncCheckpointEngine][Rank {self.rank}] Async checkpoint waiting failed with error: {e}")
                sys.exit(-1)
        logger.info(f"[TSNAsyncCheckpointEngine] Shutdown took time {time.time()-t}")
        return True

    def __del__(self):
        self.shutdown()