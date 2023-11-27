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



class TSNAsyncCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params, r):
        super().__init__(config_params, r)
        self.rank = r
        # print("<<<<<<<<<<< Inited on rank ", self.rank)
        self.prev_sn = deque()

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
        # logger.info(f"[TSNAsyncCheckpointEngine][Rank {self.rank}] Saving {path}...")
        t = time.time()
        try:
            # x = self._to_statedict(state_dict, {})
            p = Snapshot.async_take(path=path, 
                                    app_state={"objects": StateDict(ckpt=state_dict)},
                                    replicated=[]
                                    )
            self.prev_sn.append((path, p))
            # p.wait()
            # Snapshot.take(path=path, app_state=x)
            # logger.info(f"[TSNAsyncCheckpointEngine][Rank {self.rank}] Saved {path}. in time {time.time()-t}")
        except Exception as e:
            print(f"TSNAsyncCheckpointEngine][Rank {self.rank}] Async checkpoint failed with error: {e}")
            sys.exit(-1)
        # import pdb; pdb.set_trace()
        return None

    def load(self, path: str, map_location=None):
        logger.info(f"[TSNAsyncCheckpointEngine] Loading checkpoint from {path}...")
        partition = torch.load(path, map_location=map_location)
        logger.info(f"[TSNAsyncCheckpointEngine] Loaded checkpoint from {path}.")
        return partition

    def commit(self, tag):
        logger.info(f"[TSNAsyncCheckpointEngine] Checkpoint {tag} is ready now!")
        return True

    def wait(self, prev_version = -1):
        # while len(self.prev_sn) > 0:
        #     try:
        #         (path, p) = self.prev_sn.popleft()
        #         # logger.info(f"[TSNAsyncCheckpointEngine][Rank {self.rank}] In wait for {len(self.prev_sn)} for path {path}.")
        #         # for i, (x, y) in enumerate(self.prev_sn):
        #         #     print(i, x, y, y.done())
        #         if not p.done():
        #             # logger.info(f"[TSNAsyncCheckpointEngine] Waiting for {path}.")
        #             p.wait()
        #         # for i, (x, y) in enumerate(self.prev_sn):
        #         #     if y.done():
        #         #         logger.info(f"[TSNAsyncCheckpointEngine] Done checkpointing {i}, {x}, {y}.")
        #         #         del self.prev_sn[i]
        #         #         break
        #     except Exception as e:
        #         print(f"TSNAsyncCheckpointEngine][Rank {self.rank}] Async checkpoint waiting failed with error: {e}")
        #         sys.exit(-1)
        return True
    
    def shutdown(self):
        # self.wait()
        t = time.time()
        while len(self.prev_sn) > 0:
            try:
                inner_t = time.time()
                (path, p) = self.prev_sn.popleft()
                # logger.info(f"[TSNAsyncCheckpointEngine][Rank {self.rank}] In wait for {len(self.prev_sn)} for path {path}.")
                # for i, (x, y) in enumerate(self.prev_sn):
                #     print(i, x, y, y.done())
                if not p.done():
                    # logger.info(f"[TSNAsyncCheckpointEngine] Waiting for {path}.")
                    p.wait()
                logger.info(f"[TSNAsyncCheckpointEngine][Rank {self.rank}] time {time.time()-inner_t} len {len(self.prev_sn)} for path {path}.")
                # for i, (x, y) in enumerate(self.prev_sn):
                #     if y.done():
                #         logger.info(f"[TSNAsyncCheckpointEngine] Done checkpointing {i}, {x}, {y}.")
                #         del self.prev_sn[i]
                #         break
            except Exception as e:
                print(f"TSNAsyncCheckpointEngine][Rank {self.rank}] Async checkpoint waiting failed with error: {e}")
                sys.exit(-1)
        logger.info(f"[TSNAsyncCheckpointEngine] Shutdown took time {time.time()-t}")
        return True