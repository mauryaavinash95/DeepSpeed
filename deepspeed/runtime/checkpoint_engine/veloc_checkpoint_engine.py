# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.utils import logger, log_dist, instrument_w_nvtx
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine
from threading import Thread, Condition, Lock
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from collections import OrderedDict
import os, time
import shutil
from deepspeed.ops.op_builder import AsyncIOBuilder
from deepspeed.runtime.swap_tensor.constants import *

class VELOCCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params=None):
        super().__init__(config_params)
        self.d2h_stream = torch.cuda.Stream()
        self.d2h_queue = Queue()
        self.d2h_processing = 0
        self.h2l_queue = Queue()
        self.h2l_processing = 0
        self.d2h_futures = []
        self.h2l_futures = []
        d2h_lock = Lock()
        h2l_lock = Lock()
        self.d2h_cond = Condition(lock=d2h_lock)
        self.h2l_cond = Condition(lock=h2l_lock)
        # aio_op = AsyncIOBuilder().load()
        # aio_config = config_params["aio_config"]
        # self.aio_handle = aio_op.aio_handle(aio_config[AIO_BLOCK_SIZE], aio_config[AIO_QUEUE_DEPTH], aio_config[AIO_SINGLE_SUBMIT], aio_config[AIO_OVERLAP_EVENTS], aio_config[AIO_THREAD_COUNT])

    @instrument_w_nvtx
    def _d2h_copy(self):
        try:
            torch.cuda.stream(self.d2h_stream)
            while not self.d2h_queue.empty():
                with self.d2h_cond:
                    ckpt = self.d2h_queue.get()
                    self.d2h_cond.notify_all()
                # ckpt["data"] = self._to_cpu(ckpt["data"], {})
                # time.sleep(500000)
                torch.save(ckpt["data"], f"/dev/shm/{os.path.basename(ckpt['path'])}")
                logger.info(f"[VELOC] From _d2h_copy, completed torch.save to /dev/shm for {os.path.basename(ckpt['path'])}")
                with self.d2h_cond:
                    self.d2h_processing -= 1
                    self.d2h_cond.notify_all()
                with self.h2l_cond:
                    self.h2l_processing += 1
                    # self.h2l_queue.put(ckpt['path'])
                    self.h2l_queue.put(ckpt)
                    self.h2l_cond.notify_all()
                if (self.h2l_processing == 1):
                    executor = ThreadPoolExecutor(max_workers=1)
                    f = executor.submit(self._d2h_copy)
                    self.h2l_futures.append(f)
                logger.info(f"[VELOC] From _d2h_copy, completed notifying to h2f thread for {os.path.basename(ckpt['path'])}")
        except Exception as exc:
            logger.info(f"[VELOC] From _d2h_copy, generated exception: {exc}")

    @instrument_w_nvtx
    def _h2l_copy(self):
        try:
            torch.cuda.stream(self.d2h_stream)
            while not self.h2l_queue.empty():
                with self.h2l_cond:
                    v = self.h2l_queue.get(block=True)
                    self.h2l_cond.notify()
                shutil.move(f"/dev/shm/{os.path.basename(v)}", v)
                logger.info(f"[VELOC] From _h2l_copy, completed transfer of {os.path.basename(v)}")
                with self.h2l_cond:
                    self.h2l_processing-=1
                    self.h2l_cond.notify()
                logger.info(f"[VELOC] From _h2l_copy, decremented count after {os.path.basename(v)}")
        except Exception as exc:
            logger.info(f"[VELOC] From _d2h_copy, generated exception: {exc}")

    def _l2r_copy(self):
        pass

    def create(self, tag):
        log_dist(f"[VELOC] Checkpoint {tag} is about to be saved!", ranks=[0])

    @instrument_w_nvtx
    def _to_cpu(self, ele, snapshot):
        try:
            torch.cuda.stream(self.d2h_stream)
            if torch.is_tensor(ele) and ele.device.type == 'cuda':
                snapshot = ele.cpu()
            elif isinstance(ele, dict) and not isinstance(ele, OrderedDict):
                snapshot = {}
                for (k, v) in ele.items():
                    snapshot[k] = None
                    snapshot[k] = self._to_cpu(v, snapshot[k])
            elif isinstance(ele, list):
                snapshot = [None for _ in range(len(ele))]
                for (idx, v) in enumerate(ele):
                    snapshot[idx] = self._to_cpu(v, snapshot[idx])
            else:
                snapshot = ele
            return snapshot
        except Exception as exc:
            logger.info(f"[VELOC] From _to_cpu, generated exception: {exc}")

    @instrument_w_nvtx
    def save(self, state_dict, path: str):
        try:
            t = time.time()
            with self.d2h_cond:
                logger.info(f"[VELOC] Time to get d2h_cond lock {path} in {time.time()-t}")    
                self.d2h_queue.put({'data': state_dict, 'path': path})
                self.d2h_processing += 1
                self.d2h_cond.notify_all()
            logger.info(f"[VELOC] Time to get d2h_cond put and notify {path} in {time.time()-t}")    
            if (self.d2h_processing == 1):
                logger.info(f"[VELOC] Starting threadpoolexecutor {path} in {time.time()-t}")    
                executor = ThreadPoolExecutor(max_workers=1)
                f = executor.submit(self._d2h_copy)
                self.d2h_futures.append(f)
            logger.info(f"[VELOC] Added to background checkpointing {path} in {time.time()-t}")
            return None
        except Exception as exc:
            logger.info(f"[VELOC] From save, generated exception: {exc}")


    def load(self, path: str, map_location=None):
        logger.info(f"[VELOC] Loading checkpoint from {path}...")
        partition = torch.load(path, map_location=map_location)
        logger.info(f"[VELOC] Loaded checkpoint from {path}.")
        return partition

    def commit(self, tag):
        logger.info(f"[VELOC] Checkpoint {tag} is ready now!")
        return True
