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
import time
from deepspeed.ops.op_builder import VelocCkptBuilder
import sys

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
        self.ckpt_engine = VelocCkptBuilder().load().veloc_ckpt_handle(config_params["veloc_config"]["host_cache"] << 30)
        # aio_op = AsyncIOBuilder().load()
        # aio_config = config_params["aio_config"]
        # self.aio_handle = aio_op.aio_handle(aio_config[AIO_BLOCK_SIZE], aio_config[AIO_QUEUE_DEPTH], aio_config[AIO_SINGLE_SUBMIT], aio_config[AIO_OVERLAP_EVENTS], aio_config[AIO_THREAD_COUNT])

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
            # import pdb; pdb.set_trace()
            t = time.time()
            self.ckpt_engine.ckpt(dict(state_dict), path)
            logger.info(f"[VELOC] Added to background checkpointing {path} in {time.time()-t}")
            return None
        except Exception as exc:
            logger.info(f"[VELOC] From save, generated exception: {exc}")
            sys.exit(-1)


    def load(self, path: str, map_location=None):
        logger.info(f"[VELOC] Loading checkpoint from {path}...")
        partition = torch.load(path, map_location=map_location)
        logger.info(f"[VELOC] Loaded checkpoint from {path}.")
        return partition

    def commit(self, tag):
        logger.info(f"[VELOC] Checkpoint {tag} is ready now!")
        return True
