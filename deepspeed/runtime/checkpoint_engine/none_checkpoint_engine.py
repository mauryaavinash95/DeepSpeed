# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.utils import logger, log_dist
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine
import time

class NoneCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params, r):
        super().__init__(config_params, r)
        self.rank = r

    def create(self, tag):
        log_dist(f"[None] Checkpoint {tag} is about to be saved!", ranks=[0])

    def save(self, state_dict, path: str):
        logger.info(f"[None] Saved {path}. in time {0}")
        return None

    def load(self, path: str, map_location=None):
        logger.info(f"[None] Loading checkpoint from {path}...")
        partition = torch.load(path, map_location=map_location)
        logger.info(f"[None] Loaded checkpoint from {path}.")
        return partition

    def commit(self, tag):
        return True

    def wait(self, prev_version = -1):
        return True
    
    def shutdown(self):
        return True