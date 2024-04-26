
from deepspeed.utils import logger, log_dist
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine
from datastates import DataStates


class DataStatesCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params, rank):
        super().__init__(config_params, rank)
        self.ckpt_engine = DataStates(config_params, rank)
    
    def create(self, tag):
        log_dist(f"[DataStates] Checkpoint {tag} is about to be saved!", ranks=[0])
        return None

    def save(self, state_dict, path: str):
        return self.ckpt_engine.save(state_dict, path)
    
    def load(self, path: str, map_location=None):
        return self.ckpt_engine.load(path, map_location)
    
    def commit(self, tag):
        return self.ckpt_engine.commit(tag)

    def wait(self):
        return self.ckpt_engine.wait()

    def __del__(self):
        return self.ckpt_engine.shutdown()

   
