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
import pickle
import numpy as np
import ctypes
import io
from deepspeed.runtime.swap_tensor.constants import *
import copy

SIZE_UINT64 = np.array(0, dtype=np.uint64).nbytes
ASYNC_CKPT_SIZE_MIN = 1<<25
class RedirectStdout:
    def __init__(self):
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, text):
        # Forward text to Python's print
        self._stdout.write(text)

    def flush(self):
        pass  # This is necessary to prevent the AttributeError

redirect = RedirectStdout()

class VELOCCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params, r):
        try:
            super().__init__(config_params, r)
            self.rank = r
            self.ckpt_engine = VelocCkptBuilder().load().veloc_ckpt_handle(
                    int(config_params["host_cache"] << 30), 
                    int(torch.cuda.current_device()),
                    int(config_params["writer_threads"])
                    )
            self.futures = None
            self.executor = ThreadPoolExecutor(max_workers=int(config_params["writer_threads"]))
        except Exception as exc2:
            print("[ERROR]Got exception during VELOC init ", exc2)
            sys.exit(-1)


    def create(self, tag):
        log_dist(f"[VELOC] Checkpoint {tag} is about to be saved!", ranks=[0])

    # @instrument_w_nvtx
    def _parse_dict(self, ele, snapshot, async_copies_list):
        try:
            if isinstance(ele, np.ndarray): # and ele.nbytes > ASYNC_CKPT_SIZE_MIN:
                print("Got a numpy array")
                # import pdb; pdb.set_trace();
                data_device = -1
                snapshot = f"{len(async_copies_list)}-pickled-numpy"
                # Storing in async_copies_list values: data_ptr, size_in_bytes, device_id, file_offset
                async_copies_list.append([ele.ctypes.data, ele.nbytes, -1, 0])
            elif torch.is_tensor(ele) and ele.device.type == 'cuda':
                if (ele.numel()*ele.element_size() > ASYNC_CKPT_SIZE_MIN):
                    data_device = ele.device.index if ele.device.type == 'cuda' else -1
                    snapshot = f"{len(async_copies_list)}-pickled-tensor"
                    async_copies_list.append([ele, ele.numel()*ele.element_size(), data_device, 0])
                else:
                    snapshot = ele.cpu()
            elif isinstance(ele, dict):
                snapshot = {}
                for (k, v) in ele.items():
                    snapshot[k] = None
                    snapshot[k], async_copies_list = self._parse_dict(v, snapshot[k], async_copies_list)
            elif isinstance(ele, list):
                snapshot = [None for _ in range(len(ele))]
                for (idx, v) in enumerate(ele):
                    snapshot[idx], async_copies_list = self._parse_dict(v, snapshot[idx], async_copies_list)
            else:
                log_dist(f"[VELOC] Got in parse dict of type {type(ele)}: {ele}")
                snapshot = ele # copy.deepcopy(ele)
            return snapshot, async_copies_list
        except Exception as exc:
            logger.info(f"[VELOC][ERROR] From _to_cpu, generated exception: {exc}")

    @instrument_w_nvtx
    def save_background(self, state_dict, path: str):
        try:
            start_time = time.time()
            version = int(path.split("/")[-2].replace('global_step', ''))
            new_state_dict = {}
            async_copies_list = []
            t_begin = time.time()
            # if (self.rank == 0):
            #     import pdb; pdb.set_trace();
            new_state_dict, async_copies_list = self._parse_dict(state_dict, new_state_dict, async_copies_list)
            serialized_dict = pickle.dumps(new_state_dict, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"[VELOC] Time to serialize {path}, {time.time()-t_begin}, {len(serialized_dict)}")
            
            t_begin = time.time()
            headers = np.zeros((len(async_copies_list)+1, 2), dtype=np.uint64) #[(ctypes.c_uint64(0), ctypes.c_uint64(0))]*(len(async_copies_list)+1)
            header_size = pickle.dumps(headers, protocol=pickle.HIGHEST_PROTOCOL)
            

            file_offset = SIZE_UINT64        # Count the number of bytes require to write header_size in bytes
            file_offset += len(header_size)                     # Add the serialized header_size
            
            headers[0] = (file_offset, file_offset+len(serialized_dict))
            file_offset += len(serialized_dict)                 # Add offset for writing the serial-dict
            
            # self.ckpt_engine.ckpt_header_size(0, ctypes.sizeof(ctypes.c_uint64), header_size, path) # Write size of header list
            self.ckpt_engine.ckpt_header_size(version, 0, SIZE_UINT64, len(headers), path) # Write size of header list
            # After this we should be writing headers, but we do not have file offsets yet.
            # === This was being written separately.
            # self.ckpt_engine.ckpt_pickle(version, headers[0][0], headers[0][1], serialized_dict, path)  # Start writing the serialized_dict
            # logger.info(f"[VELOC] Time to write header size and serialized dict, {version}, {time.time()-t_begin}, {path}")
            
            for i, t in enumerate(async_copies_list):
                t[3] = file_offset      # v[3] represents the file_write_offset
                file_offset += t[1]     # v[1] represents the size of the object
                headers[i+1] = ((t[3], file_offset))
                if (torch.is_tensor(t[0])):
                    self.ckpt_engine.ckpt_tensor(version, headers[i+1][0], headers[i+1][1], t[0], t[1], t[2], t[3], path)
                else:
                    self.ckpt_engine.ckpt_obj(version, headers[i+1][0], headers[i+1][1], t[0], t[1], t[2], t[3], path)
            
            headers = pickle.dumps(headers, protocol=pickle.HIGHEST_PROTOCOL)
            # logger.info(f"[VELOC] Starting to write ckpt_pickle for header now for path {path}")
            # ==== This was being written separately.
            # self.ckpt_engine.ckpt_pickle(version, SIZE_UINT64, SIZE_UINT64+len(header_size), headers, path)

            combined_header = headers + serialized_dict
            self.ckpt_engine.ckpt_pickle(version, SIZE_UINT64, SIZE_UINT64+len(header_size)+len(serialized_dict), combined_header, path)

            # sys.stdout = redirect._stdout
            # logger.info(f"[VELOC] Version {version} in {time.time()-start_time}, path {path}")
            logger.info(f"[VELOC] In background meta-data thread saved {path} in time {time.time()-start_time}")
            return None
        except Exception as exc:
            logger.info(f"[VELOC][ERROR] From VELOC save_background, generated exception: {exc}")
            sys.exit(-1)

    def save(self, state_dict, path: str):
        try:
            start_time = time.time()
            self.executor.submit(self.save_background, state_dict, path)
            logger.info(f"[VELOC] Saved {path}. in time {time.time()-start_time}")
            return True
        except Exception as exc:
            logger.info(f"[VELOC][ERROR] From save, generated exception: {exc}")
            sys.exit(-1)
            
        

    def load(self, path: str, map_location=None):
        logger.info(f"[VELOC] Loading checkpoint from {path}...")
        partition = torch.load(path, map_location=map_location)
        logger.info(f"[VELOC] Loaded checkpoint from {path}.")
        return partition

    def commit(self, tag):
        logger.info(f"[VELOC] Checkpoint {tag} is ready now!")
        return True

    def wait(self, prev_version:int = -1):
        try:
            t = time.time()
            self.ckpt_engine.wait(prev_version)
            logger.info(f"[VELOC] Wait time in checkpointing engine {time.time()-t}")
        except Exception as exc:
            logger.info(f"[VELOC][ERROR] From wait, generated exception: {exc}")
            sys.exit(-1)
        return 
    
    def shutdown(self):
        self.executor.shutdown(True)
        return self.ckpt_engine.shutdown()
