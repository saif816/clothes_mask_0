import os
import re
from typing import List, Tuple, Optional

import tensorrt as trt
from cuda import cudart
import numpy as np
import model.cuda_common as common
import torch

class InferenceEngine(object):
    def __init__(self, engine_path: os.PathLike):
        """
        TensorRT runner for any generic model with dynamic batching
        Pre-allocates memory for the minimal input shape by-default
        Assumes default optimization profile is used
        :param engine_path: Path to the serialized trt engine
        """
        logger = trt.Logger(trt.Logger.VERBOSE)
        logger.min_severity = trt.Logger.Severity.VERBOSE
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger, '')
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.allocations = []
        profile_idx = 0 # assume it's built with 1 profile only'
        self.max_batch_size = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            min_shape = self.engine.get_tensor_profile_shape(name, profile_idx)[0]
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            if is_input:
                self.batch_size = min_shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in min_shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(self._dims_to_tuple(min_shape)),
                'allocation': allocation,
                'size': size
            }
            self.allocations.append(allocation)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if self.max_batch_size is None:
                    self.max_batch_size = self.engine.get_tensor_profile_shape(name, profile_idx)[-1][0]
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        self.set_batch_size(np.random.rand(*self.inputs[0]['shape']), force=True)

    def _dims_to_tuple(self, d):
        n = getattr(d, "nbDims", None) or getattr(d, "nb_dims", None)
        if isinstance(n, int) and n > 0:
            return tuple(d[i] for i in range(n))
        return tuple(int(x) for x in re.findall(r"-?\d+", str(d)))

    def output_spec(self) -> List[Tuple[List[int], np.dtype]]:
        """
        Builds an output specification from the output bindings
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def set_batch_size(self, batched_input: np.ndarray, force=False):
        """
        Adjusts shapes for input allocations to support batch size for the inference input.
        Allocates memory for input bindings with the adjusted batch size.
        :param batched_input: Inference input of shape [B, ...], where B is the batch size
        :param force: Force batch size adjustment
        :return: None
        """
        batch_size = batched_input.shape[0]
        if batch_size > self.max_batch_size:
            batch_size = self.max_batch_size
        if batch_size == self.batch_size and not force:
            return
        self.allocations = []
        for binding in self.inputs:
            _, *tail = binding["shape"]
            ok = self.context.set_input_shape(binding['name'], [batch_size, *tail])
            if not ok:
                mn, _, mx = self.engine.get_tensor_profile_shape(binding['name'], 0)
                raise RuntimeError(f"batch {batch_size} not in [{mn[0]}, {mx[0]}]")
        if not self.context.all_binding_shapes_specified:
            raise RuntimeError("Not all binding shapes were specified")

        for bind in (self.inputs + self.outputs):
            common.cuda_call(cudart.cudaFree(bind['allocation']))
            name = bind['name']
            dtype = bind['dtype']
            shape = tuple(self.context.get_tensor_shape(name))
            nbytes = int(np.prod(shape) * dtype.itemsize)
            print(shape)
            bind['shape'] = list(shape)
            bind['size'] = nbytes
            ptr = common.cuda_call(cudart.cudaMalloc(nbytes))
            bind['allocation'] = ptr
            self.allocations.append(ptr)

        self.batch_size = batch_size

    def __call__(self, inputs: List[np.ndarray]):
        return self.infer(inputs, True, torch.device("cuda:0"))

    def split_in_max_batches(self, inputs: List[np.ndarray]) -> List[List[np.ndarray]]:
        if not inputs:
            return []
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")

        batched_inputs: List[List[np.ndarray]] = []
        for inp in inputs:
            n = inp.shape[0]
            chunks = [inp[i:i + self.max_batch_size] for i in range(0, n, self.max_batch_size)]
            batched_inputs.append(chunks)

        n_mini_batches = max(len(chunks) for chunks in batched_inputs) if batched_inputs else 0

        grouped: List[List[np.ndarray]] = []
        for i in range(n_mini_batches):
            group = [chunks[i] for chunks in batched_inputs if i < len(chunks)]
            if group:
                grouped.append(group)

        return grouped

    def mini_batch_infer(
        self, inputs: List[np.ndarray],
        convert_out_to_torch: bool = True,
        torch_device: Optional[torch.device] = None
    ):
        inputs = self.split_in_max_batches(inputs)
        outputs = []
        for mini_batch in inputs:
            outputs.append(self.infer(mini_batch, False, torch_device)[0])
        if convert_out_to_torch:
            outputs = torch.tensor(np.concatenate(outputs, axis=0), device=torch_device)
            print(outputs.size())
        return outputs


    def infer(self, inputs: List[np.ndarray],
              convert_out_to_torch: bool = True,
              torch_device: Optional[torch.device] = None
        ) -> List[np.ndarray]:
        """
        Runs inference on a list of input arrays.
        :param inputs: samples - float array of shape [B, C, H, W], timesteps - int array of shape [B]
        :param convert_out_to_torch: If true, convert output to torch tensors
        :param torch_device: Device where to allocate tensors in pytorch, only used if convert_out_to_torch = True
        :return: Generated noise prediction
        """
        self.set_batch_size(inputs[0])
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        for binding, host_arr in zip(self.inputs, inputs):
            host_arr_contig = np.ascontiguousarray(host_arr, dtype=binding['dtype'])
            common.memcpy_host_to_device(binding['allocation'], host_arr_contig)

        self.context.execute_v2(self.allocations)
        for binding, host_arr in zip(self.outputs, outputs):
            common.memcpy_device_to_host(host_arr, binding['allocation'])

        if convert_out_to_torch:
            outputs = torch.tensor(np.array(outputs), device=torch_device)
        return outputs