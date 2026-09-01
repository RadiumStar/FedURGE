"""
:file: compressors.py
:date: 2025-07-17
:description: Compressor classes including Sparsification & Quantization for Federated Learning and Unlearning
"""

from typing import Dict, Any, Optional, Tuple

import torch


class Compressor:
    """Base Compressor Class for Federated Learning"""
    def __init__(self):
        self.communication_cost = 0.

    @torch.no_grad()
    def compress(self, params: Dict[str, torch.Tensor]) -> Any:
        # [WARNING] need to adapt both `named_parameters` and `state_dict` for compatibility
        """Compress model parameters and return compressed representation."""
        self.communication_cost = self.compute_communication_cost(*[param for param in params.values()])
        return params 

    @torch.no_grad()
    def decompress(self, compressed: Any) -> Dict[str, torch.Tensor]:
        """Decompress to model parameters (in full precision)."""
        return compressed

    def __call__(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compress and decompress model parameters."""
        compressed = self.compress(params)
        return self.decompress(compressed) 
    
    def compute_communication_cost(self, *args) -> float:
        """Compute the communication cost of the compressed parameters(B)
        
        :param args: compressed parameters (torch.Tensor s)
        """
        return sum(
            param.element_size() * param.numel() for param in args if param is not None
        )
    

class Sparsification(Compressor): 
    def __init__(self, sparsity: Optional[float] = None, k: Optional[int] = None, layerwise: bool = False): 
        """ Initialize Sparsification compressor

        :param sparsity: percentage of non-zero elements
        :param k: number of non-zero elements to keep; if `layerwise` is `True`, this is per layer number, otherwise it is the total number of non-zero elements
        :param layerwise: layer-wise sparsification if True, otherwise global sparsification
        :raises ValueError: must provide either `sparsity` or `k`
        """
        super().__init__()
        if sparsity is None and k is None:
            raise ValueError("Either sparsity or k must be provided.")
        self.sparsity = sparsity
        self.k = k
        self.layerwise = layerwise

    @torch.no_grad()
    def compress(self, params: Dict[str, torch.Tensor]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, int, torch.Size]]:
        """compress model parameters via sparsification

        :param params: model parameters to compress(state_dict)
        :return: `dict[name, (values, indices, original_flat_size, original_shape)]`
        """
        cost = 0.

        if self.layerwise:
            compressed = {}
            for name, tensor in params.items():
                if tensor is None:
                    compressed[name] = (None, None, 0, torch.Size([]))
                else: 
                    values, indices = self.sparsify(tensor.view(-1))
                    compressed[name] = (values, indices, tensor.numel(), tensor.shape)
                    cost += self.compute_communication_cost(values, indices)
        else: 
            shapes = {name: tensor.shape for name, tensor in params.items()}
            sizes = {name: tensor.numel() for name, tensor in params.items()}
            flat_tensors = [params[name].view(-1) for name in params]
            offsets = {}
            total_len = 0
            for name in params:
                offsets[name] = total_len
                total_len += sizes[name]
            all_tensor = torch.cat(flat_tensors)

            values, indices = self.sparsify(all_tensor)

            compressed = {}
            for name in params:
                start = offsets[name]
                end = start + sizes[name]
                mask = (indices >= start) & (indices < end)
                layer_indices = indices[mask] - start
                layer_values = values[mask]
                compressed[name] = (layer_values, layer_indices, sizes[name], shapes[name])

            cost += self.compute_communication_cost(values, indices)

        self.communication_cost = cost 

        return compressed

    @torch.no_grad()
    def decompress(self, compressed: Dict[str, Tuple[torch.Tensor, torch.Tensor, int, torch.Size]]) -> Dict[str, torch.Tensor]:
        decompressed = {}
        for name, (values, indices, size, shape) in compressed.items():
            if values is None or indices is None:
                decompressed[name] = None
            else: 
                flat_tensor = torch.zeros(size, device=values.device)
                flat_tensor[indices.to(torch.int64)] = values
                decompressed[name] = flat_tensor.view(shape) 
        return decompressed

    def sparsify(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return selected values and their indices (both 1D)"""
        raise NotImplementedError
    

class TopK(Sparsification):
    def __init__(self, k: Optional[int] = None, sparsity: Optional[float] = None, layerwise: bool = False):
        super().__init__(sparsity, k, layerwise)

    @torch.no_grad()
    def sparsify(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: 
        k = self.k if self.k is not None else max(1, int(self.sparsity * tensor.numel()))

        if k >= tensor.numel(): 
            return tensor, torch.arange(tensor.numel(), device=tensor.device)
        
        _, indices = torch.topk(tensor.abs(), k, sorted=False)
        values = tensor[indices]
        indices = indices.to(torch.int32)
        return values, indices
    

class RandK(Sparsification):
    def __init__(self, k: Optional[int] = None, sparsity: Optional[float] = None, layerwise: bool = False):
        super().__init__(sparsity, k, layerwise)

    @torch.no_grad()
    def sparsify(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k = self.k if self.k is not None else max(1, int(self.sparsity * tensor.numel()))

        if k >= tensor.numel():
            return tensor, torch.arange(tensor.numel(), device=tensor.device)
        
        indices = torch.randperm(tensor.numel(), device=tensor.device)[:k]
        values = tensor[indices]
        indices = indices.to(torch.int32)
        return values, indices


class SignSGD(Compressor): 
    """
    :paper: signSGD: Compressed Optimisation for Non-Convex Problems (ICML 2018); Error Feedback Fixes SignSGD and other Gradient Compression Schemes (ICML 2019)
    """
    def __init__(self, fixed: bool = True): 
        super().__init__()
        self.fixed = fixed
        self.bits_per_element = 1 

    @torch.no_grad()
    def compress(self, params: Dict[str, torch.Tensor]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Size]]:
        compressed = {}
        cost = 0.

        params_flat = torch.cat([tensor.view(-1) for tensor in params.values()])
        scale = params_flat.abs().mean()    # l1-norm fixed scale SignSGD

        # Quantize with stochastic rounding
        signs = params_flat.sign()

        offset = 0
        for name, tensor in params.items():
            size = tensor.numel()
            local_signs = signs[offset:offset + size]
            shape = tensor.shape

            # Store quantized signs and scale
            compressed[name] = (local_signs, scale, shape)
            offset += size

            cost += self.compute_communication_cost(local_signs, ) / local_signs.element_size() * self.bits_per_element / 8.0

        cost += self.compute_communication_cost(scale)  # scale is scalar
        self.communication_cost = cost

        return compressed

    @torch.no_grad()
    def decompress(self, compressed: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Size]]) -> Dict[str, torch.Tensor]:
        decompressed = {}
        for name, (signs, scale, shape) in compressed.items():
            flat_recon = scale * signs.view(-1)
            decompressed[name] = flat_recon.view(shape)
        return decompressed
    
    
class QSGD(Compressor): 
    """
    :paper: QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding (NIPS 2017)
    """
    def __init__(self, levels: Optional[int] = 16, bits: Optional[int] = None, stochastic: bool = True):
        super().__init__()

        if bits is not None: 
            levels = 2 ** bits
        if (levels & (levels - 1)) != 0:
            raise ValueError("levels must be a power of 2 for efficient encoding.") 
        
        self.levels = levels
        self.bits = (levels - 1).bit_length()
        self.stochastic = stochastic

    @torch.no_grad()
    def compress(self, params: Dict[str, torch.Tensor]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Size]]: 
        compressed = {}
        cost = 0.

        params_flat = torch.cat([tensor.view(-1) for tensor in params.values()])
        scale = params_flat.norm(p=2) 

        quantized = self.quantize(params_flat, scale)

        offset = 0
        for name, tensor in params.items():
            size = tensor.numel()
            quantized_tensor = quantized[offset:offset + size]
            compressed[name] = (quantized_tensor, scale, tensor.shape)
            offset += size
            cost += self.compute_communication_cost(quantized_tensor, ) / quantized_tensor.element_size() * self.bits / 8.
        
        cost += self.compute_communication_cost(scale, )
        self.communication_cost = cost

        return compressed 

    @torch.no_grad()
    def quantize(self, tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if scale.item() == 0:
            return torch.zeros_like(tensor) 
        
        scale_tensor = tensor.abs() * (self.levels - 1) / scale
        floor = scale_tensor.floor() 
        fractional = scale_tensor - floor 

        if self.stochastic: 
            rnd = torch.rand_like(fractional)
            quantized_abs = torch.where(rnd < fractional, floor + 1, floor)
        else:
            quantized_abs = torch.round(scale_tensor)

        # Clamp to [0, levels - 1]
        quantized_abs = torch.clamp(quantized_abs, 0, self.levels - 1)
        quantized = quantized_abs * tensor.sign() 

        return quantized
        
    @torch.no_grad()
    def decompress(self, compressed: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Size]]) -> Dict[str, torch.Tensor]: 
        decompressed = {}
        for name, (values, scale, shape) in compressed.items():
            flat_tensor = values.view(-1) * scale / (self.levels - 1)
            decompressed[name] = flat_tensor.view(shape)
        return decompressed


class TernGrad(Compressor):
    """
    :paper: TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning (NIPS 2017)
    
    This compressor quantizes gradients to {-1, 0, +1} using stochastic rounding.
    The scaling factor is the maximum absolute value of the gradient.
    """
    def __init__(self, stochastic: bool = True):
        super().__init__()
        self.stochastic = stochastic
        self.bits_per_element = 2 

    @torch.no_grad()
    def compress(self, params: Dict[str, torch.Tensor]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Size]]:
        compressed = {}
        cost = 0.

        params_flat = torch.cat([tensor.view(-1) for tensor in params.values()])
        scale = params_flat.abs().max()

        # Quantize with stochastic rounding
        signs = self.quantize(params_flat, scale)

        offset = 0
        for name, tensor in params.items():
            size = tensor.numel()
            local_signs = signs[offset:offset + size]
            shape = tensor.shape

            # Store quantized signs and scale
            compressed[name] = (local_signs, scale, shape)
            offset += size

            cost += self.compute_communication_cost(local_signs, ) / local_signs.element_size() * self.bits_per_element / 8.0

        cost += self.compute_communication_cost(scale)  # scale is scalar
        self.communication_cost = cost

        return compressed

    @torch.no_grad()
    def quantize(self, tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        signs = tensor.sign()  # {+1, -1}
        abs_tensor = tensor.abs()
        
        probabilities = abs_tensor / scale

        if self.stochastic:
            rand = torch.rand_like(probabilities)
            mask = probabilities >= rand  # stochastic thresholding
        else:
            mask = probabilities >= 0.5  # deterministic thresholding

        # Apply mask to signs
        quantized = signs * mask.float()
        return quantized

    @torch.no_grad()
    def decompress(self, compressed: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Size]]) -> Dict[str, torch.Tensor]:
        decompressed = {}
        for name, (signs, scale, shape) in compressed.items():
            flat_recon = scale * signs.view(-1)
            decompressed[name] = flat_recon.view(shape)
        return decompressed


if __name__ == "__main__":
    params = {
        'fc1.weight': torch.randn(5, 10),
        'fc2.bias': torch.randn(10),
    }

    params = {
        'fc1.weight': torch.tensor([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]]),
    }

    
    compressor = SignSGD()
    # print("Original parameters:", params)
    compressed = compressor.compress(params)
    # print(compressor.communication_cost)
    # print("Compressed parameters:", compressed)
    recovered = compressor.decompress(compressed)
    print(recovered.items())

    # for bit in [1, 2, 3, 4]: 
    #     compressor = QSGD(bits=bit, stochastic=True)
    #     # print("Original parameters:", params)
    #     compressed = compressor.compress(params)
    #     # print(compressor.communication_cost)
    #     # print("Compressed parameters:", compressed)
    #     recovered = compressor.decompress(compressed)
    #     print(f"bit={bit}:", recovered.items())

