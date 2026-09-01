"""
:file: communicators.py
:date: 2025-07-18
:description: Communicator classes for Federated Learning Error feedback 
"""

from typing import Dict, Optional, Literal

import torch 

from .compressors import Compressor 


class Communicator: 
    def __init__(self, compressor: Compressor = Compressor(), unit: Literal['B', 'KB', 'MB', 'GB'] = 'MB'): 
        """Base Communicator Class for Federated Learning

        :param compressor: compressor for model parameters, defaults to Compressor()
        :param unit: unit for communication cost, defaults to 'MB'
        """
        self.compressor = compressor
        self.communication_cost = 0.
        self.unit_name = unit.upper()
        self.unit_scale = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}[self.unit_name]

    @torch.no_grad()
    def __call__(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]: 
        """Compress and decompress model parameters

        :param params: model parameters to compress
        :return: decompressed model parameters
        """
        compressed_params = self.compressor.compress(params)
        self.communication_cost += self.compressor.communication_cost
        decompressed_params = self.compressor.decompress(compressed_params)
        return decompressed_params
    
    def compute_communication_cost(self, *args) -> float:
        """Compute communication cost for given model parameters."""
        return self.compressor.compute_communication_cost(*args)
    
    def get_communication_cost(self, unit: Optional[Literal['B', 'KB', 'MB', 'GB']] = 'MB') -> float:
        """Return total communication cost in selected unit."""
        if unit is not None: 
            return self.communication_cost / {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}[unit.upper()]
        return self.communication_cost / self.unit_scale

    def reset_communication_cost(self) -> None:
        """Reset accumulated communication cost."""
        # self.communication_cost = 0.0
        return 

    def reset(self) -> None: 
        self.reset_communication_cost()


class EF14(Communicator): 
    """EF14

    :paper: Sparsified SGD with Memory (NIPS 2018)
    :link: https://arxiv.org/abs/1809.07599
    """
    def __init__(self, compressor: Compressor = Compressor(), unit: Literal['B', 'KB', 'MB', 'GB'] = 'MB', is_init_compressed: bool = False): 
        """EF14 Communicator for Federated Learning

        :param compressor: compressor for model parameters, defaults to Compressor()
        :param unit: unit for communication cost, defaults to 'MB'
        :param is_init_compressed: whether the initial gradient is compressed, defaults to False
        """
        super().__init__(compressor, unit)
        self.error: Dict[str, torch.Tensor] = None
        self.is_init_compressed = is_init_compressed
    
    @torch.no_grad()
    def __call__(self, present_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]: 
        if self.error is None: 
            # first call, initialize error with present update
            self.error = {name: torch.zeros_like(val) for name, val in present_update.items()}
            if not self.is_init_compressed:
                self.communication_cost += self.compressor.compute_communication_cost(*[val for val in present_update.values()])
                return present_update

        # compress present gradient + error
        compressed_item = {}
        for name in present_update.keys():
            compressed_item[name] = present_update[name] + self.error[name]

        compressed_item = self.compressor.compress(compressed_item)
        self.communication_cost += self.compressor.communication_cost
        decompressed = self.compressor.decompress(compressed_item)

        # update error buffer
        for name in present_update.keys():
            self.error[name] += present_update[name] - decompressed[name]

        return decompressed
    
    def reset(self): 
        super().reset()
        self.error: Dict[str, torch.Tensor] = None


class EF21(Communicator):
    """EF21
    
    :paper: EF21: A New, Simpler, Theoretically Better, and Practically Faster Error Feedback (ICML 2021)
    :link: https://arxiv.org/abs/2106.05203
    """
    def __init__(self, compressor: Compressor = Compressor(), unit: Literal['B', 'KB', 'MB', 'GB'] = 'MB', is_init_compressed: bool = False): 
        """EF21 Communicator for Federated Learning

        :param compressor: compressor for model parameters, defaults to Compressor()
        :param unit: unit for communication cost, defaults to 'MB' 
        :param is_init_compressed: whether the initial gradient is compressed, defaults to False
        """
        super().__init__(compressor, unit)
        self.g: Dict[str, torch.Tensor] = None
        self.is_init_compressed = is_init_compressed

    @torch.no_grad()
    def __call__(self, present_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.g is None: 
            # first call, initialize g with present update
            if not self.is_init_compressed:
                self.g = {name: val.clone().detach() for name, val in present_update.items()}
                self.communication_cost += self.compressor.compute_communication_cost(*[val for val in present_update.values()])
                return present_update
            else:
                self.g = {name: torch.zeros_like(val) for name, val in present_update.items()}
        
        # compress the difference between current update and previous gradient
        compressed_item = {}
        for name in present_update.keys(): 
            compressed_item[name] = present_update[name] - self.g[name]

        compressed_item = self.compressor.compress(compressed_item)
        self.communication_cost += self.compressor.communication_cost
        decompressed = self.compressor.decompress(compressed_item)

        # update the compressed gradient
        for name in present_update.keys(): 
            self.g[name] += decompressed[name]

        return {key: val.clone().detach() for key, val in self.g.items()}
    
    def reset(self):
        super().reset()
        self.g: Dict[str, torch.Tensor] = None


class EF21Plus(EF21): 
    def __init__(self, compressor: Compressor = Compressor(), unit: Literal['B', 'KB', 'MB', 'GB'] = 'MB', is_init_compressed: bool = False):
        """EF21 Plus Communicator for Federated Learning

        :param compressor: compressor for model parameters, defaults to Compressor()
        :param unit: unit for communication cost, defaults to 'MB'
        :param is_init_compressed: whether the initial gradient is compressed, defaults to False
        """
        super().__init__(compressor, unit, is_init_compressed)
        
    @torch.no_grad()
    def __call__(self, present_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.g is None: 
            # first call, initialize g with present update
            if not self.is_init_compressed:
                self.g = {name: val.clone().detach() for name, val in present_update.items()}
                self.communication_cost += self.compressor.compute_communication_cost(*[val for val in present_update.values()])
                return present_update
            else:
                self.g = {name: torch.zeros_like(val) for name, val in present_update.items()}

        # compress the gradient itself
        b_compressed_item = {}
        for name in present_update.keys():
            b_compressed_item[name] = present_update[name]

        b_compressed_item = self.compressor.compress(b_compressed_item)
        b_communication_cost = self.compressor.communication_cost
        b_decompressed = self.compressor.decompress(b_compressed_item)

        # compress the difference between current update and previous gradient
        m_compressed_item = {}
        for name in present_update.keys(): 
            m_compressed_item[name] = present_update[name] - self.g[name]

        m_compressed_item = self.compressor.compress(m_compressed_item)
        m_communication_cost = self.compressor.communication_cost
        m_decompressed = self.compressor.decompress(m_compressed_item)

        # update the compressed gradient
        for name in present_update.keys(): 
            m_decompressed[name] += self.g[name]
        
        b_flat = torch.cat([b_decompressed[name].reshape(-1) for name in b_decompressed.keys()])
        m_flat = torch.cat([m_decompressed[name].reshape(-1) for name in m_decompressed.keys()])
        update_flat = torch.cat([present_update[name].reshape(-1) for name in present_update.keys()])
        B = torch.norm(b_flat - update_flat, p=2)
        M = torch.norm(m_flat - update_flat, p=2)

        if M <= B: 
            # print(f"Gradient Difference")
            self.g = m_decompressed
            self.communication_cost += m_communication_cost
        else:
            # print(f"Gradient Compression")
            self.g = b_decompressed
            self.communication_cost += b_communication_cost

        return {key: val.clone().detach() for key, val in self.g.items()}


class EF21Momentum(EF21):
    """EF21 with Momentum
    
    :paper: Momentum Provably Improves Error Feedback! (NIPS 2023)
    :link: https://arxiv.org/abs/2305.15155
    """
    def __init__(self, compressor: Compressor = Compressor(), unit: Literal['B', 'KB', 'MB', 'GB'] = 'MB', is_init_compressed: bool = False, momentum: float = 0.9):
        """EF21 Momentum Communicator for Federated Learning

        :param compressor: compressor for model parameters, defaults to Compressor()
        :param unit: unit for communication cost, defaults to 'MB'
        :param is_init_compressed: whether the initial gradient is compressed, defaults to False
        :param momentum: momentum factor, defaults to 0.9
        """
        super().__init__(compressor, unit, is_init_compressed)
        self.v: Dict[str, torch.Tensor] = None
        self.momentum = momentum

    @torch.no_grad()
    def __call__(self, present_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.g is None: 
            # first call, initialize g and v with present update
            self.v = {name: val.clone().detach() for name, val in present_update.items()}
            if not self.is_init_compressed:
                self.g = {name: val.clone().detach() for name, val in present_update.items()}
                self.communication_cost += self.compressor.compute_communication_cost(*[val for val in present_update.values()])
                return present_update
            else: 
                self.g = {name: torch.zeros_like(val) for name, val in present_update.items()}

        # update momentum estimator
        for name in present_update.keys():
            self.v[name] = self.momentum * self.v[name] + (1 - self.momentum) * present_update[name]

        # compress the difference between current update and previous gradient
        compressed_item = {}
        for name in present_update.keys():
            compressed_item[name] = self.v[name] - self.g[name]
        compressed_item = self.compressor.compress(compressed_item)
        self.communication_cost += self.compressor.communication_cost
        decompressed = self.compressor.decompress(compressed_item)

        # update the compressed gradient
        for name in present_update.keys():
            self.g[name] += decompressed[name]
        
        return {key: val.clone().detach() for key, val in self.g.items()}
    
    def reset(self):
        super().reset()
        self.v: Dict[str, torch.Tensor] = None


class EF21DoubleMomentum(EF21Momentum):
    """EF21 with Double Momentum
    
    :paper: Momentum Provably Improves Error Feedback! (NIPS 2023)
    :link: https://arxiv.org/abs/2305.15155
    """
    def __init__(self, compressor: Compressor = Compressor(), unit: Literal['B', 'KB', 'MB', 'GB'] = 'MB', is_init_compressed: bool = False, momentum: float = 0.9):
        super().__init__(compressor, unit, is_init_compressed, momentum)
        self.u: Dict[str, torch.Tensor] = None

    @torch.no_grad()
    def __call__(self, present_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.g is None: 
            # first call, initialize g and v with present update
            self.v = {name: val.clone().detach() for name, val in present_update.items()}
            self.u = {name: val.clone().detach() for name, val in present_update.items()}
            if not self.is_init_compressed:
                self.g = {name: val.clone().detach() for name, val in present_update.items()}
                self.communication_cost += self.compressor.compute_communication_cost(*[val for val in present_update.values()])
                return present_update
            else: 
                self.g = {name: torch.zeros_like(val) for name, val in present_update.items()}

        # update momentum estimator
        for name in present_update.keys():
            self.v[name] = self.momentum * self.v[name] + (1 - self.momentum) * present_update[name]
            self.u[name] = self.momentum * self.u[name] + (1 - self.momentum) * self.v[name]

        # compress the difference between current update and previous gradient
        compressed_item = {}
        for name in present_update.keys():
            compressed_item[name] = self.u[name] - self.g[name]
        compressed_item = self.compressor.compress(compressed_item)
        self.communication_cost += self.compressor.communication_cost
        decompressed = self.compressor.decompress(compressed_item)

        # update the compressed gradient
        for name in present_update.keys():
            self.g[name] += decompressed[name]
        
        return {key: val.clone().detach() for key, val in self.g.items()}
    
    def reset(self):
        super().reset()
        self.u: Dict[str, torch.Tensor] = None
        

class EControl(Communicator):
    """EControl
    
    :paper: EControl: Fast Distributed Optimization with Compression and Error Control (ICLR 2024)
    :link: https://arxiv.org/abs/2311.05645
    """
    def __init__(self, compressor: Compressor = Compressor(), unit: Literal['B', 'KB', 'MB', 'GB'] = 'MB', is_init_compressed: bool = False, eta: float = 0.001):
        """EControl Communicator for Federated Learning

        :param compressor: compressor for model parameters, defaults to Compressor()
        :param unit: unit for communication cost, defaults to 'MB'
        :param is_init_compressed: whether the initial gradient is compressed, defaults to False
        :param eta: error control parameter, defaults to 0.1
        """
        super().__init__(compressor, unit)
        self.is_init_compressed = is_init_compressed
        self.eta = eta
        self.g: Dict[str, torch.Tensor] = None
        self.error: Dict[str, torch.Tensor] = None

    @torch.no_grad()
    def __call__(self, present_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.error is None: 
            # first call, initialize error with present update
            self.error = {name: torch.zeros_like(val) for name, val in present_update.items()}
            if not self.is_init_compressed:
                self.g = {name: val.clone().detach() for name, val in present_update.items()}
                self.communication_cost += self.compressor.compute_communication_cost(*[val for val in present_update.values()])
                return present_update
            else: 
                self.g = {name: torch.zeros_like(val) for name, val in present_update.items()}

        # compress present gradient + error - previous gradient
        compressed_item = {}
        for name in present_update.keys(): 
            compressed_item[name] = self.eta * self.error[name] + present_update[name] - self.g[name]

        compressed_item = self.compressor.compress(compressed_item)
        self.communication_cost += self.compressor.communication_cost
        decompressed = self.compressor.decompress(compressed_item)

        # update the compressed gradient and error buffer
        for name in present_update.keys():
            self.error[name] += present_update[name] - self.g[name] - decompressed[name]
            self.g[name] += decompressed[name]

        return {key: val.clone().detach() for key, val in self.g.items()}
    
    def reset(self):
        super().reset()
        self.g: Dict[str, torch.Tensor] = None
        self.error: Dict[str, torch.Tensor] = None
        

class EFFACE(EControl): 
    def __init__(self, compressor: Compressor = Compressor(), unit: Literal['B', 'KB', 'MB', 'GB'] = 'MB', is_init_compressed: bool = False, eta: float = 0.001):
        """EF communicator for Federated unleArning ComprEssion

        :param compressor: compressor for model parameters, defaults to Compressor()
        :param unit: unit for communication cost, defaults to 'MB'
        :param is_init_compressed: whether the initial gradient is compressed, defaults to False
        """
        super().__init__(compressor, unit, is_init_compressed, eta)
        self.eta_b = eta
        self.eta_m = eta
        
    @torch.no_grad()
    def __call__(self, present_update: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.g is None: 
            # first call, initialize g with present update
            self.error = {name: torch.zeros_like(val) for name, val in present_update.items()}
            if not self.is_init_compressed:
                self.g = {name: val.clone().detach() for name, val in present_update.items()}
                self.communication_cost += self.compressor.compute_communication_cost(*[val for val in present_update.values()])
                return present_update
            else:
                self.g = {name: torch.zeros_like(val) for name, val in present_update.items()}

        # compress the gradient itself
        b_compressed_item = {}
        for name in present_update.keys():
            b_compressed_item[name] = present_update[name] + self.eta_b * self.error[name]

        b_compressed_item = self.compressor.compress(b_compressed_item)
        b_communication_cost = self.compressor.communication_cost
        b_decompressed = self.compressor.decompress(b_compressed_item)

        # compress the difference between current update and previous gradient
        m_compressed_item = {}
        for name in present_update.keys(): 
            m_compressed_item[name] = present_update[name] - self.g[name] + self.eta_m * self.error[name]

        m_compressed_item = self.compressor.compress(m_compressed_item)
        m_communication_cost = self.compressor.communication_cost
        m_decompressed = self.compressor.decompress(m_compressed_item)

        # update the compressed gradient
        for name in present_update.keys(): 
            m_decompressed[name] += self.g[name]

        b_flat, m_flat = {}, {}

        for name in present_update.keys():
            # new version of selection mechanism of EFFACE: | u - \eta * error - current_update |
            b_flat[name] = b_decompressed[name] - self.eta_b * self.error[name]
            m_flat[name] = m_decompressed[name] - self.eta_m * self.error[name]
        
        b_flat = torch.cat([b_flat[name].reshape(-1) for name in b_flat.keys()])
        m_flat = torch.cat([m_flat[name].reshape(-1) for name in m_flat.keys()])
        
        # b_flat = torch.cat([b_decompressed[name].reshape(-1) for name in b_decompressed.keys()])
        # m_flat = torch.cat([m_decompressed[name].reshape(-1) for name in m_decompressed.keys()])
        
        update_flat = torch.cat([present_update[name].reshape(-1) for name in present_update.keys()])
        B = torch.norm(b_flat - update_flat, p=2)
        M = torch.norm(m_flat - update_flat, p=2)

        if M <= B: 
            # print(f"Gradient Difference")
            self.g = m_decompressed
            for name in self.g.keys():
                self.error[name] += present_update[name] - m_decompressed[name]
            self.communication_cost += m_communication_cost
        else:
            # print(f"Gradient Compression")
            self.g = b_decompressed
            for name in self.g.keys():
                self.error[name] += present_update[name] - b_decompressed[name]
            self.communication_cost += b_communication_cost

        return {key: val.clone().detach() for key, val in self.g.items()}