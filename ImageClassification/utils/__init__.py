"""
:file: __init__.py
:date: 2025-07-15
:description: utility functions for federated unlearning
"""

from .misc import *
from .compressors import *
from .communicators import *

# from .loss_function import *

def get_communicator_class(
        communicator_name: Literal['Communicator', 'EF14', 'EF21', 'EF21Momentum', 'EF21DoubleMomentum', 'EControl', 'EF21Plus', 'EFFACE'] = 'Communicator', 
        compressor_name: Literal['Compressor', 'TopK', 'RandK', 'SignSGD', 'QSGD', 'TernGrad'] = 'Compressor', 
        k: Optional[int] = None, 
        sparsity: Optional[float] = None, 
        unit: Optional[Literal['B', 'KB', 'MB', 'GB']] = 'MB', 
        is_init_compressed: bool = False, 
        momentum: Optional[float] = 0.9, 
        eta: Optional[float] = 0.1,
        levels: Optional[int] = 16, 
        bits: Optional[int] = None, 
        stochastic: bool = True
    ) -> type[Communicator]:
    """Get the communicator class based on the name and compressor type

    :param communicator_name: Name of the communicator
    :param compressor_name: Name of the compressor
    :param k: Top-K or Rand-K for communication
    :param sparsity: Sparsity for sparsification compressor
    :return: Communicator class
    """
    # build the compressor instance based on the name and parameters
    if compressor_name == 'Compressor': 
        compressor = Compressor()
    elif compressor_name == 'TopK' or compressor_name == 'RandK':
        compressor = eval(f"{compressor_name}(k={k}, sparsity={sparsity})")
    elif compressor_name == 'SignSGD':
        compressor = SignSGD()
    elif compressor_name == 'TernGrad':
        compressor = TernGrad()
    elif compressor_name == 'QSGD':
        compressor = QSGD(levels=levels, bits=bits, stochastic=stochastic)
    else:
        raise ValueError(f"Unsupported compressor: {compressor_name}; Available compressors: ['Compressor', 'TopK', 'RandK', 'SignSGD', 'QSGD']")

    # return the appropriate communicator class based on the name
    if communicator_name == 'Uncompressed':
        return Communicator(compressor=Compressor(), unit=unit)
    elif communicator_name == 'Communicator' or communicator_name == 'Compressed':
        return Communicator(compressor=compressor, unit=unit)
    elif communicator_name == 'EF14':
        return EF14(compressor=compressor, unit=unit, is_init_compressed=is_init_compressed)
    elif communicator_name == 'EF21':
        return EF21(compressor=compressor, unit=unit, is_init_compressed=is_init_compressed)
    elif communicator_name == 'EF21Plus' or communicator_name == 'EF21+':
        return EF21Plus(compressor=compressor, unit=unit, is_init_compressed=is_init_compressed)
    elif communicator_name == 'EF21Momentum':
        return EF21Momentum(compressor=compressor, unit=unit, momentum=momentum, is_init_compressed=is_init_compressed)
    elif communicator_name == 'EF21DoubleMomentum':
        return EF21DoubleMomentum(compressor=compressor, unit=unit, momentum=momentum, is_init_compressed=is_init_compressed)
    elif communicator_name == 'EControl':
        return EControl(compressor=compressor, unit=unit, is_init_compressed=is_init_compressed, eta=eta)
    elif communicator_name == 'EFFACE':
        return EFFACE(compressor=compressor, unit=unit, is_init_compressed=is_init_compressed, eta=eta)
    else:
        raise ValueError(f"Unsupported communicator: {communicator_name}")