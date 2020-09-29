"""
dummy_shm.py
KD Sept 2020

creates a 'dummy' shm on the local machine. Default location (from pymilk) is /tmp

"""
import numpy as np
import matplotlib.pyplot as plt
from pyMilk.interfacing.isio_shmlib import SHM
import ImageStreamIOWrap as ISIO


class ShmParams:
    """
    settings related to the shared memory file in pyMilk. The SHM has the following format (documentation from
    the pyMilk SHM code itself https://github.com/milk-org/pyMilk/blob/master/pyMilk/interfacing/isio_shmlib.py

    SHM(fname: str,                     fname: name of the shm file
                                        the resulting name will be $MILK_SHM_DIR/<fname>.im.shm
        data: numpy.ndarray = None,     a numpy array (1, 2 or 3D of data)
                                        alternatively, a tuple ((int, ...), dtype) is accepted
                                        which provides shape and datatype (string or np type)
        nbkw: int = 0,                  number of keywords to be appended to the data structure (optional)
        shared: bool = True,            True if the memory is shared among users
        location: int = -1,             -1 for CPU RAM, >= 0 provides the # of the GPU.
        verbose: bool = False,          arguments "packed" and "verbose" are unused.
        packed=False,                   arguments "packed" and "verbose" are unused.
        autoSqueeze: bool = True,       Remove singleton dimensions between C-side and user side. Otherwise,
                                        we take the necessary steps to squeeze / pad the singleton dimensions.
                                        Warning: [data not None] assumes [autoSqueeze=False].
                                        If you're creating a SHM, it's assumed you know what you're passing.
        symcode: int = 4                A symcode parameter enables to switch through all transposed-flipped
                                        representations of rectangular images. The default is 4 as to provide
                                        retro-compatibility. We currently lack compatibility with fits files.
        )
    """
    def __init__(self):
        # self.shm_name = 'MECshm'  # use default cacao shm channel for speckle nulling
        self.shm_name = 'dm00disp06'  # use default cacao shm channel for speckle nulling

        self.shm_buffer = np.empty((50, 50), dtype=np.float32)  # this is where the probe pattern goes, so allocate the appropriate size
        self.location = -1  # -1 for CPU RAM, >= 0 provides the # of the GPU.
        self.shared = True  # if true then a shared memory buffer is allocated. If false, only local storage is used.


def new_shm():
    """
    creates a new shm
    WARNING: don't actually use this on the scexao_rtc. Use this to test on your own machine. default location
    is in /tmp/

    :return: img-- the image struct
    """
    sp = ShmParams()

    img = ISIO.Image()
    img.create(sp.shm_name, sp.shm_buffer, sp.location, sp.shared)
    print(f'\nCreated shared memory buffer at\n\t /tmp/{sp.shm_name}')

    return img


if __name__ == '__main__':
    img = new_shm()
