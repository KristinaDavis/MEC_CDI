"""
test_ptp.py
Kristina Davis
7/3/20

A mashup of code from the CDI module of MEDIS with compatability with pyMILK for writing to the shared memory buffer of
CACAO on SCExAO. pyMILK interface tested with beambar.py used for beammapping using the DM.

This module creates the 2D DM maps to be sent to the shared memory buffer of CACAO which interfaces using the
pyMilk library, specifically the SHM module. SHM then interfaces with the  ImageStreamIOWrap sub-module, which
does the bulk of the cython-python formatting of numpy into the correct C-type struct.

Most of the functionality for pyMilk can be found on the README
https://github.com/milk-org/pyMilk
or else in the SHM code itself
https://github.com/milk-org/pyMilk/blob/master/pyMilk/interfacing/isio_shmlib.py

"""
import numpy as np
import os
import warnings
import time
import pickle
import datetime
import matplotlib.pyplot as plt

from pyMilk.interfacing.isio_shmlib import SHM
from cdi_plots import plot_quick_coord_check, plot_probe_cycle, plot_probe_response

import matplotlib
# matplotlib.use('QT5Agg')
# plt.ion()


######################################################
# CDI
######################################################
class Slapper:
    """Stole this idea from falco. Just passes an object you can slap stuff onto"""
    pass


class CDI_params():
    """
    contains the parameters of the CDI probes and phase sequence to apply to the DM
    """
    def __init__(self):
        # General
        self.plot = True  # False , flag to plot phase probe or not
        self.save_to_disk = True

        self.probe_time = 1 # [s]  duration of each probe
        # time between repeating probe cycles (data to be nulled using probe info)
        self.end_probes_after_time = 60 # 60 * 4.5  # [sec] probing repeats for x seconds until stopping
        self.end_probes_after_ncycles = 800  # [int] probe repeats until it has completed x full cycles
        self.n_probes = 50  # need this just to save the probes--placeholder

        self.time_limit = self.probe_time * self.end_probes_after_ncycles
        if self.time_limit > self.end_probes_after_time:
            self.time_limit = self.end_probes_after_time
            warnings.warn(f'Ending CDI probes after {self.end_probes_after_time:0.2f} sec rather than after '
                          f'{self.end_probes_after_ncycles} cycles')

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __name__(self):
        return self.__str__().split(' ')[0].split('.')[-1]

    def init_out(self, nact):
        """
        Initialize a return structure to save probes and timeseries of commands sent

        Output structure contains:
        DM_cmd_cycle: [n_probes+1, n_act, n_act] series of applied DM probes, each with a different phase (plus last flat)

        """
        out = Slapper()
        out.probe = Slapper()
        out.ts = Slapper()

        # Timeseries Info
        out.ts.start = 0
        out.ts.probe_time = self.probe_time
        out.ts.elapsed_time = 0
        out.ts.n_cycles = 0
        out.ts.n_cmds = self.end_probes_after_ncycles
        # print(f'{cout.n_commands} = cout.n_commands')
        out.ts.cmd_tstamps = np.zeros((out.ts.n_cmds,),  dtype='datetime64[ns]')

        out.probe.DM_cmd_cycle = np.zeros((self.n_probes + 1, nact, nact))

        return out

    def save_probe(self, out, ix, probe):
        if ix <= self.n_probes:
            out.probe.DM_cmd_cycle[ix] = probe


    def save_tseries(self, out, it, t):
        out.ts.cmd_tstamps[it] = t

    def save_out_to_disk(self, out, save_location='CDI_tseries'):
        """
        saves output structure data locally on disk as a pkl file

        :param out: output structure
        :param save_location: file name of saved .pkl data
        :param plot: flag to generate plots
        :return: nothing returned explicitly but data is saved to disk at save_location
        """
        #
        nw = datetime.datetime.now()
        save_location = save_location + f"_{nw.month}-{nw.day}-{nw.year}_T{nw.hour}:{nw.minute}.pkl"
        print(f'saving file {save_location}')
        with open(save_location, 'wb') as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


def config_probe(cdi, nact):
    """
    create a probe shape to apply to the DM for CDI processing

    The probe applied to the DM to achieve CDI is that originally proposed in Giv'on et al 2011, doi: 10.1117/12.895117;
    and was used with proper in Matthews et al 2018, doi:  10.1117/1.JATIS.3.4.045001.

    The pupil coordinates used in those equations relate to the sampling of the pupil (plane of the DM). However, in
    Proper, the prop_dm uses a DM map that is the size of (n_ao_act, n_ao_act). This map was resampled from its
    original size to the actuator spacing, and the spacing in units of [m] is supplied as a keyword to the prop_dm
    function during the call. All that is to say, we apply the CDI probe using the coordinates of the DM actuators,
    and supply the probe height as an additive height to the DM map, which is passed to the prop_dm function.

    :param theta: phase of the probe
    :param nact: number of actuators in the mirror, should change if 'woofer' or 'tweeter'
    :param : ncycle
    :return: height of phase probes to add to the DM map in adaptive.py
    """
    x = np.linspace(-1/2, 1/2, nact, dtype=np.float32)
    y = np.linspace(-1/2, 1/2, nact, dtype=np.float32)
    X,Y = np.meshgrid(x, y)

    probe = 0.5 * np.sinc(50 * Y) \
            * np.cos(2*np.pi * X)  # np.sinc(cdi.probe_w * X) *

    return probe


######################################################
# Interfacing with SCExAO DM
######################################################
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


def MEC_CDI():
    """
    interfaces with the shared memory buffer of remote DM

    Here we create the interface between the shm and this code to apply new offsets for the remote DM. The interfacing
    is handled by pyMILK found https://github.com/milk-org/pyMilk.

    We also create the probe by calling config_probe, and send that offset map to the shm in a loop based on the CDI
    settings in CDI_params. We read in the time the probe pattern was applied, and save it and other cdi params to a
    structure called out.

    You can set both cdi.end_probes_after_ncycles and cdi.end_probes_after_time, and the loop will break at the shorter
    of the two, but if it hits the time limit it finishes the last full cycle before breaking (eg will wait the full
    length to complete the probe cycle and null time if the cdi.end_probes_after_time hits in the middle of a cycle).

    :return: nothing explicitly returned but probe is applied (will persist on DM until it is externally cleared,
            eg by the RTC computer on SCExAO). DM flat is sent as last command after cycle is terminated.
    """
    # Create shared memory (shm) interface
    sp = ShmParams()
    MECshm = SHM(sp.shm_name)  # create c-type interface using pyMilk's ISIO wrapper
    data = MECshm.get_data()  # used to determine size of struct (removes chance of creating wrong probe size)

    # Initialize
    cdi = CDI_params()
    out = cdi.init_out(data.shape[0])

    # Create Flat
    flat = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
    MECshm.set_data(flat)  # Apply flat probe to get start time
    pt_start = MECshm.IMAGE.md.lastaccesstime  # probe time start

    # Run Probe Cycles
    n_cycles = 0
    n_cmd = 0
    while n_cycles < cdi.end_probes_after_ncycles:
        olt = datetime.datetime.now()  # outer loop time
        if (olt-pt_start).total_seconds() > cdi.time_limit:
            print('Max time Elapsed. Ending probe cycles')
            break

        probe = config_probe(cdi, nact=data.shape[0])
        # Send flat and wait until next cycle begins
        MECshm.set_data(probe)  #
        pt_sent = MECshm.IMAGE.md.lastaccesstime  # probe time sent
        cdi.save_tseries(out, n_cmd, pt_sent)
        cdi.save_probe(out, n_cmd, probe)
        n_cmd += 1
        time.sleep(n_cycles+1)

        # Send flat
        MECshm.set_data(flat)  #
        pt_sent = MECshm.IMAGE.md.lastaccesstime  # probe time sent
        cdi.save_tseries(out, n_cmd, pt_sent)
        cdi.save_probe(out, n_cmd, flat)
        n_cmd += 1
        time.sleep(2)
        n_cycles += 1
        # while True:
        #     if (datetime.datetime.now() - pt_sent).total_seconds() > cdi.null_time:
        #         # print(f"n_cmd = {n_cmd}\n "
        #         #       f'cycle time = {(datetime.datetime.now() - olt).total_seconds()}s vs expected '
        #         #       f'probe+null time= {cdi.time_probe_plus_null:.2f}s')
        #         n_cycles += 1
        #         break

    # Wrapping up
    print(f'\ntotal time elapsed = {(datetime.datetime.now()-pt_start).total_seconds():.2f} sec')
    out.ts.start = pt_start
    out.ts.n_cycles = n_cycles
    out.ts.n_cmds = n_cmd
    out.ts.cmd_tstamps = out.ts.cmd_tstamps[0:n_cmd]
    out.ts.elapsed_time = (datetime.datetime.now() - pt_start).total_seconds()

    # Make sure shm is clear
    MECshm.set_data(flat)

    # Saving Probe and timestamp together
    if cdi.save_to_disk:
        cdi.save_out_to_disk(out)

    # Fig
    if cdi.plot:
        # plot_probe_cycle(out)
        # plot_probe_response_cycle(out)
        # plot_probe_response(out, 0)
        plot_quick_coord_check(out, 2)
        plt.show()

    return MECshm, out


def send_flat(channel):
    """
    sends a DM flat (array of 0's) to the DM channel specified

    SCExAO DM channel names have the format "dm00dispXX" where XX is the channel number. Generally CDI probes or any
    active nulling takes place on channel 06 or 07. I think 00 is the 'observed' DM flat (may not be an array of 0s
    due to voltage anomolies on the DM itself).

    :return: nothing explicitly returned but probe is applied (will persist on DM until it is externally cleared,
            eg by the RTC computer on SCExAO).
    """
    # Create shared memory (shm) interface
    sp = ShmParams()
    # channel = sp.shm_name
    MECshm = SHM(channel)  # create c-type interface using pyMilk's ISIO wrapper
    data = MECshm.get_data()  # used to determine size of struct (removes chance of creating wrong probe size)

    # Create Flat
    flat = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
    MECshm.set_data(flat)  # Apply flat probe to get start time

##
def read_data(cdi_zip):
    """

    :param cdi_zip:
    :return:
    """
    dm_file = '/work/kkdavis/src/MEC_CDI/CDI_tseries_11-16-2021_T12:31.pkl'

    with open(dm_file, 'rb') as handle:
        cdi_zip = pickle.load(handle)

    tstamps_as_unix = (cdi_zip.ts.cmd_tstamps.astype('float64') / 1e9)
    tstamps_from_file_start = tstamps_as_unix - tstamps_as_unix[0]

    print(f'\nDM Probe Series Info\n\t'
              f'Timing: Probe Integration {cdi_zip.ts.probe_time} sec\n\t'
              f'# Cycles: {cdi_zip.ts.n_cycles}, # Commands: {cdi_zip.ts.n_cmds} \n\t'
              f'Total Elapsed Time: {cdi_zip.ts.elapsed_time/60:.2f} min ({cdi_zip.ts.elapsed_time:.4f} sec)')
    print(f'Timestamps for probe/flat: {tstamps_from_file_start}')
##
if __name__ == '__main__':
    print(f"\nTesting CDI probe command cycle\n")
    mecshm, out = MEC_CDI()
    # plot_quick_coord_check(out, 2)
    send_flat('dm00disp06')



##

