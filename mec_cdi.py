"""
mec_cdi.py
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
        self.save_to_disk = False

        # Probe Dimensions (extent in pupil plane coordinates)
        self.probe_ax = 'Y'  # 'Y' # direction of the probe
        self.probe_amp = 0.2  # [um] probe amplitude, scale should be in units of actuator height limits
        self.probe_w = 30  # [actuator coordinates] width of the probe
        self.probe_h = 15  # [actuator coordinates] height of the probe
        self.probe_shift = [10, 10]  # [actuator coordinates] center position of the probe (should move off-center to
        # avoid coronagraph)
        self.probe_spacing = 8  # distance from the focal plane center to edge of the rectangular probed region

        # Phase Sequence of Probes
        self.phs_intervals = np.pi / 3  # [rad] phase interval over [0, 2pi]
        self.phase_integration_time = 0.1  # [s]  How long in sec to apply each probe in the sequence
        self.null_time = 5 * self.phase_integration_time  # [s]  # 5 * self.phase_integration_time
        # time between repeating probe cycles (data to be nulled using probe info)
        self.end_probes_after_time = 1.2 # 60 * 4.5  # [sec] probing repeats for x seconds until stopping
        self.end_probes_after_ncycles = 800  # [int] probe repeats until it has completed x full cycles

        # Check Coords
        if self.probe_ax == 'X' or self.probe_ax == 'x':
            if self.probe_h < self.probe_w:
                raise ValueError("Set probe_h > probe_w if probe_ax = X")
            if self.probe_spacing < self.probe_w/2:
                raise ValueError("probe_spacing must be greater than probe_w//2")
        elif self.probe_ax == 'Y' or self.probe_ax == 'y':
            if self.probe_h > self.probe_w:
                raise ValueError("Set probe_h > probe_w if probe_ax = Y")
            if self.probe_spacing < self.probe_h/2:
                raise ValueError("probe_spacing must be greater than probe_h//2")

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __name__(self):
        return self.__str__().split(' ')[0].split('.')[-1]

    def gen_phaseseries(self):
        """
        generate an array of phases per timestep for the CDI algorithm

        phase_series is used to populate cdi.phase_series, which may be longer than cdi.phase_cycle if multiple cycles
        are run, or probes may last for longer than one single timestep

        currently, I assume the timestream is not that long. Should only use this for short timestreams, and use a more
        efficient code for long simulations (scale of minutes or more)

        :return: phase_series  array of phases of CDI probes to apply to DM
        """
        # Number of probes
        self.phase_cycle = np.arange(0, 2 * np.pi, self.phs_intervals)  # FYI not inclusive of 2pi endpoint
        self.n_probes = len(self.phase_cycle)  # number of phase probes
        if self.n_probes % 2 != 0:
            raise ValueError(f"must have even number of phase probes\n\tchange cdi.phs_intervals")

        # Check total length of one cycle
        self.time_probe_plus_null = self.n_probes * self.phase_integration_time + self.null_time
        if self.time_probe_plus_null >= 1.0:
            pass
        else:
            raise ValueError(f"Need to reconfigure probe sequence for cycles shorter than 1 second")

        # Check total probing time
        self.time_limit = self.time_probe_plus_null * self.end_probes_after_ncycles
        if self.time_limit > self.end_probes_after_time:
            self.time_limit = self.end_probes_after_time
            warnings.warn(f'Ending CDI probes after {self.end_probes_after_time:0.2f} sec rather than after '
                          f'{self.end_probes_after_ncycles} cycles')

        return self.phase_cycle

    def init_out(self, nact):
        """
        Initialize a return structure to save probes and timeseries of commands sent

        Output structure contains:
        DM_cmd_cycle: [n_probes+1, n_act, n_act] series of applied DM probes, each with a different phase (plus last flat)

        """
        out = Slapper()
        out.probe = Slapper()
        out.ts = Slapper()

        # Probe Info
        out.probe.direction = self.probe_ax
        out.probe.amp = self.probe_amp
        out.probe.width = self.probe_w
        out.probe.height = self.probe_h
        out.probe.shift = self.probe_shift
        out.probe.spacing = self.probe_spacing
        out.probe.DM_cmd_cycle = np.zeros((self.n_probes + 1, nact, nact))
        out.probe.phs_interval = self.phs_intervals

        # Timeseries Info
        out.ts.start = 0
        out.ts.n_probes = self.n_probes
        out.ts.phase_cycle = self.phase_cycle
        out.ts.phase_integration_time = self.phase_integration_time
        out.ts.t_one_cycle = self.time_probe_plus_null
        out.ts.null_time = self.null_time
        out.ts.time_limit = self.time_limit
        out.ts.elapsed_time = 0
        out.ts.n_cycles = 0
        out.ts.n_cmds = (self.n_probes+1) * self.end_probes_after_ncycles
        # print(f'{cout.n_commands} = cout.n_commands')
        out.ts.cmd_tstamps = np.zeros((out.ts.n_cmds,),  dtype='datetime64[ns]')

        return out

    def save_probe(self, out, ix, probe):
        if ix <= self.n_probes:
            out.probe.DM_cmd_cycle[ix] = probe

            # Testing FF propagation
            # if self.verbose:
            #     plot_probe_response(out, ix)

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
        with open(save_location, 'wb') as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


def config_probe(cdi, theta, nact):
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
    x = np.linspace(-1/2-cdi.probe_shift[0]/nact, 1/2-cdi.probe_shift[0]/nact, nact, dtype=np.float32)
    y = np.linspace(-1/2-cdi.probe_shift[1]/nact, 1/2-cdi.probe_shift[1]/nact, nact, dtype=np.float32)
    X,Y = np.meshgrid(x, y)

    if cdi.probe_ax == 'X' or cdi.probe_ax == 'x':
        dir = X
    elif cdi.probe_ax == 'Y' or cdi.probe_ax == 'y':
        dir = Y
    else:
        raise ValueError('probe direction value not understood; must be string "X" or "Y"')
    probe = cdi.probe_amp * np.sinc(cdi.probe_w * X) * np.sinc(cdi.probe_h * Y) \
            * np.cos(2*np.pi*cdi.probe_spacing * dir + theta)

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
    cdi.gen_phaseseries()
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
            # print(f'elasped={(olt-pt_start).total_seconds()}, total test time={cdi.time_limit},\n'
            #       f'ncycles={n_cycles}, n_cmds={n_cmd}')
            print('Max time Elapsed. Ending probe cycles')
            break
        for ip, theta in enumerate(cdi.phase_cycle):
            ilt = datetime.datetime.now()  # inner loop time
            probe = config_probe(cdi, theta, data.shape[0])
            MECshm.set_data(probe)  # Apply Probe
            pt_sent = MECshm.IMAGE.md.lastaccesstime      # probe time sent
            cdi.save_tseries(out, n_cmd, pt_sent)
            if n_cmd <= cdi.n_probes:
                cdi.save_probe(out, n_cmd, probe)
            n_cmd += 1
            while True:
                if (datetime.datetime.now() - ilt).total_seconds() > cdi.phase_integration_time:
                    break

        # Send flat and wait until next cycle begins
        MECshm.set_data(flat)  # effectively clearing SHM
        pt_sent = MECshm.IMAGE.md.lastaccesstime  # probe time sent
        cdi.save_tseries(out, n_cmd, pt_sent)
        cdi.save_probe(out, n_cmd, flat)
        n_cmd += 1
        while True:
            if (datetime.datetime.now() - pt_sent).total_seconds() > cdi.null_time:
                # print(f"n_cmd = {n_cmd}\n "
                #       f'cycle time = {(datetime.datetime.now() - olt).total_seconds()}s vs expected '
                #       f'probe+null time= {cdi.time_probe_plus_null:.2f}s')
                n_cycles += 1
                break

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
        plot_probe_cycle(out)
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


if __name__ == '__main__':
    print(f"\nTesting CDI probe command cycle\n")
    mecshm, out = MEC_CDI()
    print(f'probe dir is {out.probe.direction}')
    plot_quick_coord_check(out, 2)
    send_flat('dm00disp06')


