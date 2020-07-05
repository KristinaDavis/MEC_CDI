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
import datetime
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import copy
from pyMilk.interfacing.isio_shmlib import SHM


######################################################
# Interfacing
######################################################
class ShmParams():
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
    is handled by pyMILK found https://github.com/milk-org/pyMilk. We also create the probe by calling config_probe, and
    send that offset map to the shm. We read in the time the probe pattern was applied. Functinality exists to save
    the probe pattern and timestamp together

    #TODO need to figure out how to save this

    :return: nothing explicitly returned but probe is applied (will persist on DM until it is externally cleared,
            eg by the RTC computer on SCExAO). Saving capability not currently implemented.
    """
    # Create shared memory (shm) interface
    sp = ShmParams()
    MECshm = SHM(sp.shm_name)  # create c-type interface using pyMilk's ISIO wrapper
    data = MECshm.get_data()  # used to determine size of struct (removes chance of creating wrong probe size)

    # Initialize
    cdi = CDI_params()
    cdi.gen_phaseseries()
    out = cdi.init_cout(data.shape[0])

    # Create Flat
    flat = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
    MECshm.set_data(flat)  # Apply flat probe to get start time
    pt_start = MECshm.IMAGE.md.lastaccesstime  # probe time start

    # Run Probe Cycles
    n_cycles = 0
    n_cmd = 0
    while n_cycles <= cdi.end_probes_after_ncycles:
        olt = datetime.datetime.now()  # outer loop time
        for ip, theta in enumerate(cdi.phase_cycle):
            ilt = datetime.datetime.now()  # inner loop time
            probe = config_probe(cdi, theta, data.shape[0])
            MECshm.set_data(probe)  # Apply Probe
            pt_sent = MECshm.IMAGE.md.lastaccesstime      # probe time sent
            cdi.save_tseries(out, n_cmd, pt_sent)
            print(f'ncmd={n_cmd}, last access time = {pt_sent}, ip={ip} ')
            n_cmd += 1

            while True:
                if (datetime.datetime.now() - ilt).seconds > cdi.phase_integration_time:
                    break
        # Send flat and wait until next cycle begins
        MECshm.set_data(flat)  # effectively clearing SHM
        pt_sent = MECshm.IMAGE.md.lastaccesstime  # probe time sent
        cdi.save_tseries(out, n_cmd, pt_sent)
        n_cmd += 1

        while True:
            if (time.time() + pt_sent.second) > cdi.time_probe_plus_null:
                n_cycles += 1
                break
    print(f'total time = {pt_sent-pt_start}')

    # Saving Probe and timestamp together
    # ap = AppliedProbe(bp, probe, t_sent)

    return MECshm, out

######################################################
# CDI
######################################################

class CDIOut():
    '''Stole this idea from falco. Just passes an object you can slap stuff onto'''
    pass


class CDI_params():
    """
    contains the parameters of the CDI probes and phase sequence to apply to the DM
    """
    def __init__(self):
        # General
        self.verbose = False  # False , flag to plot phase probe or not

        # Probe Dimensions (extent in pupil plane coordinates)
        self.probe_amp = 2e-6  # [m] probe amplitude, scale should be in units of actuator height limits
        self.probe_w = 10  # [actuator coordinates] width of the probe
        self.probe_h = 30  # [actuator coordinates] height of the probe
        self.probe_shift = [0, 0]  # [actuator coordinates] center position of the probe (should move off-center to
        # avoid coronagraph)
        self.probe_spacing = 10  # distance from the focal plane center to edge of the rectangular probed region

        # Phase Sequence of Probes
        self.phs_intervals = np.pi / 3  # [rad] phase interval over [0, 2pi]
        self.phase_integration_time = 0.1  # [s]  How long in sec to apply each probe in the sequence
        self.null_time = 1  # [s]  time between repeating probe cycles (data to be nulled using probe info)
        self.end_probes_after_time = 60  # [sec] probing repeats for x seconds until stopping
        self.end_probes_after_ncycles = 3  # [int] probe repeats until it has completed x full cycles

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
        self.total_time = self.time_probe_plus_null * self.end_probes_after_ncycles
        if self.total_time > self.end_probes_after_time:
            self.total_time = self.end_probes_after_time
            warnings.warn(f'Ending CDI probes after non-integer number of cycles')

        return self.phase_cycle

    def init_cout(self, nact):
        """
        Initialize a return structure to save probes and timeseries of commands sent

        Output structure contains:
        DM_probe_series: [n_probes, n_act, n_act] series of applied DM probes, each with a different phase

        """
        cout = CDIOut()
        cout.phase_cycle = self.phase_cycle
        cout.phase_integration_time = self.phase_integration_time
        cout.DM_probe_series = np.zeros((self.n_probes, nact, nact))
        cout.total_time = self.total_time
        cout.n_commands = (self.n_probes+1) * self.end_probes_after_ncycles + 1
        # print(f'{cout.n_commands} = cout.n_commands')
        cout.probe_tseries = np.zeros((cout.n_commands,),  dtype='datetime64[s]')

        return cout

    def save_probe(self, cout, ix, probe):
        cout.DM_probe_series[ix] = probe

    def save_tseries(self, cout, it, t):
        cout.probe_tseries[it-1] = t

    def save_cout_to_disk(self, cout, plot=False):


        # Fig
        if plot:
            if self.n_probes >= 4:
                nrows = 2
                ncols = self.n_probes//2
                figheight = 5
            else:
                nrows = 1
                ncols = self.n_probes
                figheight = 12

            fig, subplot = plt.subplots(nrows, ncols, figsize=(12, figheight))
            fig.subplots_adjust(wspace=0.5)

            fig.suptitle('Probe Series')

            for ax, ix in zip(subplot.flatten(), range(self.n_probes)):
                # im = ax.imshow(self.DM_probe_series[ix], interpolation='none', origin='lower')
                im = ax.imshow(cout.probe_series[ix], interpolation='none', origin='lower')

                ax.set_title(f"Probe " + r'$\theta$=' + f'{cout.DM_phase_series[ix]/np.pi:.2f}' + r'$\pi$')

            cb = fig.colorbar(im)  #
            cb.set_label('um')

            plt.show()


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
    :param iw: index of wavelength number in ap.wvl_range
    :return: height of phase probes to add to the DM map in adaptive.py
    """
    x = np.linspace(-1/2-cdi.probe_shift[0]/nact, 1/2-cdi.probe_shift[0]/nact, nact, dtype=np.float32)
    y = np.linspace(-1/2-cdi.probe_shift[1]/nact, 1/2-cdi.probe_shift[1]/nact, nact, dtype=np.float32)
    X,Y = np.meshgrid(x, y)

    probe = cdi.probe_amp * np.sinc(cdi.probe_w * X) * np.sinc(cdi.probe_h * Y) \
            * np.sin(2*np.pi*cdi.probe_spacing*X + theta)

    # Testing FF propagation
    if cdi.verbose:  # and theta == cdi.phase_series[0]
        probe_ft = (1/np.sqrt(2*np.pi)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(probe)))

        # fig, ax = plt.subplots(1, 3, figsize=(12, 5))
        # fig.subplots_adjust(wspace=0.5)
        # ax1, ax2, ax3 = ax.flatten()
        #
        # fig.suptitle(f"spacing={cdi.probe_spacing}, Dimensions {cdi.probe_w}x{cdi.probe_h} "
        #              f"\nProbe Amp = {cdi.probe_amp}, " + r'$\theta$' + f"={theta/np.pi:.3f}" + r'$\pi$' + '\n')
        #
        # im1 = ax1.imshow(probe, interpolation='none', origin='lower')
        # ax1.set_title(f"Probe on DM \n(dm coordinates)")
        # cb = fig.colorbar(im1, ax=ax1)
        #
        # im2 = ax2.imshow(np.sqrt(probe_ft.imag ** 2 + probe_ft.real ** 2), interpolation='none', origin='lower')
        # ax2.set_title("Focal Plane Amplitude")
        # cb = fig.colorbar(im2, ax=ax2)
        #
        # ax3.imshow(np.arctan2(probe_ft.imag, probe_ft.real), interpolation='none', origin='lower', cmap='hsv')
        # ax3.set_title("Focal Plane Phase")

        # plt.show()

        # Fig 2
        fig, ax = plt.subplots(1, 3, figsize=(12, 5))
        fig.subplots_adjust(wspace=0.5)
        ax1, ax2, ax3 = ax.flatten()
        fig.suptitle(f'Real & Imaginary Probe Response in Focal Plane\n'
                     f' '+r'$\theta$'+f'={theta/np.pi:.3f}'+r'$\pi$'+f', n_actuators = {nact}\n')

        im1 = ax1.imshow(probe, interpolation='none', origin='lower')
        ax1.set_title(f"Probe on DM \n(dm coordinates)")
        cb = fig.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(probe_ft.real, interpolation='none', origin='lower')
        ax2.set_title(f"Real FT of Probe")

        im3 = ax3.imshow(probe_ft.imag, interpolation='none', origin='lower')
        ax3.set_title(f"Imag FT of Probe")

        plt.show()

    # Saving Probe in the series
    # cdi.save_probe(ip, probe)

    return probe


def cdi_postprocess(cdi, fp_seq, sampling, plot=False):
    """
    this is the function that accepts the timeseries of intensity images from the simuation and returns the processed
    single image. This function calculates the speckle amplitude phase, and then corrects for it to create the dark
    hole over the specified region of the image.

    :param fp_seq: timestream of 2D images (intensity only) from the focal plane complex field
    :param sampling: focal plane sampling
    :return:
    """
    n_pairs = cdi.n_probes//2  # number of deltas (probe differentials)
    n_nulls = sp.numframes - cdi.n_probes
    delta = np.zeros((n_pairs, sp.grid_size, sp.grid_size), dtype=float)
    absDeltaP = np.zeros((n_pairs, sp.grid_size, sp.grid_size), dtype=float)
    phsDeltaP = np.zeros((n_pairs, sp.grid_size, sp.grid_size), dtype=float)
    Epupil = np.zeros((n_nulls, sp.grid_size, sp.grid_size), dtype=float)

    fp_seq = np.sum(fp_seq, axis=1)  # sum over wavelength

    for ix in range(n_pairs):
        # Compute deltas
        delta[ix] = np.copy(fp_seq[ix] - fp_seq[ix+n_pairs])

        for xn in range(n_nulls):
            print(f'for computing reDeltaP: xn={xn}')
            # Real Part of deltaP
            # absDeltaP[xn] = np.sqrt((fp_seq[ix] + fp_seq[ix+n_pairs])/2 - fp_seq[xn])
            # probe_ft = (1/np.sqrt(2*np.pi)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(cdi.cout.DM_probe_series[ix])))
            # phsDeltaP[xn] = np.arctan2(probe_ft.imag, probe_ft.real)

            # Least Squares Solution
            # Epupil[xn] = np.linalg.solve((-ImDeltaP[xn] + ReDeltaP[xn], delta[ix]))
            # duVecNby1 = -dmfac * np.linalg.solve((10 ** log10reg * np.diag(cvar.EyeGstarGdiag) + cvar.GstarG_wsum),
            #                                      cvar.RealGstarEab_wsum)
            # duVec = duVecNby1.reshape((-1,))

            # Fig 2
    if plot:
        fig, subplot = plt.subplots(1, n_pairs, figsize=(14,5))
        fig.subplots_adjust(wspace=0.5, right=0.85)

        fig.suptitle('Deltas for CDI Probes')

        for ax, ix in zip(subplot.flatten(), range(n_pairs)):
            im = ax.imshow(delta[ix]*1e6, interpolation='none', origin='lower',
                           norm=SymLogNorm(linthresh=1e-2),
                           vmin=-1, vmax=1) #, norm=SymLogNorm(linthresh=1e-5))
            ax.set_title(f"Diff Probe\n" + r'$\theta$' + f'={cdi.phase_series[ix]/np.pi:.3f}' +
                         r'$\pi$ -$\theta$' + f'={cdi.phase_series[ix+n_pairs]/np.pi:.3f}' + r'$\pi$')

        cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        cb.set_label('Intensity')

        plt.show()


if __name__ == '__main__':
    print(f"Testing CDI probe")
    mecshm, cout = MEC_CDI()

    dumm=0