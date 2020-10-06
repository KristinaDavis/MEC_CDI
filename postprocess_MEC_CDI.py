"""
postprocess_MEC_CDI
Kristina Davis
Aug 26 2020


"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import os
import pickle
import time
import datetime
import pytz

from mec_cdi import CDI_params, Slapper
# out = Slapper()
# out.probe = Slapper()
# out.ts = Slapper()


def open_MEC_tseries(CDI_tseries='CDI_tseries.pkl'):
    """opens existing MEC CDI timeseries .pkl file and return it"""
    with open(CDI_tseries, 'rb') as handle:
        CDI_meta = pickle.load(handle)
    return CDI_meta


def first_tstep(meta):
    """returns the first timestep time from pkl file. This is useful to tell the mkidpipeline when to start the obs"""
    first_t = meta.ts.cmd_tstamps[-1]   # [0]
    return (first_t - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')


def last_tstep(meta):
    """returns the end timestep time from pkl file. This is useful to tell the mkidpipeline when to stop the obs"""
    last_t = meta.ts.cmd_tstamps[-3]   #  [-1]
    return (last_t - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')


def cdi_postprocess(fp_seq, meta, plot=False):
    """
    this is the function that accepts the timeseries of intensity images from the simulation and returns the processed
    single image. This function calculates the speckle amplitude phase, and then corrects for it to create the dark
    hole over the specified region of the image.

    :param fp_seq: focal plane sequence (temporal cube) [nx, ny, time]
    :param sampling: focal plane sampling
    :return:
    """
    print(f'\nStarting CDI post-processing')
    st = time.time()

    nx = fp_seq.shape[0]
    ny = fp_seq.shape[1]

    # Defining Matrices
    n_pairs = meta.ts.n_probes // 2  # number of deltas (probe differentials)
    n_nulls = meta.ts.n_cmds - meta.ts.n_probes
    delta = np.zeros((n_pairs, nx, ny), dtype=float)
    Epupil = np.zeros((n_nulls*2, nx, ny), dtype=complex)
    H = np.zeros((n_pairs, 2), dtype=float)
    b = np.zeros((n_pairs, 1))

    # Differance Images (delta)
    for ip in range(n_pairs):
        delta[ip] = np.copy(fp_seq[:,:,ip] - fp_seq[:,:,ip + n_pairs])

    for i in range(nx):
            for j in range(ny):

                for xn in range(n_nulls):
                    for ip in range(n_pairs):
                        absDeltaP = np.abs((fp_seq[i,j,ip] + fp_seq[i,j,ip + n_pairs]) / 2
                                           - fp_seq[i,j,meta.ts.n_probes+xn])
                        absDeltaP = np.sqrt(absDeltaP)
                        # probe_ft = (1/np.sqrt(2*np.pi)) *
                        # np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(cdi.cout.DM_probe_series[ix])))
                        phsDeltaP = np.arctan2(fp_seq[i,j,ip].imag - fp_seq[meta.ts.n_probes+xn,i,j].imag,
                                               fp_seq[ip,i,j].real - fp_seq[meta.ts.n_probes+xn,i,j].real)
                        cpxDeltaP = absDeltaP*np.exp(1j*phsDeltaP)

                        H[ip, :] = [-cpxDeltaP.imag, cpxDeltaP.real]
                        b[ip] = delta[ip,i,j]

                    a = 2 * H
                    Exy, res, rnk, s = sl.lstsq(a, b)  # returns tuple, not array
                    Epupil[xn, i, j] = Exy[0] + (1j * Exy[1])
                    cats = 5

    et = time.time()
    print(f'\tCDI post-processing for {n_nulls} null-images finished in {et-st:.1f} sec == {(et-st)/60:.2f} min')

    if plot:
        ####################
        # Difference Images
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

        ##############################
        # Focal Plane E-field Estimate
        fig, subplot = plt.subplots(1, n_nulls, figsize=(14, 5))
        fig.subplots_adjust(wspace=0.5, right=0.85)

        fig.suptitle('Estimated Focal-Plane E-field')

        for ax, ix in zip(subplot, range(n_nulls)):
            im = ax.imshow(np.abs(Epupil[ix])**2 * 1e6, interpolation='none', origin='lower',
                           norm=LogNorm())
            ax.set_title(f'Estimation timestep {ix}')

        cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        # cb.set_label('Intensity')

        plt.show()

        #####################
        # Subtracted E-field
        fig, subplot = plt.subplots(1, n_nulls, figsize=(14, 5))
        fig.subplots_adjust(wspace=0.05, right=0.85)

        fig.suptitle('Subtracted E-field')

        for ax, ix in zip(subplot, range(n_nulls)):
            im = ax.imshow(I_fp[ix]-np.abs(Epupil[ix]) ** 2, interpolation='none', origin='lower')#,
                           # norm=LogNorm())
            ax.set_title(f'Estimation timestep {ix}')

        cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        # cb.set_label('Intensity')

        plt.show()


if __name__ == '__main__':
    # dm_header = open_MEC_tseries('/home/captainkay/mazinlab/MEC_data/CDI_tseries3-9-2020_hour0_min39.pkl')
    dm_meta = open_MEC_tseries('/work/kkdavis/scratch/CDI_tseries_3-9-2020_hour0_min11.pkl')

    print(f'First Timestep = {first_tstep(dm_meta):.0f}\nLast Timestep = {last_tstep(dm_meta):.0f}')

    fp_seq = np.load('cdi1_timeCube3.npy', allow_pickle=True)
    grail = cdi_postprocess(fp_seq, dm_meta, plot=True)

    dumm=0
