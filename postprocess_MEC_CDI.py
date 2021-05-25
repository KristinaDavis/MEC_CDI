##
"""
postprocess_MEC_CDI
Kristina Davis
Aug 26 2020


"""

import numpy as np
import os
import time
import scipy.linalg as sl
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

from mec_cdi import Slapper
from read_photons import open_MEC_tseries

##
def cdi_postprocess(fp_seq, cdi_zip, plot=False):
    """
    this is the function that accepts the timeseries of intensity images from the simulation and returns the processed
    single image. This function calculates the speckle amplitude phase, and then corrects for it to create the dark
    hole over the specified region of the image.

    :param fp_seq: focal plane sequence (temporal cube) [nx, ny, time]
    :param cdi_zip: cdi_zipdata from .pkl file from DM settings and timestream created at runtime
    :param plot: toggle to display plots or not
    :return: nothing
    """
    ##
    print(f'\nStarting CDI post-processing')
    st = time.time()

    nx = fp_seq.shape[0]
    ny = fp_seq.shape[1]
    dm_act = cdi_zip.probe.DM_cmd_cycle.shape[-1]

    # Defining Matrices
    n_pairs = cdi_zip.ts.n_probes // 2  # number of deltas (probe differentials)
    n_nulls = np.int(cdi_zip.ts.null_time / cdi_zip.ts.phase_integration_time)
    delta = np.zeros((n_pairs, nx, ny), dtype=float)
    phs_delta = np.zeros((n_pairs*2, nx, ny), dtype=float)
    E_pupil = np.zeros((n_nulls*2, nx, ny), dtype=complex)
    H = np.zeros((n_pairs, 2), dtype=float)
    b = np.zeros((n_pairs, 1))

    # Masking
    mask2D, imsk, jmsk, irng, jrng, imx, imn, jmx, jmn = get_fp_mask(cdi_zip)


    ## Differance Images (delta)
    # (I_ip+ - I_ip-)/2
    for ip in range(n_pairs):
        delta[ip] = (np.abs(fp_seq[:,:,ip])**2 - np.abs(fp_seq[:,:,ip + n_pairs])**2) / 2

##
    for i, j in zip(irng, jrng):
            for xn in range(n_nulls-1):
                for ip in range(n_pairs):
                    # absDeltaP = (fp_seq[i,j,ip] + fp_seq[i,j,ip + n_pairs] / 2
                    #                    - fp_seq[i,j,cdi_zip.ts.n_probes+xn])
                    # absDeltaP = np.sqrt(absDeltaP)
                    # phsDeltaP = phs_delta[ip,i,j]
                    # cpxDeltaP = absDeltaP*np.exp(1j*phsDeltaP)
                    # Amplitude DeltaP
                    Ip = fp_seq[i, j, ip]
                    Im = fp_seq[i, j, ip + n_pairs]
                    Io = fp_seq[i, j, cdi_zip.ts.n_probes + xn]
                    abs = (Ip + Im) / 2 - Io
                    if abs < 0:
                        abs = 0
                    absDeltaP = np.sqrt(abs)
                    # absDeltaP = np.sqrt(np.abs((Ip + Im) / 2 - Io))

                    # Phase DeltaP
                    # The phase of the change in the focal plane of the probe applied to the DM
                    # First subtract Eo vector from each probe phase to make new field vectors dEa, dEb,
                    # then take the angle between the two
                    dEp = fp_seq[i, j, ip] - fp_seq[i, j, cdi_zip.ts.n_probes + xn]
                    dEm = fp_seq[i, j, ip + n_pairs] - fp_seq[i, j, cdi_zip.ts.n_probes + xn]
                    phsDeltaP = np.arctan2(dEp.imag - dEm.imag, dEp.real - dEm.real)

                    cpxDeltaP = absDeltaP * np.exp(1j * phsDeltaP)
                    H[ip, :] = [-cpxDeltaP.imag, cpxDeltaP.real]  # [n_pairs, 2]
                    b[ip] = delta[ip, i, j]  # [n_pairs, 1]

                a = 2 * H
                Exy = sl.lstsq(a, b)[0]  # returns tuple, not array
                E_pupil[xn, i, j] = Exy[0] + (1j * Exy[1])

    et = time.time()
    print(f'\tCDI post-processing for {n_nulls} null-images finished in {et-st:.1f} sec == {(et-st)/60:.2f} min')
##
    if plot:
        ########################
        # Phase Delta Images
        fig, subplot = plt.subplots(1, n_pairs, figsize=(14, 5))
        fig.subplots_adjust(wspace=0.5, right=0.85)

        fig.suptitle('Phase Interpolated')

        for ax, ix in zip(subplot.flatten(), range(n_pairs)):
            im = ax.imshow(phs_delta[ix].T * 1e6, interpolation='none', origin='lower',
                           # norm=SymLogNorm(linthresh=1e-2),
                           vmin=-1, vmax=1)  # , norm=SymLogNorm(linthresh=1e-5))
            ax.set_title(f"Phase\n" + r'$\theta$' + f'={cdi_zip.ts.phase_cycle[ix] / np.pi:.3f}' +
                         r'$\pi$ -$\theta$' + f'={cdi_zip.ts.phase_cycle[ix + n_pairs] / np.pi:.3f}' + r'$\pi$')

        # cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        # cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        # cb.set_label('Intensity')

        plt.show()
        ####################
        # Difference Images
        fig, subplot = plt.subplots(1, n_pairs, figsize=(14,5))
        fig.subplots_adjust(wspace=0.5, right=0.85)

        fig.suptitle('Deltas for CDI Probes')

        for ax, ix in zip(subplot.flatten(), range(n_pairs)):
            im = ax.imshow(delta[ix].T*1e6, interpolation='none', origin='lower',
                           # norm=SymLogNorm(linthresh=1e-2),
                           vmin=-1, vmax=1) #, norm=SymLogNorm(linthresh=1e-5))
            ax.set_title(f"Diff Probe\n" + r'$\theta$' + f'={cdi_zip.ts.phase_cycle[ix]/np.pi:.3f}' +
                         r'$\pi$ -$\theta$' + f'={cdi_zip.ts.phase_cycle[ix+n_pairs]/np.pi:.3f}' + r'$\pi$')

        # cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        # cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        # cb.set_label('Intensity')

        plt.show()

        ##############################
        # Focal Plane E-field Estimate
        fig, subplt = plt.subplots(1, 3, figsize=(14, 5))
        fig.subplots_adjust(wspace=0.5, right=0.85)

        fig.suptitle('Estimated Focal-Plane E-field')
        for ax, ix in zip(subplt, range(3)):
            print(f'ax={ax}, ix={ix}')
            im = ax.imshow(np.abs(E_pupil[ix].T)**2 * 1e6, interpolation='none', origin='lower')  #,
                           # norm=LogNorm())
            ax.set_title(f'Estimation timestep {ix}')

        # cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        # cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        # # cb.set_label('Intensity')

        plt.show()

        # #####################
        # Subtracted E-field
        fig, subplot = plt.subplots(2, 5, figsize=(14, 5))
        fig.subplots_adjust(wspace=0.05, right=0.85)

        fig.suptitle('Subtracted E-field')

        for ax, ix in zip(subplot, range(n_nulls)):
            im = ax.imshow(fp_seq[:,:,ix].T-np.abs(E_pupil[ix].T) ** 2, interpolation='none', origin='lower')#,
                           # norm=LogNorm())
            ax.set_title(f'Estimation timestep {ix}')

        # cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        # cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        # # cb.set_label('Intensity')

        plt.show()

##
def get_fp_mask(cdi, thresh=1e-7):
    """
    returns a mask of the CDI probe pattern in focal plane coordinates

    :param cdi: structure containing all CDI probe parameters
    :return:
    """
    nx = 140
    ny = 146
    dm_act = cdi.probe.DM_cmd_cycle.shape[1]

    probe_ft = (1 / np.sqrt(2 * np.pi)) *\
               np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(cdi.probe.DM_cmd_cycle[0])))

    Ar = interpolate.interp2d(range(dm_act), range(dm_act), probe_ft.real, kind='cubic')
    Ai = interpolate.interp2d(range(dm_act), range(dm_act), probe_ft.imag, kind='cubic')
    ArI = Ar(np.linspace(0, dm_act, ny), np.linspace(0, dm_act, nx))
    AiI = Ai(np.linspace(0, dm_act, ny), np.linspace(0, dm_act, nx))

    fp_probe = np.sqrt(ArI**2 + AiI**2)
    fp_mask = (fp_probe > thresh)
    (imsk, jmsk) = (fp_probe > thresh).nonzero()

    irng = range(min(imsk), max(imsk), 1)
    jrng = range(min(jmsk), max(jmsk), 1)

    imx = max(irng) - 1  # -1 is to get index values for plotting purposes
    imn = min(irng) - 1
    jmx = max(jrng) - 1
    jmn = min(jrng) - 1

    return fp_mask, imsk, jmsk, irng, jrng, imx, imn, jmx, jmn

##
if __name__ == '__main__':
    target_name = 'Jan2021_test8'  # None
    cdi_zip = open_MEC_tseries('/darkdata/kkdavis/mec/Jan2021/CDI_tseries_1-30-2021_T4:27.pkl')
    fp_seq = np.load('/darkdata/kkdavis/mec/Jan2021/Jan2021_test1_1611978617_temporalCube_regBins.npy', allow_pickle=True)
    grail = cdi_postprocess(fp_seq, cdi_zip, plot=True)

    dumm=1
