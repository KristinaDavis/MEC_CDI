##
"""
postprocess_MEC_CDI
Kristina Davis
Aug 26 2020


"""

import numpy as np
import os
import time
import datetime
import pickle
import scipy.linalg as sl
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

from mec_cdi import Slapper
from read_photons import open_MEC_tseries

##
def cdi_postprocess(fp_seq, meta, plot=False):
    """
    this is the function that accepts the timeseries of intensity images from the simulation and returns the processed
    single image. This function calculates the speckle amplitude phase, and then corrects for it to create the dark
    hole over the specified region of the image.

    :param fp_seq: focal plane sequence (temporal cube) [nx, ny, time]
    :param meta: metadata from .pkl file from DM settings and timestream created at runtime
    :param plot: toggle to display plots or not
    :return: nothing
    """
    ##
    print(f'\nStarting CDI post-processing')
    st = time.time()

    nx = fp_seq.shape[0]
    ny = fp_seq.shape[1]
    dm_act = meta.probe.DM_cmd_cycle.shape[-1]

    # Defining Matrices
    n_pairs = meta.ts.n_probes // 2  # number of deltas (probe differentials)
    n_nulls = np.int(meta.ts.null_time / meta.ts.phase_integration_time)
    delta = np.zeros((n_pairs, nx, ny), dtype=float)
    phs_delta = np.zeros((n_pairs*2, nx, ny), dtype=float)
    Epupil = np.zeros((n_nulls*2, nx, ny), dtype=complex)
    H = np.zeros((n_pairs, 2), dtype=float)
    b = np.zeros((n_pairs, 1))

    ## Differance Images (delta)
    for ip in range(n_pairs):
        delta[ip] = np.copy(fp_seq[:,:,ip] - fp_seq[:,:,ip + n_pairs])
        # FFT and Interpolate DM Map onto MEC coordinates
        fftA = (1/np.sqrt(2*np.pi) *\
                                   np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(meta.probe.DM_cmd_cycle[ip]))))
        fftB = (1/np.sqrt(2*np.pi) *\
                                   np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(meta.probe.DM_cmd_cycle[ip+n_pairs]))))
        Ar = interpolate.interp2d(range(dm_act), range(dm_act), fftA.real, kind='cubic')
        Ai = interpolate.interp2d(range(dm_act), range(dm_act), fftA.imag, kind='cubic')
        Br = interpolate.interp2d(range(dm_act), range(dm_act), fftB.real, kind='cubic')
        Bi = interpolate.interp2d(range(dm_act), range(dm_act), fftB.imag, kind='cubic')
        ArI = Ar(np.linspace(0,dm_act,ny), np.linspace(0,dm_act, nx))
        AiI = Ai(np.linspace(0,dm_act,ny), np.linspace(0,dm_act, nx))
        BrI = Br(np.linspace(0,dm_act,ny), np.linspace(0,dm_act, nx))
        BiI = Bi(np.linspace(0,dm_act,ny), np.linspace(0,dm_act, nx))
        phs_delta[ip,:,:] = np.arctan2(AiI-BiI, ArI-BrI)


    # for ip in range(meta.ts.n_probes):
    #     phs_fft = np.angle(1/np.sqrt(2*np.pi) *\
    #                                np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(meta.probe.DM_cmd_cycle[ip]))))
    #     f = interpolate.interp2d(range(phs_fft.shape[0]), range(phs_fft.shape[0]), phs_fft, kind='cubic')
    #     ipd = f(np.linspace(0,phs_fft.shape[0],ny), np.linspace(0,phs_fft.shape[0], nx))
    #     phs_delta[ip,:,:] = np.roll(np.roll(ipd,30,axis=0),10,axis=1)
##
    for i in range(nx):
            for j in range(ny):

                for xn in range(n_nulls-1):
                    for ip in range(n_pairs):
                        absDeltaP = np.abs((fp_seq[i,j,ip] + fp_seq[i,j,ip + n_pairs]) / 2
                                           - fp_seq[i,j,meta.ts.n_probes+xn])
                        absDeltaP = np.sqrt(absDeltaP)
                        phsDeltaP = phs_delta[ip,i,j]
                        # phsDeltaP = np.arctan2(fp_seq[i,j,ip].imag - fp_seq[i,j, meta.ts.n_probes+xn].imag,
                        #                        fp_seq[i,j,ip].real - fp_seq[i,j, meta.ts.n_probes+xn].real)
                        cpxDeltaP = absDeltaP*np.exp(1j*phsDeltaP)

                        H[ip, :] = [-cpxDeltaP.imag, cpxDeltaP.real]
                        b[ip] = delta[ip,i,j]

                    a = 2 * H
                    Exy, res, rnk, s = sl.lstsq(a, b)  # returns tuple, not array
                    Epupil[xn, i, j] = Exy[0] + (1j * Exy[1])
                    cats = 5

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
            im = ax.imshow(phs_delta[ix] * 1e6, interpolation='none', origin='lower',
                           # norm=SymLogNorm(linthresh=1e-2),
                           vmin=-1, vmax=1)  # , norm=SymLogNorm(linthresh=1e-5))
            ax.set_title(f"Phase\n" + r'$\theta$' + f'={meta.ts.phase_cycle[ix] / np.pi:.3f}' +
                         r'$\pi$ -$\theta$' + f'={meta.ts.phase_cycle[ix + n_pairs] / np.pi:.3f}' + r'$\pi$')

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
            im = ax.imshow(delta[ix]*1e6, interpolation='none', origin='lower',
                           # norm=SymLogNorm(linthresh=1e-2),
                           vmin=-1, vmax=1) #, norm=SymLogNorm(linthresh=1e-5))
            ax.set_title(f"Diff Probe\n" + r'$\theta$' + f'={meta.ts.phase_cycle[ix]/np.pi:.3f}' +
                         r'$\pi$ -$\theta$' + f'={meta.ts.phase_cycle[ix+n_pairs]/np.pi:.3f}' + r'$\pi$')

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
            im = ax.imshow(np.abs(Epupil[ix])**2 * 1e6, interpolation='none', origin='lower')  #,
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
            im = ax.imshow(fp_seq[:,:,ix]-np.abs(Epupil[ix]) ** 2, interpolation='none', origin='lower')#,
                           # norm=LogNorm())
            ax.set_title(f'Estimation timestep {ix}')

        # cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        # cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        # # cb.set_label('Intensity')

        plt.show()


if __name__ == '__main__':
    dm_meta = open_MEC_tseries('CDI2/CDI_tseries_10-6-2020_hour5_min27.pkl')
    # dm_meta = open_MEC_tseries('/work/kkdavis/scratch/old/CDI_tseries_3-9-2020_hour0_min11.pkl')
    # data = np.load('CDI2/CDI2_config_1601962002.npz', allow_pickle=True)
    fp_seq = np.load('CDI2_tcube_fullcycle_1601962002.npy', allow_pickle=True)
    grail = cdi_postprocess(fp_seq, dm_meta, plot=True)

    dumm=1
