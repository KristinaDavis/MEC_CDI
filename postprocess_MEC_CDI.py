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

from mec_cdi import CDI_params, CDIOut



def open_MEC_tseries(CDI_tseries='CDI_tseries.pkl'):
    """opens existing MEC CDI timeseries .pkl file and return it"""
    with open(CDI_tseries, 'rb') as handle:
        CDI_meta =pickle.load(handle)
    return CDI_meta


def cdi_postprocess(cpx_seq, plot=False):
    """
    this is the function that accepts the timeseries of intensity images from the simulation and returns the processed
    single image. This function calculates the speckle amplitude phase, and then corrects for it to create the dark
    hole over the specified region of the image.

    :param cpx_seq: full complex field data from the simulation
    :param sampling: focal plane sampling
    :return:
    """
    print(f'\nStarting CDI post-processing')
    st = time.time()

    # Defining Matrices
    n_pairs = cdi.n_probes // 2  # number of deltas (probe differentials)
    n_nulls = sp.numframes - cdi.n_probes
    delta = np.zeros((n_pairs,sp.grid_size, sp.grid_size), dtype=float)
    Epupil = np.zeros((n_nulls*2, sp.grid_size, sp.grid_size), dtype=complex)
    H = np.zeros((n_pairs, 2), dtype=float)
    b = np.zeros((n_pairs, 1))

    # Differance Images (delta)
    for ip in range(n_pairs):
        delta[ip] = np.copy(I_fp[ip] - I_fp[ip + n_pairs])

    for i in range(sp.grid_size):
            for j in range(sp.grid_size):

                for xn in range(n_nulls):
                    for ip in range(n_pairs):
                        absDeltaP = np.abs((I_fp[ip,i,j] + I_fp[ip + n_pairs,i,j]) / 2 - I_fp[cdi.n_probes+xn,i,j])
                        absDeltaP = np.sqrt(absDeltaP)
                        # probe_ft = (1/np.sqrt(2*np.pi)) *
                        # np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(cdi.cout.DM_probe_series[ix])))
                        phsDeltaP = np.arctan2(fp_seq[ip,i,j].imag - fp_seq[cdi.n_probes+xn,i,j].imag,
                                               fp_seq[ip,i,j].real - fp_seq[cdi.n_probes+xn,i,j].real)
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


        kitten=1

if __name__ == '__main__':
    cp = open_MEC_tseries('CDI_tseries.pkl')

    dumm=0
