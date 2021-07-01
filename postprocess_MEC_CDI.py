##
"""
postprocess_MEC_CDI
Kristina Davis
Aug 26 2020


"""

import numpy as np
import os
import time
import bisect
import scipy.linalg as sl
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

from mec_cdi import Slapper
from read_photons import open_MEC_tseries

##
def cdi_postprocess(fp_seq, cdi_zip, plot=False, debug=False):
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
    mask2D, imsk, jmsk, irng, jrng, imx, imn, jmx, jmn = get_fp_mask(cdi_zip, thresh=1, shft=[25,10])

    if debug:
        fig, ax = plt.subplots(1,1)
        fig.suptitle(f'Masked FP in CDI probe Region')
        im = ax.imshow((fp_seq[:,:,0]*mask2D).T)

    ## Differance Images (delta)
    # (I_ip+ - I_ip-)/2
    for ip in range(n_pairs):
        delta[ip] = (np.abs(fp_seq[:,:,ip])**2 - np.abs(fp_seq[:,:,ip + n_pairs])**2) / 2

##
    irng = range(140)
    jrng = range(140)
    for i in irng:
        for j in jrng:
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
            im = ax.imshow(phs_delta[ix].T * 1e6, interpolation='none', origin='lower')  #,
                           # norm=SymLogNorm(linthresh=1e-2),
                           # vmin=-1, vmax=1)  # , norm=SymLogNorm(linthresh=1e-5))
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
            im = ax.imshow(delta[ix].T*1e6, interpolation='none', origin='lower')#,
                           # norm=SymLogNorm(linthresh=1e-2),
                           # vmin=-1, vmax=1) #, norm=SymLogNorm(linthresh=1e-5))
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
            im = ax.imshow(np.abs(E_pupil[ix].T)**2, interpolation='none' ,
                           norm=LogNorm())
            ax.set_title(f'Estimation timestep {ix}')

        # # Focal Plane E-field Estimate
        # fig, subplt = plt.subplots(1, 1, figsize=(6, 5))
        # fig.subplots_adjust(wspace=0.5, right=0.85)
        # fig.suptitle('Estimated Focal-Plane E-field')
        # im = subplt.imshow(np.abs(E_pupil[ix].T)**2 * 1e-6, interpolation='none',
        #                norm=LogNorm())
        # #####################
        # Subtracted E-field
        fig, subplot = plt.subplots(1, 3, figsize=(10, 5))
        fig.subplots_adjust(wspace=0.1, right=0.85)
        fig.suptitle(f' Subtracted E-field')

        I_processed = np.floor(np.abs(E_pupil[ix].T)**2 * 1e-6)
        for ax, ix in zip(subplot, range(3)):
            imsx = ax.imshow(fp_seq[:,:,ix+n_nulls].T-I_processed, interpolation='none',
                             vmin=-1, vmax=2000)
                           # norm=LogNorm())
            ax.set_title(f'Estimation timestep {ix}')

        cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        cb = fig.colorbar(imsx, orientation='vertical', cax=cax)  #
        # cb.set_label('Intensity')
        ######################################################
        # Subtracted E-field
        fig, subplot = plt.subplots(2, 3, figsize=(14, 10))
        fig.subplots_adjust(wspace=0.1, right=0.85)
        fig.suptitle(f'{target_name} Subtracted E-field')
        ax1, ax2, ax3, ax4, ax5, ax6 = subplot.flatten()

        I_processed = np.floor(np.abs(E_pupil[ix].T) ** 2 * 1e-6)

        ax1.imshow(fp_seq[:, :, 0 + n_nulls].T, interpolation='none', vmin=-1, vmax=2000)
        ax1.set_title(f'Null Step 1')
        ax2.imshow(fp_seq[:, :, 1 + n_nulls].T, interpolation='none', vmin=-1, vmax=2000)
        ax2.set_title(f'Null Step 2')
        ax3.imshow(fp_seq[:, :, 2 + n_nulls].T, interpolation='none', vmin=-1, vmax=2000)
        ax3.set_title(f'Null Step 3')
        ax4.imshow(fp_seq[:,:,0+n_nulls].T-I_processed, interpolation='none', vmin=-1, vmax=2000)
        ax4.set_title(f'CDI Subtracted Null 1')
        ax5.imshow(fp_seq[:, :, 1 + n_nulls].T - I_processed, interpolation='none', vmin=-1, vmax=2000)
        ax5.set_title(f'CDI Subtracted Null 2')
        ax6.imshow(fp_seq[:, :, 2+ n_nulls].T - I_processed, interpolation='none', vmin=-1, vmax=2000)
        ax6.set_title(f'CDI Subtracted Null 3')
        plt.show()

##
def basic_fft(cdi_zip,)


def load_matched_DM_data(tstamp, map_dir, txt_list, txt_times):
    """
    Given a unix timestamp, will search for the DM telemetry file name that is the best match (immediately prior or
    equal to the given timestamp), load in the timestream and 2D DM maps best fit match data, and return it

    :param tstamp: unix timestamp of a cdi command (from cdi_zip)
    :param map_dir: directory where the DM telemetry is stored
    :param txt_list: list of .txt files in the specified dir
    :return: map_ts, dm_map: map_ts is a list of the unix timestamps of each DM map, dm_map is the 2D map
    """
    # Finding earlier best match
    idm = find_rt(txt_times, tstamp)
    map_file_match = txt_list[idm]

    r2 = os.path.basename(map_file_match)
    dmTel_name_parts = os.path.splitext(r2)
    dmtxt_file = os.path.join(map_dir, map_file_match)
    dmfits_file = os.path.join(map_dir, dmTel_name_parts[0] + '.fits')

    # Fits Import
    from astropy.io import fits
    hdul = fits.open(dmfits_file)
    # hdul.info()
    hdr = hdul[0].header
    # hdr
    dm_map = hdul[0].data

    # Load Timing info from .txt file
    """
    col0 : datacube frame index
    col1 : Main index
    col2 : Time since cube origin
    col3 : Absolute time
    col4 : stream cnt0 index
    col5 : stream cnt1 index
    col6 : time difference between consecutive frames
    """
    dmtxt_data = np.loadtxt(dmtxt_file)
    map_ts = dmtxt_data[:, 3]

    return map_ts, dm_map


def find_rt(a, x):
    'Find rightmost value less than x'
    i = bisect.bisect_right(a, x)
    return i-1



##
def get_fp_mask(cdi_zip, thresh=1e-7, shft=[None,None]):
    """
    returns a mask of the CDI probe pattern in focal plane coordinates

    :param cdi: structure containing all CDI probe parameters
    :return:
    """
    nx = 140
    ny = 146
    dm_act = cdi_zip.probe.DM_cmd_cycle.shape[1]

    probe_ft = (1 / np.sqrt(2 * np.pi)) *\
               np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(cdi_zip.probe.DM_cmd_cycle[0])))

    Ar = interpolate.interp2d(range(dm_act), range(dm_act), probe_ft.real, kind='cubic')
    Ai = interpolate.interp2d(range(dm_act), range(dm_act), probe_ft.imag, kind='cubic')
    ArI = Ar(np.linspace(0, dm_act, ny), np.linspace(0, dm_act, nx))
    AiI = Ai(np.linspace(0, dm_act, ny), np.linspace(0, dm_act, nx))

    fp_probe = np.sqrt(ArI**2 + AiI**2)
    fp_mask = (fp_probe > thresh)
    (imsk, jmsk) = (fp_probe > thresh).nonzero()

    if shft[0] is not None:
        fp_mask = np.roll(fp_mask, shft[0], axis=0)
        imsk = np.roll(imsk, shft[0], axis=0)
    if shft[1] is not None:
        fp_mask = np.roll(fp_mask,shft[1], axis=1)
        jmsk = np.roll(jmsk,shft[1], axis=1)

    irng = range(min(imsk), max(imsk), 1)
    jrng = range(min(jmsk), max(jmsk), 1)

    imx = max(irng) - 1  # -1 is to get index values for plotting purposes
    imn = min(irng) - 1
    jmx = max(jrng) - 1
    jmn = min(jrng) - 1

    return fp_mask, imsk, jmsk, irng, jrng, imx, imn, jmx, jmn

##
##########################################
# DM Telemetry (referred to as map)
##########################################
# Converting .txt string to Unix to compare with MEC command timestamp
map_dir = '/darkdata/kkdavis/mec/May2021c/dm_telemetry/dm00disp03/'
fn = sorted(os.listdir(map_dir))
txts = [x for x in fn if ".fits" not in x]
tt = [x.replace('dm00disp03_','') for x in txts]
tt = [x.replace('.txt','') for x in tt]
ymd = cdi_zip.ts.cmd_tstamps[0].astype('datetime64[D]')
ymd = ymd.astype('<U18')
tt = [(ymd+'T'+x) for x in tt]
tt = [datetime_to_unix(np.datetime64(x)) for x in tt]
# #---------------------
# # MEC Initial Flat Timestep
# flat = datetime_to_unix(cdi_zip.ts.cmd_tstamps[0])  # just find the first command, which should be a flat


## Plot DM Map Probe Sequence
msk = np.tile(np.append(False, np.repeat(True, cdi_zip.ts.n_probes)), cdi_zip.ts.n_cycles)
cmds_probe_only = cdi_zip.ts.cmd_tstamps[msk]

DM_maps = np.zeros((cdi_zip.ts.n_probes, cdi_zip.probe.DM_cmd_cycle.shape[1], cdi_zip.probe.DM_cmd_cycle.shape[2]))
import datetime
fig, subplot = plt.subplots(2,3, figsize=(12,8))
fig.suptitle(f'DM Telemetry Data \n'
                 f'target = {target_name}, {h5_name_parts[0]}{h5_name_parts[1]}\n'
                 f' N probes={cdi_zip.ts.n_probes}, '
                 f'N null steps={np.int(cdi_zip.ts.null_time / cdi_zip.ts.phase_integration_time)}, '
                 f'integration time={cdi_zip.ts.phase_integration_time} sec')

for ax, ix in zip(subplot.flatten(), range(cdi_zip.ts.n_probes)):
    cdi_cmd = datetime_to_unix(cmds_probe_only[ix])
    map_ts, dm_map = load_matched_DM_data(cdi_cmd, map_dir, txts, tt)
    # Check that the MEC command falls within the time range of the telemetry file
    try:
        if not (cdi_cmd < map_ts[-1]) & (cdi_cmd > map_ts[0]):
            raise ValueError('MEC command NOT within range of opened .txt file')
    except(ValueError):
        map_ts, dm_map = load_matched_DM_data(cdi_cmd, map_dir, txts, tt)
    ixsync = find_lt(map_ts, cdi_cmd)

    im = ax.imshow(dm_map[ixsync, :, :],
              vmax=cdi_zip.probe.amp, vmin=-cdi_zip.probe.amp
             )
    ax.set_title(f'{map_ts[ixsync] - datetime_to_unix(cdi_zip.ts.cmd_tstamps[0]):.2f}')

cax = fig.add_axes([0.91, 0.2, 0.02, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
cb.set_label('DM voltage (relative scale)')

# plt.savefig(f'{dm_path}/{target_name}_dmMap_seq.png')


##
if __name__ == '__main__':
    target_name = 'Vega_2021_run7'  # None
    cdi_zip = open_MEC_tseries('/darkdata/kkdavis/mec/May2021c/CDI_tseries_5-25-2021_T11:58.pkl')
    fp_seq = np.load('/darkdata/kkdavis/mec/May2021c/Vega_2021_run7_1621943770_temporalCube_regBins.npy',
                     allow_pickle=True)
    grail = cdi_postprocess(fp_seq, cdi_zip, plot=True)

    dumm=1
