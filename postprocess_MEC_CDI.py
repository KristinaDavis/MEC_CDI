##
"""
postprocess_MEC_CDI
Kristina Davis
Aug 26 2020

regular bins:
        a temporal cube where each bin is spaced by the probe integration time. The unprobed "null steps" are
        split up into multiple time bins, rather than being lumped into one longer duration time bin.

irregular bins:
        a temporal cube where each null timestep has a single bin that corresponds to a full null cycle, meaning
        that the timestep is a longer time duration than a probed step
"""

import numpy as np
import os
import time
import bisect
import scipy.linalg as sl
import warnings
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import proper

from mec_cdi import Slapper
from read_photons import open_MEC_tseries, datetime_to_unix
from cdi_plots import plot_probe_response_cycle, plot_quick_coord_check, \
    plot_probe_response, plot_probe_cycle, get_fp_mask
from matplotlib.collections import LineCollection


##
def cdi_postprocess(fp_seq, cdi_zip, map_dir=None, plot=False, debug=False):
    """
    this is the function that accepts the timeseries of intensity images from the simulation and returns the processed
    single image. This function calculates the speckle amplitude phase, and then corrects for it to create the dark
    hole over the specified region of the image.

    :param fp_seq: focal plane sequence (temporal cube) [nx, ny, time]
    :param cdi_zip: cdi_zipdata from .pkl file from DM settings and timestream created at runtime
    :param map_dir: directory where the DM telemetry data (generated by scexaortc) is stored
    :param plot: toggle to display plots or not
    :return: nothing, lots of plots though
    """
    ##
    print(f'\nStarting CDI post-processing')
    st = time.time()

    nx = fp_seq.shape[1]  # size of MEC images
    ny = fp_seq.shape[2]
    dm_act = cdi_zip.probe.DM_cmd_cycle.shape[-1]  # number of DM actuators
    n_pairs = cdi_zip.ts.n_probes // 2  # number of deltas (probe differentials)
    n_nulls = int(cdi_zip.ts.null_time / cdi_zip.ts.phase_integration_time)
    n_cycles = cdi_zip.ts.n_cycles
    n_probes = cdi_zip.ts.n_probes

    # MEC Probe Commands (removes flat (null step) commands )
    msk = np.tile(np.append(np.repeat(True, n_probes), False), n_cycles)  # there is no beginning 'flat' command
    cmds_probe_only = cdi_zip.ts.cmd_tstamps[msk]
    cmds_nulls_only = cdi_zip.ts.cmd_tstamps[~msk]

    # # Sorting fp_seq
    if 'irreg' in datf:
        # Each 'null' step is much longer in time than the probe commands; matches the format of the probe timesteps
        # since the null was sent with a single flat command and lasts the length of a null as determined by mec_cdi.py
        reg_msk = msk
    else:
        # this assumes fp_seq was created with the null steps spaced out over time via photontable.temporal cube
        # regular bins in read_photons.py. Must reformat mask if there is one large null step, as in irreg_bins
        reg_msk = np.tile(np.append(np.repeat(True, n_probes), np.repeat(False, n_nulls)),
                          n_cycles)
        # raise ValueError(f'No longer working for regular binned data, use irreg instead')

    if len(reg_msk) > fp_seq.shape[0]:  # h5 might be cutoff mid-cycle if the dataset was long
        reg_msk = reg_msk[0:fp_seq.shape[0]]
    mec_probed = fp_seq[ reg_msk]
    mec_nulls = fp_seq[~reg_msk]
    intensity_counter(mec_probed, mec_nulls)

    # Define arrays for sorting probe vs null steps
    sim_grid = 256  # phase propagation simulation grid size nxn
    extracted = [nx,ny]
    probe_tstamps = np.zeros((len(cmds_probe_only)))
    probe_maps = np.zeros((len(cmds_probe_only), dm_act, dm_act))
    null_tstamps = np.zeros((len(cmds_nulls_only)))
    null_maps = np.zeros((len(cmds_nulls_only), dm_act, dm_act))

    ## Loading or simulating DM voltage maps
    lst = time.time()
    # # Complex Map
    # fp_mask, edges = get_fp_mask(cdi_zip)
    # cl = LineCollection(edges, colors='r')

    if map_dir:
        # Load DM Telemetry Data => dm voltage maps saved by scexaortc (referred to as map)
        # cpx_dm_sim = np.zeros((len(cmds_probe_only), nx, ny), dtype=complex)
        # cpx_null_sim = np.zeros((len(cmds_nulls_only), nx, ny), dtype=complex)
        cpx_dm = np.zeros((len(cmds_probe_only), nx, ny), dtype=complex)
        cpx_null = np.zeros((len(cmds_nulls_only), nx, ny), dtype=complex)

        # Converting .txt string to Unix to compare with MEC command timestamp
        fn = sorted(os.listdir(map_dir))
        txts = [x for x in fn if ".fits" not in x]
        tt = [x.replace('dm00disp_', '') for x in txts]  # 'dm00disp03_'
        tt = [x.replace('.txt', '') for x in tt]
        ymd = cdi_zip.ts.cmd_tstamps[0].astype('datetime64[D]')
        ymd = ymd.astype('<U18')
        tt = [(ymd + 'T' + x) for x in tt]
        tt = [datetime_to_unix(np.datetime64(x)) for x in tt]
        # #---------------------
        # Saving DM map that matches closest prior probe command
        flat = get_standard_flat(debug=False)
        # Probes
        for ix in range(len(cmds_probe_only)):  # len(cmds_probe_only)
            map_ts, dm_map, ixsync = sync_tstep(ix, cmds_probe_only, txts, tt)
            probe_tstamps[ix] = map_ts[ixsync]
            probe_maps[ix] = dm_map[ixsync, :, :]
            # cpx_dm[ix] = basic_fft(dm_map[ixsync, :, :], nx, ny)
            cpx_dm[ix], smp = proper.prop_run('scexao_model', .9, sim_grid,  # for some reason 0.9 gives 900 nm
                                             PASSVALUE={'map':probe_maps[ix]*1e-6, 'psf_size':extracted,
                                                        'verbose':debug, 'ix':ix}, QUIET=True)
        # Nulls
        for ix in range(len(cmds_nulls_only)):  # len(cmds_nulls_only)
            map_ts, dm_map, ixsync = sync_tstep(ix, cmds_nulls_only, txts, tt)
            null_tstamps[ix] = map_ts[ixsync]
            null_maps[ix] = dm_map[ixsync, :, :]-flat
            # cpx_null[ix] = basic_fft(null_maps[ix], nx, ny)
            cpx_null[ix], smp = proper.prop_run('scexao_model', .9, sim_grid,  # for some reason 0.9 gives 900 nm
                                          PASSVALUE={'map': null_maps[ix]*1e-6, 'psf_size':extracted,
                                                     'verbose':False, 'ix':ix}, QUIET=True)
        # # Rescaling to match MEC focal plane dimensions
        # for ix in range(cpx_dm_sim.shape[0]):
        #     cpx_dm[ix]   = resample_cpx(cpx_dm_sim[ix],   nx, ny)
        # for ix in range(cpx_null_sim.shape[0]):
        #     cpx_null[ix] = resample_cpx(cpx_null_sim[ix], nx, ny)

        elt = time.time()
        print(
            f'Phase propagation finished in {elt - lst:.1f} sec == {(elt - lst) / 60:.2f} min')

    else:
        # Just use the DM probe pattern as the voltage map
        cpx_all_sim = np.zeros((len(cdi_zip.probe.DM_cmd_cycle), extracted[0], extracted[1]), dtype=complex)
        cpx_all_1 = np.zeros((n_pairs*2+n_nulls, nx, ny), dtype=complex)
        cpx_all = np.zeros((len(reg_msk), nx, ny), dtype=complex)

        # flat = get_standard_flat(debug=False)  # standard flat  900e-9/1e6 *
        flat = np.zeros((50,50),dtype=float)  # empty
        # cos function
        # dmx, dmy = np.meshgrid(
        #     np.linspace(-0.5, 0.5, 50),
        #     np.linspace(-0.5, 0.5, 50))
        #
        # xm = dmx * 12 * 2.0 * np.pi
        # ym = dmy * 2.0 * np.pi
        # flat = 1e-8*np.sin(xm)
        # # # Test Probe
        # x = np.linspace(-1/2 - 5/50, 1/2 - 5/50, 50, dtype=np.float32)
        # y = np.linspace(-1/2 - 5/50, 1/2 - 5/50, 50, dtype=np.float32)
        # X,Y = np.meshgrid(x,y)
        # flat =  np.sinc(10 * X) * np.sinc(5 * Y) * np.cos(2 * np.pi * 10 * Y + np.pi/4)

        # complex of all maps from the probe pattern
        # last probe in the command cycle is the null
        for ix in range(len(cdi_zip.probe.DM_cmd_cycle)):
            map = 1e-6 * (flat + cdi_zip.probe.DM_cmd_cycle[ix])   #TODO replace flat + cmd
            # cpx_all_sim[ix] = basic_fft(map, 128, 128)
            cpx_all_sim[ix], smp = proper.prop_run('scexao_model', .9, sim_grid,
                                              PASSVALUE={'map': map, 'psf_size':extracted,
                                                         'verbose': debug, 'ix':ix}, QUIET=True)

        # Copying FP Null image for n_nulls steps
        ix=0
        while ix < n_nulls-1:
            cpx_all_sim = np.concatenate((cpx_all_sim, np.array([cpx_all_sim[-1,:,:]])),axis=0)
            ix+=1

        # repeat cycle of n_probes+n_nulls for n_cycles
        # for ix in range(len(cpx_all_sim)):
        #     cpx_all_1[ix] = resample_cpx(cpx_all_sim[ix], nx, ny)
        # cycle through the probe cycle for length of observation
        cpx_all = np.tile(cpx_all_sim, (cdi_zip.ts.n_cycles, 1, 1))

        # Probes
        cpx_dm = cpx_all[reg_msk]
        probe_maps = (cdi_zip.probe.DM_cmd_cycle[0:cdi_zip.ts.n_probes] + flat)
        probe_tstamps = cdi_zip.ts.cmd_tstamps[msk].astype('float')/1e9
        # Nulls
        cpx_null = cpx_all[~reg_msk]
        null_maps = np.tile(cdi_zip.probe.DM_cmd_cycle[-1] + flat, (cdi_zip.ts.n_cycles, 1, 1))
        null_tstamps = cdi_zip.ts.cmd_tstamps[~msk].astype('float')/1e9

        elt = time.time()
        print(
            f'\nPhase propagation finished in {elt - lst:.1f} sec == {(elt - lst) / 60:.2f} min\n')

    if debug:
        # Plot DM Map Probe Sequence
        fig, subplot = plt.subplots(2, cdi_zip.ts.n_probes//2, figsize=(14, 8))
        fig.suptitle(f'DM Telemetry Data, Probe Commands: '
                     f'target = {target_name}\n'
                     f' N probes={cdi_zip.ts.n_probes}, '
                     f'N null steps={int(cdi_zip.ts.null_time / cdi_zip.ts.phase_integration_time)}, '
                     f'integration time={cdi_zip.ts.phase_integration_time} sec', fontweight='bold', fontsize=14)

        for ax, ix in zip(subplot.flatten(), range(len(cdi_zip.probe.DM_cmd_cycle))):
            im = ax.imshow(probe_maps[ix],   #
                           vmax=np.max(probe_maps[0]), vmin=-np.max(probe_maps[0]) # np.min(probe_maps[0])
                           )
            ax.set_title(f"t={probe_tstamps[ix] - (cdi_zip.ts.cmd_tstamps[0]).astype('float')/1e9:.2f}")

        cax = fig.add_axes([0.91, 0.2, 0.02, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        cb.set_label('DM voltage (relative scale)')
        plt.savefig(f'{plt_path}/{target_name}_dmMap_probeseq.png')

        # Null DM Map
        fig, subplot = plt.subplots(2, len(cmds_nulls_only) // 2, figsize=(14, 8))
        fig.suptitle(f'DM Telemetry Data, Non-probed Timesteps (nulls): '
                     f'target = {target_name}\n'
                     f' N probes={n_probes}, '
                     f'N null steps={int(n_nulls)}, '
                     f'integration time={cdi_zip.ts.phase_integration_time} sec', fontweight='bold', fontsize=14)

        for ax, ix in zip(subplot.flatten(), range(len(cmds_nulls_only))):
            im = ax.imshow(null_maps[ix],  #
                           vmax=np.max(null_maps[0]), vmin=-np.max(null_maps[0])  # 0
                           )
            ax.set_title(f"t={null_tstamps[ix] - (cdi_zip.ts.cmd_tstamps[0]).astype('float') / 1e9:.2f}")
        cax = fig.add_axes([0.91, 0.2, 0.02, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        cb.set_label(f'DM actuator height (' + r'$\mu$' + 'm, uncalibrated)')
        plt.savefig(f'{plt_path}/{target_name}_dmMap_nullseq.png')

    # Re-center array to match mec images
    cpx_dm   = np.roll(cpx_dm , (-3, 20),(1,2))
    cpx_null = np.roll(cpx_null,(-3, 20),(1,2))
    if plot:
        # Complex Map
        # fp_mask, edges = get_fp_mask(cdi_zip)
        # cl = LineCollection(edges, colors='r')

        fig, subplot = plt.subplots(2, n_probes // 2, figsize=(14, 8))
        fig.suptitle(f'Propogated Intensity through SCExAO Model (probes): '
                     f'target = {target_name}\n'
                     f' N probes={cdi_zip.ts.n_probes}, '
                     f'N null steps={int(cdi_zip.ts.null_time / cdi_zip.ts.phase_integration_time)}, '
                     f'integration time={cdi_zip.ts.phase_integration_time} sec', fontweight='bold', fontsize=14)

        for ax, ix in zip(subplot.flatten(), range(n_probes)):
            im = ax.imshow(np.abs(cpx_dm[ix])**2, interpolation='none',
                  norm=LogNorm(vmin=1e-5,vmax=1e-2)
                           )
            # ax.add_collection(cl)
            ax.set_title(f"t={probe_tstamps[ix] - (cdi_zip.ts.cmd_tstamps[0]).astype('float')/1e9:.2f}")

        cax = fig.add_axes([0.91, 0.2, 0.02, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        cb.set_label(f'Intensity')
        plt.savefig(f'{plt_path}/{target_name}_simulated_focalplane.png')

    # plt.show()
    ######################
    # CDI Algorithm
    # ####################
    # Defining Matrices
    delta = np.zeros((n_pairs, nx, ny), dtype=float)
    E_pupil = np.zeros((n_nulls*n_cycles, nx, ny), dtype=complex)
    H = np.zeros((n_pairs, 2), dtype=float)
    b = np.zeros((n_pairs, 1))

    # Masking
    mask2D, imsk, jmsk, irng, jrng, imx, imn, jmx, jmn = get_fp_mask(cdi_zip, thresh=1e-3, shft=[-3, 20])  # , shft=[25,10]

    if debug:
        fig, ax = plt.subplots(1,1)
        fig.suptitle(f'Masked FP in CDI probe Region', fontweight='bold', fontsize=14)
        im = ax.imshow((mec_probed[0,:,:]*mask2D).T)

        cax = fig.add_axes([0.85, 0.2, 0.02, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        cb.set_label(f'Counts')

    cdit = time.time()
    for cy in range(1):  # range(cdi_zip.ts.n_cycles)
        cycle = cy*n_nulls
        for i in range(140):  #  range(80,85)  irng range(140)
            for j in range(140):  #  range(80,85)  jrng range(140)
                for xn in range(n_nulls):
                    for ip in range(n_pairs):
                        # print(f'writing to [{cycle + xn}, i={i}, j={j}], cycle={cycle}, cy={cy}, xn={xn}')
                        # Differance Images (delta)
                        # (I_ip+ - I_ip-)/2
                        delta[ip,:,:] = (mec_probed[cycle+ip, :, :] - mec_probed[cycle + ip + n_pairs, :, :]) / 2
                            #TODO does this get overwritten each cycle?

                        # Amplitude DeltaP
                        Ip = mec_probed[cycle + ip, i, j]
                        Im = mec_probed[cycle + ip + n_pairs, i, j]
                        Io = mec_nulls[cycle + xn, i, j] #TODO fix this to run over all nulls in series
                        abs = (Ip + Im) / 2 - Io
                        if abs < 0:
                            abs = 0
                        absDeltaP = np.sqrt(abs)
                        # absDeltaP = np.sqrt(np.abs((Ip + Im) / 2 - Io))

                        # Phase DeltaP
                        # The phase of the change in the focal plane of the probe applied to the DM
                        # First subtract Eo vector from each probe phase to make new field vectors dEa, dEb,
                        # then take the angle between the two
                        dEp = cpx_dm[cycle + ip, i, j] - cpx_null[cycle + xn, i, j]
                        dEm = cpx_dm[cycle+ip+n_pairs, i, j] - cpx_null[cycle + xn, i, j]
                        # dEp = fp_seq[i, j, ip] - fp_seq[i, j, cdi_zip.ts.n_probes + xn]
                        # dEm = fp_seq[i, j, ip + n_pairs] - fp_seq[i, j, cdi_zip.ts.n_probes + xn]
                        phsDeltaP = np.arctan2(dEp.imag - dEm.imag, dEp.real - dEm.real)

                        # print(f'dEp={dEp}, dEm={dEm}, xn={xn}\n'
                        #       f'abs={absDeltaP}, phs={phsDeltaP}')

                        cpxDeltaP = absDeltaP * np.exp(1j * phsDeltaP)
                        H[ip, :] = [-cpxDeltaP.imag, cpxDeltaP.real]  # [n_pairs, 2]
                        b[ip] = delta[ip, i, j]  # [n_pairs, 1]

                    a = 2 * H
                    Exy = sl.lstsq(a, b)[0]  # returns tuple, not array
                    # print(f'writing to [{cycle + xn}, i={i}, j={j}], cycle={cycle}, cy={cy}, xn={xn}')
                    E_pupil[cycle + xn, i, j] = Exy[0] + (1j * Exy[1])

    et = time.time()
    print(f'CDI post-processing for {n_nulls*cdi_zip.ts.n_cycles} null-images'
          f' finished in {et-cdit:.1f} sec == {(et-cdit)/60:.2f} min\n')

    # I_processed
    I_processed = np.zeros((len(E_pupil), nx, ny))
    for ix in range(len(E_pupil)):
        I_processed[ix] = np.floor(np.abs(E_pupil[ix]) ** 2 )

    ##
    if plot:
        ####################
        # Difference Images
        fig, subplot = plt.subplots(1, n_pairs, figsize=(14,5))
        fig.subplots_adjust(wspace=0.3, right=0.90, left=0.05)

        fig.suptitle((r'$I_i^+ - I_i^-$' ' for CDI Probes'), fontweight='bold', fontsize=14)

        for ax, ix in zip(subplot.flatten(), range(n_pairs)):
            im = ax.imshow(delta[ix].T, interpolation='none',
                           # norm=SymLogNorm(linthresh=1e-2),
                           vmin=-1, vmax=1) #, norm=SymLogNorm(linthresh=1e-5))
            ax.set_title(f"Diff Probe\n" + r'$\theta$' + f'={cdi_zip.ts.phase_cycle[ix]/np.pi:.3f}' +
                         r'$\pi$ -$\theta$' + f'={cdi_zip.ts.phase_cycle[ix+n_pairs]/np.pi:.3f}' + r'$\pi$')

        # cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        # cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        # cb.set_label('Intensity')


        ##############################
        # Focal Plane E-field Estimate
        fig, subplt = plt.subplots(2, 4, figsize=(14, 7))
        fig.subplots_adjust(wspace=0.3, right=0.85, left=0.05)

        fig.suptitle('Estimated Focal-Plane E-field', fontweight='bold', fontsize=14)
        for ax, ix in zip(subplt.flatten(), range(8)):
            im = ax.imshow(I_processed[ix].T, interpolation='none',  # ,
                           vmin=-1, vmax=200)  # ,
                           # norm=LogNorm())
            ax.set_title(f'Estimation timestep {ix}')
        cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        cb.set_label('Intensity')

        ######################
        # Subtracted E-field
        fig, subplot = plt.subplots(1, 3, figsize=(14, 5))
        fig.subplots_adjust(wspace=0.1, right=0.85, left=0.05)
        fig.suptitle(f' Subtracted E-field', fontweight='bold', fontsize=14)

        for ax, ix in zip(subplot, range(3)):
            imsx = ax.imshow(mec_nulls[ix,:,:].T-I_processed[ix].T, interpolation='none',
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
        fig.suptitle(f'{target_name} Subtracted E-field', fontweight='bold', fontsize=14)
        ax1, ax2, ax3, ax4, ax5, ax6 = subplot.flatten()

        ax1.imshow(fp_seq[0 + n_nulls].T, interpolation='none', vmin=-1, vmax=2000)
        ax1.set_title(f'Null Step 1')
        ax2.imshow(fp_seq[1 + n_nulls].T, interpolation='none', vmin=-1, vmax=2000)
        ax2.set_title(f'Null Step 2')
        ax3.imshow(fp_seq[2 + n_nulls].T, interpolation='none', vmin=-1, vmax=2000)
        ax3.set_title(f'Null Step 3')
        ax4.imshow(fp_seq[0 + n_nulls].T-I_processed[0].T, interpolation='none', vmin=-1, vmax=2000)
        ax4.set_title(f'CDI Subtracted Null 1')
        ax5.imshow(fp_seq[1 + n_nulls].T - I_processed[0].T, interpolation='none', vmin=-1, vmax=2000)
        ax5.set_title(f'CDI Subtracted Null 2')
        ax6.imshow(fp_seq[ 2 + n_nulls].T - I_processed[0].T, interpolation='none', vmin=-1, vmax=2000)
        ax6.set_title(f'CDI Subtracted Null 3')

        # plt.show()
        ##
        # DeltaP
        fig, subplot = plt.subplots(2, n_pairs, figsize=(12, 8))
        fig.subplots_adjust(wspace=0.5, right=0.85)
        fig.suptitle(r'$\frac{I_i^+ + I_i^-}{2} - I_{null}$ for CDI Probes  ' +
                     f'target = {target_name}, \n'
                     f'Probe Amp = {cdi_zip.probe.amp}, N probes={cdi_zip.ts.n_probes}, '
                     f'N null steps={int(cdi_zip.ts.null_time / cdi_zip.ts.phase_integration_time)}, '
                     f'integration time={cdi_zip.ts.phase_integration_time} sec\n', fontweight='bold', fontsize=14)

        cnt = np.tile(range(n_pairs), 2)
        xn = 1
        for ax, ix in zip(subplot.flatten(), range(2 * n_pairs)):
            Ip = fp_seq[cnt[ix] ]
            Im = fp_seq[cnt[ix] + n_pairs]
            Io = fp_seq[cdi_zip.ts.n_probes + (ix // n_pairs + 0)]

            absDP = (Ip + Im) / 2 - Io
            # if absDP.any() < 0:
            #     absDP = 0
            # absDeltaP = np.sqrt(absDP)
            im = ax.imshow(absDP.T, interpolation='none',  #
                           vmin=-1, vmax=1,
                           cmap='plasma',
                           )  # , norm=SymLogNorm(linthresh=1e-5))
            ax.set_title(f"probe phase pair {cnt[ix] + 1},\nnull step {ix // n_pairs + 1 + 0}")
            # cl = LineCollection(edges, colors='r')
            # ax.add_collection(cl)
            if ix + 1 > xn * n_pairs - 1:
                xn += 1

        cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
        cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
        cb.set_label('Intensity')

    ## Probe Response (from mec_cdi.py)
    # save_set = {'save':False, 'plt': plt_path, 'target': target_name, 'h5': run_data['h5_unix'].tolist()}
    # plot_probe_response(cdi_zip, 0, save_set)
    # plot_probe_cycle(cdi_zip, save_set)
    # # plot_probe_response_cycle(cdi_zip)
    # plot_quick_coord_check(cdi_zip, 0, save_set)

##################################################
# Functions
###################################################
def sync_tstep(ii, cmds, txt_list, txt_times):
    """
    given a unix timestamp that a command was sent--from the perspective of the scexao rtc computer-- first search for
    the DM telemetry files (.txt file containing timing data and the .fits file containing the 2D maps) that correspond
    to that command timestep. Then, open those files to search for the best timestep match within the opened
    telemetry data, best meaning closest prior timestep

    :param ii: iteration (comes from external for loop)
    :param cmds: list of CDI commands, either probes or nulls from cdi_zip
    :param txt_list: list of .txt files in the specified dir (eg txts)
    :param txt_times: unixtimes within the directory (eg tt)
    :return: map_ts map_ts is a list of the unix timestamps of each DM map
    :return: dm_map: dm_map is the 2D map
    :return: ixsync: array index location within map_ts/dm_map that best matches the cdi_zip command timestamp
    """
    cdi_cmd = datetime_to_unix(cmds[ii])  # cmds_probe_only[ix]
    map_ts, dm_map = load_matched_DM_data(cdi_cmd, map_dir, txt_list, txt_times)
    # Check that the MEC command falls within the time range of the telemetry file
    try:
        if not (cdi_cmd < map_ts[-1]) & (cdi_cmd > map_ts[0]):
            raise ValueError('MEC command NOT within range of opened .txt file')
    except(ValueError):
        map_ts, dm_map = load_matched_DM_data(cdi_cmd, map_dir, txt_list, txt_times)
    ixsync = find_lt(map_ts, cdi_cmd)
    return map_ts, dm_map, ixsync


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

    hdul.close()
    return map_ts, dm_map


def find_rt(a, x):
    'Find rightmost value less than x'
    i = bisect.bisect_right(a, x)
    return i-1


def find_lt(a, x):
    'Find rightmost value less than x'
    i = bisect.bisect_left(a, x)
    return i


def resample_cpx(cpx, nx, ny):
    """
    resample the cpx data returned from the proper model simulation to the size of the MEC images

    :param cpx: nxn square array from the model
    :param nx: number of x steps in the MEC image
    :param ny: number of y steps in the MEC image
    :return: the complex data is interpolated to a 2D nx, ny complex plane
    """

    Ar = interpolate.interp2d(range(cpx.shape[0]), range(cpx.shape[1]), cpx.real, kind='cubic')
    Ai = interpolate.interp2d(range(cpx.shape[0]), range(cpx.shape[1]), cpx.imag, kind='cubic')
    ArI = Ar(np.linspace(0, cpx.shape[0], ny), np.linspace(0, cpx.shape[1], nx))  # linspace(start, stop, n_points)
    AiI = Ai(np.linspace(0, cpx.shape[0], ny), np.linspace(0, cpx.shape[1], nx))

    return ArI * np.exp(1j * AiI)


def basic_fft(map, nx=140, ny=146, use_flat=False):
    """
    performs a 2D fft of input 2D array (eg DM map)  and interpolates it onto MEC coordinates

    :param cdi_zip:
    :param map: 2D array of pupil plane
    :return:
    """
    if use_flat:
        flat = np.squeeze(get_standard_flat(debug=True))
        map += flat

        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.suptitle('DM flat + CDI probe', fontweight='bold', fontsize=14)
        ax.imshow(map, interpolation='none')

    probe_ft = (1 / np.sqrt(2 * np.pi)) * \
               np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(map)))

    return resample_cpx(probe_ft, nx, ny)  # np.sqrt(ArI ** 2 + AiI ** 2)


def get_fp_mask(cdi_zip, thresh=1e-7, shft=[None,None]):
    """
    returns a mask of the CDI probe pattern in focal plane coordinates

    :param cdi_zip: structure containing all CDI probe parameters
    :return:
    """
    fp_probe = basic_fft(cdi_zip.probe.DM_cmd_cycle[0])
    fp_intensity = np.abs(fp_probe)**2

    if shft[0] is not None:
        fp_mask = np.roll(fp_intensity, shft[0], axis=0)
    if shft[1] is not None:
        fp_mask = np.roll(fp_intensity,shft[1], axis=1)

    fp_mask = (fp_intensity > thresh)
    (imsk, jmsk) = (fp_intensity > thresh).nonzero()

    irng = range(min(imsk), max(imsk), 1)
    jrng = range(min(jmsk), max(jmsk), 1)

    imx = max(irng) - 1  # -1 is to get index values for plotting purposes
    imn = min(irng) - 1
    jmx = max(jrng) - 1
    jmn = min(jrng) - 1

    return fp_mask, imsk, jmsk, irng, jrng, imx, imn, jmx, jmn


def get_standard_flat(debug=False):
    """Load a standard SCExAO DM flat map from a fits file"""
    warnings.warn(f'\nUsing flat map from Sept2021. Update if changed\n')
    # dmfits_file = '/darkdata/kkdavis/mec/May2021Sci/dm_20210518/dm00disp00_12:00:33.796602287.fits'
    dmfits_file = '/darkdata/kkdavis/mec/Sept2021sci/dm_data/dm00disp00/dm00disp00_11:25:17.842164902.fits'
    # Fits Import
    from astropy.io import fits
    hdul = fits.open(dmfits_file)
    dm_map = hdul[0].data
    hdul.close()

    if debug:  # Plot flat
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.suptitle('DM flat', fontweight='bold', fontsize=14)
        ax.imshow(dm_map[0], interpolation='none')

    return np.squeeze(dm_map)

def intensity_counter(probes, nulls):
    """determines the intensity of probed vs unprobed images"""
    I_probed = np.zeros(len(probes))
    I_nulls = np.zeros(len(nulls))
    for ix in range(len(probes)):
        I_probed[ix] = np.sum(probes[ix])
    for ix in range(len(nulls)):
        I_nulls[ix] = np.sum(nulls[ix])

    print(f'Avg Intensity for probed images is   = {np.mean(I_probed)}\n'
          f'Avg intensity for unprobed images is = {np.mean(I_nulls)}')

##
if __name__ == '__main__':
    target_name = 'Sept2021sci_run3'
    datf ='/darkdata/kkdavis/mec/Sept2021sci/pkl/Sept2021sci_run3_1631274604_processed_shft0.5.npz'
    run_data = np.load(datf, allow_pickle=True)
    fp_seq = run_data['tcube_regcycle']  #.tolist()
    cdi_zip = open_MEC_tseries('/darkdata/kkdavis/mec/Sept2021sci/pkl/CDI_tseries_9-10-2021_T11:51.pkl')
    map_dir = None
    # map_dir = '/darkdata/kkdavis/mec/Sept2021sci/dm_data/dm00disp'
    h5_name = run_data['h5_unix'].tolist()
    plt_path = os.path.join(run_data['base_pth'].tolist(),'plots')

    if not 'h5' in run_data.files and 'file_h5' in run_data.files:
        r2 = os.path.basename(run_data['file_h5'].tolist())
        h5_name = os.path.splitext(r2)
        h5_name = h5_name[0]

    grail = cdi_postprocess(fp_seq, cdi_zip, map_dir=map_dir,  plot=True, debug=True)
    plt.show()
    dumm=1
