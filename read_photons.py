##
"""
read_photons.py
Krisitna Davis
09/2020

code to read in and correlate CDI timestream data (from mec_cdi.py) and coordinate it with processed temporal cubes
generated as mkidpipeline.PhotonTable. Temporal cubes are generated and saved for loading into the
postprocess_MEC_CDI.py function.

"""

import numpy as np
import os
import pickle
import time
import warnings
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm, SymLogNorm

from cdi_plots import plot_probe_response_cycle, plot_quick_coord_check, plot_probe_response, plot_probe_cycle
import mkidpipeline as pipe
from mec_cdi import CDI_params, Slapper  # need this to open .pkl files

# Color Definitions (for terminal output warnings)
CPRP = '\033[35m'
CGRN = '\033[36m'
CCYN = '\033[96m'
CEND = '\033[0m'

def open_MEC_tseries(CDI_tseries='CDI_tseries.pkl'):
    """opens existing MEC CDI timeseries .pkl file and return it"""
    with open(CDI_tseries, 'rb') as handle:
        CDI_meta = pickle.load(handle)
    return CDI_meta


def first_tstep(meta):
    """returns the first timestep time from pkl file. This is useful to tell the mkidpipeline when to start the obs"""
    first_t = meta.ts.cmd_tstamps[0]  #  [-1]
    return (first_t - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')


def last_tstep(meta):
    """returns the end timestep time from pkl file. This is useful to tell the mkidpipeline when to stop the obs"""
    last_t = meta.ts.cmd_tstamps[-1]   # [-3]
    return (last_t - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')


def datetime_to_unix(tstamp):
    """returns a unix timestep given a np.datetime input timestamp"""
    return (tstamp - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
##


if __name__ == '__main__':
    ##
    target_name = 'Jan2021_test2'  # None
    dm_file = '/darkdata/kkdavis/mec/Jan2021/CDI_tseries_1-30-2021_T3:57.pkl'  #
    file_h5 = '/work/kkdavis/pipeline_out/20210129/1611979028.h5'

    # Nifty File Name Extraction--makes nice print statements and gets unix timestamp name for file saving later
    r1 = os.path.basename(dm_file)
    dm_path = os.path.dirname(dm_file)
    dm_name_parts = os.path.splitext(r1)
    r2 = os.path.basename(file_h5)
    h5_name_parts = os.path.splitext(r2)
    h5_path = os.path.dirname(file_h5)
    h5_start = float(h5_name_parts[0])
    scratch_path = '/work/kkdavis/scratch'

    ## Open DM .pkl file
    dm_header = open_MEC_tseries(dm_file)

    firstUnixTstep = first_tstep(dm_header)
    lastUnixTstep = last_tstep(dm_header)
    total_h5_seconds = lastUnixTstep - firstUnixTstep
    print(
        f'\n\n{target_name} {h5_name_parts[0]}\n{dm_name_parts[0]}:\n\t'
        f'First Timestep = {first_tstep(dm_header):.6f}\n\tLast Timestep = {last_tstep(dm_header):.6f}\n'
        f'Duration = {dm_header.ts.elapsed_time/60:.2f} min ({dm_header.ts.elapsed_time:.4f} sec)')

    ## Printing DM Info
    print(f'\nDM Probe Series Info\n\t'
          f'Probe Dir {dm_header.probe.direction}\n\t'
          f'Probe phase intervals {dm_header.probe.phs_interval/np.pi:.2f}pi\n\t'
          f'Timing: Phase Integration {dm_header.ts.phase_integration_time} sec, Null Time {dm_header.ts.null_time} sec\n\t'
          f'# probes {dm_header.ts.n_probes}, # Cycles: {dm_header.ts.n_cycles}, # Commands: {dm_header.ts.n_cmds} \n\t'
          f'Time for one cycle: {dm_header.ts.t_one_cycle}\n\t'
          f'Total Elapsed Time: {dm_header.ts.elapsed_time/60:.2f} min ({dm_header.ts.elapsed_time:.4f} sec)')

    ## Load tCube from saved file
    tcube1 = np.load('/work/kkdavis/cdi/ScienceOct2020/SciOct2020_tcube_0-30sec_1602072901.npy')
    tcube_fullcycle = np.load(f'{dm_path}/{target_name}_{h5_name_parts[0]}_temporalCube_regBins.npy', allow_pickle=True)  # SciOct2020_tcube_fullcycle_1602072901.npy
    tcube_regcycle = np.load(f'{dm_path}/{target_name}_{h5_name_parts[0]}_temporalCube_irregBins.npy', allow_pickle=True)
    cimg1 = np.load('/work/kkdavis/cdi/ScienceOct2020/SciOct2020_piximg_1602072901.npy')

    ## Check Datasets Match
    if float(h5_name_parts[0])*1e9 > dm_header.ts.cmd_tstamps[-1].astype('float') or \
        float(h5_name_parts[0])*1e9 < (dm_header.ts.cmd_tstamps[0].astype('float')-5e10):
        warnings.warn(f"{CCYN}h5 file does not match DM commanded range{CEND}\n"
                      f"{h5_name_parts[0]}.h5 not within "
                      f"{first_tstep(dm_header)-5:.0f} to {last_tstep(dm_header):.0f}")
    else:
        print(f'\n\tDatasets match\n')

    ## Timestamp Conversion & Syncing
    tstamps_as_unix = dm_header.ts.cmd_tstamps.astype('float64') / 1e9
    tstamps_from_h5_start = tstamps_as_unix - h5_start
    if tstamps_from_h5_start[0] < 0:
        plus = tstamps_from_h5_start > 0
        # tstamps_from_h5_start = tstamps_from_h5_start[plus]#np.delete(tstamps_from_h5_start, plus)

    tstamps_from_tstamps_start = dm_header.ts.cmd_tstamps - dm_header.ts.cmd_tstamps[0]
    tstamps_from_tstamps_start = tstamps_from_tstamps_start.astype('float64') / 1e9

    ##  Create Photontable from h5
    table1 = pipe.Photontable(file_h5)

    ## Make Image Cube
    print(f'\nMaking Total Intensity Image')
    cimg_start = time.time()

    cimg1 = table1.getPixelCountImage()['image']  # total integrated over the whole time

    cimg_end = time.time()
    duration_make_cimg = cimg_end - cimg_start
    print(f'time to make count image is {duration_make_cimg/60:.2f} minutes')


    ## Make Temporal Cube- one cycle
    """
    firstSec = [seconds after first tstep in the h5 file]
    integrationTime = [seconds after first tstep in the h5 file], duration of the cube (second of last tstep after firstSec)
    timeslice = bin width of the timescale axis (integration time of each bin along z axis of cube)

    ** note here, you do not enter the times in unix timestamp, rather by actual seconds where 0 is the start of the
    h5 file
    """
    print(f'\nMaking Temporal Cube-One Cycle')
    start_make_cube = time.time()

    cycle_Nsteps = np.int(dm_header.ts.t_one_cycle/dm_header.ts.probe_integration_time)

    tcube_1cycle = table1.getTemporalCube(timeslices=tstamps_from_h5_start[0:3*cycle_Nsteps])

    # tcube_1cycle = table1.getTemporalCube(firstSec=0,
    #                                 # integrationTime=dm_header.ts.t_one_cycle,
    #                                 integrationTime=24,
    #                                 timeslice=0.1)  # dm_header.ts.probe_integration_time

    end_make_cube = time.time()
    duration_make_tcube = end_make_cube - start_make_cube
    print(f'time to make one-cycle temporal cube is {duration_make_tcube/60:.2f} minutes '
          f'({duration_make_tcube:.2f} sec)')
    #
    ## Temporal Cube full dataset--regular bins
    print(f'\nMaking Temporal Cube-Full h5 Duration, bin spacing set by probe integration time')
    start_make_cube = time.time()
    tcube_regcycle = table1.getTemporalCube(firstSec=tstamps_from_h5_start[0],
                                          integrationTime=dm_header.ts.elapsed_time,
                                          timeslice=dm_header.ts.phase_integration_time)

    end_make_cube = time.time()
    duration_make_tcube = end_make_cube - start_make_cube
    print(f'time to make full h5 duration temporal cube is {duration_make_tcube / 60:.2f} minutes'
          f'({duration_make_tcube:.2f} sec)')

    ## Temporal Cube full dataset--irregular bins
    print(f'\nMaking Temporal Cube-Full h5 Duration, bin spacing set by DM timestamps')
    start_make_cube = time.time()
    tcube_fullcycle = table1.getTemporalCube(timeslices=tstamps_from_h5_start)

    end_make_cube = time.time()
    duration_make_tcube = end_make_cube - start_make_cube
    print(f'time to make full h5 duration temporal cube is {duration_make_tcube / 60:.2f} minutes'
          f'({duration_make_tcube:.2f} sec)')

    ## Saving Created Data
    # Save several together
    np.savez(f'{dm_path}/{target_name}_{h5_name_parts[0]}_processed',
             tcube_regcycle=tcube_regcycle['cube'],
             tcube_fullcycle=tcube_fullcycle['cube'],
             dm_header=dm_header,
             file_h5=file_h5,
             dm_file=dm_file)

    # Save just the 1 cycle temporal cube
    np.save(f'SciOct2020_{firstUnixTstep}_tcube_firstcycles_test1', tcube_1cycle['cube'])

    # Save the full cycles temporal cube
    np.save(f'{dm_path}/{target_name}_{h5_name_parts[0]}_temporalCube_regBins', tcube_regcycle['cube'])
    np.save(f'{dm_path}/{target_name}_{h5_name_parts[0]}_temporalCube_irregBins', tcube_fullcycle['cube'])

#===============================================================
# Plotting
#===============================================================
    ##  Pixel Count Image (cmg1)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(f'Total Pixel Count Image: {h5_name_parts[0]}')
    ax.imshow(cimg1, interpolation='none')  # [70:140,10:90,:]

    ## Probe Response (from mec_cdi.py)
    plot_probe_response(dm_header, 0)
    plot_probe_cycle(dm_header)
    plot_probe_response_cycle(dm_header)
    plot_quick_coord_check(dm_header, 0)

    ## Beam Image
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(table1.beamImage, interpolation='none', origin='lower')
    plt.show()

    ## Timeslices CDI 1 Cycle

    if dm_header.ts.n_probes >= 4:
        nrows = 2
        ncols = dm_header.ts.n_probes // 2
        figheight = 6
    else:
        nrows = 1
        ncols = dm_header.ts.n_probes
        figheight = 2

    fig, subplot = plt.subplots(nrows, ncols, figsize=(14, 25))
    # fig.subplots_adjust(left=0.02, hspace=.4, wspace=0.2)

    fig.suptitle(f'MEC CDI Probe Response of {h5_name_parts[0]}{h5_name_parts[1]}, target= {target_name}\n'
                 f' N probes={dm_header.ts.n_probes}, '
                 f'N null steps={np.int(dm_header.ts.null_time / dm_header.ts.probe_integration_time)}, '
                 f'integration time={dm_header.ts.probe_integration_time} sec')

    for ax, ix in zip(subplot.flatten(), range(dm_header.ts.n_probes)):
        im = ax.imshow(tcube_fullcycle['cube'][:,:,ix], interpolation='none')  # [55:140,25:125,ix], [:,:,ix],
        ax.set_title(f"Probe " + r'$\theta$=' + f'{dm_header.ts.phase_cycle[ix] / np.pi:.2f}' + r'$\pi$')

    # warnings.simplefilter("ignore", category=UserWarning)
    cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])  # Add axes for colorbar @ position [left,bottom,width,height]
    cb = fig.colorbar(im, cax=cbar_ax, orientation='vertical')  #
    cb.set_label(f'Counts', fontsize=12)


    ## Separating Probes & Null Steps by Temporal Cube type
    """
    because of the way mec_cdi.py is structured, the null step is of arbitrary length compared to the length of the 
    integration time of each probe pattern. thus, dm_header.ts.probe_integration_time ~= dm_header.ts.null_time, 
    and there is no check to make sure the null time is an integer number of phase integration times. This leads to 
    the null time being a single long step rather than an integer number of timesteps of the probe_integration_time. 
    
    We can treat this in two ways. The first is to make the temporal cube with regular bin spacing, meaning that when
    the temporal cube is made, we give it the first second of the probe applied to the DM and then make regular spaced
    bins from then on (which assumes the time the null was applied is n*dm_header.ts.probe_integration_time). The other
    way we can handle this is to make irregular spaced bins, meaning that when the temporal cube is made, we send in
    each timestep that we record from the DM output, which gives us a cut down to the us and is more accurate. However, 
    since there was only one 'null command', there is only one 'null timestep'. This is only bothersome when you want 
    to try to compare the count during a probe phase vs the null time; a null timestep 3x as long as any one phase 
    integration is going to look much higher than when the probe was applied. Conversely, this way it is easier to 
    remove the null steps as single points rather than groups of points from the cube. 
    
    TL/dr if the temporal cube was made with regular bins, each full cycle of the probe has n_probes + n_null steps,
    otherwise it a full cycle of the probe has n_probes + 1 null step
    """
    # Plot Data Length
    plt_cycles = dm_header.ts.n_cycles  # plot a subset of the full length of the temporal cube
    bins = 'regular'  # 'regular' or 'irregular'

    # oc -> original cube; tax -> time axis
    if bins == 'regular':
        n_nulls = dm_header.ts.null_time / dm_header.ts.phase_integration_time
        if n_nulls.is_integer():
            n_nulls = np.int(n_nulls)
            plt_length = 42  #(dm_header.ts.n_probes + n_nulls) * plt_cycles
            # oc = tcube_regcycle[:, :, 0:plt_length]  # if loading form npz
            oc = tcube_regcycle['cube'][:, :, 0:plt_length]
            # oc = tcube_regcycle_bigger[:, :, 0:plt_length]
            tax = np.linspace(tstamps_from_h5_start[0], tstamps_from_h5_start[-1],  # [plt_length]
                            plt_length)
        else:
            bins = 'irregular'
            warnings.warn(f'\n\n{CCYN}'
                          f'Warning: Nulls are {n_nulls}x the length of phase probe integration times (non-integer)\n'
                          f'{CPRP}Setting bins to irreg (time bins selected by DM timestamps){CEND}')

    if bins == 'irregular':
        n_nulls = 1
        plt_length = np.int((dm_header.ts.n_probes+1)*plt_cycles)
        oc = tcube_fullcycle[:, :, 0:plt_length]  # if loading form npz
        # oc = tcube_fullcycle['cube'][:, :, 0:plt_length]
        tax = tstamps_from_h5_start[0:plt_length]

    # # Separating Probes & Null Steps
    probe_mask = np.append(np.repeat(True, dm_header.ts.n_probes), np.repeat(False, n_nulls))
    probe_mask = np.tile(probe_mask, plt_cycles)
    probe_mask = probe_mask[0:plt_length]

    ## Pixel Count Image (Temporal Image Cube summed over oc length)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(f'Total Pixel Count Image: {h5_name_parts[0]}{h5_name_parts[1]}\n'
                 f'{target_name}')
    ax.imshow(np.sum(oc, axis=2).T, interpolation='none')  # [70:140,10:90,:]

    ## Pixel Count Image Subarray, summed over full plot length
    rowstart = 60
    rowend = 140
    colstart = 30
    colend = 110

    xr = slice(rowstart, rowend)
    yr = slice(colstart, colend)
    # subarr = oc[xr, yr, 0:dm_header.ts.n_probes-1]
    subarr = oc[xr, yr, :]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 25))
    fig.subplots_adjust(left=0.05, hspace=.4, wspace=0.2)

    fig.suptitle(f'MEC Summed 1 Cycle\n {h5_name_parts[0]}{h5_name_parts[1]}, target= {target_name}\n'
                 f' N probes={dm_header.ts.n_probes}, '
                 f'N null steps={np.int(dm_header.ts.null_time / dm_header.ts.phase_integration_time)}, '
                 f'integration time={dm_header.ts.phase_integration_time} sec')
    im = ax.imshow(np.sum(subarr, axis=2), interpolation='none')  # [70:140,10:90,:]
    ax.set_xticks(np.linspace(0, subarr.shape[1], 10, dtype=np.int))
    ax.set_yticks(np.linspace(0, subarr.shape[0], 10, dtype=np.int))
    ax.set_xticklabels(np.linspace(colstart, colend, 10, dtype=np.int))
    ax.set_yticklabels(np.linspace(rowstart, rowend, 10, dtype=np.int))

    # warnings.simplefilter("ignore", category=UserWarning)
    cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])  # Add axes for colorbar @ position [left,bottom,width,height]
    cb = fig.colorbar(im, cax=cbar_ax, orientation='vertical')  #
    cb.set_label(f'Counts', fontsize=12)


    ## Pixel Count Image Sequence (subarray)
    rowstart = 15
    rowend = 125
    colstart = 30
    colend = 140

    xr = slice(colstart, colend)
    yr = slice(rowstart, rowend)
    subarr = oc[xr, yr, :]

    if dm_header.ts.n_probes >= 4:
        nrows = 2
        ncols = dm_header.ts.n_probes // 2
        figheight = 6
    else:
        nrows = 1
        ncols = dm_header.ts.n_probes
        figheight = 2

    fig, subplot = plt.subplots(nrows, ncols, figsize=(14, 25))
    fig.subplots_adjust(left=0.05, hspace=.4, wspace=0.2)

    fig.suptitle(f'MEC CDI Probe Response of {h5_name_parts[0]}{h5_name_parts[1]}, target= {target_name}\n'
                 f' N probes={dm_header.ts.n_probes}, '
                 f'N null steps={np.int(dm_header.ts.null_time / dm_header.ts.phase_integration_time)}, '
                 f'integration time={dm_header.ts.phase_integration_time} sec')
    for ax, ix in zip(subplot.flatten(), range(dm_header.ts.n_probes)):
        im = ax.imshow(subarr[:,:,ix].T, interpolation='none')  # [70:140,10:90,:]
        ax.set_title(f"Probe " + r'$\theta$=' + f'{dm_header.ts.phase_cycle[ix] / np.pi:.2f}' + r'$\pi$')
        ax.set_xticks(np.linspace(0, subarr.shape[0], 10, dtype=np.int))
        ax.set_yticks(np.linspace(0, subarr.shape[1], 10, dtype=np.int))
        ax.set_xticklabels(np.linspace(colstart, colend, 10, dtype=np.int))
        ax.set_yticklabels(np.linspace(rowstart, rowend, 10, dtype=np.int))

    # warnings.simplefilter("ignore", category=UserWarning)
    cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])  # Add axes for colorbar @ position [left,bottom,width,height]
    cb = fig.colorbar(im, cax=cbar_ax, orientation='vertical')  #
    cb.set_label(f'Counts', fontsize=12)

    plt.show()

    ## Time Stream from Selected Pixels: Nulls + Probes Different Colors

    fig, axs = plt.subplots(4, 1, figsize=(10, 40))
    fig.subplots_adjust(wspace=0.3, hspace=0.5)
    ax1, ax2, ax3, ax4 = axs.flatten()
    colors = ['blue', 'orange']
    label = ('Probe', 'Null')
    fig.suptitle(f'Timestreams from Selected Pixels, {bins} Bins \n'
                 f'target = {target_name}, {h5_name_parts[0]}{h5_name_parts[1]}\n'
                 f' N probes={dm_header.ts.n_probes}, '
                 f'N null steps={np.int(dm_header.ts.null_time / dm_header.ts.phase_integration_time)}, '
                 f'integration time={dm_header.ts.phase_integration_time} sec')
    # Test 8
    pix1 = [104, 47]
    pix2 = [121, 52]  #
    pix3 = [86, 61]  #
    pix4 = [90, 50]  #

    for s, l in zip((probe_mask, ~probe_mask), label):
        ax1.plot(tax[s], oc[pix1[0], pix1[1], s], '.', label=l)
        ax1.legend()
    ax1.set_title(f'Pixel {pix1}, CDI')

    for s, l in zip((probe_mask, ~probe_mask), label):
        ax2.plot(tax[s], oc[pix2[0], pix2[1], s], '.', label=l)
    ax2.set_title(f'Pixel {pix2}, CDI')

    for s, l in zip((probe_mask, ~probe_mask), label):
        ax3.plot(tax[s], oc[pix3[0], pix3[1], s], '.', label=l)
    ax3.set_title(f'Pixel {pix3}, Not Speckle')

    for s, l in zip((probe_mask, ~probe_mask), label):
        ax4.plot(tax[s], oc[pix4[0], pix4[1], s], '.', label=l)
    ax4.set_title(f'Pixel {pix4}, CDI')

    ##
    # Plot length of each timestep over time

    diffs = np.zeros(dm_header.ts.n_cmds)
    for it in range(dm_header.ts.n_cmds-1):
        diffs[it] = (dm_header.ts.cmd_tstamps[it+1] - dm_header.ts.cmd_tstamps[it]) * 1e-9  # 1e-9 converts from ns to sec


    fig, ax = plt.subplots(1,1)
    # ax.plot(dm_header.ts.cmd_tstamps, np.ones(len(dm_header.ts.cmd_tstamps)),'r.')
    # ax.plot(diffs,'b.')
    ax.plot(tax[probe_mask], np.ones(tax[probe_mask].size), 'r.')
    # ax.set_ylim(bottom=1.9e-1,top=2.1e-1)

    ## Animation
    import matplotlib.animation as animation

    rowstart = 15
    rowend = 125
    colstart = 30
    colend = 140

    xr = slice(colstart, colend)
    yr = slice(rowstart, rowend)
    subarr = oc[xr, yr, :]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # fig.subplots_adjust(left=0.05, hspace=.4, wspace=0.2)

    ims = []
    phs_cnt = 0
    for ix in range(plt_length):  #
        if (ix / (dm_header.ts.n_probes + n_nulls)).is_integer() and ix != 0:
            phs_cnt += (dm_header.ts.n_probes + n_nulls)
            print('Increasing Phase Counter')
        print(f'ix={ix}, phs_cnt={phs_cnt}')
        if probe_mask[ix]:
            phase = f'{dm_header.ts.phase_cycle[ix-phs_cnt] / np.pi:.2f}'
            color = 'k'
        else:
            phase = f' NULL '
            color = 'cyan'

        im = ax.imshow(subarr[:, :, ix].T)  # [70:140,10:90,:]
        ttl = plt.text(0.5, 1.01, f"{target_name}, file {h5_name_parts[0]}{h5_name_parts[1]}\n"
                       f' N probes={dm_header.ts.n_probes}, '
                       f'N null steps={np.int(dm_header.ts.null_time / dm_header.ts.phase_integration_time)}, '
                       f'integration time={dm_header.ts.phase_integration_time} sec \n'
                       f"Probe " + r'$\theta$=' + phase + r'$\pi$',
                       horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes,
                       fontsize='large', color=color)
        ax.set_xticks(np.linspace(0, subarr.shape[1], 10, dtype=np.int))
        ax.set_yticks(np.linspace(0, subarr.shape[0], 10, dtype=np.int))
        ax.set_xticklabels(np.linspace(colstart, colend, 10, dtype=np.int))
        ax.set_yticklabels(np.linspace(rowstart, rowend, 10, dtype=np.int))

        # cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])  # Add axes for colorbar @ position [left,bottom,width,height]
        # cb = fig.colorbar(ax, cax=cbar_ax, orientation='vertical')  #
        # cb.set_label(f'Counts', fontsize=12)

        ims.append([im, ttl])

    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=False)  # time in miliseconds 0.001 s
    # ani.save(f'{dm_path}/{target_name}_withRegularSpacedNulls.gif')


##
"""
    # ts1 = oc[81,79,:]
    # ts2 = oc[90,97,:]
    # ts3 = oc[102,101,:]
    # ts4 = oc[110,110,:]
    # tsc = ts1+ts2+ts3+ts4
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # ax.plot(tsc/4)
    # plt.show()
    pix1 = [84, 38]
    ax1.plot(oc[84, 38, :])
    ax1.set_title(f'Pixel {pix1}, CDI Region')

    pix2 = [79, 19]
    ax2.plot(range(oc.shape[2]), oc[79, 19, :])
    ax2.set_title(f'Pixel {pix2}, Astrogrid')

    pix3 = [108, 22]
    im3 = ax3.plot(oc[108, 22, :])
    ax3.set_title(f'Pixel {pix3}, Non-CDI region')

    pix4 = [126, 40]
    ax4.plot(oc[126, 40, :])
    ax4.set_title(f'Pixel {pix4}, bottom CDI region')
"""



"""
CDI2 Pixels
pix1 = [90,82]
    ax1.plot(oc[90,82,:])
    ax1.set_title(f'Pixel {pix1}, CDI Region')

    pix2 = [99,82]
    ax2.plot(range(oc.shape[2]), oc[99,82,:])
    ax2.set_title(f'Pixel {pix2}, CDI-speckle')

    pix3 = [106,58]
    im3 = ax3.plot(oc[106,58,:])
    ax3.set_title(f'Pixel {pix3}, Non-CDI region')

    pix4 = [99,52]
    ax4.plot(oc[99,52,:])
    ax4.set_title(f'Pixel {pix4}, Non-CDI region, speckle')



"""

