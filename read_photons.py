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
from matplotlib.colors import LogNorm, SymLogNorm

from cdi_plots import plot_probe_response_cycle, plot_quick_coord_check, plot_probe_response, plot_probe_cycle
# import mkidpipeline.photontable as pt
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


def check_h5_duration(table):
    photons = table.photonTable.read()
    frst_photon = photons['Time'].min()
    last_photon = photons['Time'].max()
    print(f'Last Photon {last_photon/1e6:.2f} sec after h5 start')
    return last_photon


def list_keys(file):
    """
    prints the names of the variables stored in a .npz file without pre-loading in the entire array
    :param file: name of a .npz file
    :return:
    """
    data_in = np.load(file, mmap_mode='r')
    list(data_in.files)
    print(data_in.files)

##
if __name__ == '__main__':

    ##
    # target_name = 'Jan2021_test5'  # None
    # dm_file = '/darkdata/kkdavis/mec/Jan2021/CDI_tseries_1-30-2021_T4:16.pkl'
    # file_h5 = '/work/kkdavis/pipeline_out/20210129/1611980038.h5'
    target_name = 'Vega_2021_run5'  # None
    dm_file = '/darkdata/kkdavis/mec/May2021c/CDI_tseries_5-25-2021_T11:53.pkl'
    file_h5 = '/darkdata/kkdavis/mec/May2021c/h5s/1621943458.h5'

    # target_name = 'Hip79124'  # None
    # file_h5 ='/darkdata/steiger/MEC/20200705/Hip79124/1594023761.h5'  #

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
    cdi_zip = open_MEC_tseries(dm_file)

    firstUnixTstep = first_tstep(cdi_zip)
    lastUnixTstep = last_tstep(cdi_zip)
    total_h5_seconds = lastUnixTstep - firstUnixTstep
    print(
        # f'\n\n{target_name} {h5_name_parts[0]}\n
        f'{dm_name_parts[0]}:\n\t'
        f'First Timestep = {first_tstep(cdi_zip):.6f}\n\tLast Timestep = {last_tstep(cdi_zip):.6f}\n'
        f'Duration = {cdi_zip.ts.elapsed_time/60:.2f} min ({cdi_zip.ts.elapsed_time:.4f} sec)')

    ## Printing DM Info
    print(f'\nDM Probe Series Info\n\t'
          f'Probe Dir {cdi_zip.probe.direction}, Amp = {cdi_zip.probe.amp}\n\t'
          f'Probe phase intervals {cdi_zip.probe.phs_interval/np.pi:.2f}pi\n\t'
          f'Timing: Phase Integration {cdi_zip.ts.phase_integration_time} sec, Null Time {cdi_zip.ts.null_time} sec\n\t'
          f'# probes {cdi_zip.ts.n_probes}, # Cycles: {cdi_zip.ts.n_cycles}, # Commands: {cdi_zip.ts.n_cmds} \n\t'
          f'Time for one cycle: {cdi_zip.ts.t_one_cycle:.4f} sec\n\t'
          f'Total Elapsed Time: {cdi_zip.ts.elapsed_time/60:.2f} min ({cdi_zip.ts.elapsed_time:.4f} sec)')

    ## Check Datasets Match
    if float(h5_name_parts[0])*1e9 > cdi_zip.ts.cmd_tstamps[-1].astype('float') or \
        float(h5_name_parts[0])*1e9 < (cdi_zip.ts.cmd_tstamps[0].astype('float')-5e10):
        warnings.warn(f"{CCYN}h5 file does not match DM commanded range{CEND}\n"
                      f"{h5_name_parts[0]}.h5 not within "
                      f"{first_tstep(cdi_zip)-5:.0f} to {last_tstep(cdi_zip):.0f}")
    else:
        print(f'\n\tDatasets match\n')

    ## Timestamp Conversion & Syncing
    tstamps_as_unix = cdi_zip.ts.cmd_tstamps.astype('float64') / 1e9
    tstamps_from_h5_start = tstamps_as_unix - h5_start
    if tstamps_from_h5_start[0] < 0:
        plus = tstamps_from_h5_start > 0
        # tstamps_from_h5_start = tstamps_from_h5_start[plus]#np.delete(tstamps_from_h5_start, plus)

    tstamps_from_tstamps_start = cdi_zip.ts.cmd_tstamps - cdi_zip.ts.cmd_tstamps[0]
    tstamps_from_tstamps_start = tstamps_from_tstamps_start.astype('float64') / 1e9

    ## Load tCube from saved file
    existing = f'{dm_path}/{target_name}_{h5_name_parts[0]}_processed.npz'
    if os.path.isfile(existing):
        list_keys(existing)
        loaded = np.load(existing, allow_pickle=True)
        tcube_fullcycle = loaded['tcube_fullcycle'].tolist()
        tcube_regcycle = loaded['tcube_regcycle'].tolist()
        # tcube_fullcycle = np.load(f'{dm_path}/{target_name}_{h5_name_parts[0]}_temporalCube_regBins.npy', allow_pickle=True)  # SciOct2020_tcube_fullcycle_1602072901.npy
        # tcube_regcycle = np.load(f'{dm_path}/{target_name}_{h5_name_parts[0]}_temporalCube_irregBins.npy', allow_pickle=True)
        # cimg1 = np.load('/work/kkdavis/cdi/ScienceOct2020/SciOct2020_piximg_1602072901.npy')
    else:
        pass

        ##  Create Photontable from h5
        table1 = pt.Photontable(file_h5)
        last_sec = cdi_zip.ts.elapsed_time + tstamps_from_h5_start[0]
        if last_sec > check_h5_duration(table1):
            warnings.warn(f'\n{CCYN}CDI Test Exceedes h5 duration')

        ## Make Image Cube
        # print(f'\nMaking Total Intensity Image')
        # cimg_start = time.time()
        #
        # cimg1 = table1.getPixelCountImage()['image']  # total integrated over the whole time
        #
        # cimg_end = time.time()
        # duration_make_cimg = cimg_end - cimg_start
        # print(f'time to make count image is {duration_make_cimg/60:.2f} minutes')


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

        cycle_Nsteps = np.int(cdi_zip.ts.t_one_cycle/cdi_zip.ts.phase_integration_time)

        tcube_1cycle = table1.getTemporalCube(timeslices=tstamps_from_h5_start[0:3*cycle_Nsteps])

        end_make_cube = time.time()
        duration_make_tcube = end_make_cube - start_make_cube
        print(f'time to make one-cycle temporal cube is {duration_make_tcube/60:.2f} minutes '
              f'({duration_make_tcube:.2f} sec)')
        np.save(f'{dm_path}/{target_name}_{h5_name_parts[0]}_temporalCube_quick', tcube_1cycle['cube'])

        ## Temporal Cube full dataset--regular bins
        """
        makes a temporal cube where each bin is spaced by the probe integration time. This way, an unprobed length 
        of time is split up into multiple time bins, rather than being lumped into one longer duration time bin.  
        This is useful for comparing intensity in a single probed timestep to an equivalent 'null' timestep
        
        integration time is a misnomer for getTemporalCube. What it really wants is the last timestep that you want 
        included in the cube, in units of seconds after the h5 file start
        """
        print(f'\nMaking Temporal Cube-Full h5 Duration, bin spacing set by probe integration time')
        start_make_cube = time.time()
        tcube_regcycle = table1.getTemporalCube(firstSec=tstamps_from_h5_start[0],
                                              integrationTime=last_sec,
                                              timeslice=cdi_zip.ts.phase_integration_time)

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
                 tcube_regcycle=tcube_regcycle,
                 tcube_fullcycle=tcube_fullcycle,
                 # cmd_tstamps=cdi_zip.ts.cmd_tstamps,
                 h5_fullpath=file_h5,
                 h5_unix=h5_name_parts[0],
                 rtc_zipfile=dm_file)

        # np.savez(f'{scratch_path}/MEC_{target_name}_{h5_name_parts[0]}_forJessica',
        #          file_h5=file_h5,
        #          intensity_image=cimg1)

        # Save just the 1 cycle temporal cube
        # np.save(f'SciOct2020_{firstUnixTstep}_tcube_firstcycles_test1', tcube_1cycle['cube'])

        # Save the full cycles temporal cube
        np.save(f'{dm_path}/{target_name}_{h5_name_parts[0]}_temporalCube_regBins', tcube_regcycle['cube'])
        np.save(f'{dm_path}/{target_name}_{h5_name_parts[0]}_temporalCube_irregBins', tcube_fullcycle['cube'])


#===============================================================
# Plotting
#===============================================================
    # ##  Pixel Count Image (cmg1)
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # fig.suptitle(f'Total Pixel Count Image: {h5_name_parts[0]}')
    # ax.imshow(cimg1, interpolation='none')  # [70:140,10:90,:]
    # plt.show()
    ## Probe Response (from mec_cdi.py)
    save_set = {'save': True, 'dm': dm_path, 'target': target_name, 'h5': h5_name_parts[0]}
    plot_probe_response(cdi_zip, 0, save_set)
    # plot_probe_cycle(cdi_zip)
    # plot_probe_response_cycle(cdi_zip)
    plot_quick_coord_check(cdi_zip, 0, save_set)

    ## Separating Probes & Null Steps by Temporal Cube type
    """
    because of the way mec_cdi.py is structured, the null step is of arbitrary length compared to the length of the 
    integration time of each probe pattern. thus, cdi_zip.ts.phase_integration_time ~= cdi_zip.ts.null_time, 
    and there is no check to make sure the null time is an integer number of phase integration times. This leads to 
    the null time being a single long step rather than an integer number of timesteps of the phase_integration_time. 
    
    We can treat this in two ways. The first is to make the temporal cube with regular bin spacing, meaning that when
    the temporal cube is made, we give it the first second of the probe applied to the DM and then make regular spaced
    bins from then on (which assumes the time the null was applied is n*cdi_zip.ts.phase_integration_time). The other
    way we can handle this is to make irregular spaced bins, meaning that when the temporal cube is made, we send in
    each timestep that we record from the DM output, which gives us a cut down to the us and is more accurate. However, 
    since there was only one 'null command', there is only one 'null timestep'. This is only bothersome when you want 
    to try to compare the count during a probe phase vs the null time; a null timestep 3x as long as any one phase 
    integration is going to look much higher than when the probe was applied. Conversely, this way it is easier to 
    apply the reconstructed E-field to the entire null step. It is also easier to remove the null steps as single 
    points rather than groups of points from the cube. 
    
    TL/dr if the temporal cube was made with regular bins, each full cycle of the probe has n_probes + n_null steps,
    otherwise it a full cycle of the probe has n_probes + 1 null step
    """
    # Plot Data Length
    # plt_cycles = cdi_zip.ts.n_cycles  # plot a subset of the full length of the temporal cube
    plt_cycles = 5
    bins = 'regular'  # 'regular' or 'irregular'

    if target_name == 'Jan2021_test2':
        tcube_regcycle = {'cube':np.array(tcube_regcycle)}

    # oc -> original cube; tax -> time axis
    if bins == 'regular':
        n_nulls = cdi_zip.ts.null_time / cdi_zip.ts.phase_integration_time
        if not n_nulls.is_integer():
            n_nulls = np.int(n_nulls)
        plt_length = np.int((cdi_zip.ts.n_probes + n_nulls) * plt_cycles)
        if plt_length > tcube_regcycle['cube'].shape[2]:
            plt_length = tcube_regcycle['cube'].shape[2]
        # plt_length = np.int(np.floor(260 / cdi_zip.ts.t_one_cycle))
        oc = tcube_regcycle['cube'][:, :, 0:plt_length]
        tax = np.linspace(tstamps_from_h5_start[0], tstamps_from_h5_start[-1],  # [plt_length]
                        plt_length)
    else:
        bins = 'irregular'
        warnings.warn(f'\n\n{CCYN}'
                      f'Warning: Nulls are {n_nulls}x the length of phase probe integration times (non-integer)\n'
                      f'{CPRP}Setting bins to irreg (time bins selected by DM timestamps){CEND}')

    if bins == 'irregular':
        n_nulls = 1
        plt_length = np.int((cdi_zip.ts.n_probes+1)*plt_cycles)
        oc = tcube_fullcycle['cube'][:, :, 0:plt_length]
        tax = tstamps_from_h5_start[0:plt_length]

    # # Separating Probes & Null Steps
    probe_mask = np.append(np.repeat(True, cdi_zip.ts.n_probes), np.repeat(False, n_nulls))
    probe_mask = np.tile(probe_mask, plt_cycles)
    probe_mask = probe_mask[0:plt_length]

    ## Pixel Count Image (Temporal Image Cube summed over oc length)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(f'Total Pixel Count Image: {h5_name_parts[0]}{h5_name_parts[1]}\n'
                 f'{target_name}', fontweight='bold', fontsize=14)
    ax.imshow(np.sum(oc, axis=2).T, interpolation='none')  # [70:140,10:90,:]

    ## Pixel Count Image Subarray, summed over full plot length
    rowstart = 15
    rowend = 125
    colstart = 30
    colend = 140

    xr = slice(colstart, colend)
    yr = slice(rowstart, rowend)
    # subarr = oc[xr, yr, 0:cdi_zip.ts.n_probes-1]
    subarr = oc[xr, yr, :]

    # msk_x = slice(25,60)
    # msk_y = slice(59,139)
    # fp_mask = (np.sum(subarr, axis=2) > 2e5)
    # screwed = np.sum(subarr, axis=2) - (fp_mask*2e5)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 25))
    fig.subplots_adjust(left=0.05, hspace=.4, wspace=0.2)

    fig.suptitle(f'MEC Nulls Summed Pre CDI\n {h5_name_parts[0]}{h5_name_parts[1]}, target= {target_name}\n'
                 f' N probes={cdi_zip.ts.n_probes}, '
                 f'N null steps={np.int(cdi_zip.ts.null_time / cdi_zip.ts.phase_integration_time)}, '
                 f'integration time={cdi_zip.ts.phase_integration_time} sec', fontweight='bold', fontsize=14)
    im = ax.imshow(np.sum(subarr, axis=2).T, interpolation='none')  # np.sum(subarr, axis=2) [70:140,10:90,:]
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

    if cdi_zip.ts.n_probes >= 4:
        nrows = 2
        ncols = cdi_zip.ts.n_probes // 2
        figheight = 6
    else:
        nrows = 1
        ncols = cdi_zip.ts.n_probes
        figheight = 2

    fig, subplot = plt.subplots(nrows, ncols, figsize=(14, 25))
    fig.subplots_adjust(left=0.05, hspace=.4, wspace=0.2)

    fig.suptitle(f'MEC CDI Probe Response of {h5_name_parts[0]}{h5_name_parts[1]}, target= {target_name}\n'
                 f' N probes={cdi_zip.ts.n_probes}, '
                 f'N null steps={np.int(cdi_zip.ts.null_time / cdi_zip.ts.phase_integration_time)}, '
                 f'integration time={cdi_zip.ts.phase_integration_time} sec', fontweight='bold', fontsize=14)
    for ax, ix in zip(subplot.flatten(), range(cdi_zip.ts.n_probes)):
        im = ax.imshow(subarr[:,:,ix].T, interpolation='none')  # [70:140,10:90,:]
        ax.set_title(f"Probe " + r'$\theta$=' + f'{cdi_zip.ts.phase_cycle[ix] / np.pi:.2f}' + r'$\pi$')
        ax.set_xticks(np.linspace(0, subarr.shape[0], 10, dtype=np.int))
        ax.set_yticks(np.linspace(0, subarr.shape[1], 10, dtype=np.int))
        ax.set_xticklabels(np.linspace(colstart, colend, 10, dtype=np.int))
        ax.set_yticklabels(np.linspace(rowstart, rowend, 10, dtype=np.int))

    # warnings.simplefilter("ignore", category=UserWarning)
    cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])  # Add axes for colorbar @ position [left,bottom,width,height]
    cb = fig.colorbar(im, cax=cbar_ax, orientation='vertical')  #
    cb.set_label(f'Counts', fontsize=12)

    ## Checking Pixel locations for timestream

    pix1 = [107, 66]
    pix2 = [100, 55]  #
    pix3 = [74, 47]  # not a speckle
    pix4 = [88, 63]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle(f'Timestreams from Selected Pixels, {bins} Bins \n'
                 f'target = {target_name}, {h5_name_parts[0]}{h5_name_parts[1]}\n'
                 f' N probes={cdi_zip.ts.n_probes}, '
                 f'N null steps={np.int(cdi_zip.ts.null_time / cdi_zip.ts.phase_integration_time)}, '
                 f'integration time={cdi_zip.ts.phase_integration_time} sec', fontweight='bold', fontsize=14)
    ax.imshow(oc[:, :, 0].T, interpolation='none')
    plt.plot(pix1[0], pix1[1], 'r*')
    plt.plot(pix2[0], pix2[1], 'r*')
    plt.plot(pix3[0], pix3[1], 'r*')
    plt.plot(pix4[0], pix4[1], 'r*')

    plt.savefig(f'{dm_path}/plots/{target_name}_{h5_name_parts[0]}_stars.png')


    ## Time Stream from Selected Pixels: Nulls + Probes Different Colors

    fig, axs = plt.subplots(4, 1, figsize=(10, 40))
    fig.subplots_adjust(wspace=0.3, hspace=0.5)
    ax1, ax2, ax3, ax4 = axs.flatten()
    colors = ['blue', 'orange']
    label = ('Probe', 'Null')
    fig.suptitle(f'Timestreams from Selected Pixels, {bins} Bins \n'
                 f'target = {target_name}, {h5_name_parts[0]}{h5_name_parts[1]}\n'
                 f' N probes={cdi_zip.ts.n_probes}, '
                 f'N null steps={np.int(cdi_zip.ts.null_time / cdi_zip.ts.phase_integration_time)}, '
                 f'integration time={cdi_zip.ts.phase_integration_time} sec', fontweight='bold', fontsize=14)

    for s, l in zip((probe_mask, ~probe_mask), label):
        ax1.plot(tax[s], oc[pix1[0], pix1[1], s], '.', label=l)
        ax1.legend()
    ax1.set_title(f'Pixel {pix1}, Probe Region')

    for s, l in zip((probe_mask, ~probe_mask), label):
        ax2.plot(tax[s], oc[pix2[0], pix2[1], s], '.', label=l)
    ax2.set_title(f'Pixel {pix2}, Probe Region')

    for s, l in zip((probe_mask, ~probe_mask), label):
        ax3.plot(tax[s], oc[pix3[0], pix3[1], s], '.', label=l)
    ax3.set_title(f'Pixel {pix3}, Unprobed Region')

    for s, l in zip((probe_mask, ~probe_mask), label):
        ax4.plot(tax[s], oc[pix4[0], pix4[1], s], '.', label=l)
    ax4.set_title(f'Pixel {pix4}, Probe Region')

    plt.savefig(f'{dm_path}/plots/{target_name}_{h5_name_parts[0]}_tstream_pix.png')


    # ## Animation
    # import matplotlib.animation as animation
    #
    # rowstart = 15
    # rowend = 125
    # colstart = 30
    # colend = 140
    #
    # xr = slice(colstart, colend)
    # yr = slice(rowstart, rowend)
    # subarr = oc[xr, yr, :]
    #
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # # fig.subplots_adjust(left=0.05, hspace=.4, wspace=0.2)
    #
    # ims = []
    # phs_cnt = 0
    # for ix in range(plt_length):  #
    #     if (ix / (cdi_zip.ts.n_probes + n_nulls)).is_integer() and ix != 0:
    #         phs_cnt += (cdi_zip.ts.n_probes + n_nulls)
    #     if probe_mask[ix]:
    #         phase = f'{cdi_zip.ts.phase_cycle[np.int(ix-phs_cnt)] / np.pi:.2f}'
    #         color = 'k'
    #     else:
    #         phase = f' NULL '
    #         color = 'cyan'
    #
    #     im = ax.imshow(subarr[:, :, ix].T)  # [70:140,10:90,:]
    #     ttl = plt.text(0.5, 1.01, f"{target_name}, file {h5_name_parts[0]}{h5_name_parts[1]}\n"
    #                    f' N probes={cdi_zip.ts.n_probes}, '
    #                    f'N null steps={np.int(cdi_zip.ts.null_time / cdi_zip.ts.phase_integration_time)}, '
    #                    f'integration time={cdi_zip.ts.phase_integration_time} sec \n'
    #                    f"Probe " + r'$\theta$=' + phase + r'$\pi$',
    #                    horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes,
    #                    fontsize='large', color=color)
    #     ax.set_xticks(np.linspace(0, subarr.shape[1], 10, dtype=np.int))
    #     ax.set_yticks(np.linspace(0, subarr.shape[0], 10, dtype=np.int))
    #     ax.set_xticklabels(np.linspace(colstart, colend, 10, dtype=np.int))
    #     ax.set_yticklabels(np.linspace(rowstart, rowend, 10, dtype=np.int))
    #
    #     # cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])  # Add axes for colorbar @ position [left,bottom,width,height]
    #     # cb = fig.colorbar(ax, cax=cbar_ax, orientation='vertical')  #
    #     # cb.set_label(f'Counts', fontsize=12)
    #
    #     ims.append([im, ttl])
    #
    # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=False)  # time in miliseconds 0.001 s
    # ani.save(f'{dm_path}/plots/{target_name}_withRegularSpacedNulls.gif')

## Delta Probe Focal Plane (absDP)
    fp_sequence = tcube_regcycle['cube'][:, :, 0:plt_length]

    n_pairs = cdi_zip.ts.n_probes // 2  # number of deltas (probe differentials)
    n_nulls = np.int(cdi_zip.ts.null_time // cdi_zip.ts.phase_integration_time)
    n_nulls=2

    # fp_mask, edges, _, _, _, _ = get_fp_mask(cdi_zip, thresh=1e-5)

    # DeltaP
    fig, subplot = plt.subplots(n_nulls, n_pairs, figsize=(12, 8))
    fig.subplots_adjust(wspace=0.5, right=0.85)
    fig.suptitle(r'$\frac{I_i^+ + I_i^-}{2} - I_{null}$ for CDI Probes  ' +
                 f'target = {target_name}, {h5_name_parts[0]}{h5_name_parts[1]}\n'
                 f'Probe Amp = {cdi_zip.probe.amp}, N probes={cdi_zip.ts.n_probes}, '
                 f'N null steps={np.int(cdi_zip.ts.null_time / cdi_zip.ts.phase_integration_time)}, '
                 f'integration time={cdi_zip.ts.phase_integration_time} sec\n', fontweight='bold', fontsize=14)
    cnt = np.tile(range(n_pairs),n_nulls)
    nx=1
    for ax, ix in zip(subplot.flatten(), range(n_nulls*n_pairs)):
        Ip = fp_sequence[:,:,cnt[ix]]
        Im = fp_sequence[:,:,cnt[ix] + n_pairs]
        Io = fp_sequence[:,:,cdi_zip.ts.n_probes + (ix//n_pairs+0)]

        absDP = (Ip + Im) / 2 - Io
        # if absDP.any() < 0:
        #     absDP = 0
        # absDeltaP = np.sqrt(absDP)
        im = ax.imshow(absDP.T, interpolation='none',  #
                       vmin=-1, vmax=1,
                       cmap='plasma',
                       )  # , norm=SymLogNorm(linthresh=1e-5))
        ax.set_title(f"probe phase pair {cnt[ix]+1},\nnull step {ix//n_pairs+1+0}")
        # cl = LineCollection(edges, colors='r')
        # ax.add_collection(cl)
        if ix+1 > nx*n_pairs-1:
            nx+=1

    cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
    cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
    cb.set_label('Intensity')

    plt.savefig(f'{dm_path}/plots/{target_name}_{h5_name_parts[0]}_probe_Delta_null_1_2.png')

    ## Compare Probes to Nulls

    if cdi_zip.ts.n_probes >= 4:
        nrows = 2
        ncols = cdi_zip.ts.n_probes // 2
        figheight = 6
    else:
        nrows = 1
        ncols = cdi_zip.ts.n_probes
        figheight = 2

    intensity = np.zeros((nrows*ncols))

    fig, subplot = plt.subplots(nrows, ncols, figsize=(12, 8))
    fig.subplots_adjust(left=0.05, hspace=.4, wspace=0.2)

    fig.suptitle(f'MEC CDI Probe Response of {h5_name_parts[0]}{h5_name_parts[1]}, target= {target_name}\n'
                 f' N probes={cdi_zip.ts.n_probes}, '
                 f'N null steps={np.int(cdi_zip.ts.null_time / cdi_zip.ts.phase_integration_time)}, '
                 f'integration time={cdi_zip.ts.phase_integration_time} sec', fontweight='bold', fontsize=14)
    for ax, ix in zip(subplot.flatten(), range(cdi_zip.ts.n_probes)):
        if ix < cdi_zip.ts.n_probes //2:
            im = ax.imshow(oc[:, :, ix].T, interpolation='none')  # [70:140,10:90,:]
            ax.set_title(f"Probe " + r'$\theta$=' + f'{cdi_zip.ts.phase_cycle[ix] / np.pi:.2f}' + r'$\pi$')
            # print(f'Total Intensity at step {ix} = {np.sum(oc[:, :, ix]):.1f}')
            intensity[ix] = np.sum(oc[:, :, ix])
        else:
            im = ax.imshow(oc[:, :, ix + cdi_zip.ts.n_probes].T, interpolation='none')  # [70:140,10:90,:]
            ax.set_title(f"Null {ix-cdi_zip.ts.n_probes //2}")
            # print(f'Total Intensity at step {ix} = {np.sum(oc[:, :, ix + cdi_zip.ts.n_probes])}')
            intensity[ix] = np.sum(oc[:, :, ix + cdi_zip.ts.n_probes])
    print(f'Avg Intensity for probed images is = {np.mean(intensity[0:ncols-1])}\n'
          f'Avg intensity for unprobed images is = {np.mean(intensity[ncols:-1])}')

    cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])  # Add axes for colorbar @ position [left,bottom,width,height]
    cb = fig.colorbar(im, cax=cbar_ax, orientation='vertical')  #
    cb.set_label(f'Counts', fontsize=12)

    plt.savefig(f'{dm_path}/plots/{target_name}_{h5_name_parts[0]}_probe_vs_null.png')


##
    plt.show()

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


 ##
    # Plot length of each timestep over time

    diffs = np.zeros(cdi_zip.ts.n_cmds)
    for it in range(cdi_zip.ts.n_cmds-1):
        diffs[it] = (cdi_zip.ts.cmd_tstamps[it+1] - cdi_zip.ts.cmd_tstamps[it]) * 1e-9  # 1e-9 converts from ns to sec


    fig, ax = plt.subplots(1,1)
    # ax.plot(cdi_zip.ts.cmd_tstamps, np.ones(len(cdi_zip.ts.cmd_tstamps)),'r.')
    # ax.plot(diffs,'b.')
    ax.plot(tax[probe_mask], np.ones(tax[probe_mask].size), 'r.')
    # ax.set_ylim(bottom=1.9e-1,top=2.1e-1)
    
## Beam Image --don't quite know what this is, but its like a row plot
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(table1.beamImage, interpolation='none', origin='lower')
    plt.show()

"""

