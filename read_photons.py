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
import datetime
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm, SymLogNorm

from mec_cdi import plot_probe_response_cycle, plot_quick_coord_check, plot_probe_response, plot_probe_cycle
import mkidpipeline as pipe


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
    target_name = ''  # None
    # file_h5 = '/darkdata/kkdavis/mec/SciOct2020/1602048860.h5'
    # dm_file = '/darkdata/kkdavis/mec/timingOct2020/CDI_tseries_10-20-2020_T21:23.pkl'  #
    file_h5 = '/darkdata/captainkay/mec/SciOct2020/1602048860.h5'
    dm_file = '/darkdata/captainkay/mec/SciOct2020/SciOct2020_config1_dummy2.pkl'

    # Nifty File Name Extraction--makes nice print statements and gets unix timestamp name for file saving later
    r1 = os.path.basename(dm_file)
    dm_path = os.path.dirname(dm_file)
    dm_name_parts = os.path.splitext(r1)
    r2 = os.path.basename(file_h5)
    h5_name_parts = os.path.splitext(r2)
    h5_path = os.path.dirname(file_h5)
    scratch_path = '/work/kkdavis/scratch'

    ## Load tCube from saved file
    tcube1 = np.load('/work/kkdavis/cdi/ScienceOct2020/SciOct2020_tcube_0-30sec_1602072901.npy')
    tcubeFull = np.load('/work/kkdavis/cdi/ScienceOct2020/SciOct2020_tcube_fullcycle_1602072901.npy')  # SciOct2020_tcube_fullcycle_1602072901.npy
    cimg1 = np.load('/work/kkdavis/cdi/ScienceOct2020/SciOct2020_piximg_1602072901.npy')
    
    ##  Load Photontable
    table1 = pipe.Photontable(file_h5)

    ## Open DM .pkl file
    dm_header = open_MEC_tseries(dm_file)

    firstUnixTstep = first_tstep(dm_header)
    lastUnixTstep = last_tstep(dm_header)
    total_h5_seconds = lastUnixTstep - firstUnixTstep
    print(
        f'\n\n{h5_name_parts[0]}\n{dm_name_parts[0]}:\n\t'
        f'First Timestep = {first_tstep(dm_header):.6f}\n\tLast Timestep = {last_tstep(dm_header):.6f}')

    

    ## Check Datasets Match

    ## Make Image Cube
    print(f'\nMaking Total Intensity Image')
    cimg_start = time.time()

    cimg1 = table1.getPixelCountImage()['image']  # total integrated over the whole time

    cimg_end = time.time()
    duration_make_cimg = cimg_end - cimg_start
    print(f'time to make count image is {duration_make_cimg/60:.2f} minutes')


    ## Make Temporal Cube
    """
    firstSec = [seconds after first tstep in the h5 file]
    integrationTime = [seconds after first tstep in the h5 file], duration of the cube (second of last tstep after firstSec)
    timeslice = bin width of the timescale axis (integration time of each bin along z axis of cube)

    ** note here, you do not enter the times in unix timestamp, rather by actual seconds where 0 is the start of the
    h5 file
    """
    print(f'\nMaking Temporal Cube-One Cycle')
    start_make_cube = time.time()

    tstamps_as_seconds = dm_header.ts.cmd_tstamps - dm_header.ts.cmd_tstamps[0]
    tstamps_as_seconds = tstamps_as_seconds.astype('float64') / 1e9
    cycle_Nsteps = np.int(dm_header.ts.t_one_cycle/dm_header.ts.phase_integration_time)

    tcube_1cycle = table1.getTemporalCube(timeslices=tstamps_as_seconds[0:3*cycle_Nsteps])

    # tcube_1cycle = table1.getTemporalCube(firstSec=0,
    #                                 # integrationTime=dm_header.ts.t_one_cycle,
    #                                 integrationTime=24,
    #                                 timeslice=0.1)  # dm_header.ts.phase_integration_time

    end_make_cube = time.time()
    duration_make_tcube = end_make_cube - start_make_cube
    print(f'time to make one-cycle temporal cube is {duration_make_tcube/60:.2f} minutes '
          f'({duration_make_tcube:.2f} sec)')
    #
    ## Temporal Cube full dataset
    print(f'\nMaking Temporal Cube-Full h5 Duration')
    start_make_cube = time.time()
    tcube_fullcycle = table1.getTemporalCube(firstSec=0,
                                          integrationTime=dm_header.ts.elapsed_time,
                                          timeslice=dm_header.ts.phase_integration_time)

    end_make_cube = time.time()
    duration_make_tcube = end_make_cube - start_make_cube
    print(f'time to make full h5 duration temporal cube is {duration_make_tcube / 60:.2f} minutes'
          f'({duration_make_tcube:.2f} sec)')


    ## Saving Created Data
    # Save several together
    np.savez(f'CDI2/CDI2_config_{firstUnixTstep}',
             table=table1,
             tcube_1c=tcube_1cycle['cube'],
             tcube_fc=tcube_fullcycle['cube'],
             meta=dm_header)

    # Save just the 1 cycle temporal cube
    # np.save(f'CDI2/CDI2_tcube_1cycle_{firstUnixTstep}', tcube_1cycle['cube'])
    np.save(f'SciOct2020_{firstUnixTstep}_tcube_firstcycles_test1', tcube_1cycle['cube'])

    # Save the full cycles temporal cube
    # np.save(f'SciOct2020_tcube_fullcycle_{firstUnixTstep}', tcube_fullcycle['cube'])
    np.save(f'/data0/captainkay/mec/CDI2/CDI2_tcube_fullcycle_{firstUnixTstep}', tcube_fullcycle['cube'])


    # Save just the pixel count img
    # np.save(f'{h5_path}/SciOct2020_piximg_{h5_name_parts[0]}', cimg1)
    np.save(f'{scratch_path}/{target_name}_{h5_name_parts[0]}_cimg', cimg1)

    # Pickling

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

    ## Recovery of Phase from timestream data
    # Data
    oc = tcube_1cycle['cube']
    # oc = tcubeFull[:,:,0:52]     # tcube_1cycle['cube']
    # oc = tcube_fullcycle['cube'][:, :, 0:52]

    # Time Axis
    tax = tstamps_as_seconds[0:3*cycle_Nsteps]


    ## Total Pixel Count Image In Subregion (total temporal cube)
    fig, subplt = plt.subplots(nrows=2, ncols=3)
    fig.suptitle(f'Temporal Cube: {h5_name_parts[0]}')
    for ax, p in zip(subplt.flatten(), range(2 * 3)):
        ax.imshow(oc, interpolation='none')
        # ax.imshow(tcube_fullcycle['cube'][:,:,p], interpolation='none')

    plt.show()

    ##
    rowstart = 70
    rowend = 140
    colstart = 10
    colend = 90
    # rowstart = 73
    # rowend = 140
    # colstart = 46
    # colend = 122

    xr = slice(rowstart, rowend)
    yr = slice(colstart, colend)
    subarr = oc[xr, yr, :]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(f'Total Pixel Count Image: {h5_name_parts[0]}')
    ax.imshow(np.sum(oc, axis=2), interpolation='none')  # [70:140,10:90,:]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(f'Total Pixel Count Image: {h5_name_parts[0]}{h5_name_parts[1]}\n'
                 f'N probes={dm_header.ts.n_probes}, '
                 f'N null steps={np.int(dm_header.ts.null_time/dm_header.ts.phase_integration_time)}')
    ax.imshow(np.sum(subarr, axis=2), interpolation='none')  # [70:140,10:90,:]
    ax.set_xticks(np.linspace(0, subarr.shape[1], 10, dtype=np.int))
    ax.set_yticks(np.linspace(0, subarr.shape[0], 10, dtype=np.int))
    ax.set_xticklabels(np.linspace(colstart, colend, 10, dtype=np.int))
    ax.set_yticklabels(np.linspace(rowstart, rowend, 10, dtype=np.int))

    plt.show()

    ##
    fig, axs = plt.subplots(4,1, figsize=(10,40))
    labels = ["{0:.3f}".format(x) for x in np.linspace(tax[0], tax[-1], 10)]
    fig.subplots_adjust(wspace=0.3, hspace=0.5)
    ax1, ax2, ax3, ax4 = axs.flatten()
    fig.suptitle('Timestreams from Selected Pixels\n'
                 f'{h5_name_parts[0]}{h5_name_parts[1]},     N probes={dm_header.ts.n_probes}, '
                 f'N null steps={np.int(dm_header.ts.null_time/dm_header.ts.phase_integration_time)}')
    pix1 = [84, 38]
    ax1.plot(oc[84, 38, :])
    ax1.set_title(f'Pixel {pix1}, CDI Region')
    ax1.set_xticks(np.linspace(0, len(tax), 10))
    ax1.set_xticklabels(labels)

    pix2 = [79, 19]
    ax2.plot(range(oc.shape[2]), oc[79, 19, :])
    ax2.set_title(f'Pixel {pix2}, Astrogrid')
    ax2.set_xticks(np.linspace(0, len(tax), 10))
    ax2.set_xticklabels(labels)

    pix3 = [102, 27]
    im3 = ax3.plot(oc[102, 27, :])
    ax3.set_title(f'Pixel {pix3}, Non-CDI region')
    ax3.set_xticks(np.linspace(0, len(tax), 10))
    ax3.set_xticklabels(labels)

    pix4 = [126, 40]
    ax4.plot(oc[126, 40, :])
    ax4.set_title(f'Pixel {pix4}, bottom CDI region')
    ax4.set_xticks(np.linspace(0, len(tax), 10))
    ax4.set_xticklabels(labels)
    plt.show()

    ##
    # Plot timestream

    diffs = np.zeros(dm_header.ts.n_cmds)
    for it in range(dm_header.ts.n_cmds-1):
        diffs[it] = (dm_header.ts.cmd_tstamps[it+1] - dm_header.ts.cmd_tstamps[it]) * 1e-9  # 1e-9 converts from ns to sec


    fig, ax = plt.subplots(1,1)
    # ax.plot(dm_header.ts.cmd_tstamps, np.ones(len(dm_header.ts.cmd_tstamps)),'r.')
    ax.plot(diffs,'b.')
    # ax.set_ylim(bottom=1.9e-1,top=2.1e-1)




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

