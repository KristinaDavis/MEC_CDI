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
from matplotlib.colors import LogNorm, SymLogNorm

from mec_cdi import CDI_params, Slapper
import mkidpipeline as pipe


def open_MEC_tseries(CDI_tseries='CDI_tseries.pkl'):
    """opens existing MEC CDI timeseries .pkl file and return it"""
    with open(CDI_tseries, 'rb') as handle:
        CDI_meta = pickle.load(handle)
    return CDI_meta


def first_tstep(meta):
    """returns the first timestep time from pkl file. This is useful to tell the mkidpipeline when to start the obs"""
    first_t = meta.ts.cmd_tstamps[0]  #  [-1]
    return (first_t - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')


def last_tstep(meta):
    """returns the end timestep time from pkl file. This is useful to tell the mkidpipeline when to stop the obs"""
    last_t = meta.ts.cmd_tstamps[-1]   # [-3]
    return (last_t - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
##

if __name__ == '__main__':
    ##
    # file_h5 = '/data0/captainkay/mec/SciOct2020/1602049681.h5'
    # dm_file = '/data0/captainkay/mec/SciOct2020/SciOct2020_config3_dummy.pkl'  #'/work/kkdavis/scratch/old/CDI_tseries_3-9-2020_hour0_min11.pkl'
    file_h5 = '/data0/captainkay/mec/SciOct2020/1602049820.h5'
    dm_file = '/data0/captainkay/mec/SciOct2020/SciOct2020_config3_dummy.pkl'
    # Nifty File Name Extraction--makes nice print statements and gets unix timestamp name for file saving later
    r1 = os.path.basename(dm_file)
    dm_name_parts = os.path.splitext(r1)
    r2 = os.path.basename(file_h5)
    h5_name_parts = os.path.splitext(r2)

    ## Load tCube from saved file
    # tcube1 = np.load('cdi1_timeCube3')
    
    ##  Load Photontable
    table1 = pipe.Photontable(file_h5)

    ## Open DM .pkl file
    dm_header = open_MEC_tseries(dm_file)
    # dm_header = open_MEC_tseries('/work/kkdavis/scratch/old/CDI_tseries_3-9-2020_hour0_min11.pkl')   # CDI_tseries_2-9-2020_hour23_min58.pkl

    firstUnixTstep = np.int(first_tstep(dm_header))
    lastUnixTstep = np.int(last_tstep(dm_header))
    total_h5_seconds = lastUnixTstep - firstUnixTstep
    print(
        f'\n\n{h5_name_parts[0]}\n{dm_name_parts[0]}:\n\t'
        f'First Timestep = {first_tstep(dm_header):.0f}\n\tLast Timestep = {last_tstep(dm_header):.0f}')

    

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

    tcube_1cycle = table1.getTemporalCube(firstSec=60,
                                    # integrationTime=dm_header.ts.t_one_cycle,
                                    integrationTime=90,
                                    timeslice=dm_header.ts.phase_integration_time)

    end_make_cube = time.time()
    duration_make_tcube = end_make_cube - start_make_cube
    print(f'time to make one-cycle temporal cube is {duration_make_tcube/60:.2f} minutes '
          f'({duration_make_tcube:.2f} sec)')

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
    np.save(f'SciOct2020_tcube_60sec_{firstUnixTstep}', tcube_1cycle['cube'])

    # Save the full cycles temporal cube
    np.save(f'SciOct2020_tcube_fullcycle_{firstUnixTstep}', tcube_fullcycle['cube'])

    # Save just the pixel count img
    np.save(f'SciOct2020_piximg_{firstUnixTstep}', cimg1)

    # Pickling


    ##
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # ax.imshow(table1.beamImage, interpolation='none', origin='lower')
    # plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(f'Total Pixel Count Image: {h5_name_parts[0]}')
    ax.imshow(cimg1, interpolation='none', origin='lower')
    plt.show()

    fig, subplt = plt.subplots(nrows=2, ncols=3)
    fig.suptitle(f'Temporal Cube: {h5_name_parts[0]}')
    for ax, p in zip(subplt.flatten(), range(2*3)):
        ax.imshow(tcube1['cube'][:,:,p], interpolation='none', origin='lower')

    plt.show()


    kittens=0


