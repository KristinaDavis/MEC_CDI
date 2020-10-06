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
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import sys
import pickle
import time
import datetime
import pytz

from mec_cdi import CDI_params, Slapper
import mkidpipeline as pipe


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


##
dm_header = open_MEC_tseries('/work/kkdavis/scratch/CDI_tseries_3-9-2020_hour0_min11.pkl')   # CDI_tseries_2-9-2020_hour23_min58.pkl
firstUnixTstep = np.int(first_tstep(dm_header))
lastUnixTstep = np.int(last_tstep(dm_header))
total_h5_seconds = lastUnixTstep - firstUnixTstep

##  Load Photontable
table1 = pipe.Photontable('/data0/captainkay/mec/CDI1/config3/1599091228.h5')

## Check Datasets Match

## Make Image Cube
# tcube1 = table1.getTemporalCube()  #firstSec=firstUnixTstep
# cimg_start = time.time()
#
# cimg1 = table1.getPixelCountImage()['image']  # total integrated over the whole time
#
# cimg_end = time.time()
# duration_make_cimg = cimg_end - cimg_start
# print(f'time to make count image is {duration_make_cimg/60:.2f}')


## Load tCube from saved file
tcube1 = np.load('cdi1_timeCube3')


## Make Temporal Cube
"""
firstSec = seconds of first second in the cube, entered as number of seconds after start of the h5 file
integrationTime = seconds, duration of the cube (second of last tstep after firstSec)
timeslice = bin width of the timescale axis (integration time of each bin along z axis of cube)

** note here, you do not enter the times in unix timestamp, rather by actual seconds where 0 is the start of the
h5 file
"""
# want to use timeslice=dm_header.ts.phase_integration_time but since it was messed up for CDI1 round of
# white light tests we will hand-tune it for now
tcube_start = time.time()

one_cycle_time = (dm_header.ts.n_probes + dm_header.ts.null_time)  # *dm_header.ts.phase_integration_time
tcube1 = table1.getTemporalCube(firstSec=0,
                                integrationTime=one_cycle_time,
                                timeslice=1)

tcube_end = time.time()
duration_make_tcube = tcube_end - tcube_start
print(f'time to make temporal cube is {duration_make_tcube/60:.2f} minutes')


## Saving Created Data
# Save several together
np.savez(f'cubes_CDI2_config1_{firstUnixTstep}', table=table1, tcube=tcube1['cube'], meta=dm_header)

# Save just the temporal cube
np.save(f'cdi2_confg1_{firstUnixTstep}', tcube1['cube'])

# Pickling


##
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.imshow(table1.beamImage, interpolation='none', origin='lower')
plt.show()

fig, subplt = plt.subplots(nrows=2, ncols=3)
for ax, p in zip(subplt.flatten(), range(2*3)):
    ax.imshow(tcube1['cube'][:,:,p], interpolation='none', origin='lower')

plt.show()



# print(np.nonzero(c1[cube]))
kittens=0


if __name__ == '__main__':
    dm_meta = open_MEC_tseries('/home/scexao/mkids/CDI/20201005/CDI_tseries_10-6-2020_hour3_min52.pkl')
    print(f'First Timestep = {first_tstep(dm_meta):.0f}\nLast Timestep = {last_tstep(dm_meta):.0f}')
