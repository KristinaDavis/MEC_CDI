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
import postprocess_MEC_CDI as pp
import mkidpipeline as pipe

##
dm_header = pp.open_MEC_tseries('/work/kkdavis/scratch/CDI_tseries_3-9-2020_hour0_min11.pkl')   # CDI_tseries_2-9-2020_hour23_min58.pkl
firstUnixTstep = np.int(pp.first_tstep(dm_header))
lastUnixTstep = np.int(pp.last_tstep(dm_header))
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
np.savez('cubes_CDI1_config3', table=table1, tcube=tcube1['cube'], meta=dm_header)

# Save just the temporal cube
np.save('cdi1_timeCube3', tcube1['cube'])

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
