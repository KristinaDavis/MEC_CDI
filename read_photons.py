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

dm_header = pp.open_MEC_tseries('/home/captainkay/mazinlab/MEC_data/CDI_tseries_2-9-2020_hour23_min58.pkl')
firstUnixTstep = pp.first_tstep(dm_header)
lastUnixTstep = pp.last_tstep(dm_header)

table1 = pipe.Photontable('/mnt/captainkay/mec/CDI1/config3/1599084915.h5')  # '/mnt/data0/captainkay/mec/CDI1/config3/1599084765.h5'
# tcube1 = table1.getTemporalCube()  #firstSec=firstUnixTstep
img = table1.getPixelCountImage()['image']

# print(np.nonzero(c1[cube]))
kittens=0
