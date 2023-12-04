#!/usr/bin/env python

import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import h5py
from oasis.functions import deconvolve
from oasis import oasisAR1, oasisAR2
import pyute as ut
from importlib import reload
reload(ut)
import scipy.ndimage.filters as sfi
import scipy.stats as sst
import scipy.ndimage.measurements as snm
from mpl_toolkits.mplot3d import Axes3D
import retinotopy_analysis as rt
reload(rt)
import pdb

procname = 'procfiles/vip_ret_proc.hdf5'
dsname = '../shared_data/vip_l23_data_struct.hdf5'

reload(rt)
def tack_on(thisfold,thisfile,rg=(2,-10),criterion=lambda x:np.abs(x)>100):
    folds.append(thisfold)
    files.append(thisfile)
    rgs.append(rg)
    criteria.append(criterion)
    
folds = []
files = []
rgs = []
criteria = []

thisfold = '180321/M7955/'
thisfile = 'M7955_000_001'
tack_on(thisfold,thisfile)

thisfold = '180412/M7955/'
thisfile = 'M7955_150_003'
tack_on(thisfold,thisfile)

# # thisfold = '180509/M8959/'
# # thisfile = 'M8959_170_002'
# # do_process(thisfold,thisfile,criterion=lambda x:np.abs(x)>100) 

# # thisfold = '180511/M8959/'
# # thisfile = 'M8959_140_003'
# # do_process(thisfold,thisfile,criterion=lambda x:np.abs(x)>100) 

# # WORSE IMAGING QUALITY BEFORE THIS POINT

# thisfold = '180513/M8959/' # only gray screen with running expt on this day
# thisfile = 'M8959_120_002'
# do_process(thisfold,thisfile,criterion=lambda x:np.abs(x)>100) 

thisfold = '180516/M8956/'
thisfile = 'M8956_115_002'
tack_on(thisfold,thisfile)

thisfold = '180519/M8959/'
thisfile = 'M8959_135_002'
tack_on(thisfold,thisfile)

thisfold = '180528/M8959/'
thisfile = 'M8959_100_003'
tack_on(thisfold,thisfile)

thisfold = '180531/M8961/'
thisfile = 'M8961_135_003'
tack_on(thisfold,thisfile)

thisfold = '180618/M8956/'
thisfile = 'M8956_145_002'
tack_on(thisfold,thisfile)

thisfold = '180719/M8961/'
thisfile = 'M8961_150_006'
tack_on(thisfold,thisfile)

thisfold = '180720/M8961/'
thisfile = 'M8961_100_003'
tack_on(thisfold,thisfile)

thisfold = '180903/M8961/'
thisfile = 'M8961_150_002'
tack_on(thisfold,thisfile,rg=(1,-10))

thisfold = '190710/M0208/'
thisfile = 'M0208_095_006'
tack_on(thisfold,thisfile,rg=(1,-10))

thisfold = '190723/M0311/'
thisfile = 'M0311_100_008'
tack_on(thisfold,thisfile,rg=(1,-10))

reload(rt)
reload(ut)
#ret,paramdict,pval,trialrun,has_inverse,nbydepth,proc = rt.analyze_everything(folds,files,rgs,criteria)

reload(rt)

if os.path.exists(procname):
    os.remove(procname)
if os.path.exists(dsname):
    os.remove(dsname)
keylist = rt.analyze_simply(folds,files,rgs=rgs,datafoldbase='/media/mossing/backup_1/data/2P/',stimfoldbase='/home/mossing/modulation/visual_stim/',procname=procname)

with h5py.File(procname,mode='r') as proc:
    grouplist = rt.add_data_struct_h5_simply(dsname,cell_type='Vip', keylist=keylist, frame_rate_dict=None, proc=proc, nbefore=8, nafter=8)
