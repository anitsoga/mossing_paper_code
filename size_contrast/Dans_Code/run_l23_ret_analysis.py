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
import matplotlib
import scipy.stats as sst
from mpl_toolkits.mplot3d import Axes3D
import retinotopy_analysis as rt
reload(rt)
import scipy.optimize as sop
import pdb

procname = 'procfiles/pyr_l23_ret_proc.hdf5'
dsname = '/home/mossing/Documents/notebooks/shared_data/pyr_l23_data_struct.hdf5'

ret = {}
paramdict = {}
pval_ret = {}
trialrun = {}
nbydepth = {}
spont = {}
has_inverse = {}

def tack_on(thisfold,thisfile,adjust_fn=None,rg=(2,-10),criterion=lambda x:np.abs(x)>100):
    folds.append(thisfold)
    files.append(thisfile)
    rgs.append(rg)
    adjust_fns.append(adjust_fn)
    criteria.append(criterion)
    
folds = []
files = []
rgs = []
criteria = []
adjust_fns = []

# thisfold = '180220/M7254/'
# thisfile = 'M7254_130_004'
# tack_on(thisfold,thisfile)

# thisfold = '180405/M8570/'
# thisfile = 'M8570_000_001'
# tack_on(thisfold,thisfile)

# thisfold = '180530/M8174/'
# thisfile = 'M8174_130_002'
# tack_on(thisfold,thisfile)

### standardized expts starting here

thisfold = '181205/M10130/'
thisfile = 'M10130_175_003'
rg = (1,-10)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '181213/M8536/'
thisfile = 'M8536_155_001'
rg = (1,-10)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '181214/M10130/'
thisfile = 'M10130_135_001'
rg = (1,-10)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '190128/M8982/'
thisfile = 'M8982_200_002'
rg = (1,-11)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '190130/M9667/'
thisfile = 'M9667_135_001'
rg = (1,-10)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '190131/M9355/'
thisfile = 'M9355_165_001'
rg = (1,-10)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '190208/M9355/'
thisfile = 'M9355_175_003'
rg = (1,-11)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '190212/M8536/'
thisfile = 'M8536_150_004'
rg = (1,-11)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

#thisfold = '190304/M10077/'
#thisfile = 'M10077_500_002'
#rg = (1,-11)
#criterion = lambda x:np.abs(x)>0
#tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

#something weird about triggers in the size-contrast part of this

thisfold = '190102/M10130/'
thisfile = 'M10130_110_001'
rg = (1,-10)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '181209/M8536/'
thisfile = 'M8536_165_002'
rg = (1,-10)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '191002/M0293/'
thisfile = 'M0293_145_003'
rg = (1,-10)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '191108/M0403/'
thisfile = 'M0403_150_004'
rg = (1,-10)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)


reload(rt)
if os.path.exists(procname):
    os.remove(procname)
# ret,paramdict,pval,trialrun,has_inverse,nbydepth,spont = rt.analyze_everything(folds,files,rgs,criteria,datafoldbase='/media/mossing/backup_1/data/2P/',stimfoldbase='/home/mossing/modulation/visual_stim/')
keylist = rt.analyze_simply(folds,files,adjust_fns=adjust_fns,rgs=rgs,datafoldbase='/media/mossing/backup_1/data/2P/',stimfoldbase='/home/mossing/modulation/visual_stim/',procname=procname)

reload(rt)
import analysis_template as at
reload(at)

with h5py.File(procname,mode='r') as proc:
    grouplist = rt.add_data_struct_h5_simply(dsname,cell_type='PyrL23', keylist=keylist, frame_rate_dict=None, proc=proc, nbefore=8, nafter=8)
