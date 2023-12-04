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

procname = 'procfiles/pv_ret_proc.hdf5'
dsname = '/home/mossing/Documents/notebooks/shared_data/pv_l23_data_struct.hdf5'

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

thisfold = '191105/M0589/'
thisfile = 'M0589_220_004'
rg = (1,-10)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '191220/M0589/'
thisfile = 'M0589_185_005'
rg = (1,-10)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '191229/M0892/'
thisfile = 'M0892_220_001'
rg = (1,-10)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '200102/M0892/'
thisfile = 'M0892_200_002'
rg = (1,-10)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)





reload(rt)
if os.path.exists(procname):
    os.remove(procname)
# ret,paramdict,pval,trialrun,has_inverse,nbydepth,spont = rt.analyze_everything(folds,files,rgs,criteria,datafoldbase='/media/mossing/backup_1/data/2P/',stimfoldbase='/home/mossing/modulation/visual_stim/')
keylist = rt.analyze_simply(folds,files,adjust_fns=adjust_fns,rgs=rgs,datafoldbase='/home/mossing/modulation/matfiles/',stimfoldbase='/home/mossing/modulation/visual_stim/',procname=procname)

reload(rt)
import analysis_template as at
reload(at)

with h5py.File(procname,mode='r') as proc:
    grouplist = rt.add_data_struct_h5_simply(dsname,cell_type='Pv', keylist=keylist, frame_rate_dict=None, proc=proc, nbefore=8, nafter=8)
