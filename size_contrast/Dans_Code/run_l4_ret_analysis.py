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
import matplotlib
import scipy.stats as sst
from mpl_toolkits.mplot3d import Axes3D
import retinotopy_analysis as rt
import scipy.optimize as sop
import pdb
import glob

procname = 'procfiles/pyr_l4_ret_proc.hdf5'
dsname = '/home/mossing/Documents/notebooks/shared_data/pyr_l4_data_struct.hdf5'

ret = {}
paramdict = {}
pval_ret = {}
trialrun = {}
nbydepth = {}
spont = {}
has_inverse = {}

def compute_rg(thisfold,thisfile,lb=50,ub=70):
    matfile = datafoldbase + thisfold + thisfile + '.mat'
    info = ut.loadmat(matfile,'info')
    frame = info['frame'][()]
    total_frames = len(frame)
    looks_good = (np.where((np.diff(frame)>lb) & (np.diff(frame)<ub))[0])
    rg = (looks_good[0],looks_good[-1]-total_frames+1)
    print(frame)
    return rg

datafoldbase='/home/mossing/modulation/matfiles/'

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

thisfold = '181127/M10073/'
thisfile = 'M10073_385_002'
rg = compute_rg(thisfold,thisfile,lb=20,ub=40)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '190202/M10075/'
thisfile = 'M10075_350_004'
rg = (1,-11)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '190304/M10077/'
thisfile = 'M10077_500_002'
rg = (1,-11)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

# #thisfold = '190411/M10077/'
# #thisfile = 'M10077_350_004'
# #rg = (1,-11)
# #criterion = lambda x:np.abs(x)>0
# #tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

# thisfold = '190620/M10619/'
# thisfile = 'M10619_285_001'
# rg = (1,-10)
# criterion = lambda x:np.abs(x)>0
# tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '190607/M10443/'
thisfile = 'M10443_320_004'
rg = compute_rg(thisfold,thisfile)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '190620/M10619/'
thisfile = 'M10619_285_003'
rg = (1,-11)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '190620/M10616/'
thisfile = 'M10616_320_003'
rg = (1,-11)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '190624/M10615/'
thisfile = 'M10615_355_004'
rg = compute_rg(thisfold,thisfile)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '190627/M10616/'
thisfile = 'M10616_365_006'
rg = compute_rg(thisfold,thisfile)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

thisfold = '190701/M10615/'
thisfile = 'M10615_290_004'
rg = compute_rg(thisfold,thisfile)
criterion = lambda x:np.abs(x)>0
tack_on(thisfold,thisfile,rg=rg,criterion=criterion)

reload(rt)
if os.path.exists(procname):
    os.remove(procname)
# ret,paramdict,pval,trialrun,has_inverse,nbydepth,spont = rt.analyze_everything(folds,files,rgs,criteria,datafoldbase='/media/mossing/backup_1/data/2P/',stimfoldbase='/home/mossing/modulation/visual_stim/')
keylist = rt.analyze_simply(folds,files,adjust_fns=adjust_fns,rgs=rgs,datafoldbase=datafoldbase,stimfoldbase='/home/mossing/modulation/visual_stim/',procname=procname)

if os.path.exists(dsname):
    os.remove(dsname)
reload(rt)
import analysis_template as at
reload(at)

with h5py.File(procname,mode='r') as proc:
    grouplist = rt.add_data_struct_h5_simply(dsname,cell_type='PyrL4', keylist=keylist, frame_rate_dict=None, proc=proc, nbefore=8, nafter=8)
