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
import size_contrast_analysis as sca
reload(sca)
import retinotopy_analysis as rt
import scipy.optimize as sop
import sklearn
import pdb
import analysis_template as at

procname = 'procfiles/pyr_l4_size_contrast_proc.hdf5'
dsname = '/home/mossing/Documents/notebooks/shared_data/pyr_l4_data_struct.hdf5'

datafoldbase = '/home/mossing/modulation/matfiles/'

def compute_rg(thisfold,thisfile,lb=27,ub=34):
    matfile = datafoldbase + thisfold + thisfile + '.mat'
    info = ut.loadmat(matfile,'info')
    frame = info['frame'][()]
    total_frames = len(frame)
    looks_good = (np.where((np.diff(frame)>lb) & (np.diff(frame)<ub))[0])
    rg = (looks_good[0],looks_good[-1]-total_frames+2)
#     print(frame[-10:])
    return rg

folds = []
files = []
rets = []
adjust_fns = []
rgs = []
criteria = []
datafoldbases = []

def tack_on(thisfold,thisfile,retnumber,frame_adjust=None,rg=(1,0),criterion=lambda x: np.abs(x)>100,datafoldbase=None):
    folds.append(thisfold)
    files.append(thisfile)
    rets.append(retnumber)
    adjust_fns.append(frame_adjust)
    rgs.append(rg)
    criteria.append(criterion)
    datafoldbases.append(datafoldbase)

thisfold = '181127/M10073/'
thisfile = 'M10073_385_003'
retnumber = '002'
# datafoldbase = '/media/mossing/data_ssd/data/2P/'
tack_on(thisfold,thisfile,retnumber,criterion=lambda x: np.abs(x)>100,datafoldbase=datafoldbase) #,frame_adjust=frame_adjust)

thisfold = '190202/M10075/'
thisfile = 'M10075_350_005'
retnumber = '004'
# datafoldbase = '/media/mossing/backup_1/data/2P/'
thisrg = (1,-1)
tack_on(thisfold,thisfile,retnumber,rg=thisrg,criterion=lambda x: np.abs(x)>100,datafoldbase=datafoldbase) #,frame_adjust=frame_adjust)

thisfold = '190304/M10077/'
thisfile = 'M10077_500_003'
retnumber = '002'
# datafoldbase = '/media/mossing/backup_1/data/2P/'
thisrg = (1,0)
tack_on(thisfold,thisfile,retnumber,rg=thisrg,criterion=lambda x: np.abs(x)>100,datafoldbase=datafoldbase)

thisfold = '190607/M10443/'
thisfile = 'M10443_320_002'
retnumber = '004'
thisrg = compute_rg(thisfold,thisfile)
tack_on(thisfold,thisfile,retnumber,rg=thisrg,criterion=lambda x: np.abs(x)>100,datafoldbase=datafoldbase)

thisfold = '190620/M10619/'
thisfile = 'M10619_285_002'
retnumber = '003'
# datafoldbase = '/media/mossing/backup_1/data/2P/'
thisrg = (1,-1)
tack_on(thisfold,thisfile,retnumber,rg=thisrg,criterion=lambda x: np.abs(x)>100,datafoldbase=datafoldbase)

thisfold = '190620/M10616/'
thisfile = 'M10616_320_002'
retnumber = '003'
# datafoldbase = '/media/mossing/backup_1/data/2P/'
thisrg = (1,0)
tack_on(thisfold,thisfile,retnumber,rg=thisrg,criterion=lambda x: np.abs(x)>100,datafoldbase=datafoldbase)

thisfold = '190624/M10615/'
thisfile = 'M10615_355_003'
retnumber = '004'
thisrg = compute_rg(thisfold,thisfile)
tack_on(thisfold,thisfile,retnumber,rg=thisrg,criterion=lambda x: np.abs(x)>100,datafoldbase=datafoldbase)

thisfold = '190627/M10616/'
thisfile = 'M10616_365_005'
retnumber = '006'
thisrg = compute_rg(thisfold,thisfile)
tack_on(thisfold,thisfile,retnumber,rg=thisrg,criterion=lambda x: np.abs(x)>100,datafoldbase=datafoldbase)

thisfold = '190701/M10615/'
thisfile = 'M10615_290_003'
retnumber = '004'
thisrg = compute_rg(thisfold,thisfile)
tack_on(thisfold,thisfile,retnumber,rg=thisrg,criterion=lambda x: np.abs(x)>100,datafoldbase=datafoldbase)

if os.path.exists(procname):
    os.remove(procname)
# if os.path.exists(dsname):
#     os.remove(dsname)

reload(sca)
reload(at)
# soriavg,ret_vars = sca.analyze_everything_by_criterion(folds,files,rets,adjust_fns,rgs,criteria=criteria,datafoldbase=datafoldbases,stimfoldbase='/home/mossing/modulation/visual_stim/',criterion_cutoff=0.2,procname='pyr_l23_size_contrast_proc.hdf5')
keylist = sca.analyze_simply(folds,files,rets,adjust_fns,rgs,datafoldbase=datafoldbases,stimfoldbase='/home/mossing/modulation/visual_stim/',procname=procname)

reload(sca)
with h5py.File(procname,mode='r') as proc:
    #keylist = list([x for x in proc.keys() if not x[0]=='_'])
    #print([key for key in proc[keylist[0]]['ret_vars'].keys()])
#     key = keylist[0]
#     print(proc[key]['ret_vars']['paramdict_normal']['xo'])
#     print([key for key in proc[keylist[0]].keys()])
    grouplist = sca.add_data_struct_h5_simply(dsname,cell_type='PyrL4', keylist=keylist, frame_rate_dict=None, proc=proc, nbefore=8, nafter=8)
