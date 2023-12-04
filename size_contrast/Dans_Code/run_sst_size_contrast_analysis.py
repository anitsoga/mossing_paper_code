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
import scipy.optimize as sop
import pdb
import size_contrast_analysis as sca
reload(sca)
import retinotopy_analysis as rt
reload(rt)
import pickle as pkl

procname = 'procfiles/sst_size_contrast_proc.hdf5'
dsname = '../shared_data/sst_l23_data_struct.hdf5'

folds = []
files = []
rets = []
adjust_fns = []
rgs = []
criteria = []
datafoldbases = []

def tack_on(thisfold,thisfile,retnumber,frame_adjust=None,rg=(1,0),criterion=lambda x: np.abs(x)>100,datafoldbase='/home/mossing/scratch/2Pdata/'):
    folds.append(thisfold)
    files.append(thisfile)
    rets.append(retnumber)
    adjust_fns.append(frame_adjust)
    rgs.append(rg)
    criteria.append(criterion)
    datafoldbases.append(datafoldbase)

thisfold = '180713/M9053/'
thisfile = 'M9053_125_002'
retnumber = '001'
tack_on(thisfold,thisfile,retnumber,datafoldbase='/media/mossing/backup_1/data/2P/') 

thisfold = '180714/M9053/'
thisfile = 'M9053_140_004'
retnumber = '003'
tack_on(thisfold,thisfile,retnumber,datafoldbase='/media/mossing/backup_1/data/2P/')

thisfold = '180802/M9053/'
thisfile = 'M9053_135_003'
retnumber = '002'
tack_on(thisfold,thisfile,retnumber,datafoldbase='/media/mossing/backup_1/data/2P/')

thisfold = '180821/M9417/'
thisfile = 'M9417_140_005'
retnumber = '006'
tack_on(thisfold,thisfile,retnumber,datafoldbase='/media/mossing/backup_1/data/2P/')

thisfold = '181117/M10039/'
thisfile = 'M10039_150_003'
retnumber = '002'
tack_on(thisfold,thisfile,retnumber,datafoldbase='/media/mossing/backup_1/data/2P/')

thisfold = '181120/M10039/'
thisfile = 'M10039_140_006'
retnumber = '005'
tack_on(thisfold,thisfile,retnumber,datafoldbase='/media/mossing/backup_1/data/2P/')

thisfold = '181121/M10039/'
thisfile = 'M10039_135_005'
retnumber = '004'
tack_on(thisfold,thisfile,retnumber,datafoldbase='/media/mossing/backup_1/data/2P/')

if os.path.exists(procname):
    os.remove(procname)
import analysis_template as at
reload(at)
reload(sca)
keylist = sca.analyze_simply(folds,files,rets,adjust_fns,rgs,datafoldbase=datafoldbases,stimfoldbase='/home/mossing/modulation/visual_stim/',procname=procname)
with h5py.File(procname,mode='r') as proc:
    grouplist = sca.add_data_struct_h5_simply(dsname,cell_type='Sst', keylist=keylist, frame_rate_dict=None, proc=proc, nbefore=8, nafter=8)
