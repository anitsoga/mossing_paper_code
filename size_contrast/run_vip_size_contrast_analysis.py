#!/usr/bin/env python

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import h5py
from oasis.functions import deconvolve
from oasis import oasisAR1, oasisAR2
import os
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
import analysis_template as at

folds = []
files = []
rets = []
adjust_fns = []
rgs = []
criteria = []
datafoldbases = []

def tack_on(thisfold,thisfile,retnumber,frame_adjust=None,rg=(1,0),criterion=lambda x: np.abs(x)>100, datafoldbase='/media/mossing/backup_1/data/2P/'):
    folds.append(thisfold)
    files.append(thisfile)
    rets.append(retnumber)
    adjust_fns.append(frame_adjust)
    rgs.append(rg)
    criteria.append(criterion)
    datafoldbases.append(datafoldbase)

thisfold = '180412/M7955/'
thisfile = 'M7955_150_004'
retnumber = '003'
tack_on(thisfold,thisfile,retnumber)

thisfold = '180516/M8956/'
thisfile = 'M8956_115_003'
retnumber = '002'
tack_on(thisfold,thisfile,retnumber)

# NEW
thisfold = '180519/M8959/'
thisfile = 'M8959_135_003'
retnumber = '002'
tack_on(thisfold,thisfile,retnumber)
# NEW

thisfold = '180528/M8959/'
thisfile = 'M8959_100_005'
retnumber = '003'
tack_on(thisfold,thisfile,retnumber)

thisfold = '180531/M8961/'
thisfile = 'M8961_135_005'
retnumber = '003'
tack_on(thisfold,thisfile,retnumber)#,frame_adjust)

thisfold = '180618/M8956/'
thisfile = 'M8956_145_003'
retnumber = '002'
frame_adjust = lambda x: np.delete(x,6)
tack_on(thisfold,thisfile,retnumber,frame_adjust=frame_adjust)

# NOT MUCH RUNNING IN THESE TWO

thisfold = '180719/M8961/'
thisfile = 'M8961_150_007'
retnumber = '006'
tack_on(thisfold,thisfile,retnumber)

thisfold = '180720/M8961/'
thisfile = 'M8961_100_004'
retnumber = '003'
tack_on(thisfold,thisfile,retnumber)

# NOT MUCH RUNNING IN THESE TWO

thisfold = '180903/M8961/'
thisfile = 'M8961_150_003'
retnumber = '002'
tack_on(thisfold,thisfile,retnumber)#,frame_adjust) os

thisfold = '190710/M0208/'
thisfile = 'M0208_095_005'
retnumber = '006'
tack_on(thisfold,thisfile,retnumber)#,frame_adjust) os

thisfold = '190723/M0311/'
thisfile = 'M0311_100_007'
retnumber = '008'
tack_on(thisfold,thisfile,retnumber)#,frame_adjust) os

reload(sca)
procname = 'procfiles/vip_l23_size_contrast_proc.hdf5'
dsname = '../shared_data/vip_l23_data_struct.hdf5'
if os.path.exists(procname):
    os.remove(procname)

reload(sca)
reload(at)
keylist = sca.analyze_simply(folds,files,rets,adjust_fns,rgs,datafoldbase=datafoldbases,stimfoldbase='/home/mossing/modulation/visual_stim/',procname=procname)
with h5py.File(procname,mode='r') as proc:
    grouplist = sca.add_data_struct_h5_simply(dsname,cell_type='Vip', keylist=keylist, frame_rate_dict=None, proc=proc, nbefore=8, nafter=8)
