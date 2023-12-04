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
import pdb
import retinotopy_analysis as rt
reload(rt)
import scipy.optimize as sop
import analysis_template as at

procname = 'procfiles/sst_ret_proc.hdf5'
dsname = '../shared_data/sst_l23_data_struct.hdf5'

ret = {}
paramdict = {}
pval = {}
trialrun = {}
nbydepth = {}
spont = {}
has_inverse = {}

reload(rt)
reload(ut)

def tack_on(thisfold,thisfile,rg=(2,-10),criterion=lambda x:np.abs(x)>100,datafoldbase='/media/mossing/backup_1/data/2P/'):
    folds.append(thisfold)
    files.append(thisfile)
    rgs.append(rg)
    criteria.append(criterion)
    datafoldbases.append(datafoldbase)
    
folds = []
files = []
rgs = []
criteria = []
datafoldbases = []

thisfold = '180713/M9053/'
thisfile = 'M9053_125_'#002'
retnumber = '001'
tack_on(thisfold,thisfile+retnumber)

thisfold = '180714/M9053/'
thisfile = 'M9053_140_'#004'
retnumber = '003'
tack_on(thisfold,thisfile+retnumber)

thisfold = '180802/M9053/'
thisfile = 'M9053_135_' #003'
retnumber = '002'
tack_on(thisfold,thisfile+retnumber)

thisfold = '180821/M9417/'
thisfile = 'M9417_140_'#005'
retnumber = '006'
tack_on(thisfold,thisfile+retnumber,rg=(1,-10))

thisfold = '181117/M10039/'
thisfile = 'M10039_150_'#003'
retnumber = '002'
tack_on(thisfold,thisfile+retnumber,rg=(1,-10),datafoldbase='/media/mossing/backup_1/data/2P/')

thisfold = '181120/M10039/'
thisfile = 'M10039_140_'#006'
retnumber = '005'
tack_on(thisfold,thisfile+retnumber,rg=(1,-10),datafoldbase='/media/mossing/backup_1/data/2P/')

thisfold = '181121/M10039/'
thisfile = 'M10039_135_'#005'
retnumber = '004'
tack_on(thisfold,thisfile+retnumber,rg=(1,-10),datafoldbase='/media/mossing/backup_1/data/2P/')

# do_process(thisfold,thisfile+retnumber,criterion=lambda x:np.abs(x)>100,rg=(1,-10))

# ret,paramdict,pval,trialrun,has_inverse,nbydepth,spont = rt.analyze_everything(folds,files,rgs,criteria,datafoldbase=datafoldbases,stimfoldbase='/home/mossing/modulation/visual_stim/')

reload(rt)
reload(at)

if os.path.exists(procname):
    os.remove(procname)
if os.path.exists(dsname):
    os.remove(dsname)
keylist = rt.analyze_simply(folds,files,rgs=rgs,datafoldbase=datafoldbases,stimfoldbase='/home/mossing/modulation/visual_stim/',procname=procname)

with h5py.File(procname,mode='r') as proc:
    grouplist = rt.add_data_struct_h5_simply(dsname,cell_type='Sst', keylist=keylist, frame_rate_dict=None, proc=proc, nbefore=8, nafter=8)
