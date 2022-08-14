
# coding: utf-8

# In[12]:


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
reload(rt)
import naka_rushton_analysis as nra
import pdb
import sklearn
import pickle as pkl


# In[16]:


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
    
thisfold = '181205/M10130/'
thisfile = 'M10130_175_004'
retnumber = '003'
tack_on(thisfold,thisfile,retnumber,datafoldbase='/media/mossing/backup_1/data/2P/') 

thisfold = '181213/M8536/'
thisfile = 'M8536_155_002'
retnumber = '001'
tack_on(thisfold,thisfile,retnumber,datafoldbase='/media/mossing/backup_1/data/2P/')

thisfold = '181214/M10130/'
thisfile = 'M10130_135_002'
retnumber = '001'
tack_on(thisfold,thisfile,retnumber,datafoldbase='/media/mossing/backup_1/data/2P/')

thisfold = '190128/M8982/'
thisfile = 'M8982_200_003'
retnumber = '002'
tack_on(thisfold,thisfile,retnumber,datafoldbase='/media/mossing/backup_1/data/2P/')

thisfold = '190130/M9667/'
thisfile = 'M9667_135_002'
retnumber = '001'
rg = (1,-1)
tack_on(thisfold,thisfile,retnumber,datafoldbase='/media/mossing/backup_1/data/2P/',rg=rg)

thisfold = '190131/M9355/'
thisfile = 'M9355_165_002'
retnumber = '001'
tack_on(thisfold,thisfile,retnumber,datafoldbase='/media/mossing/backup_1/data/2P/')

thisfold = '190208/M9355/'
thisfile = 'M9355_175_004'
retnumber = '003'
tack_on(thisfold,thisfile,retnumber,datafoldbase='/media/mossing/backup_1/data/2P/')

thisfold = '190212/M8536/'
thisfile = 'M8536_150_005'
retnumber = '004'
tack_on(thisfold,thisfile,retnumber,datafoldbase='/media/mossing/backup_1/data/2P/')

thisfold = '190102/M10130/'
thisfile = 'M10130_110_002'
retnumber = '001'
rg = (1,-1)
frame_adjust = lambda x: np.concatenate((x[1::4][np.newaxis],x[2::4][np.newaxis]),axis=0).T.flatten()
tack_on(thisfold,thisfile,retnumber,datafoldbase='/media/mossing/backup_1/data/2P/',rg=rg,frame_adjust=frame_adjust)

thisfold = '181209/M8536/'
thisfile = 'M8536_165_003'
retnumber = '002'
tack_on(thisfold,thisfile,retnumber,datafoldbase='/media/mossing/backup_1/data/2P/')


# In[17]:


reload(sca)
soriavg,ret_vars,proc = sca.analyze_everything_by_criterion(folds,files,rets,adjust_fns,rgs,criteria=criteria,datafoldbase=datafoldbases,stimfoldbase='/home/mossing/modulation/visual_stim/',criterion_cutoff=0.2)


# In[ ]:


keylist = list([x for x in soriavg.keys() if not x[0]=='_'])


# In[ ]:


# keylist = list(soriavg.keys())
data_struct = sca.gen_full_data_struct(cell_type='PyrL23', keylist=keylist, frame_rate_dict=None, proc=proc, ret_vars=ret_vars, nbefore=4, nafter=4)


# In[ ]:


# sio.savemat('syn_sst_tomato_data_struct.mat',data_struct)
sio.savemat('pyr_l23_full_data_struct.mat',data_struct)



