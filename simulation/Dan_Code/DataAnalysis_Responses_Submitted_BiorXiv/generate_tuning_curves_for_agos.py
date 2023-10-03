#!/usr/bin/env python
# coding: utf-8

import pyute as ut
import autograd.numpy as np
import matplotlib.pyplot as plt
import sklearn
import h5py
import pdb
import scipy.optimize as sop
from autograd import elementwise_grad as egrad
from mpl_toolkits.mplot3d import Axes3D
import sklearn.discriminant_analysis as skd
import autograd.scipy.special as ssp
from autograd import jacobian
import size_contrast_analysis as sca
import scipy.stats as sst
import utils


# In[2]:


dsbase ='/Users/agos/ProjectsCluster2/AdesnikData/DatasetDec11/GenerateTuningCurves/'
dsnames = [dsbase+x+'_data_struct.hdf5' for x in ['pyr_l4','pyr_l23','sst_l23','vip_l23']]


# In[3]:


nsize,ncontrast = 5,6


# In[4]:


tunings = []
uparams = []
displacements = []
pvals = []
for dsname in dsnames:
    print(dsname)
    this_tuning,this_uparam,this_displacement,this_pval = utils.compute_tuning(dsname)
    tunings.append(this_tuning)
    uparams.append(this_uparam)
    displacements.append(this_displacement)
    pvals.append(this_pval)

with ut.hdf5read(dsnames[1]) as ds:
    keylist = list(ds.keys())

selection = utils.default_selection()
rs = utils.gen_rs(selection=selection)

def sum_to_1(r):
    R = r.reshape((r.shape[0],-1))
    R = R/np.nansum(R,axis=1)[:,np.newaxis]
    return R
Rs = [[None,None] for i in range(len(rs))]
for iR,r in enumerate(rs):
    for ialign in range(2):
        Rs[iR][ialign] = sum_to_1(r[ialign])

