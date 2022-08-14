#!/usr/bin/env python

import pyute as ut
import autograd.numpy as np
import sklearn
import h5py
import pdb
import scipy.optimize as sop
from autograd import elementwise_grad as egrad

    def compute_tuning(dsfile):
        with h5py.File(dsfile,mode='r') as f:
            keylist = [key for key in f.keys()]
            tuning = [None]*len(keylist)
            uparam = [None]*len(keylist)
            for ikey in range(len(keylist)):
                session = f[keylist[ikey]]
                print(session)

                if 'size_contrast_0' in session:
                    sc0 = session['size_contrast_0']
                    data = sc0['decon'][:]
                    stim_id = sc0['stimulus_id'][:]
                    nbefore = sc0['nbefore'][()]
                    nafter = sc0['nafter'][()]
                    trialrun = sc0['running_speed_cm_s'][:,nbefore:-nafter].mean(-1)>10 #
                    print(sc0['running_speed_cm_s'].shape)
                    print(np.nanmean(trialrun))
                    if np.nanmean(trialrun)>0.4:
                        tuning[ikey] = ut.compute_tuning(data,stim_id,trial_criteria=trialrun)[:]
                    for param in sc0['stimulus_parameters']:
                        uparam[ikey] = sc0[param][:]
        return tuning,uparam

def average_up(arr):
    return arr[:,:,:,:,8:-8].mean(-1).mean(-1) #.reshape((arr.shape[0],-1))

def columnize(arr):
    output = np.nanmean(arr,0).flatten()
    output = output/output.max()
    return output

dsname_pc = '/home/mossing/Documents/notebooks/shared_data/pyr_l23_data_struct.hdf5'
dsname_sst = '/home/mossing/Documents/notebooks/shared_data/sst_l23_data_struct.hdf5'
dsname_vip = '/home/mossing/Documents/notebooks/shared_data/vip_l23_data_struct.hdf5'

tuning_pc,uparam_pc = compute_tuning(dsname_pc)
tuning_sst,uparam_sst = compute_tuning(dsname_sst)
tuning_vip,uparam_vip = compute_tuning(dsname_vip)

