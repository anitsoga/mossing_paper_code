#!/usr/bin/env python
# coding: utf-8

import pyute as ut
import autograd.numpy as np
import matplotlib.pyplot as plt
import sklearn
import h5py
import pdb
import scipy.optimize as sop
from mpl_toolkits.mplot3d import Axes3D
import sklearn.discriminant_analysis as skd
import autograd.scipy.special as ssp
from autograd import elementwise_grad as egrad
from autograd import grad
from autograd import jacobian
from autograd import hessian
import size_contrast_analysis as sca
import scipy.stats as sst
import sim_utils
from importlib import reload
reload(sim_utils)
import calnet.utils
import calnet.fitting_spatial_feature_model
import opto_utils

dsbase = '/Users/dan/Documents/notebooks/mossing-PC/shared_data/'
dsnames = [dsbase+x+'_data_struct.hdf5' for x in ['pyr_l4','pyr_l23','sst_l23','vip_l23','pv_l23']]
dsnames = [dsnames[1]]

nsize,ncontrast = 6,6

to_exclude = ['session_'+exptname for exptname in ['180714_M9053','180321_M7955', '180519_M8959', '180531_M8961', '180618_M8956','190202_M10075', '190620_M10619']]
tunings_decon,uparams_decon,displacements,pvals = [[None,None] for ivar in range(4)]
for irun,run_bool in enumerate([False,True]):
    tunings_decon[irun],uparams_decon[irun],displacements[irun],pvals[irun] = [[] for ivar in range(4)]
    for dsname in dsnames:
        new_vars = sim_utils.compute_tuning(dsname,datafield='decon',running=run_bool,expttype='size_contrast_0')
        tunings_decon[irun] = tunings_decon[irun] + [new_vars[0]]
        uparams_decon[irun] = uparams_decon[irun] + [new_vars[1]]
        displacements[irun] = displacements[irun] + [new_vars[2]]
        pvals[irun] = pvals[irun] + [new_vars[3]]

reload(sim_utils)
to_exclude = ['session_'+exptname for exptname in ['180714_M9053','180321_M7955', '180519_M8959', '180531_M8961', '180618_M8956','190202_M10075', '190620_M10619']]
ret_info,uparams_sc,displacements,pvals = [[None,None] for ivar in range(4)]
for irun,run_bool in enumerate([False,True]):
    ret_info[irun] = []
    for dsname in dsnames:
        new_vars = sim_utils.get_ret_info(dsname,expttype='size_contrast_0')
        ret_info[irun] = ret_info[irun] + [new_vars]

uangle = np.arange(0,360,45)
def nansem(data):
    return np.nanstd(data)/np.sqrt(np.sum(~np.isnan(data)))
def n_non_nan(data):
    return np.sum(~np.isnan(data))

ncutoff = 10 # if there are less than this many cells, fill in nan
rf_dist = True # if True, use rf center for radial distance; if False, use cell center
dmax = 30 # max radial dist to consider, in vis. degrees
nbin = 5 # number of radial distance pixels

nsize,ncontrast,nangle = 6,6,8
irun,itype,ialign = 0,0,0
nexpt = len(ret_info[0][0])
bins = np.linspace(0,dmax,nbin+1) # boundaries for radial distance pixels

data_bin = np.nan*np.ones((nexpt,nbin,nsize,ncontrast))
data_sem = np.nan*np.ones((nexpt,nbin,nsize,ncontrast))
data_n = np.nan*np.ones((nexpt,nbin,nsize,ncontrast))

# pshift = np.nan*np.ones((nexpt,nbin-1,nsize,ncontrast))
for iexpt in range(nexpt):
    if not ret_info[0][0][iexpt]['ret_map_loc'] is None and not tunings_decon[0][0][iexpt] is None:
	if not rf_dist:
		distance = np.sqrt(np.sum(ret_info[0][0][iexpt]['ret_map_loc']**2,1))
	else:
        	distance = np.sqrt(np.sum(ret_info[0][0][iexpt]['rf_center']**2,1))
        this_data = np.nanmean(tunings_decon[0][0][iexpt][:,:,[0,-5,-4,-3,-2,-1],:,8:-8],4) # some recordings have 3% contrast; ignore this value
	# this_data: roi x size x contrast x angle
	# data: roi x size x contrast
        data = np.nanmean(this_data,3)
        data = data/np.nanmean(data)
        ori_data = ut.circ_align_to_pref(this_data,axis=3) # align so that preferred direction is in the 0th position
        non_nan = ~np.isnan(np.nanmean(np.nanmean(data,1),1)) #& ~np.isnan(np.nanmean(np.nanmean(np.nanmean(ori_data,1),1),1))
        sig_driven = (ret_info[0][0][iexpt]['pval'] < 0.05) & (ret_info[0][0][iexpt]['amplitude'] > 0) & (ret_info[0][0][iexpt]['sigma'] > 3.3) & (ret_info[0][0][iexpt]['sqerror'] < 0.5)
        lkat = non_nan & sig_driven
        this_nsize = data.shape[1]
        if np.sum(lkat):
            for isize in range(this_nsize):
                for icontrast in range(ncontrast):
                    data_bin[iexpt,:,isize,icontrast] = sst.binned_statistic(distance[lkat],data[lkat,isize,icontrast],statistic=np.nanmean,bins=bins).statistic
                    data_sem[iexpt,:,isize,icontrast] = sst.binned_statistic(distance[lkat],data[lkat,isize,icontrast],statistic=nansem,bins=bins).statistic
                    data_n[iexpt,:,isize,icontrast] = sst.binned_statistic(distance[lkat],data[lkat,isize,icontrast],statistic=n_non_nan,bins=bins).statistic
data_bin[data_n < ncutoff] = np.nan
data_sem[data_n < ncutoff] = np.nan
data_bin[:,:,:,:] = data_bin[:,:,:,:]-data_bin[:,:,:,0:1] # compute evoked event rate

usize = np.array((5,8,13,22,36,60))
ucontrast = np.array((0,6,12,25,50,100))

colors = plt.cm.viridis(np.linspace(0,1,6))
x = 0.5*(bins[1:]+bins[:-1])
x = np.concatenate((-x[::-1],x))
plt.figure(figsize=(7.5,5))
for isize in range(nsize):
    plt.subplot(2,3,isize+1)
    for icontrast in range(1,ncontrast):
        lkat = data_n[:,:,isize,icontrast]
        data = np.nanmean(data_bin[:,:,isize,icontrast],0)
        data = np.concatenate((data[::-1],data))
        this_sem = np.sqrt(np.nansum(data_sem[:,:,isize,icontrast]**2,0))/np.sum(~np.isnan(data_sem[:,:,isize,icontrast]),0)
        this_sem = np.concatenate((this_sem[::-1],this_sem))
        plt.errorbar(x,data,this_sem,c=colors[icontrast],label='%d%%'%ucontrast[icontrast])
    plt.ylim((-0.5,6))
    plt.fill_between((-usize[isize]/2,usize[isize]/2),(-0.35,-0.35),(-0.15,-0.15),facecolor='k',alpha=0.5)
    ut.erase_top_right()
    plt.title('%d$^o$ size'%usize[isize])
    plt.axhline(0,c='k',linestyle='dashed',alpha=0.5)
for iplot in range(3,6):
    plt.subplot(2,3,iplot+1)
    plt.xlabel('retinotopic location')
for iplot in range(0,6,3):
    plt.subplot(2,3,iplot+1)
    plt.ylabel('evoked event rate/mean')
plt.subplot(2,3,6)
plt.legend(ncol=2)
plt.tight_layout()
# plt.savefig('figures/spatial_evoked_by_size.jpg',dpi=300)

colors = plt.cm.viridis(np.linspace(0,1,6))
x = 0.5*(bins[1:]+bins[:-1])
x = np.concatenate((-x[::-1],x))
plt.figure(figsize=(7.5,5))
for isize in range(nsize):
    for icontrast in range(0,ncontrast):
        plt.subplot(2,3,icontrast+1)
        lkat = data_n[:,:,isize,icontrast]
        data = np.nanmean(data_bin[:,:,isize,icontrast],0)
        data = np.concatenate((data[::-1],data))
        this_sem = np.sqrt(np.nansum(data_sem[:,:,isize,icontrast]**2,0))/np.sum(~np.isnan(data_sem[:,:,isize,icontrast]),0)
        this_sem = np.concatenate((this_sem[::-1],this_sem))
        plt.errorbar(x,data,this_sem,c=colors[isize],label='%d$^o$'%usize[isize])
        plt.ylim((-0.5,6))
        ut.erase_top_right()
        if icontrast>0:
            plt.title('%d%% contrast'%ucontrast[icontrast])
            plt.axhline(0,c='k',linestyle='dashed',alpha=0.5)
#     plt.fill_between((-usize[isize]/2,usize[isize]/2),(-0.35,-0.35),(-0.15,-0.15),facecolor='k',alpha=0.5)
for iplot in range(3,6):
    plt.subplot(2,3,iplot+1)
    plt.xlabel('retinotopic location')
for iplot in [1,3]:
    plt.subplot(2,3,iplot+1)
    plt.ylabel('evoked event rate/mean')
plt.subplot(2,3,1)
plt.legend(ncol=2)
plt.axis('off')
plt.ylim((0.1,0.2))
plt.tight_layout()
# plt.savefig('figures/spatial_evoked_by_contrast.jpg',dpi=300)