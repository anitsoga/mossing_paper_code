#!/usr/bin/env python
# coding: utf-8

#######################################
# I/O
import os
import sys
sys.path.insert(0,"/Users/agos/Dropbox/ColumbiaProjects/Data_NewDanFitting/Data/data_scripts/OASIS-master/")
sys.path.insert(0, "/Users/agos/Dropbox/ColumbiaProjects/Dans_Data_Package/analysis_agos/")
OutputDir='./Output/'
import pickle

#######################################
# Basics
import autograd.numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sst
from importlib import reload
import matplotlib.ticker as ticker
#######################################
# Dan
#import pyute as ut
#import sim_utils
#reload(sim_utils)
#import opto_utils

#######################################
# Colormaps
import matplotlib.colors as mc
import matplotlib._color_data as mcd
import colorsys
import matplotlib.colors as mcolors

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

##################################
# General
import itertools
from scipy.special import erf
from scipy import optimize
import math
import random
import scipy as sp
from tqdm import tqdm
from scipy.stats import norm
import h5py
from scipy import optimize
import sklearn.metrics as skm
import sklearn
import sklearn.linear_model
#######################################
# Auxiliary funcs

def nansem(data):
    return np.nanstd(data)/np.sqrt(np.sum(~np.isnan(data)))
def n_non_nan(data):
    return np.sum(~np.isnan(data))


def lighten_color(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        try:
            c= mcd.XKCD_COLORS[color].upper()
        except:
            c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


#####################################################################################
#####################################################################################
# This is a coopy paste of Dans functions, modified to control locomotion parameters
#####################################################################################
#####################################################################################


default_running_pct_cutoff = 0.4
default_running_speed_threshold=10


def compute_tuning(dsfile,datafield='decon',running=True,expttype='size_contrast_0',running_pct_cutoff=default_running_pct_cutoff,fill_nans_under_cutoff=True,running_speed_threshold=default_running_speed_threshold):
    # take in an HDF5 data struct, and convert to an n-dimensional matrix
    # describing the tuning curve of each neuron. For size-contrast stimuli,
    # the dimensions of this matrix are ROI index x size x contrast x direction x time.
    # This outputs a list of such matrices, where each element is one imaging session
    
    
    with h5py.File(dsfile,mode='r') as f:
        keylist = [key for key in f.keys()]
        tuning = [None for i in range(len(keylist))]
        uparam = [None for i in range(len(keylist))]
        displacement = [None for i in range(len(keylist))]
        pval = [None for i in range(len(keylist))]
        for ikey in range(len(keylist)):
            session = f[keylist[ikey]]
            print(session)
            #print([key for key in session.keys()])
            if expttype in session and datafield in session[expttype]:
                sc0 = session[expttype]
                print(datafield)
                data = sc0[datafield][:]
                stim_id = sc0['stimulus_id'][:]
                nbefore = sc0['nbefore'][()]
                nafter = sc0['nafter'][()]
                if running:
                    # we want the number of running trials is bigger than running_pct_cutoff_
                    pct_cutoff=running_pct_cutoff
                    trialrun = sc0['running_speed_cm_s'][:,nbefore:-nafter].mean(-1)>running_speed_threshold #
                else:
                    # and therefore the number of stationary ones will be 1-those
                    #pct_cutoff=1-running_pct_cutoff
                    pct_cutoff=running_pct_cutoff# this is how dan had it

                    trialrun = sc0['running_speed_cm_s'][:,nbefore:-nafter].mean(-1)<running_speed_threshold
                print(np.nanmean(trialrun))
                if np.nanmean(trialrun)>pct_cutoff:
                    tuning[ikey] = ut_compute_tuning(data,stim_id,trial_criteria=trialrun)[:]
                elif fill_nans_under_cutoff:
                    tuning[ikey] = ut_compute_tuning(data,stim_id,trial_criteria=trialrun)[:]
                    tuning[ikey] = np.nan*np.ones_like(tuning[ikey])
                uparam[ikey] = []
                for param in sc0['stimulus_parameters']:
                    uparam[ikey] = uparam[ikey]+[sc0[param][:]]
                if 'rf_displacement_deg' in sc0:
                    pval[ikey] = sc0['rf_mapping_pval'][:]
                    sqerror = session['retinotopy_0']['rf_sq_error'][:]
                    sigma = session['retinotopy_0']['rf_sigma'][:]
                    X = session['cell_center'][:]
                    y = sc0['rf_displacement_deg'][:].T
                    rf_conditions = [ut_k_and(~np.isnan(X[:,0]),~np.isnan(y[:,0])),sqerror<0.75,sigma>3.3,pval[ikey]<0.1]
                    lkat = np.ones((X.shape[0],),dtype='bool')
                    for cond in rf_conditions:
                        lkat_new = (lkat & cond)
                        if lkat_new.sum()>=5:
                            lkat = lkat_new.copy()
                    linreg = sklearn.linear_model.LinearRegression().fit(X[lkat],y[lkat])
                    displacement[ikey] = np.zeros_like(y)
                    displacement[ikey][~np.isnan(X[:,0])] = linreg.predict(X[~np.isnan(X[:,0])])
    return tuning,uparam,displacement,pval
    
    
    
    
def compute_tuning_w_speed(dsfile,datafield='decon',running=True,expttype='size_contrast_0',running_pct_cutoff=default_running_pct_cutoff,fill_nans_under_cutoff=True,running_speed_threshold=default_running_speed_threshold):
    # take in an HDF5 data struct, and convert to an n-dimensional matrix
    # describing the tuning curve of each neuron. For size-contrast stimuli,
    # the dimensions of this matrix are ROI index x size x contrast x direction x time.
    # This outputs a list of such matrices, where each element is one imaging session
    
    
    with h5py.File(dsfile,mode='r') as f:
        keylist = [key for key in f.keys()]
        tuning = [None for i in range(len(keylist))]
        uparam = [None for i in range(len(keylist))]
        displacement = [None for i in range(len(keylist))]
        pval = [None for i in range(len(keylist))]
        for ikey in range(len(keylist)):
            session = f[keylist[ikey]]
            print(session)
            #print([key for key in session.keys()])
            if expttype in session and datafield in session[expttype]:
                sc0 = session[expttype]
                print(datafield)
                data = sc0[datafield][:]
                stim_id = sc0['stimulus_id'][:]
                nbefore = sc0['nbefore'][()]
                nafter = sc0['nafter'][()]
                if running:
                    # we want the number of running trials is bigger than running_pct_cutoff_
                    pct_cutoff=running_pct_cutoff
                    trialrun = sc0['running_speed_cm_s'][:,nbefore:-nafter].mean(-1)>running_speed_threshold #
                else:
                    # and therefore the number of stationary ones will be 1-those
                    #pct_cutoff=1-running_pct_cutoff
                    pct_cutoff=running_pct_cutoff# this is how dan had it

                    trialrun = sc0['running_speed_cm_s'][:,nbefore:-nafter].mean(-1)<running_speed_threshold
                print(np.nanmean(trialrun))
                if np.nanmean(trialrun)>pct_cutoff:
                    tuning[ikey] = ut_compute_tuning(data,stim_id,trial_criteria=trialrun)[:]
                elif fill_nans_under_cutoff:
                    tuning[ikey] = ut_compute_tuning(data,stim_id,trial_criteria=trialrun)[:]
                    tuning[ikey] = np.nan*np.ones_like(tuning[ikey])
                uparam[ikey] = []
                for param in sc0['stimulus_parameters']:
                    uparam[ikey] = uparam[ikey]+[sc0[param][:]]
                if 'rf_displacement_deg' in sc0:
                    pval[ikey] = sc0['rf_mapping_pval'][:]
                    sqerror = session['retinotopy_0']['rf_sq_error'][:]
                    sigma = session['retinotopy_0']['rf_sigma'][:]
                    X = session['cell_center'][:]
                    y = sc0['rf_displacement_deg'][:].T
                    rf_conditions = [ut_k_and(~np.isnan(X[:,0]),~np.isnan(y[:,0])),sqerror<0.75,sigma>3.3,pval[ikey]<0.1]
                    lkat = np.ones((X.shape[0],),dtype='bool')
                    for cond in rf_conditions:
                        lkat_new = (lkat & cond)
                        if lkat_new.sum()>=5:
                            lkat = lkat_new.copy()
                    linreg = sklearn.linear_model.LinearRegression().fit(X[lkat],y[lkat])
                    displacement[ikey] = np.zeros_like(y)
                    displacement[ikey][~np.isnan(X[:,0])] = linreg.predict(X[~np.isnan(X[:,0])])
    return tuning,uparam,displacement,pval
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def get_ret_info(dsfile,expttype='size_contrast_0'):
    # take in an HDF5 data struct, and convert to an n-dimensional matrix
    # describing the tuning curve of each neuron. For size-contrast stimuli,
    # the dimensions of this matrix are ROI index x size x contrast x direction x time.
    # This outputs a list of such matrices, where each element is one imaging session
    with h5py.File(dsfile,mode='r') as f:
        keylist = [key for key in f.keys()]
        ret_info = [None for i in range(len(keylist))]
        for ikey in range(len(keylist)):
            session = f[keylist[ikey]]
            print(session)
            #print([key for key in session.keys()])
            if expttype in session:
                sc0 = session[expttype]
                if 'rf_displacement_deg' in sc0:
                    pval = sc0['rf_mapping_pval'][:]
                    sqerror = session['retinotopy_0']['rf_sq_error'][:]
                    sigma = session['retinotopy_0']['rf_sigma'][:]
                    try:
                        amplitude = session['retinotopy_0']['rf_amplitude'][:]
                    except:
                        amplitude = np.nan*np.ones_like(sigma)
                    cell_center = session['cell_center'][:]
                    rf_center = sc0['rf_displacement_deg'][:].T
                    X = cell_center
                    y = rf_center
                    rf_conditions = [ut_k_and(~np.isnan(X[:,0]),~np.isnan(y[:,0])),sqerror<0.75,sigma>3.3,pval<0.1]
                    lkat = np.ones((X.shape[0],),dtype='bool')
                    for cond in rf_conditions:
                        lkat_new = (lkat & cond)
                        if lkat_new.sum()>=5:
                            lkat = lkat_new.copy()
                    linreg = sklearn.linear_model.LinearRegression().fit(X[lkat],y[lkat])
                    ret_map_loc = np.zeros_like(y)
                    ret_map_loc[~np.isnan(X[:,0])] = linreg.predict(X[~np.isnan(X[:,0])])
                    ret_info[ikey] = {'pval':pval,'sqerror':sqerror,'sigma':sigma,\
                            'cell_center':cell_center,'rf_center':rf_center,\
                            'ret_map_loc':ret_map_loc,'amplitude':amplitude}
    return ret_info


def ut_k_and(*args):
    if len(args)>2:
        return np.logical_and(args[0],k_and(*args[1:]))
    elif len(args)==2:
        return np.logical_and(args[0],args[1])
    else:
        return args[0]


def ut_compute_tuning(data,stim_id,cell_criteria=None,trial_criteria=None):
    ndims = stim_id.shape[0]
    maxind = tuple(stim_id.max(1).astype('int')+1)
    if cell_criteria is None:
        cell_criteria = np.ones((data.shape[0],),dtype='bool')
    if trial_criteria is None:
        trial_criteria = np.ones((data.shape[1],),dtype='bool')
    nparams = len(maxind)
    ntrialtypes = int(np.prod(maxind))
    tuning = np.zeros((data[cell_criteria].shape[0],ntrialtypes)+data.shape[2:])
    for itype in range(ntrialtypes):
        imultitype = np.unravel_index(itype,maxind)
        these_trials = trial_criteria.copy()
        for iparam in range(nparams):
            these_trials = np.logical_and(these_trials,stim_id[iparam]==imultitype[iparam])
        tuning[:,itype] = np.nanmean(data[cell_criteria][:,these_trials],1)
    tuning = np.reshape(tuning,(tuning.shape[0],)+maxind+tuning.shape[2:])
    return tuning
