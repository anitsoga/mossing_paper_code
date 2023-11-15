#!/usr/bin/env python

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os

run_cutoff = 7
ctr_cutoff = 10
trial_frac_running_cutoff = 0.33
rf_mapping_pval_cutoff = 0.05

modal_usize = np.array([  5., 8.21865442, 13.50925609, 22.20558144, 36.5])
modal_ucontrast = np.array([0, 6, 12, 25, 50, 100])

### HELPER FNS

# 
def compute_tuning_curves(data_struct,run_cutoff=run_cutoff,ctr_cutoff=ctr_cutoff,rf_mapping_pval_cutoff=rf_mapping_pval_cutoff):
#     trialavg = {}
    strialavg = {}
    usize = {}
    ucontrast = {}
    session_ids = [x for x in data_struct.keys() if not x[0]=='_']
    for session_id in session_ids:
        ds = data_struct[session_id] #[()]
        #trialavg[session_id] = [None]*2
        strialavg[session_id] = [None]*2
        usize[session_id] = ds['stimulus_size_deg'][()]
        ucontrast[session_id] = 100*ds['stimulus_contrast'][()]
        for run_status in range(2):
            running = (ds['running_speed_cm_s'][()]>run_cutoff)==run_status # true where animal's running status agrees with the one being analyzed
            if np.nanmean(running)>trial_frac_running_cutoff: # include only sessions where there were enough trials in the relevant running condition
                nsize = len(ds['stimulus_size_deg'][()])
                ncontrast = len(ds['stimulus_contrast'][()])
                nangle = len(ds['stimulus_direction'][()])
                responsive = ds['rf_mapping_pval'][()]<rf_mapping_pval_cutoff
                centered = ds['rf_distance_deg'][()]<ctr_cutoff
                gd_roi = np.logical_and(responsive,centered)
                nroi = ds['decon'][()][gd_roi].shape[0]
#                 trialavg[session_id][run_status] = np.zeros((nroi,nsize,ncontrast,nangle))
                strialavg[session_id][run_status] = np.zeros((nroi,nsize,ncontrast,nangle))
                for i in range(nsize):
                    for j in range(ncontrast):
                        for k in range(nangle):
                            size_id = ds['stimulus_id'][()][0]
                            contrast_id = ds['stimulus_id'][()][1]
                            angle_id = ds['stimulus_id'][()][2]
                            gd_stimulus = np.logical_and(angle_id==k,np.logical_and(size_id==i,contrast_id==j))
                            gd_trial = np.logical_and(gd_stimulus,running)
                            strialavg[session_id][run_status][:,i,j,k] = np.nanmean(ds['decon'][()][gd_roi][:,gd_trial],axis=1)
                            #trialavg[session_id][run_status][:,i,j,k] = np.nanmean(ds['calcium_responses_au'][()][gd_roi][:,gd_trial],axis=1)
    return strialavg,usize,ucontrast

# normalize each ROI's tuning curve to its max value, before averaging across ROIs
def norm_to_max(arr):
    data = np.nanmean(arr,-1)
    data = data/np.nanmax(np.nanmax(data,-1),-1)[:,np.newaxis,np.newaxis]
    return data

# add additional data to the 0th axis of an existing array, or start a new array if existing array is empty
def tack_on(starting,to_add):
    if not starting.size:
        return to_add
    else:
        return np.concatenate((starting,to_add),axis=0)

# show mean size/contrast tuning across ROIs, for each animal, in the running and/or non-running condition
def plot_tuning_curves(strialavg):
    plt.figure()
    keylist = list(strialavg.keys())
    for i in range(len(keylist)):
        plt.subplot(2,len(keylist),i+1)
        try:
            data = np.nanmean(strialavg[keylist[i]][0],-1)
            data = data/np.nanmax(np.nanmax(data,-1),-1)[:,np.newaxis,np.newaxis]
            plt.imshow(np.nanmean(data,0))
        except:
            print('no non-running')
        plt.axis('off')
    # plt.figure()
    for i in range(len(keylist)):
        plt.subplot(2,len(keylist),len(keylist)+i+1)
        try:
            data = np.nanmean(strialavg[keylist[i]][1],-1)
            data = data/np.nanmax(np.nanmax(data,-1),-1)[:,np.newaxis,np.newaxis]
            plt.imshow(np.nanmean(data,0))
        except:
            print('no running')
        plt.axis('off')

# normalize by ROI, and concatenate data across sessions into one big array
def compute_normalized_summary_tuning(strialavg, usize, modal_usize, ucontrast, modal_ucontrast,run_status=1):
    snorm = np.array(())
    keylist = list(strialavg.keys())
    for key in keylist:
        try:
            data = norm_to_max(strialavg[key][run_status])
            take_these_sizes = np.in1d(np.round(usize[key]),np.round(modal_usize))
            put_here_sizes = np.in1d(np.round(modal_usize),np.round(usize[key]))
            take_these_contrasts = np.in1d(np.round(ucontrast[key]),np.round(modal_ucontrast))
            put_here_contrasts = np.in1d(np.round(modal_ucontrast),np.round(ucontrast[key]))
            if np.all(put_here_sizes) and np.all(put_here_contrasts):
                snorm = tack_on(snorm,data[:,take_these_sizes][:,:,take_these_contrasts])
        except:
            if run_status:
                print('no running in ' + key)
            else:
                print('no non-running in ' + key)
    return snorm

# compute bootstrapped 95 pct. confidence intervals for fn (usu. the mean) along a specified axis
def bootstrap(arr,fn,axis=0,nreps=1000,pct=(2.5,97.5)):
    # given arr 1D of size N, resample nreps sets of N of its elements with replacement. Compute fn on each of the samples
    # and report percentiles pct
    N = arr.shape[axis]
    c = np.random.choice(np.arange(N),size=(N,nreps))
    L = len(arr.shape)
    resamp=np.rollaxis(arr,axis,0)
    resamp=resamp[c]
    resamp=np.rollaxis(resamp,0,axis+2) # plus 1 due to rollaxis syntax. +1 due to extra resampled axis
    resamp=np.rollaxis(resamp,0,L+1)
    stat = fn(resamp,axis=axis)
    lb = np.percentile(stat,pct[0],axis=-1) # resampled axis rolled to last position
    ub = np.percentile(stat,pct[1],axis=-1) # resampled axis rolled to last position
    return lb,ub

# compute bootstrapped lower and upper bounds, and plot them
def plot_bootstrapped_errorbars_w_dots(x,arr,pct=(2.5,97.5),colors=None,linewidth=None,markersize=None):
    mn_tgt = np.nanmean(arr,0)
    lb_tgt,ub_tgt = bootstrap(arr,fn=np.nanmean,pct=pct)
    plot_errorbars_w_dots(x,mn_tgt,lb_tgt,ub_tgt,colors=colors,linewidth=linewidth,markersize=markersize)
    
# given a set of errorbars, plot them
def plot_errorbars_w_dots(x,mn_tgt,lb_tgt,ub_tgt,colors=None,linewidth=None,markersize=None):
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0,1,mn_tgt.shape[0]))
    for i in range(mn_tgt.shape[0]):
        plot_errorbar_w_dots(x,mn_tgt[i],lb_tgt[i],ub_tgt[i],c=colors[i],linewidth=linewidth,markersize=markersize)
        
# plot a single errorbar
def plot_errorbar_w_dots(x,mn_tgt,lb_tgt,ub_tgt,plot_options=None,c=None,linestyle=None,linewidth=None,markersize=None):
    opt_keys = ['c','linestyle','linewidth','markersize']
    opt = parse_options(plot_options,opt_keys,c,linestyle,linewidth,markersize)
    c,linestyle,linewidth,markersize = [opt[key] for key in opt_keys]

    errorplus = ub_tgt-mn_tgt
    errorminus = mn_tgt-lb_tgt
    errors = np.concatenate((errorplus[np.newaxis],errorminus[np.newaxis]),axis=0)
    plt.errorbar(x,mn_tgt,yerr=errors,c=c,linestyle=linestyle,linewidth=linewidth)
    plt.scatter(x,mn_tgt,c=c,s=markersize)
    
# wrapper for specifying plot options
def parse_options(opt,opt_keys,*args):
    # create a dict opt with keys opt_keys specifying the options listed
    # options specified in *args will overwrite the original entries of opt if they are not None

    if opt is None:
        opt = {}

    for i,key in enumerate(opt_keys):
        if not args[i] is None or not key in opt:
            opt[key] = args[i]

    for key in opt_keys:
        if not key in opt:
            opt[key] = None

    return opt



### MAIN PART

data_struct_files = ['pyr_l23_data_struct.mat','sst_data_struct.mat','vip_data_struct.mat']

if not os.path.exists('results'):
	os.mkdir('results')

for filename in data_struct_files:
    cell_type = filename.split('_data_struct.mat')[0]
    data_struct = sio.loadmat(filename,struct_as_record=True,squeeze_me=True)
    strialavg,usize,ucontrast = compute_tuning_curves(data_struct)
    
    for run_status in range(2):
        snorm = compute_normalized_summary_tuning(strialavg, usize, modal_usize, ucontrast, modal_ucontrast,run_status=run_status)
        plt.figure()
        if run_status:
            to_append = 'running'
        else:
            to_append = 'non-running'
        plt.title(cell_type + ' contrast response by size, '+to_append)
        plot_bootstrapped_errorbars_w_dots(modal_ucontrast,snorm)
        plt.legend([str(int(np.round(x)))+'$^o$' for x in modal_usize])
        plt.xlabel('contrast (%)')
        plt.ylabel('event rate / max event rate')
        plt.savefig('results/'+cell_type+'_contrast_by_size_'+to_append+'.pdf')

    for run_status in range(2):
        snorm = compute_normalized_summary_tuning(strialavg, usize, modal_usize, ucontrast, modal_ucontrast,run_status=run_status)
        plt.figure()
        if run_status:
            to_append = 'running'
        else:
            to_append = 'non-running'
        plt.title(cell_type + ' size response by contrast, '+to_append)
        gray = np.tile(snorm[:,:,0].mean(1)[:,np.newaxis,np.newaxis],(1,1,len(modal_ucontrast)))
        usize0 = np.concatenate(((0,),modal_usize))
        snorm0 = np.concatenate((gray,snorm),axis=1)
        plot_bootstrapped_errorbars_w_dots(usize0,snorm0[:,:,1:].transpose((0,2,1)))
        plt.legend([str(int(np.round(x)))+'%' for x in modal_ucontrast[1:]])
        plt.xlabel('size ($^o$)')
        plt.ylabel('event rate / max event rate')
        plt.savefig('results/'+cell_type+'_size_by_contrast_'+to_append+'.pdf')
