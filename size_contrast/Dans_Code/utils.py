#!/usr/bin/env python

import autograd.numpy as np
from autograd import elementwise_grad as egrad
import scipy.optimize as sop
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import pandas as pd
import pyute as ut
import matplotlib.pyplot as plt
import autograd.scipy.special as ssp
import scipy.stats as sst

def f_miller_troyer(k,mu,s2):
    u = mu/np.sqrt(2*s2)
    A = 0.5*mu*(1+ssp.erf(u))
    B = np.sqrt(s2)/np.sqrt(2*np.pi)*np.exp(-u**2)
    return k*(A + B)

def f_mt(inp):
    return f_miller_troyer(1,inp,1)

def requ(inp):
    return inp**2*(inp>0)

#def predict_output(s0,offset,fn=None,nub_var=nubs_active):
#    inp = (s0[np.newaxis,:]*nub_var).sum(1) + offset
#    return fn(inp)

def unpack_theta(theta):
    s0 = theta[:-1]
    offset = theta[-1]
    return s0,offset

def unpack_theta_amplitude(theta):
    s0 = theta[:-2]
    offset = theta[-2]
    amplitude = theta[-1]
    return s0,offset,amplitude

def unpack_gaussian_theta(theta):
    mu = theta[:2]
    sigma2 = theta[2]
    amplitudeI = theta[3]
    offset1 = theta[4]
    offset2 = theta[5]
    amplitudeO = theta[6]
    return mu,sigma2,amplitudeI,offset1,offset2,amplitudeO

def compute_overlap(mu,sigma2):
    def compute_component(mu,sigma2):
        return 0.5*(ssp.erf((0.5 - mu)/np.sqrt(2*sigma2)) - ssp.erf((-0.5 - mu)/np.sqrt(2*sigma2)))
    components = [compute_component(mu[iaxis],sigma2) for iaxis in range(2)]
    return np.prod(components)

def gaussian_fit_to_nub_rf(mu,sigma2,amplitude,offset):
    nub_locs = np.array([(0,0),(1,0),(0,1),(-1,0),(0,-1)])
    rf = np.array([compute_overlap(mu - nub_locs[iloc],sigma2) for iloc in range(len(nub_locs))])
    rf = amplitude*(rf-offset)
    return rf

def fit_output_gaussian(this_response,fn=None):
    def minusL(theta):
        ypred = predict_output_gaussian_theta(theta,fn=fn)
        return ((this_response - ypred)**2).sum()
    def minusdLdtheta(theta):
        return egrad(minusL)(theta)
    single_patch_response = this_response[[16,8,4,2,1]]
    mu0 = nub_locs[np.argmax(single_patch_response)]
    A0 = np.sqrt(np.max(single_patch_response*(single_patch_response>0)))
    A1 = this_response.max()
    theta0 = np.concatenate((mu0,np.array((1,A0,0,0,A1))))
    bounds = [(-2,2),(-2,2),(1e-2,4),(-1,1),(-np.inf,np.inf),(-np.inf,np.inf),(0,np.inf)]
    thetastar = sop.fmin_l_bfgs_b(minusL,theta0,fprime=minusdLdtheta,bounds=bounds)
    return thetastar

def evaluate_gaussian(xx,yy,mu,sigma2,amplitude,offset):
    g = 1/np.sqrt(2*np.pi*sigma2)*np.exp(-0.5*((xx-mu[0])**2+(yy-mu[1])**2)/sigma2)
    return amplitude*(g-offset)

def evaluate_gaussian_theta(xx,yy,theta):
    mu,sigma2,amplitude,offset1,offset2 = unpack_gaussian_theta(theta)
    return evaluate_gaussian(xx,yy,mu,sigma2,amplitude,offset1)

def show_gaussian_fit(theta,bd=1.75,cbd=2):
    x = np.linspace(-bd,bd,100)
    y = np.linspace(bd,-bd,100)
    xx,yy = np.meshgrid(x,y)
    plt.imshow(evaluate_gaussian_theta(xx,yy,theta),extent=[-bd,bd,-bd,bd],cmap='bwr')
    plt.clim(-cbd,cbd)
    plt.scatter(nub_locs[:,0],nub_locs[:,1],c='g',marker='+')
    rects = []
    for inub in range(nub_locs.shape[0]):
        rect = Rectangle(nub_locs[inub]-np.array((0.5,0.5)),1,1)
        rects.append(rect)
    pc = PatchCollection(rects, facecolor='none', alpha=1, edgecolor='k')
    plt.gca().add_collection(pc)
    plt.xlim((-bd,bd))
    plt.ylim((-bd,bd))

def show_fit(theta,bd=1.75,cbd=1,nub_order=np.array([0,1,2,3,4])):
    x = np.linspace(-bd,bd,100)
    y = np.linspace(bd,-bd,100)
    xx,yy = np.meshgrid(x,y)
    plt.imshow(theta[nub_locs.shape[0]+1]*np.ones_like(xx),extent=[-bd,bd,-bd,bd],cmap='bwr')
    plt.clim(-cbd,cbd)
    rects = []
    facecolors = []
    this_nub_locs = nub_locs[nub_order]
    for inub in range(nub_locs.shape[0]):
        facecolor = plt.cm.bwr((theta[inub]+cbd)/cbd/2)
        rect = Rectangle(this_nub_locs[inub]-np.array((0.5,0.5)),1,1,facecolor=facecolor)
        rects.append(rect)
        facecolors.append(facecolor)
    
    pc = PatchCollection(rects, alpha=1, facecolor=facecolors, edgecolor='k')
    plt.gca().add_collection(pc)
    #plt.scatter(nub_locs[:,0],nub_locs[:,1],c='g',marker='+')
    nub_lbls = [str(n) for n in range(nub_locs.shape[0])]
    for inub in range(this_nub_locs.shape[0]):
        plt.text(this_nub_locs[inub,0],this_nub_locs[inub,1],nub_lbls[inub],c='k',horizontalalignment='center',verticalalignment='center')
    #plt.text(nub_locs[:,0],nub_locs[:,1],nub_lbls,c='g')
    plt.xlim((-bd,bd))
    plt.ylim((-bd,bd))
    plt.xticks([])
    plt.yticks([])

def select_trials(trial_info,selector,training_frac,include_all=False):
    # dict saying what to do with each trial type. If a function, apply that function to the trial info column to 
    # obtain a boolean indexing variable
    # if 0, then the tuning output should be indexed by that variable
    # if 1, then that variable will be marginalized over in the tuning output
    def gen_train_test_exptwise(ti):
        ntrials = ti[params[0]].size
        gd = np.ones((ntrials,),dtype='bool')
        for param in params:
            if callable(selector[param]): # all the values of selector that are functions, ignore trials where that function evaluates to False
                exclude = ~selector[param](ti[param])
                gd[exclude] = False
        condition_list = gen_condition_list(ti,selector) # automatically, separated out such that each half of the data gets an equivalent fraction of trials with each condition type
        condition_list = [c[gd] for c in condition_list]
        in_training_set = np.zeros((ntrials,),dtype='bool')
        in_test_set = np.zeros((ntrials,),dtype='bool')
        to_keep = output_training_test(condition_list,training_frac)
        in_training_set[gd] = to_keep
        in_test_set[gd] = ~to_keep
        if include_all:
            train_test = [in_training_set,in_test_set,np.logical_or(in_training_set,in_test_set)]
        else:
            train_test = [in_training_set,in_test_set]
        return train_test,ntrials
        
    params = list(selector.keys())
    keylist = list(trial_info.keys())
    if isinstance(trial_info[keylist[0]],dict):
        ntrials = {}
        train_test = {}
        for key in trial_info.keys():
            ti = trial_info[key]
            train_test[key],ntrials[key] = gen_train_test_exptwise(ti)
    else:
        ti = trial_info
        train_test,ntrials = gen_train_test_exptwise(ti)
        
    return train_test

def output_training_test(condition_list,training_frac):
    # output training and test sets balanced for conditions
    # condition list, generated by gen_condition_list, has a row for each condition that should be equally assorted
    if not isinstance(condition_list,list):
        condition_list = [condition_list.copy()]
    iconds,uconds = zip(*[pd.factorize(c,sort=True) for c in condition_list])
    #uconds = [np.sort(u) for u in uconds]
    nconds = np.array([u.size for u in uconds])
    in_training_set = np.zeros(condition_list[0].shape,dtype='bool')
    for iflat in range(np.prod(nconds)):
        coords = np.unravel_index(iflat,tuple(nconds))
        lkat = np.where(ut.k_and(*[iconds[ic] == coords[ic] for ic in range(len(condition_list))]))[0]
        n_train = int(np.round(training_frac*len(lkat)))
        to_train = np.random.choice(lkat,n_train,replace=False)
        in_training_set[to_train] = True
    #assert(True==False)
    return in_training_set

def gen_selector_running(run):
    selector = {}
    if run:
        selector['running'] = lambda x: x
    else:
        selector['running'] = lambda x: np.logical_not(x)
    selector['stimulus_direction_deg'] = 1
    selector['stimulus_size_deg'] = 0
    selector['stimulus_contrast'] = 0
    return selector

def compute_tuning(df,trial_info,selector,include=None):
    params = list(selector.keys())
    expts = list(trial_info.keys())
    nexpt = len(expts)
    tuning = [None for iexpt in range(nexpt)]
    if include is None:
        include = {expt:None for expt in expts}
    for iexpt,expt in enumerate(expts):
        in_this_expt = (df.session_id == expt)
        trialwise = df[in_this_expt].pivot(values='data',index='roi_index',columns='trial_index')
        nroi = trialwise.shape[0]
        ntrial = trialwise.shape[1]
        if include[expt] is None:
            include[expt] = np.ones((ntrial,),dtype='bool')
        if not isinstance(include[expt],list):
            include[expt] = [include[expt]]
        npart = len(include[expt])
        if isinstance(include[expt],list):
            tuning[iexpt] = [None for ipart in range(npart)]
        condition_list = []
        condition_list = gen_condition_list(trial_info[expt],selector,filter_selector=np.logical_not)
        iconds,uconds = zip(*[pd.factorize(c,sort=True) for c in condition_list])
        nconds = [len(u) for u in uconds]
        for ipart in range(npart):
            tuning[iexpt][ipart] = np.zeros((nroi,)+tuple(nconds))
            for iflat in range(np.prod(nconds)):
                coords = np.unravel_index(iflat,tuple(nconds))
                lkat = ut.k_and(include[expt][ipart],*[iconds[ic] == coords[ic] for ic in range(len(condition_list))])
                tuning[iexpt][ipart][(slice(None),)+coords] = np.nanmean(trialwise.iloc[:,lkat],-1)
    return tuning

def compute_tuning_df(df,trial_info,selector,include=None):
    params = list(selector.keys())
#     expts = list(trial_info.keys())
    expts = list(np.unique(df.session_id))
    nexpt = len(expts)
    tuning = pd.DataFrame()
    if include is None:
        include = {expt:None for expt in expts}
    for iexpt,expt in enumerate(expts):
        in_this_expt = (df.session_id == expt)
        #print(include[expt][0].sum())
        trialwise = df.loc[in_this_expt].pivot(values='data',index='roi_index',columns='trial_index')
        nroi = trialwise.shape[0]
        ntrial = trialwise.shape[1]
        if include[expt] is None:
            include[expt] = np.ones((ntrial,),dtype='bool')
        if not isinstance(include[expt],list):
            include[expt] = [include[expt]]
        npart = len(include[expt])
#         if isinstance(include[expt],list):
#             tuning[iexpt] = [None for ipart in range(npart)]
        condition_list = []
        condition_list = gen_condition_list(trial_info[expt],selector,filter_selector=np.logical_not)
        iconds,uconds = zip(*[pd.factorize(c,sort=True) for c in condition_list])
        nconds = [len(u) for u in uconds]
        for ipart in range(npart):
            #tip = np.zeros((nroi,)+tuple(nconds))
            #for iflat in range(np.prod(nconds)):
            #    coords = np.unravel_index(iflat,tuple(nconds))
            #    lkat = ut.k_and(include[expt][ipart],*[iconds[ic] == coords[ic] for ic in range(len(condition_list))])
            #    tip[(slice(None),)+coords] = np.nanmean(trialwise.loc[:,lkat],-1)
            #tip_df = pd.DataFrame(tip,index=np.arange(tip.shape[0]),columns=np.arange(tip.shape[1]))
            nflat = np.prod(nconds)
            tip = np.zeros((nroi,nflat))
            coords = np.zeros((len(nconds),nflat),dtype='int')
            for iflat in range(nflat):
                coords[:,iflat] = np.unravel_index(iflat,tuple(nconds))
                #print(include[expt][ipart].sum())
                lkat = ut.k_and(include[expt][ipart],*[iconds[ic] == coords[ic,iflat] for ic in range(len(condition_list))])
                #print((include[expt][ipart].sum(),ut.k_and(*[iconds[ic] == coords[ic,iflat] for ic in range(len(condition_list))]).sum(),lkat.sum()))
                #print(ut.k_and(*[iconds[ic] == coords[ic,iflat] for ic in range(len(condition_list))]).sum())
                #assert(True==False)
                tip[(slice(None),iflat)] = np.nanmean(trialwise.loc[:,lkat],-1)
            tip_df = pd.DataFrame(tip,index=np.arange(tip.shape[0]),columns=[c for c in coords])
            tip_df['partition'] = ipart
            tip_df['session_id'] = expt
            tip_df['celltype'] = trial_info[expt]['celltype']
            tuning = tuning.append(tip_df)
    return tuning #,trialwise,tip
                
def compute_bootstrap_error(df,trial_info,selector,pct=(16,84),include=None):
    params = list(selector.keys())
    expts = list(trial_info.keys())
    nexpt = len(expts)
    bounds = [None for iexpt in range(nexpt)]
    npct = len(pct)
    if include is None:
        include = {expt:None for expt in expts}
    for iexpt,expt in enumerate(expts):
        in_this_expt = (df.session_id == expt)
        trialwise = np.array(df[in_this_expt].pivot(values='data',index='roi_index',columns='trial_index'))
        nroi = trialwise.shape[0]
        ntrial = trialwise.shape[1]
        compress_flag = False
        if include[expt] is None:
            include[expt] = np.ones((ntrial,),dtype='bool')
        if not isinstance(include[expt],list):
            include[expt] = [include[expt]]
            compress_flag = True
        npart = len(include[expt])
        bounds[iexpt] = [None for ipart in range(npart)]
        condition_list = []
        condition_list = gen_condition_list(trial_info[expt],selector,filter_selector=np.logical_not)
        iconds,uconds = zip(*[pd.factorize(c,sort=True) for c in condition_list])
        nconds = [len(u) for u in uconds]
        for ipart in range(npart):
            bounds[iexpt][ipart] = [None for ipct in range(npct)]
            for ipct in range(npct):
                bounds[iexpt][ipart][ipct] = np.zeros((nroi,)+tuple(nconds))
            for iflat in range(np.prod(nconds)):
                coords = np.unravel_index(iflat,tuple(nconds))
                lkat = ut.k_and(include[expt][ipart],*[iconds[ic] == coords[ic] for ic in range(len(condition_list))])
                bds = ut.bootstrap(trialwise[:,lkat],np.nanmean,axis=1,pct=pct)
                for ipct in range(npct):
                    bounds[iexpt][ipart][ipct][(slice(None),)+coords] = bds[ipct]
        if compress_flag:
            bounds[iexpt] = bounds[iexpt][0]
    return bounds

def gen_condition_list(ti,selector,filter_selector=lambda x:True):
# ti: trial_info generated by ut.compute_tavg_dataframe
# selector: dict where each key is a param in ti.keys(), and each value is either a callable returning a boolean, 
# to be applied to ti[param], or an input to the function filter_selector
# filter selector: if filter_selector(selector[param]), the tuning curve will be separated into the unique elements of ti[param]. 
    params = list(selector.keys())
    condition_list = []
    for param in params:
        if not callable(selector[param]) and filter_selector(selector[param]):
            condition_list.append(ti[param])
    return condition_list

def run_roiwise_fn(fn,*inp):
    outp = [None for iexpt in range(len(inp[0]))]
    for iexpt in range(len(inp[0])):
        nroi = len(inp[0][iexpt])
        iroi = 0
        temp = fn(inp[0][iexpt][iroi])
        outp[iexpt] = np.zeros((nroi,temp.shape[0]))
        for iroi in range(nroi):
            print(iroi)
            these_inps = [inp1[iexpt][iroi] for inp1 in inp]
            outp[iexpt][iroi] = fn(*these_inps)
    return outp

def fdr_bh(pvals,fdr=0.05):
    sortind = np.argsort(pvals)
    M = pvals.size
    multby = M/(1+np.arange(M))
    pvals_corr = np.zeros_like(pvals)
    pvals_corr[sortind] = pvals[sortind]*multby
    sig = (pvals_corr < fdr)
    return sig

def test_sig_driven(df,roi_info,trial_info,pcutoff=0.05):
    session_ids = list(roi_info.keys())
    for expt in session_ids:
        in_this_expt = (df.session_id == expt)
        trialwise = df[in_this_expt].pivot(values='data',index='roi_index',columns='trial_index')
        roilist = np.unique(trialwise.index)
        nroi = roilist.size
        roi_info[expt]['sig_driven'] = np.zeros((nroi,),dtype='bool')
        trialcond = trial_info[expt]['stimulus_nubs_active']
        trialrun = trial_info[expt]['running']
        condlist = np.unique(trialcond)
        ncond = len(condlist)
        stim_driven = np.zeros((nroi,ncond-1))
        no_stim = (trialcond==0)&trialrun
        response_no_stim = np.array(trialwise.iloc[:,no_stim].T)
        roi_info[expt]['stim_pval'] = np.zeros((nroi,ncond-1))
        mean_evoked_dfof = trialwise.iloc[:,~no_stim].mean(1) - trialwise.iloc[:,no_stim].mean(1)
        for icond,ucond in enumerate(condlist[1:]):
            this_stim = (trialcond==ucond)&trialrun
            response_this_stim = np.array(trialwise.iloc[:,this_stim].T)
            _,roi_info[expt]['stim_pval'][:,icond] = sst.ttest_ind(response_no_stim,response_this_stim)
        for iroi,uroi in enumerate(roilist):
            different_from_0 = np.any(fdr_bh(roi_info[expt]['stim_pval'][iroi],fdr=pcutoff))
            roi_info[expt]['sig_driven'][iroi] = (different_from_0 and mean_evoked_dfof[iroi]>0.2)
    return roi_info

def compute_tuning_all(df,trial_info):
    conditions = ['running','nonrunning']
    selector = {'running': gen_nub_selector_running(True), 'nonrunning': gen_nub_selector_running(False)}
    keylist = list(trial_info.keys())
    train_test = {}
    for condition in conditions:
        train_test[condition] = {}
        for key in keylist:
            train_test[condition][key] = select_trials(trial_info[key],selector[condition],0.5,include_all=True)
    tuning = pd.DataFrame()
    ttls = np.unique(df.celltype) #['pyr_l4','pyr_l23','sst_l23','vip_l23']
    for ttl in ttls:
        for condition in conditions:
            tuning = tuning.append(compute_tuning_df(df.loc[df.celltype==ttl],trial_info,selector[condition],include=train_test[condition]))
    return tuning

def compute_tuning_many_partitionings(df,trial_info,npartitionings):
    selector_v1 = gen_selector_running(run=True)
    keylist = list(trial_info.keys())
    train_test = {}
    for key in keylist:
        train_test[key] = [None for ipartitioning in range(npartitionings)]
        for ipartitioning in range(npartitionings):
            train_test[key][ipartitioning] = select_trials(trial_info[key],selector_v1,0.5)
    tuning = pd.DataFrame()
    ttls = np.unique(df.celltype) #list(train_test.keys())
    selectors = [selector_v1 for n in range(len(ttls))]
    tt = [{k:v[ipartitioning] for k,v in zip(train_test.keys(),train_test.values())} for ipartitioning in range(npartitionings)]
    for ttl,selector in zip(ttls,selectors):
        for ipartitioning in range(npartitionings):
            #print(tt[ipartitioning][keylist[0]][0].sum())
            new_tuning = compute_tuning_df(df.loc[df.celltype==ttl],trial_info,selector,include=tt[ipartitioning])
            new_tuning['partitioning'] = ipartitioning
            tuning = tuning.append(new_tuning)
    return tuning
