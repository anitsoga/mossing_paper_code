#!/usr/bin/env python

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

def compute_tuning(dsfile,datafield='decon',running=True,running_threshold=10):
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
            if 'size_contrast_0' in session and datafield in session['size_contrast_0']:
                sc0 = session['size_contrast_0']
                print(datafield)
                data = sc0[datafield][:]
                stim_id = sc0['stimulus_id'][:]
                nbefore = sc0['nbefore'][()]
                nafter = sc0['nafter'][()]
                if running:
                    trialrun = sc0['running_speed_cm_s'][:,nbefore:-nafter].mean(-1)>running_threshold #
                else:
                    trialrun = sc0['running_speed_cm_s'][:,nbefore:-nafter].mean(-1)<running_threshold
                #print(sc0['running_speed_cm_s'].shape)
                print(np.nanmean(trialrun))
                if np.nanmean(trialrun)>0.4:
                    tuning[ikey] = ut.compute_tuning(data,stim_id,trial_criteria=trialrun)[:]
                for param in sc0['stimulus_parameters']:
                    uparam[ikey] = sc0[param][:]
                if 'rf_displacement_deg' in sc0:
                    pval[ikey] = sc0['rf_mapping_pval'][:]
                    X = session['cell_center'][:]
                    y = sc0['rf_displacement_deg'][:].T
                    lkat = ut.k_and(pval[ikey]<0.05,~np.isnan(X[:,0]),~np.isnan(y[:,0]))
                    linreg = sklearn.linear_model.LinearRegression().fit(X[lkat],y[lkat])
                    displacement[ikey] = np.zeros_like(y)
                    displacement[ikey][~np.isnan(X[:,0])] = linreg.predict(X[~np.isnan(X[:,0])])
    return tuning,uparam,displacement,pval

def compute_tunings(dsnames,datafield='decon',running=True,running_threshold=10):
    # compute tuning as above, for each of a list of HDF5 files each corresponding to a particular cell type
    tunings = []
    uparams = []
    displacements = []
    pvals = []
    for dsname in dsnames:
        print(dsname)
        this_tuning,this_uparam,this_displacement,this_pval = compute_tuning(dsname,datafield=datafield,running=running,running_threshold=running_threshold)
        tunings.append(this_tuning)
        uparams.append(this_uparam)
        displacements.append(this_displacement)
        pvals.append(this_pval)
    return tunings,uparams,displacements,pvals
        
def default_dsnames():
    print("this is default but you shouldnt be here")
    dsbase = '/Users/agos/ProjectsCluster/AdesnikData/FinalDataSet/GenerateTuningCurves/'
    dsnames = [dsbase+x+'_data_struct.hdf5' for x in ['pyr_l4','pyr_l23','pv_l23','sst_l23','vip_l23']]
    return dsnames

def default_selection():
    # select experiments which had all of the relevant stim conditions
    selection = [None, None, None, [1,2,3,4,5,9], None]
    return selection

def default_condition_inds():
    # select the relevant stim conditions for experiments that had extra ones
    slices = [slice(None,5),[0,-5,-4,-3,-2,-1]]
    return slices
    
def average_up(arr,nbefore=8,nafter=8):
    # average across time points and directions
    return np.nanmean(np.nanmean(arr[:,:,:,:,nbefore:-nafter],-1),-1)

def columnize(arr):
    output = np.nanmean(arr,0).flatten()
    output = output/output.max()
    return output

def include_aligned(displacement,dcutoff,pval,pcutoff=0.05,less=True):
    # split data up into spatial pixels, according to the distance of the RF center from the stimulus center
    if less:
        criterion = lambda x: (x**2).sum(0) < dcutoff**2
    else:
        criterion = lambda x: (x**2).sum(0) > dcutoff**2
    return np.logical_and(criterion(displacement),pval < pcutoff)

def gen_rspatial(dsnames=None,selection=None,dcutoffs=[0,5,10,15],pval_cutoff=0.05,slices=None,datafield='decon',running_threshold=10):
    # from a list of HDF5 files, split up the data into an arbitrary number of spatial pixels based on RF center location
    if dsnames is None:
        dsnames = default_dsnames()
    if selection is None:
        selection = default_selection()
    if slices is None:
        condition_inds = default_condition_inds()
        
    tunings,uparams,displacements,pvals = compute_tunings(dsnames,datafield=datafield,running_threshold=running_threshold)
    
    rs = []
    for icelltype in range(len(tunings)):
        rs.append([])
        these_tunings = tunings[icelltype]
        these_displacements = displacements[icelltype]
        these_pvals = pvals[icelltype]
        if not selection[icelltype] is None:
            sel = selection[icelltype]
        else:
            sel = np.arange(len(these_tunings))
        these_displacements = [these_displacements[i].T for i in sel if not these_tunings[i] is None]
        these_pvals = [these_pvals[i] for i in sel if not these_tunings[i] is None]
        these_tunings = [these_tunings[i] for i in sel if not these_tunings[i] is None]
        for idcutoff in range(len(dcutoffs)):
            dcutoff = dcutoffs[idcutoff]
            if len(dcutoffs)>idcutoff+1:
                dcuton = dcutoffs[idcutoff+1]
                aligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=False) & include_aligned(d,dcuton,p,pval_cutoff,less=True) for d,p in zip(these_displacements,these_pvals)]
            else:
                aligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=False) for d,p in zip(these_displacements,these_pvals)]
#         aligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=True) for d,p in zip(these_displacements,these_pvals)]
#         misaligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=False) for d,p in zip(these_displacements,these_pvals)]
            raligned = average_up(np.concatenate([x[aligned[i]][:,condition_inds[0],condition_inds[1]] for i,x in enumerate(these_tunings)],axis=0))
#         rmisaligned = average_up(np.concatenate([x[misaligned[i]][:,condition_inds[0],condition_inds[1]] for i,x in enumerate(these_tunings)],axis=0))
            rs[icelltype].append(raligned)
    return rs

def gen_rs(dsnames=None,selection=None,dcutoff=5,pval_cutoff=0.05,slices=None,running=True,running_threshold=10):
    # same specifically for case of two spatial pixels
    if dsnames is None:
        dsnames = default_dsnames()
    if selection is None:
        selection = default_selection()
    if slices is None:
        condition_inds = default_condition_inds()
        
    tunings,uparams,displacements,pvals = compute_tunings(dsnames,running=running,running_threshold=running_threshold)
    
    rs = []
    for icelltype in range(len(tunings)):
        these_tunings = tunings[icelltype]
        these_displacements = displacements[icelltype]
        these_pvals = pvals[icelltype]
        if not selection[icelltype] is None:
            sel = selection[icelltype]
        else:
            sel = np.arange(len(these_tunings))
        these_displacements = [these_displacements[i].T for i in sel if not these_tunings[i] is None]
        these_pvals = [these_pvals[i] for i in sel if not these_tunings[i] is None]
        these_tunings = [these_tunings[i] for i in sel if not these_tunings[i] is None]
        aligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=True) for d,p in zip(these_displacements,these_pvals)]
        misaligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=False) for d,p in zip(these_displacements,these_pvals)]
        raligned = average_up(np.concatenate([x[aligned[i]][:,condition_inds[0],condition_inds[1]] for i,x in enumerate(these_tunings)],axis=0))
        rmisaligned = average_up(np.concatenate([x[misaligned[i]][:,condition_inds[0],condition_inds[1]] for i,x in enumerate(these_tunings)],axis=0))
        rs.append([raligned,rmisaligned])
    return rs

def gen_size_tuning(sc):
    # add 0% contrast stimulus as if it were a 0 degree size
    gray = np.tile(sc[:,:,0].mean(1)[:,np.newaxis,np.newaxis],(1,1,sc.shape[2]))
    to_plot = np.concatenate((gray,sc),axis=1)
    print(to_plot.shape)
    return to_plot

def plot_size_tuning_by_contrast(arr):
    usize = np.array((0,5,8,13,22,36))
    ucontrast = np.array((0,6,12,25,50,100))
    arr_sz = gen_size_tuning(arr)
    arr_sz = arr_sz/arr_sz.max(1).max(1)[:,np.newaxis,np.newaxis]
    lb,ub = ut.bootstrap(arr_sz,np.mean,pct=(2.5,97.5))
    to_plot = arr_sz.mean(0)
    for ic in range(1,6):
        plt.subplot(1,5,ic)
        ut.plot_bootstrapped_errorbars_hillel(usize,arr_sz[:,:,ic].transpose((0,2,1)),colors=['k','r'],markersize=5)
#         plt.scatter((0,),to_plot[:,0,0].mean(0))
#         plt.scatter((0,),to_plot[:,0,1].mean(0))
        plt.ylim(0.5*to_plot.min(),1.2*to_plot.max())
        plt.title('%d%% contrast' % ucontrast[ic])
        plt.xlabel('size ($^o$)')
    plt.subplot(1,5,1)
    plt.ylabel('event rate / max event rate')
    plt.tight_layout()
    
def plot_size_tuning(arr,colors=None):
    usize = np.array((0,5,8,13,22,36))
    ucontrast = np.array((0,6,12,25,50,100))
    arr_sz = gen_size_tuning(arr)
    arr_sz = arr_sz/arr_sz.max(1).max(1)[:,np.newaxis,np.newaxis]
    lb,ub = ut.bootstrap(arr_sz,np.mean,pct=(2.5,97.5))
    to_plot = arr_sz.mean(0)
    ut.plot_bootstrapped_errorbars_hillel(usize,arr_sz[:,:,1::].transpose((0,2,1)),colors=colors)
    plt.ylim(to_plot.min()-0.1,to_plot.max()+0.1)
#     plt.title('%d%% contrast' % ucontrast[ic])
    plt.xlabel('size ($^o$)')
#     plt.subplot(1,2,1)
    
    plt.tight_layout()
    
def plot_size_tuning_peak_norm(arr,colors=None):
    usize = np.array((0,5,8,13,22,36))
    ucontrast = np.array((0,6,12,25,50,100))
    arr_sz = gen_size_tuning(arr)
    mx = arr_sz.max(1)[:,np.newaxis]
    mn = arr_sz.min(1)[:,np.newaxis]
    mn = 0
    arr_sz = (arr_sz-mn)/(mx-mn)
    lb,ub = ut.bootstrap(arr_sz,np.mean,pct=(2.5,97.5))
    to_plot = arr_sz.mean(0)
    ut.plot_bootstrapped_errorbars_hillel(usize,arr_sz[:,:,1::2].transpose((0,2,1)),colors=colors[::2])
    plt.ylim(to_plot.min()-0.1,to_plot.max()+0.1)
#     plt.title('%d%% contrast' % ucontrast[ic])
    plt.xlabel('size ($^o$)')
    plt.tight_layout()
    
def f_miller_troyer(mu,s2):
    # firing rate function, gaussian convolved with ReLU, derived in Miller and Troyer 2002
    u = mu/np.sqrt(2*s2)
    A = 0.5*mu*(1+ssp.erf(u))
    B = np.sqrt(s2)/np.sqrt(2*np.pi)*np.exp(-u**2)
    return A + B
#     return 0.5*mu*(1+np.exp(u)) + sigma/np.sqrt(2*np.pi)*np.exp(-u**2) # 0.5*mu*(1+ssp.erf(u))

def fprime_miller_troyer(mu,s2):
    # firing rate function, gaussian convolved with ReLU, derived in Miller and Troyer 2002
    u = mu/np.sqrt(2*s2)
    A = 0.5*(1+ssp.erf(u))
    return A

def fit_w(X,y,rate_fn,wm0=None,ws0=None,bounds=None):
    # X is (N,P), y is (N,). Finds w: (P,) weight matrix to explain y as y = f(X(wm),X(ws))
    # f is a static nonlinearity, given as a function of mean and std. of noise
    N,P = X.shape
    def parse_w(w):
        wm = w[:P]
        ws = w[P:]
#         return wm,ws,k
        return wm,ws
    def minusL(w):
#         wm,ws,k = parse_w(w)
        wm,ws = parse_w(w)
        return 0.5*np.sum((rate_fn(X @ wm,X @ ws)-y)**2) # k*
    def minusdLdw(w): 
        # sum in first dimension: (N,1) times (N,1) times (N,P)
        return egrad(minusL)(w)
    
    w0 = np.concatenate((wm0,ws0)) #,(k0,)))
    
    factr=1e7
    epsilon=1e-8
    pgtol=1e-5
    wstar = sop.fmin_l_bfgs_b(minusL,w0,fprime=minusdLdw,bounds=bounds,pgtol=pgtol,factr=factr,epsilon=epsilon)
    
    return wstar

def u_fn(X,Wx,Y,Wy,k):
    return X[0] @ Wx + X[1] @ (Wx*k) + Y[0] @ Wy + Y[1] @ (Wy*k)

def evaluate_f_mt(X,Ws,offset,k):
    # Ws: Wx,Wy,s02,Y
    return f_miller_troyer(u_fn(X,Ws[0],Ws[3],Ws[1],k)+offset,Ws[2])

def fit_w_data_loss(X,ydata,rate_fn,wm0=None,ws0=None,s020=None,k0=None,bounds=None,niter=int(1e4)):
    # X is (N,P), y is (N,). Finds w: (P,) weight matrix to explain y as y = f(X(wm),X(ws))
    # f is a static nonlinearity, given as a function of mean and std. of noise
    N,P = X[0].shape
    abd = 1

    nroi = ydata.shape[0]
    alpha_roi = sst.norm.ppf(np.arange(1,nroi+1)/(nroi+1))
    
    def sort_by_11(w):
        yalpha11 = rate_fn_wrapper(w,np.array((-abd,abd)))
        difference11 = compute_y_distance(yalpha11[np.newaxis,:,:],ydata[:,np.newaxis,:])
        sortby11 = np.argsort(difference11[:,0]-difference11[:,-1])
        return sortby11
    
    def compute_y_distance(y1,y2):
        return np.sum((y1-y2)**2,axis=-1)
    
    def compare_sorted_to_expected(w,sortind):
       
        yalpha_roi = rate_fn_wrapper(w,alpha_roi)
#         print(yalpha_roi.max())
        difference_roi = compute_y_distance(yalpha_roi,ydata[sortind])
        difference_roi_unsorted = compute_y_distance(yalpha_roi,ydata)
#         print(difference_roi.shape)
        return difference_roi

    def rate_fn_wrapper(w,alphas):
        wm,ws,s02,k = parse_w(w)
        inputs0 = [wm,np.array((0,)),s02,[np.array((0,)),np.array((0,))]]
        inputs1 = [X,ws,[np.array((0,)),np.array((0,))],np.array((0,))]
        yalpha = rate_fn(X,inputs0,alphas[:,np.newaxis]*u_fn(*inputs1,k),k)
        yalpha = normalize(yalpha)
        return yalpha

#     def compute_f_by_itself(w):

    def normalize(arr):
        arrsum = arr.sum(1)
#         arrnorm = np.ones_like(arr)
#         arrnorm = arrnorm/arrnorm.shape[1]
        well_behaved = (arrsum>0)[:,np.newaxis]
        arrnorm = well_behaved*arr/arrsum[:,np.newaxis] + (~well_behaved)*np.ones_like(arr)/arr.shape[1]
        return arrnorm
    
    def parse_w(w):
        wm = w[:P]
        ws = w[P:-2]
        s02 = w[-2]
        k = w[-1]
        return wm,ws,s02,k

    def minusL(w,sortind):
#         wm,ws,k = parse_w(w)
        difference_roi = compare_sorted_to_expected(w,sortind)
#         print(difference_roi.shape)
#         print(str(w) + ' -> ' + str(np.round(np.sum(difference_roi),decimals=2)))
        return 0.5*np.sum(difference_roi) # k*
    
    def minusdLdw(w,sortind): 
        # sum in first dimension: (N,1) times (N,1) times (N,P)
        return egrad(lambda w: minusL(w,sortind))(w)
    
    def fix_violations(w,bounds):
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        w[w<lb] = lb[w<lb]
        w[w>ub] = ub[w>ub]
        return w
    
    w0 = np.concatenate((wm0,ws0,s020,k0)) #,(k0,)))
    
    factr=1e7
    epsilon=1e-8
    pgtol=1e-5
    this_w = w0
    for i in range(niter):
        sortind = sort_by_11(this_w)
        wstar = sop.fmin_l_bfgs_b(lambda w: minusL(w,sortind),this_w,fprime=lambda w: minusdLdw(w,sortind),bounds=bounds,pgtol=pgtol,factr=factr,epsilon=epsilon,maxiter=1)
        assert(~np.isnan(wstar[1]))
        if np.isnan(wstar[1]):
            this_w = old_w
        else:
            this_w = wstar[0].copy() + np.random.randn(*this_w.shape)*0.01*np.exp(-i/niter)
            old_w = wstar[0].copy()
        this_w = fix_violations(this_w,bounds)
        print(str(i) + ': ' + str(wstar[1]))
    
    max_alpha = alpha_roi.max()
    this_yalpha = rate_fn_wrapper(this_w,np.linspace(-max_alpha,max_alpha,101))
    ydist = compute_y_distance(ydata[sortind,np.newaxis],this_yalpha[np.newaxis,:])
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(ydist/ydist.max(1)[:,np.newaxis],extent=[-max_alpha,max_alpha,0,10])
    plt.plot(-alpha_roi,10*np.arange(nroi)/nroi,c='m')
    plt.subplot(2,2,2)
    plt.imshow(this_yalpha[25].reshape((5,6)))
    plt.subplot(2,2,4)
    plt.imshow(this_yalpha[75].reshape((5,6)))
#         print(wstar)
    
    return wstar
