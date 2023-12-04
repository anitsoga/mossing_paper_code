#!/usr/bin/env python
# coding: utf-8

#######################################
# I/O
import os
import sys
sys.path.insert(0,"./OASIS-master/")
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

#print('USING DANS CODE')
#import pyute as ut
#import sim_utils
#reload(sim_utils)
#import opto_utils

import spatial_plots_Agos_utils as sim_utils
#######################################
# Colormaps
import matplotlib.colors as mc
import matplotlib._color_data as mcd
import colorsys
import matplotlib.colors as mcolors
import matplotlib
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
import scipy.io
from scipy import optimize
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

default_running_pct_cutoff=0.4
default_running_speed_threshold=10




############################################################################################################################################################

class Spatial_Data(object):
    """
        ncutoff   if there are less than this many cells, fill in nan
        rf_dist  if True, use rf center for radial distance; if False, use cell center
        dmax max radial dist to consider, in vis. degrees
        nbin  number of radial distance pixels

    """


    def   __init__(self,dir_2_data,dir_2_output, Locomotion='Stationary',pval=0.08, ncutoff = 5 ,rf_dist = False, sig_cond=True, sqe_cond=True, nbin = None, dmax=30,sigma_err=2 ):

        self.dir_2_data=dir_2_data
        self.dir_2_output=dir_2_output
        
                
        if not os.path.exists( self.dir_2_output):
            os.makedirs( self.dir_2_output)

        
        self.Locomotion=Locomotion
        self.pval=pval
        self.ncutoff=ncutoff
        self.rf_dist=rf_dist
        
        if Locomotion=='Stationary':
            self.irun=int(0)
        elif Locomotion=='Locomotion':
            self.irun=int(1)

        
        self.sig_cond=sig_cond
        self.sqe_cond=sqe_cond

        self.dmax=dmax
        self.bin_nbin=nbin
        self.smooth_nbin=100 # this is the bin for the smooth field method
        self.sigma_err=sigma_err

        self.ctr_cutoff=11
            
        
        
        # Hardcoded
        self.usize = np.array((5,8,13,22,36,60))
        self.ucontrast = np.array((0,6,12,25,50,100))
        self.uori = np.array((0,45,90,135,180,225,270,315))

        self.celltypes_names= ['pyr_l4','pyr_l23','pv_l23','sst_l23','vip_l23']
        self.nsize,self.ncontrast, self.nori, self.npops = len(self.usize),len(self.ucontrast),len( self.uori), len(self.celltypes_names )
        self.excluded_sessions=['180714_M9053','180321_M7955', '180519_M8959', '180531_M8961', '180618_M8956','190202_M10075', '190620_M10619']
        self.mycolor=['xkcd:puke','black','xkcd:turquoise','xkcd:burnt orange','xkcd:dark red']
        self.mylimits=['xkcd:puke','black','xkcd:turquoise','xkcd:burnt orange','xkcd:dark red']
        self.colormaps=[cm.get_cmap('YlGn_r', 128), cm.get_cmap('Greys', 128), cm.get_cmap('GnBu', 128), cm.get_cmap('Oranges', 128), cm.get_cmap('Reds', 128),]
        self.neutral_color='xkcd:deep purple'
        self.neutral_colormap='xkcd:deep purple'

                       
    def set_rate_field_method_configuration(self,rate_field_method):
        if rate_field_method=='smooth':
            self.nbin=self.smooth_nbin
            self.bins = np.linspace(0,self.dmax,self.smooth_nbin+1) # boundaries for radial distance pixels
            self.paramsout='-Locomotion_'+str(self.Locomotion)+'_rf_dist_'+str(self.rf_dist)+\
            '-sig_cond_'+str(self.sig_cond)+'-sqe_cond_'+str(self.sqe_cond)+'-dmax_'+str(self.dmax)+'-sigmaerr_'+str(self.sigma_err)
        elif rate_field_method=='bin':
            self.nbin=self.bin_nbin
            self.bins = np.linspace(0,self.dmax,self.nbin+1) # boundaries for radial distance pixels
            self.paramsout='-Locomotion_'+str(self.Locomotion)+'_rf_dist_'+str(self.rf_dist)+\
            '-sig_cond_'+str(self.sig_cond)+'-sqe_cond_'+str(self.sqe_cond)+'-ncutoff_'+str(self.ncutoff)+'-nbin_'+str(self.nbin)+'-dmax_'+str(self.dmax)
        
        
    def __del__(self):
        print('Destructor called, instance deleted.')

    def get_filename(self,this_name):
            dsname = self.dir_2_data+this_name+'_data_struct.hdf5'
            return dsname

    def run_all_data(self):
        for this_name in self.celltypes_names:
            
            data_evoked, data_sem=self.get_retinotopic_aligned_data(self,this_name)
            self.save_retinotopic_aligned_data(this_name, data_evoked, data_sem)
            self.save_get_retinotopic_aligned_data(data_evoked, data_sem)

    
    def extract_all_lists_from_raw(self,expttype='size_contrast_0',datafield='F'):
        nameout=[]
        for celltype_name in self.celltypes_names:
            self.extract_lists_from_raw_each(celltype_name,datafield,saveit=True)
            nameout.append('Lists_from_raw-'+'expttype='+expttype+'-datafield='+datafield+'-'+celltype_name)

        return nameout
        
        
        
    def extract_lists_from_raw_each(self,celltype_name,expttype='size_contrast_0',datafield='F',saveit=False):
        """ This is the first step of processing. It takes the full raw data and makes lists of
            - tuning curves
            - parameters
            - dispacements
            - retinotoy
        """

        dsname=self.get_filename(celltype_name)

        self.to_exclude = ['session_'+exptname for exptname in self.excluded_sessions]

        n_params=4
        tunings,uparams,displacements,pvals = [[None,None] for ivar in range(n_params)]
        for run_idx,run_bool in enumerate([False,True]):
            tunings[run_idx],uparams[run_idx],displacements[run_idx],pvals[run_idx] = [[] for ivar in range(n_params)]
            new_vars = sim_utils.compute_tuning(dsname,datafield=datafield,running=run_bool,expttype=expttype)
            tunings[run_idx] = tunings[run_idx] + [new_vars[0]]
            uparams[run_idx] = uparams[run_idx] + [new_vars[1]]
            displacements[run_idx] = displacements[run_idx] + [new_vars[2]]
            pvals[run_idx] = pvals[run_idx] + [new_vars[3]]

        reload(sim_utils)
        ret_info,uparams_sc,displacements,pvals = [[None,None] for ivar in range(n_params)]
        for run_idx,run_bool in enumerate([False,True]):
            ret_info[run_idx] = []
            new_vars = sim_utils.get_ret_info(dsname,expttype)
            ret_info[run_idx] = ret_info[run_idx] + [new_vars]


        Data2Save={}
        Data2Save['cell_type']=celltype_name
        Data2Save['usize'] = self.usize
        Data2Save['ucontrast'] = self.ucontrast
        Data2Save['tunings']=tunings
        Data2Save['uparams']=uparams
        Data2Save['displacements']=displacements
        Data2Save['pvals']= pvals
        Data2Save['ret_info']= ret_info

        nameout='Lists_from_raw-'+'expttype='+expttype+'-datafield='+datafield+'-'+celltype_name
        with open(self.dir_2_data+nameout+".pkl", 'wb') as handle_Model:
            pickle.dump(Data2Save, handle_Model, protocol=pickle.HIGHEST_PROTOCOL)
        return tunings,uparams,displacements,pvals,ret_info
        
        
        
        
        

    def load_extracted_lists_from_raw(self,celltype_name,expttype='size_contrast_0',datafield='F'):
        nameout='Lists_from_raw-'+'expttype='+expttype+'-datafield='+datafield+'-'+celltype_name
        with open(self.dir_2_data+nameout+'.pkl', 'rb') as handle_Model:
            this_extracted_lists=pickle.load(handle_Model)
        tunings = this_extracted_lists['tunings']
        uparams = this_extracted_lists['uparams']
        displacements = this_extracted_lists['displacements']
        pvals         = this_extracted_lists['pvals']
        ret_info      = this_extracted_lists['ret_info']

        return tunings,uparams,displacements,pvals,ret_info


    def save_all_orientation_preferences_OSI(self,celltype_name,expttype='size_contrast_0',datafield='F'):

        All_Data=self.get_size_contrast_ori_trial_nonspatial_data(celltype_name,expttype,datafield)
        this_data=All_Data['this_data']
        
        OSI=[]
        P_O=[]
        P_O_trialaveraged=[]
        OSI_trialaveraged=[]

        for nexpt in range(len(this_data)):
            dat_shap=this_data[nexpt].shape

            OSI.append(np.zeros(dat_shap[:-1])*np.nan)
            P_O.append(np.zeros(dat_shap[:-1])*np.nan)
            P_O_trialaveraged.append(np.zeros(dat_shap[:-2])*np.nan)
            OSI_trialaveraged.append(np.zeros(dat_shap[:-2])*np.nan)

            N=this_data[nexpt].shape[0]
            n_s=this_data[nexpt].shape[1]
            n_c=this_data[nexpt].shape[2]
            n_t=this_data[nexpt].shape[4]


            for i in range(N):
                for s in range(1,n_s):
                    for c in range(1,n_c):
                        for k in range(n_t):
                            this_trial=(this_data[nexpt][i,s,c,:4,k]-this_data[nexpt][i,0,0,:4,k]+this_data[nexpt][i,s,c,4:,k]-this_data[nexpt][i,0,0,4:,k])/2
                            OSI[nexpt][i,s,c,k]=(np.nanmax(this_trial)-np.nanmin(this_trial))/(np.nanmax(this_trial)+np.nanmin(this_trial))
                            P_O[nexpt][i,s,c,k]=np.argmax(this_trial)
                        this_mean_over_trial=np.nanmean((this_data[nexpt][i,s,c,:4,:]-this_data[nexpt][i,0,0,:4,:])/2+(this_data[nexpt][i,s,c,4:,:]-this_data[nexpt][i,0,0,4:,:])/2,-1)
                        OSI_trialaveraged[nexpt][i,s,c]=(np.nanmax(this_mean_over_trial)-np.nanmin(this_mean_over_trial))/(np.nanmax(this_mean_over_trial)+np.nanmin(this_mean_over_trial))

                        P_O_trialaveraged[nexpt][i,s,c]=np.argmax(this_mean_over_trial)


        Data2Save={}
        Data2Save['OSI']=OSI
        Data2Save['P_O'] = P_O
        Data2Save['OSI_trialaveraged'] = OSI_trialaveraged
        Data2Save['P_O_trialaveraged']=P_O_trialaveraged

        nameout='Orientation_preferences_OSI_'+'expttype='+expttype+'-datafield='+datafield+'-'+celltype_name
        with open(self.dir_2_output+nameout+".pkl", 'wb') as handle_Model:
            pickle.dump(Data2Save, handle_Model, protocol=pickle.HIGHEST_PROTOCOL)

        return nameout


    def get_orientation_preferences_OSI(self,celltype_name,expttype='size_contrast_0',datafield='F'):

        nameout='Orientation_preferences_OSI_'+'expttype='+expttype+'-datafield='+datafield+'-'+celltype_name
        try:
            with open(self.dir_2_output+nameout+'.pkl', 'rb') as handle_Model:
                Ori_Data=pickle.load(handle_Model)
        except:
            self.save_all_orientation_preferences_OSI(celltype_name,expttype,datafield)
            with open(self.dir_2_output+nameout+'.pkl', 'rb') as handle_Model:
                Ori_Data=pickle.load(handle_Model)
                
        OSI = Ori_Data['OSI']
        P_O = Ori_Data['P_O']
        OSI_trialaveraged =Ori_Data['OSI_trialaveraged']
        P_O_trialaveraged = Ori_Data['P_O_trialaveraged']

        return OSI, P_O, OSI_trialaveraged, P_O_trialaveraged


    def plot_tuning_curves(self,celltype_name,expttype='size_contrast_0',datafield='F',baseline_substracted=True,nexpt=5,ncell=2):

        # Load the matrix that is nexp x cell x size x contrast x orientation x trial
        this_data_dic=self.get_size_contrast_ori_trial_nonspatial_data(celltype_name,expttype='size_contrast_0',datafield='F')
        this_data=this_data_dic['this_data']

        # Load the matrixes of orientaiton tunning and OSI that is nexp x cell x size x contrast x orientation x trial
#        OSI, P_O, OSI_trialaveraged, P_O_trialaveraged= self.get_orientation_preferences_OSI(celltype_name,expttype='size_contrast_0',datafield='F')



        fig, axs = plt.subplots(self.ncontrast-1,self.nsize-1, figsize=(20, 10), dpi= 300, facecolor='w', edgecolor='k',sharex='col',sharey='row')
        fig.subplots_adjust(hspace = .2, wspace=0.2)
        plt.rcParams.update({'font.size': 13})


        ori_xaxis=np.arange(0,360,45)
        for isize in range(1,self.nsize):
            for icontrast in range(1,self.ncontrast):
                    
            
                if baseline_substracted:
                    axs[icontrast-1,isize-1].plot(ori_xaxis,this_data[nexpt][ncell,isize,icontrast,:,:]-this_data[nexpt][ncell,0,0,:,:],alpha=0.5);
                    this_mean_over_trial=np.nanmean((this_data[nexpt][ncell,isize,icontrast,:4,:]-this_data[nexpt][ncell,0,0,:4,:])/2+(this_data[nexpt][ncell,isize,icontrast,4:,:]-this_data[nexpt][ncell,0,0,4:,:])/2,-1)

                else:
                    axs[icontrast-1,isize-1].plot(ori_xaxis,this_data[nexpt][ncell,isize,icontrast,:,:],alpha=0.5);


                axs[icontrast-1,isize-1].plot(ori_xaxis,np.nanmean(this_data[nexpt][ncell,isize,icontrast,:,:],-1),'k',alpha=1);
                axs[icontrast-1,isize-1].axvline(ori_xaxis[int(np.argmax(this_mean_over_trial))],color='k',linestyle='--')
                axs[icontrast-1,isize-1].axvline(ori_xaxis[4+int(np.argmax(this_mean_over_trial))],color='k',linestyle='--')



                axs[icontrast-1,isize-1].xaxis.set_major_locator(ticker.MultipleLocator(45))
                axs[icontrast-1,isize-1].yaxis.set_major_locator(ticker.MultipleLocator(0.1))

                axs[icontrast-1,isize-1].set_ylim([0,0.15])
                axs[0,isize-1].set_title('%d$^o$ size'%self.usize[isize])
                axs[icontrast-1,isize-1].set_ylabel('C= '+str(self.ucontrast[icontrast]))
                axs[self.ncontrast-2,0].set_xlabel('Stim Orientation')




        plt.tight_layout()

        nameout='Orientation-evoked' + str(baseline_substracted) +'expttype='+expttype+'-datafield='+datafield+'-'+celltype_name +'_nexpt='+str(nexpt)+'_ncell='+str(ncell)
        fig.savefig(self.dir_2_output+nameout+'.pdf',dpi=300)



    def get_orientation_preference(self,this_data):
            # this_data: N x size x contrast x angle, this is a single session matrix
            # if bigger than that we also have trials, we average first

            if len(this_data.shape)>4:
                this_data=np.nanmean(this_data,-1)
                
            N=this_data.shape[0]
            n_s=this_data.shape[1]
            n_c=this_data.shape[2]
            n_o=this_data.shape[3]
        
            # Get orientation preference
            P_O_trialaveraged=np.zeros((N,n_s,n_c))*np.nan
            for i in range(N):
                for s in range(1,n_s):
                    for c in range(1,n_c):
                        this_mean_over_trial=this_data[i,s,c,:4]/2+this_data[i,s,c,4:]/2
                        P_O_trialaveraged[i,s,c]=np.argmax(this_mean_over_trial)
                        
            #Choose maximum contrast and smallish size
            P_O=P_O_trialaveraged[:,2,-1]
            return P_O
            

    def get_size_contrast_position_data(self,celltype_name,expttype='size_contrast_0',datafield='F', ori_cross_all='all', normalized_session=True, baseline_substracted=True, rate_field_method='smooth', running_pct_cutoff=default_running_pct_cutoff, running_speed_threshold=default_running_pct_cutoff,all_sessions_merged=False):


        if running_pct_cutoff is None and running_speed_threshold is None:
            #Get raw data
            print('Using default parameters for locomotion')
        
            try:
                tunings,uparams,displacements,pvals,ret_info=self.load_extracted_lists_from_raw(celltype_name,expttype,datafield)
            except:
                print(' The data is not yet extracted this is going to take a little while')
    #            self.extract_all_lists_from_raw(expttype=expttype)
    #            tunings,uparams,displacements,pvals,ret_info=self.load_extracted_lists_from_raw(celltype_name,expttype)

        elif running_pct_cutoff is not None and running_speed_threshold is not None:
            fill_nans_under_cutoff=True
            try:
                tunings,uparams,displacements,pvals,ret_info=self.load_extracted_lists_from_raw_each_locomotion_control(celltype_name, expttype , datafield,running_pct_cutoff, fill_nans_under_cutoff, running_speed_threshold)
            except:
                print(' The data is not yet extracted this is going to take a little while')



        #############################################################################
        self.set_rate_field_method_configuration(rate_field_method)


        nexpt = len(ret_info[self.irun][0])
        data_bin = np.nan*np.ones((nexpt,self.nbin,self.nsize,self.ncontrast,self.nori))
        data_sem = np.nan*np.ones((nexpt,self.nbin,self.nsize,self.ncontrast,self.nori))
        data_n   = np.nan*np.ones((nexpt,self.nbin,self.nsize,self.ncontrast,self.nori))
        
        all_rate_vec=[]
        all_distance=[]

        for isize in range(self.nsize):
            all_rate_vec.append([])
            all_distance.append([])
            for icontrast in range(self.ncontrast):
                all_rate_vec[isize].append([])
                all_distance[isize].append([])
                for iori in range(self.nori):
                    all_rate_vec[isize][icontrast].append([])
                    all_distance[isize][icontrast].append([])

                

        for iexpt in range(nexpt):
            #if not ret_info[self.irun][0][iexpt]['ret_map_loc'] is None and not tunings[self.irun][0][iexpt] is None:
            if celltype_name=='vip_l23':
                this_cond=ret_info[self.irun][0][iexpt]
            else:
                this_cond=ret_info[self.irun][0][iexpt]['ret_map_loc']

            if not this_cond is None and not tunings[self.irun][0][iexpt] is None:
                if not self.rf_dist:
                    distance = np.sqrt(np.sum(ret_info[self.irun][0][iexpt]['ret_map_loc']**2,1))
                else:
                    distance = np.sqrt(np.sum(ret_info[self.irun][0][iexpt]['rf_center']**2,1))

                this_data = np.nanmean(tunings[self.irun][0][iexpt][:,:,[0,-5,-4,-3,-2,-1],:,8:-8],4) # some recordings have 3% contrast, ignore this value

                ################################## Normalize?
                if normalized_session:
                    this_data = this_data/np.nanmean(this_data)

                ################################## Get orientation
                P_O=self.get_orientation_preference(this_data)

                ################################## Cell selection ###################################################################
                non_nan = ~np.isnan(np.nanmean(np.nanmean(np.nanmean(this_data,1),1),1))  #& ~np.isnan(np.nanmean(np.nanmean(np.nanmean(ori_data,1),1),1))
                sig_driven = (ret_info[self.irun][0][iexpt]['pval'] < self.pval)
                lkat = non_nan & sig_driven
                if self.sig_cond:
                    lkat=lkat &(ret_info[self.irun][0][iexpt]['sigma'] > 3.3)
                if self.sqe_cond:
                    lkat=lkat &(ret_info[self.irun][0][iexpt]['sqerror'] < 0.5)

                this_nsize = this_data.shape[1]
                if np.sum(lkat):
                    for iori in range(self.nori):
                        if ori_cross_all=='oriented':
                            this_ori=(P_O-np.mod(iori,4))==0
                            lkat_final=lkat &this_ori
                        elif ori_cross_all=='cross':
                            this_ortho=(P_O-np.mod(iori+2,4))==0
                            lkat_final=lkat &this_ortho
                        elif ori_cross_all=='all':
                            this_all=P_O>0
                            lkat_final=lkat &this_all

                        for isize in range(this_nsize):
                            for icontrast in range(self.ncontrast):
                                rate_vec=this_data[lkat_final,isize,icontrast,iori]
                                distance_vec=distance[lkat_final]
                                data_bin[iexpt,:,isize,icontrast,iori],data_sem[iexpt,:,isize,icontrast,iori], data_n[iexpt,:,isize,icontrast,iori]  = self.get_rate_field(rate_field_method,rate_vec,distance_vec, self.sigma_err)
#                                print('isize ', str(isize)+'icontrast ', str(icontrast)+ 'iori ', str(iori))
                                all_rate_vec[isize][icontrast][iori].append(rate_vec)
                                all_distance[isize][icontrast][iori].append(distance_vec)
        
        ######################### This will override the data_bin above. All iexp will have the same data, so we can use the same code in the plotting.
        if all_sessions_merged:
            for iexpt in range(nexpt):
                if celltype_name=='vip_l23':
                    this_cond=ret_info[self.irun][0][iexpt]
                else:
                    this_cond=ret_info[self.irun][0][iexpt]['ret_map_loc']
                if not this_cond is None and not tunings[self.irun][0][iexpt] is None:
                    this_nsize=tunings[self.irun][0][iexpt].shape[1]
                    for isize in range(this_nsize):
                        for icontrast in range(self.ncontrast):
                            for iori in range(self.nori):
                                data_bin[iexpt,:,isize,icontrast,iori],data_sem[iexpt,:,isize,icontrast,iori], data_n[iexpt,:,isize,icontrast,iori]  = \
                                self.get_rate_field(rate_field_method,np.concatenate(all_rate_vec[isize][icontrast][iori]),np.concatenate(all_distance[isize][icontrast][iori]), self.sigma_err)

        data_bin[data_n < self.ncutoff] = np.nan
        data_sem[data_n < self.ncutoff] = np.nan

        if baseline_substracted:

            data_evoked= data_bin -data_bin[:,:,:,0:1,:]
        else:
            data_evoked= data_bin

        return data_evoked, data_sem




    def get_rate_field_method_smooth(self,rate_vec,distance_vec, sigma_err_vec):
        x=np.linspace(0,self.dmax,self.smooth_nbin)
        try:
            len(sigma_err_vec)
        except:
            sigma_err=sigma_err_vec
            sigma_err_vec=np.ones_like(distance_vec)*sigma_err

        nominator=np.zeros_like(x)
        denominator=np.zeros_like(x)
        for k in range(len(rate_vec)):
            nominator+=rate_vec[k]*np.exp(-(x-distance_vec[k])**2/(2*sigma_err_vec[k]**2))
            denominator+=np.exp(-(x-distance_vec[k])**2/(2*sigma_err_vec[k]**2))

        data_mean= nominator/denominator
        data_sem= np.ones_like(data_mean)*np.nan
        data_n= np.ones_like(data_mean)*np.inf

        return data_mean , data_sem, data_n
        
        
        
    def get_rate_field_method_bin(self,rate_vec,distance_vec):
        data_mean = sst.binned_statistic(distance_vec,rate_vec,statistic=np.nanmean,bins=self.bins).statistic
        data_sem = sst.binned_statistic(distance_vec,rate_vec,statistic=nansem,bins=self.bins).statistic
        data_n = sst.binned_statistic(distance_vec,rate_vec,statistic=n_non_nan,bins=self.bins).statistic
        return data_mean , data_sem, data_n
        
        
        
    def get_rate_field(self,rate_field_method,rate_vec,distance_vec, sigma_err_vec):
        if rate_field_method=='smooth':
            data_mean , data_sem, data_n=self.get_rate_field_method_smooth(rate_vec,distance_vec, sigma_err_vec)
        elif rate_field_method=='bin':
            data_mean , data_sem, data_n=self.get_rate_field_method_bin(rate_vec,distance_vec)
        return data_mean , data_sem, data_n



            
    def get_size_contrast_ori_trial_nonspatial_data(self,celltype_name,expttype='size_contrast_0',datafield='F'):
        try:
            tunings,uparams,displacements,pvals,ret_info=self.load_extracted_lists_from_raw(celltype_name,expttype,datafield)
        except:
            print(' The data is not yet extracted this is going to take a little while')
#            self.extract_all_lists_from_raw(expttype=expttype)
#            tunings,uparams,displacements,pvals,ret_info=self.load_extracted_lists_from_raw(celltype_name,expttype)

        this_data=[]
        nexpt = len(ret_info[self.irun][0])

        for iexpt in range(nexpt):
            #if not ret_info[self.irun][0][iexpt]['ret_map_loc'] is None and not tunings[self.irun][0][iexpt] is None:
            if celltype_name=='vip_l23':
                this_cond=ret_info[self.irun][0][iexpt]
            else:
                this_cond=ret_info[self.irun][0][iexpt]['ret_map_loc']

            if not this_cond is None and not tunings[self.irun][0][iexpt] is None:
                this_data.append(tunings[self.irun][0][iexpt][:,:,[0,-5,-4,-3,-2,-1],:,8:-8]) # some recordings have 3% contrast, ignore this value
                        
        Data2Save={}
        Data2Save['this_data']=this_data
        Data2Save['celltype_name'] = celltype_name
        Data2Save['expttype'] = expttype
        Data2Sae['datafield']=datafield

        return Data2Save
    ############################################################################################################################################################
    ############################################################################################################################################################
    ############################################################################################################################################################

    def save_retinotopic_aligned_data(self,this_name,data_bin, data_evoked, data_sem):
                    
        Data2Save={}
        Data2Save['cell_type']=this_name
        Data2Save['usize'] = self.usize
        Data2Save['ucontrast'] = self.ucontrast
        Data2Save['data_bin']=data_bin
        Data2Save['data_sem']=data_sem
        Data2Save['data_evoked']=data_evoked
        Data2Save['rf_dist']=self.rf_dist
        Data2Save['ncutoff']= self.ncutoff
        Data2Save['rf_dist']= self.rf_dist
        Data2Save['dmax']= self.dmax
        Data2Save['nbin']= self.nbin
        Data2Save['pval']= self.pval

        nameout='Spatial_Data_'+this_name+self.paramsout
        with open(self.dir_2_output+nameout+".pkl", 'wb') as handle_Model:
            pickle.dump(Data2Save, handle_Model, protocol=pickle.HIGHEST_PROTOCOL)

        return self.dir_2_output+nameout+".pkl"


    def save_retinotopic_aligned_ori_data(self,this_name,data_bin, data_evoked, data_sem):
                    
        Data2Save={}
        Data2Save['cell_type']=this_name
        Data2Save['usize'] = self.usize
        Data2Save['uori'] = self.uori
        Data2Save['ucontrast'] = self.ucontrast
        Data2Save['data_bin']=data_bin
        Data2Save['data_sem']=data_sem
        Data2Save['data_evoked']=data_evoked
        Data2Save['rf_dist']=self.rf_dist
        Data2Save['ncutoff']= self.ncutoff
        Data2Save['rf_dist']= self.rf_dist
        Data2Save['dmax']= self.dmax
        Data2Save['nbin']= self.nbin
        Data2Save['pval']= self.pval

        nameout='Spatial_Data_Orientation'+this_name+self.paramsout
        with open(self.dir_2_output+nameout+".pkl", 'wb') as handle_Model:
            pickle.dump(Data2Save, handle_Model, protocol=pickle.HIGHEST_PROTOCOL)

        return self.dir_2_output+nameout+".pkl"




    def load_and_plot(self,this_name):
        
        cell_id=self.celltypes_names.index(this_name)
        nameout='Spatial_Data_'+this_name+self.paramsout
        print(self.dir_2_output+nameout)

        with open(self.dir_2_output+nameout+'.pkl', 'rb') as handle_Model:
            this_sim=pickle.load(handle_Model)
        data_sem   =this_sim['data_sem']
        data_evoked=this_sim['data_evoked']
        self.plot_retinotopic_aligned_data(cell_id, data_evoked, data_sem)


    def load_and_plot_all(self):
        for this_name in self.celltypes_names:
            try:
                self.load_and_plot(this_name)
            except:
                print('failed to oad and plot' + str(this_name))
    ############################################################################################################################################################
    ############################################################################################################################################################
    ############################################################################################################################################################


    def plot_retinotopic_aligned_ori_aligned_data(self,data_evoked, data_sem,celltype_name,expttype='size_contrast_0',datafield='F', ori_cross_all='all', normalized_session=True, baseline_substracted=True, rate_field_method='smooth'):

        cell_id=self.celltypes_names.index(celltype_name)

        plt.rcParams.update({'font.size': 15})

        fig, axs = plt.subplots(1,self.nsize, figsize=(20, 3), dpi= 300, facecolor='w', edgecolor='k',sharex='col',sharey='row')
        fig.subplots_adjust(hspace = .2, wspace=0.2)

        self.set_rate_field_method_configuration(rate_field_method)
        
        x = 0.5*(self.bins[1:]+self.bins[:-1])
        retinotopic_location= np.concatenate((-x[::-1],x))

        s_bar=np.nanmean(data_evoked)/2
        
        data_model_mean=np.zeros((len(retinotopic_location),len(self.usize),len(self.ucontrast)))
        data_model_sem=np.zeros((len(retinotopic_location),len(self.usize),len(self.ucontrast)))
        
        for isize in range(self.nsize):
            for icontrast in range(0,self.ncontrast):
            
            
                data_aux=np.nanmean(data_evoked[:,:,isize,icontrast,:],-1)
                data = np.nanmean(data_aux,0) # average over trials
                data = np.concatenate((data[::-1],data))
                data_model_mean[:,isize,icontrast]=data


                if rate_field_method=='smooth':
                    this_sem = np.nanstd(np.nanmean(data_evoked[:,:,isize,icontrast,:],-1),0)/np.sqrt(data_evoked.shape[0])
                    this_sem = np.concatenate((this_sem[::-1],this_sem))
                    data_model_sem[:,isize,icontrast]=this_sem
                    axs[isize].fill_between(retinotopic_location,data_model_mean[:,isize,icontrast]-data_model_sem[:,isize,icontrast], data_model_mean[:,isize,icontrast]+data_model_sem[:,isize,icontrast], color = lighten_color(self.mycolor[cell_id],(icontrast+1)/self.ncontrast), alpha=0.2)
                    axs[isize].plot(retinotopic_location,data_model_mean[:,isize,icontrast],label='%d%%'%self.ucontrast[icontrast], color=lighten_color(self.mycolor[cell_id],(icontrast+1)/self.ncontrast))
                elif rate_field_method=='bin':
                    this_sem = np.sqrt(np.nansum(np.nanmean(data_sem[:,:,isize,icontrast,:],-1)**2,0))/np.sum(~np.isnan(np.nanmean(data_sem[:,:,isize,icontrast,:],-1)),0)
                    this_sem = np.concatenate((this_sem[::-1],this_sem))
                    data_model_sem[:,isize,icontrast]=this_sem
                    axs[isize].errorbar(retinotopic_location,data_model_mean[:,isize,icontrast],data_model_sem[:,isize,icontrast],label='%d%%'%self.ucontrast[icontrast], color=lighten_color(self.mycolor[cell_id],(icontrast+1)/self.ncontrast))



                axs[isize].fill_between((-self.usize[isize]/2,self.usize[isize]/2),(-s_bar,-s_bar),(-s_bar*0.1,-s_bar*0.1),facecolor='k',alpha=0.5)
                axs[isize].set_title('%d$^o$ size'%self.usize[isize])
                axs[isize].axhline(0,c='k',linestyle='dashed',alpha=0.5)
                axs[isize].set_xlabel('retinotopic location')
                axs[isize].set_xlim([-40,40])
                axs[0].set_ylabel(' rate/mean')
                axs[0].set_ylabel(' rate/mean')

        plt.tight_layout()

        nameout='Spatial_Data-expttype='+expttype+'-datafield='+datafield+ '-'+ori_cross_all+ '-'+ 'normalized='+ str(normalized_session) + '-evoked' + str(baseline_substracted) + self.celltypes_names[cell_id]+'-rate_field_method='+rate_field_method+self.paramsout
        fig.savefig(self.dir_2_output+nameout+'.pdf',dpi=300)



    def get_and_plot_orientation_averaged_size_contrast_tuning_gaussian_fits(self,celltype_name,expttype='size_contrast_0',datafield='F', ori_cross_all='all', normalized_session=True, baseline_substracted=True, rate_field_method='smooth',running_pct_cutoff=None, running_speed_threshold=None,all_sessions_merged=False):

        data_evoked, data_sem=self.get_size_contrast_position_data(celltype_name,expttype, datafield,ori_cross_all, normalized_session, baseline_substracted, rate_field_method,running_pct_cutoff,running_speed_threshold,all_sessions_merged)

        if datafield=='F':
            this_lb_ylim=-0.1
            this_up_ylim=1
        elif datafield=='deconv':
            this_lb_ylim=-0.5
            this_up_ylim=6

        
        # Cell Id for color
        cell_id=self.celltypes_names.index(celltype_name)


        def gaussian(x, amplitude, stddev):
            return amplitude * np.exp(-(x**2/ 2/ stddev**2))

        eps=10**-6

        self.set_rate_field_method_configuration(rate_field_method)
        
        x = 0.5*(self.bins[1:]+self.bins[:-1])
        retinotopic_location= np.concatenate((-x[::-1],x))

        # Bins 'continuum' for the gaussian
        retinotopic_location_cont=np.linspace(np.min(retinotopic_location),np.max(retinotopic_location),100)

        # Initialize
        data_model_mean=np.zeros((len(retinotopic_location),len(self.usize),len(self.ucontrast)))
        data_model_sem=np.zeros((len(retinotopic_location),len(self.usize),len(self.ucontrast)))
        gaussian_parameters=np.zeros((2,self.nsize,self.ncontrast))*np.nan

        
        #############################################################################################
        fig, axs= plt.subplots(1,6, figsize=(25,4), dpi= 100, facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = 2, wspace=0.3)
        
        for isize in range(self.nsize):
            seed_contrast=3
            for icontrast in [seed_contrast,0,1,2,4,5]:#range(self.ncontrast-1,0,-1):

                data_aux=np.nanmean(data_evoked[:,:,isize,icontrast,:],-1)
                data = np.nanmean(data_aux,0) # average over trials
                data = np.concatenate((data[::-1],data))
                data_model_mean[:,isize,icontrast]=data


                if rate_field_method=='smooth':
                    this_sem = np.nanstd(np.nanmean(data_evoked[:,:,isize,icontrast,:],-1),0)/np.sqrt(data_evoked.shape[0])
                    this_sem = np.concatenate((this_sem[::-1],this_sem))
                    data_model_sem[:,isize,icontrast]=this_sem
                    axs[isize].fill_between(retinotopic_location,data_model_mean[:,isize,icontrast]-data_model_sem[:,isize,icontrast], data_model_mean[:,isize,icontrast]+data_model_sem[:,isize,icontrast], color=lighten_color(self.mycolor[cell_id],(icontrast+1)/self.ncontrast),alpha=0.2)
                    axs[isize].plot(retinotopic_location,data_model_mean[:,isize,icontrast],label='%d%%'%self.ucontrast[icontrast], color=lighten_color(self.mycolor[cell_id],(icontrast+1)/self.ncontrast))
                elif rate_field_method=='bin':
                    this_sem = np.sqrt(np.nansum(np.nanmean(data_sem[:,:,isize,icontrast,:],-1)**2,0))/np.sum(~np.isnan(np.nanmean(data_sem[:,:,isize,icontrast,:],-1)),0)
                    this_sem = np.concatenate((this_sem[::-1],this_sem))
                    data_model_sem[:,isize,icontrast]=this_sem
                    axs[isize].errorbar(retinotopic_location,data_model_mean[:,isize,icontrast],data_model_sem[:,isize,icontrast],label='%d%%'%self.ucontrast[icontrast], color=lighten_color(self.mycolor[cell_id],(icontrast+1)/self.ncontrast))


                
                non_nan_data=    ~np.isnan(data_model_mean[:,isize,icontrast])
                try:
                    if icontrast==seed_contrast:#(self.ncontrast-1):
                        popt, _ = optimize.curve_fit(gaussian, retinotopic_location[non_nan_data], data_model_mean[non_nan_data,isize,icontrast], bounds=([0, 0], [100,100+eps]))
                    else:
                        popt, _ = optimize.curve_fit(gaussian, retinotopic_location[non_nan_data], data_model_mean[non_nan_data,isize,icontrast], bounds=([0, gaussian_parameters[1,isize,seed_contrast]], [100, gaussian_parameters[1,isize,seed_contrast]+0.1]))

                except:
                    popt=np.array([np.nan]*2)
                
                if popt[1]>40:
                    popt[1]=np.nan
                    
                gaussian_parameters[:,isize,icontrast]=popt
                axs[isize].plot(retinotopic_location_cont,gaussian(retinotopic_location_cont,*popt),'--k')

                axs[isize].set_xlabel('retinotopic location')

                axs[isize].set_ylim((this_lb_ylim,this_up_ylim))
                axs[isize].fill_between((-self.usize[isize]/2,self.usize[isize]/2),(-0.35,-0.35),(-0.15,-0.15),facecolor='k',alpha=0.5)
                axs[isize].set_title('%d$^o$ size'%self.usize[isize],color=lighten_color(self.neutral_color,(isize+1)/self.nsize) )
                axs[isize].axhline(0,c='k',linestyle='dashed',alpha=0.5)
                
        axs[0].set_ylabel('evoked event rate/mean')

        plt.legend(ncol=2)
        plt.tight_layout()


        
        nameout='Gaussian_Curves-expttype='+expttype+'-datafield='+datafield+ '-'+ori_cross_all+ '-'+ 'normalized='+ str(normalized_session) + '-evoked' + str(baseline_substracted) + self.celltypes_names[cell_id]+'-rate_field_method='+rate_field_method+'-merged_sees='+str(all_sessions_merged)+self.paramsout
        fig.savefig(self.dir_2_output+nameout+'.pdf',dpi=300)
        
        #############################################################################################

        fig, axs= plt.subplots(2,3, figsize=(20,10), dpi= 100, facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = 0.4, wspace=0.3)

        Im1=axs[0,0].imshow(gaussian_parameters[0,:,:].transpose(),origin = 'lower',extent = [self.usize[0], self.usize[-1], self.ucontrast[0], self.ucontrast[-1]],interpolation='nearest', aspect = 0.5,cmap='viridis')#=self.colormaps[cell_id])
        axs[0,0].set_ylabel('Contrast')
        axs[0,0].set_xlabel('Size')
        axs[0,0].set_title('Gaussian Amplitude')
        plt.colorbar(Im1,ax=axs[0,0])

        Im2=axs[1,0].imshow(gaussian_parameters[1,:,:].transpose(),origin = 'lower',extent = [self.usize[0], self.usize[-1], self.ucontrast[0], self.ucontrast[-1]],interpolation='nearest', aspect = 0.5,cmap='viridis')#=self.colormaps[cell_id])
        axs[1,0].set_ylabel('Contrast')
        axs[1,0].set_xlabel('Size')
        axs[1,0].set_title('Gaussian Std')
        plt.colorbar(Im2,ax=axs[1,0])

        
        

        for icontrast in range(1,self.ncontrast):

            axs[0,1].plot(self.usize,gaussian_parameters[0,:,icontrast], color=lighten_color(self.mycolor[cell_id],(icontrast+1)/self.ncontrast))
            axs[0,1].set_ylabel('Amplitude')
            axs[0,1].set_xlabel('Size')
            axs[0,1].set_title('Gaussian Amplitude')

            axs[1,1].plot(self.usize,gaussian_parameters[1,:,icontrast], color=lighten_color(self.mycolor[cell_id],(icontrast+1)/self.ncontrast))
            axs[1,1].set_ylabel('Amplitude')
            axs[1,1].set_xlabel('Size')
            axs[1,1].set_title('Gaussian Std')

        for isize in range(self.nsize):
            axs[0,2].plot(self.ucontrast,gaussian_parameters[0,isize,:], color=lighten_color(self.neutral_color,(isize+1)/self.nsize))
            axs[0,2].set_ylabel('Amplitude')
            axs[0,2].set_xlabel('Contrast')
            axs[0,2].set_title('Gaussian Amplitude')

            axs[1,2].plot(self.ucontrast,gaussian_parameters[1,isize,:], color=lighten_color(self.neutral_color,(isize+1)/self.nsize))
            axs[1,2].set_ylabel('Amplitude')
            axs[1,2].set_xlabel('Contrast')
            axs[1,2].set_title('Gaussian Std')


        nameout='Gaussian_Fits-expttype='+expttype+'-datafield='+datafield+ '-'+ori_cross_all+ '-'+ 'normalized='+ str(normalized_session) + '-evoked' + str(baseline_substracted) + self.celltypes_names[cell_id]+'-rate_field_method='+rate_field_method+'-merged_sees='+str(all_sessions_merged)+self.paramsout
        fig.savefig(self.dir_2_output+nameout+'.pdf',dpi=300)

        Data2Save={}
        Data2Save["data_mean__distance_size_contrast"]=data_model_mean
        Data2Save["data_sem__distance_size_contrast"]=data_model_sem
        Data2Save["retinotopic_distance"]=retinotopic_location
        Data2Save["sizes"]=self.usize
        Data2Save["contrasts"]=self.ucontrast
        Data2Save["gaussian_mean__size_contrast"]=gaussian_parameters[0,:,:]
        Data2Save["gaussian_std__size_contrast"]=gaussian_parameters[1,:,:]

        with open(self.dir_2_output+nameout+".pkl", 'wb') as handle_Model:
            pickle.dump(Data2Save, handle_Model, protocol=pickle.HIGHEST_PROTOCOL)
        scipy.io.savemat(self.dir_2_output+nameout+'.mat',Data2Save)


##################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
        
        
    def extract_lists_from_raw_each_locomotion_control(self,celltype_name, expttype='size_contrast_0' ,datafield='F',running_pct_cutoff=default_running_pct_cutoff, fill_nans_under_cutoff=True, running_speed_threshold=default_running_speed_threshold, saveit=False):
        
        
        """ This is the first step of processing. It takes the full raw data and makes lists of
            - tuning curves
            - parameters
            - dispacements
            - retinotoy
            
            OBS: Here if running_speed_threshold=0 and running_pct_cutoff=0 we get all locomotion and non locomotion together under the locomotion bracket.
        """


        dsname=self.get_filename(celltype_name)

        self.to_exclude = ['session_'+exptname for exptname in self.excluded_sessions]

        n_params=4
        tunings,uparams,displacements,pvals = [[None,None] for ivar in range(n_params)]
        for run_idx,run_bool in enumerate([False,True]):
            tunings[run_idx],uparams[run_idx],displacements[run_idx],pvals[run_idx] = [[] for ivar in range(n_params)]
            new_vars = sim_utils.compute_tuning(dsname,datafield=datafield,running=run_bool,expttype=expttype,running_pct_cutoff=running_pct_cutoff,fill_nans_under_cutoff=fill_nans_under_cutoff,running_speed_threshold=running_speed_threshold)
            tunings[run_idx] = tunings[run_idx] + [new_vars[0]]
            uparams[run_idx] = uparams[run_idx] + [new_vars[1]]
            displacements[run_idx] = displacements[run_idx] + [new_vars[2]]
            pvals[run_idx] = pvals[run_idx] + [new_vars[3]]

        reload(sim_utils)
        ret_info,uparams_sc,displacements,pvals = [[None,None] for ivar in range(n_params)]
        for run_idx,run_bool in enumerate([False,True]):
            ret_info[run_idx] = []
            new_vars = sim_utils.get_ret_info(dsname,expttype)
            ret_info[run_idx] = ret_info[run_idx] + [new_vars]


        Data2Save={}
        Data2Save['cell_type']=celltype_name
        Data2Save['usize'] = self.usize
        Data2Save['ucontrast'] = self.ucontrast
        Data2Save['tunings']=tunings
        Data2Save['uparams']=uparams
        Data2Save['displacements']=displacements
        Data2Save['pvals']= pvals
        Data2Save['ret_info']= ret_info

        nameout='Lists_from_raw-'+'-expttype='+expttype+'-datafield='+datafield+'-running_pct_cutoff='+str(np.around(running_pct_cutoff,1))+'-running_speed_threshold='+str(np.around(running_speed_threshold,1))+'-'+celltype_name
        with open(self.dir_2_data+nameout+".pkl", 'wb') as handle_Model:
            pickle.dump(Data2Save, handle_Model, protocol=pickle.HIGHEST_PROTOCOL)
        return tunings,uparams,displacements,pvals,ret_info
        
        
        
    def load_extracted_lists_from_raw_each_locomotion_control(self,celltype_name, expttype='size_contrast_0' ,datafield='F',running_pct_cutoff=default_running_pct_cutoff, fill_nans_under_cutoff=False, running_speed_threshold=default_running_speed_threshold):

        nameout='Lists_from_raw-'+'-expttype='+expttype+'-datafield='+datafield+'-running_pct_cutoff='+str(running_pct_cutoff)+'-running_speed_threshold='+str(running_speed_threshold)+'-'+celltype_name
        
        print(nameout)
        with open(self.dir_2_data+nameout+'.pkl', 'rb') as handle_Model:
            this_extracted_lists=pickle.load(handle_Model)
        
        if datafield=='decon':
            tunings = this_extracted_lists['tunings_decon']
            uparams = this_extracted_lists['uparams_decon']
        else:
            tunings = this_extracted_lists['tunings']
            uparams = this_extracted_lists['uparams']

        displacements = this_extracted_lists['displacements']
        pvals         = this_extracted_lists['pvals']
        ret_info      = this_extracted_lists['ret_info']

        return tunings,uparams,displacements,pvals,ret_info
        
        


    def get_size_contrast_centered_ori_averaged_locomotion_data(self,celltype_name,expttype='size_contrast_0',datafield='F',running_pct_cutoff=default_running_pct_cutoff, running_speed_threshold=default_running_speed_threshold, which_normalization='session_mean'):

        """
        tunings[locomotion][nothing][nexpt](cells, size, contrast, orientation, trials)
        """

        try:
            tunings,uparams,displacements,pvals,ret_info=self.load_extracted_lists_from_raw_each_locomotion_control(celltype_name, expttype=expttype ,datafield=datafield, running_pct_cutoff=running_pct_cutoff, fill_nans_under_cutoff=True, running_speed_threshold=running_speed_threshold)
        except:
            nameout='Lists_from_raw-'+'-expttype='+expttype+'-datafield='+datafield+'-running_pct_cutoff='+str(running_pct_cutoff)+'-running_speed_threshold='+str(running_speed_threshold)+'-'+celltype_name
            print('Cant find ' + nameout)
            print(' The data is not yet extracted this is going to take a little while')


        irun=0 # ret_info[0]=ret_info[1] because its all nan padded,  just need to choose one
        nexpt = len(ret_info[irun][0])

        stat_loco_data_responsive=[]
        stat_loco_data_all=[]

        for iexpt in range(1,nexpt):


            if celltype_name=='vip_l23':
                this_cond=ret_info[irun][0][iexpt]
            else:
                this_cond=ret_info[irun][0][iexpt]['ret_map_loc']


            if not this_cond is None and not tunings[irun][0][iexpt] is None:

                ################################## Define how we compute the distance
                if not self.rf_dist:
                    distance = np.sqrt(np.sum(ret_info[irun][0][iexpt]['ret_map_loc']**2,1))
                else:
                    distance = np.sqrt(np.sum(ret_info[irun][0][iexpt]['rf_center']**2,1))

                ################################## Trial and Orientation average
                this_data_stationary = np.nanmean( np.nanmean(tunings[0][0][iexpt][:,:,[0,-5,-4,-3,-2,-1],:,8:-8],-1),-1) # some recordings have 3% contrast, ignore this value
                this_data_locomotion = np.nanmean( np.nanmean(tunings[1][0][iexpt][:,:,[0,-5,-4,-3,-2,-1],:,8:-8],-1),-1) # some recordings have 3% contrast, ignore this value
                this_data_all=np.stack((this_data_stationary,this_data_locomotion),axis=0)


                ################################## Cell selection ###################################################################
                non_nan = ~np.isnan(np.nanmean(np.nanmean(this_data_stationary,-1),-1))
                sig_driven = (ret_info[irun][0][iexpt]['pval'] < self.pval)


                centered = (distance < self.ctr_cutoff)
                lkat_responsive = non_nan & sig_driven & centered
                lkat_all=non_nan & centered


                this_normalized_data=self.normalize_loco_data(this_data_all,which_normalization)




                if np.sum(lkat_responsive):
                    stat_loco_data_responsive.append(this_normalized_data[:,lkat_responsive,:,:])
                    stat_loco_data_all.append(this_normalized_data[:,lkat_all,:,:])


        return stat_loco_data_responsive, stat_loco_data_all



    def get_size_contrast_centered_ori_averaged_locomotion_data_all_celltypes(self,expttype='size_contrast_0',datafield='F',running_pct_cutoff=default_running_pct_cutoff, running_speed_threshold=default_running_speed_threshold, which_normalization='session_mean'):
        stat_loco_data_responsive_all_celltypes=[]
        stat_loco_data_all_all_celltypes=[]

        for celltype_name in self.celltypes_names[1:]:
            stat_loco_data_responsive, stat_loco_data_all = self.get_size_contrast_centered_ori_averaged_locomotion_data(celltype_name,expttype,datafield,running_pct_cutoff, running_speed_threshold, which_normalization)
            
            stat_loco_data_responsive_all_celltypes.append(stat_loco_data_responsive)
            stat_loco_data_all_all_celltypes.append(stat_loco_data_all)

        return stat_loco_data_responsive_all_celltypes, stat_loco_data_all_all_celltypes




    def normalize_loco_data(self, this_data_all,which_normalization):
        
        if which_normalization=='individual_sum':
            den=np.apply_over_axes(np.nansum, this_data_all, [0,2,3]).flatten()
            out_data=this_data_all/den[np.newaxis,:,np.newaxis,np.newaxis]

        elif which_normalization=='individual_mean':
            den=np.apply_over_axes(np.nanmean, this_data_all, [0,2,3]).flatten()
            out_data=this_data_all/den[np.newaxis,:,np.newaxis,np.newaxis]

        elif which_normalization=='session_mean':
            out_data=this_data_all/np.nanmean(this_data_all)

        return out_data
        


    def plot_save_differences_loco_stat(self,expttype='size_contrast_0',datafield='F',running_pct_cutoff=default_running_pct_cutoff, running_speed_threshold=default_running_speed_threshold, which_normalization='individual_sum', cell_selec='all'):


        stat_loco_data_responsive, stat_loco_data_all=  self.get_size_contrast_centered_ori_averaged_locomotion_data_all_celltypes(expttype,datafield,running_pct_cutoff, running_speed_threshold, which_normalization)


        if cell_selec=='all':
            stat_loco_data=stat_loco_data_all
        elif cell_selec=='responsive':
            stat_loco_data=stat_loco_data_responsive
        else:
            print('your options are all cells or responsive')

        for this_size in range(self.nsize):
            ######################

            matplotlib.rc('xtick', labelsize=8)
            matplotlib.rc('ytick', labelsize=8)

            fig, axs = plt.subplots(self.npops-1, self.ncontrast, figsize=(12, 6),dpi= 250, facecolor='w', edgecolor='k',sharex=True,sharey='row')
            fig.subplots_adjust(hspace = .1, wspace=.3)
            mean_difference_loco=np.zeros((self.npops-1, self.ncontrast))


            for this_celltype in range(self.npops-1):
                for this_contrast in range(self.ncontrast):

                    nexpt=len(stat_loco_data[this_celltype])

                    this_stat=np.array([])
                    this_loco=np.array([])


                    for iexpt in range(nexpt):
                        try:
                            this_stat=np.concatenate((this_stat, stat_loco_data[this_celltype][iexpt][0,:,this_size,this_contrast]))
                            this_loco=np.concatenate((this_loco, stat_loco_data[this_celltype][iexpt][1,:,this_size,this_contrast]))
                        except:
                            pass

                    diffs=this_loco-this_stat;
                    diffs=diffs[~np.isnan(diffs)]


                    #bins=np.linspace(-max_min,max_min,nbins_data[k]);
                    bins=np.linspace(np.percentile(diffs,1),np.percentile(diffs,99),int(len(diffs)/10));

                    mean_difference_loco[this_celltype,this_contrast]=np.nanmean(diffs)
                    axs[this_celltype,this_contrast].hist(diffs,density=True,bins=bins,color=self.mycolor[this_celltype+1],edgecolor='None',alpha = 0.8)

                    axs[this_celltype,this_contrast].axvline(0,c='k',linestyle='dashed',alpha = 0.8,linewidth=0.7)
                    axs[this_celltype,this_contrast].axvline(np.nanmean(diffs),c='k',alpha = 0.8,linewidth=0.7)

                    axs[-1,this_contrast].set_xlabel("Response",fontsize=8)
                    axs[this_celltype,this_contrast].set_ylabel("Loco-Stat " + self.celltypes_names[this_celltype+1],fontsize=8)


                    axs[0,this_contrast].set_title("Contrast " + str(self.ucontrast[this_contrast]),fontsize=8)

            fig.tight_layout()
            
            nameout='Distribution_of_differences-locomotion-minus_stationary-size_'+str(self.usize[this_size])+'-expttype='+expttype+'-datafield='+datafield +'-running_pct_cutoff='+ str(running_pct_cutoff)+ '-speed_thr='+ str(running_speed_threshold) + '-normalization=' + which_normalization+ '-cell_selec='+ cell_selec
            fig.savefig(self.dir_2_output+nameout+'.pdf',dpi=300)








    def plot_Locomotion_Stationary_Differences_Vs_Stationary_and_covs(self, expttype='size_contrast_0',datafield='F',running_pct_cutoff=default_running_pct_cutoff, running_speed_threshold=default_running_speed_threshold, which_normalization='individual_sum', cell_selec='all'):


        stat_loco_data_responsive, stat_loco_data_all=  self.get_size_contrast_centered_ori_averaged_locomotion_data_all_celltypes(expttype,datafield,running_pct_cutoff, running_speed_threshold, which_normalization)

        if cell_selec=='all':
            stat_loco_data=stat_loco_data_all
        elif cell_selec=='responsive':
            stat_loco_data=stat_loco_data_responsive
        else:
            print('your options are all cells or responsive')

        mean_difference_loco=np.zeros((self.npops-1, self.ncontrast,self.nsize))
        std_difference_loco=np.zeros((self.npops-1, self.ncontrast,self.nsize))
        covariance_difference_loco=np.zeros((self.npops-1, self.ncontrast,self.nsize))



        for this_size in range(self.nsize):



            ######################


            matplotlib.rc('xtick', labelsize=8)
            matplotlib.rc('ytick', labelsize=8)

            fig, axs = plt.subplots(self.npops-1, self.ncontrast, figsize=(12, 6),dpi= 250, facecolor='w', edgecolor='k',sharex=True,sharey='row')
            fig.subplots_adjust(hspace = .1, wspace=.3)

            for this_celltype in range(self.npops-1):
                for this_contrast in range(self.ncontrast):

                    nexpt=len(stat_loco_data[this_celltype])

                    this_stat=np.array([])
                    this_loco=np.array([])


                    for iexpt in range(nexpt):
                        try:
                            this_stat=np.concatenate((this_stat, stat_loco_data[this_celltype][iexpt][0,:,this_size,this_contrast]))
                            this_loco=np.concatenate((this_loco, stat_loco_data[this_celltype][iexpt][1,:,this_size,this_contrast]))
                        except:
                            pass

                    diffs=this_loco-this_stat;
                    this_loco=this_loco[~np.isnan(diffs)]
                    this_stat=this_stat[~np.isnan(diffs)]
                    diffs=diffs[~np.isnan(diffs)]



                    #bins=np.linspace(-max_min,max_min,nbins_data[k]);
                    bins=np.linspace(np.percentile(diffs,1),np.percentile(diffs,99),int(len(diffs)/10));

                    mean_difference_loco[this_celltype,this_contrast,this_size]=np.nanmean(diffs)
                    std_difference_loco[this_celltype,this_contrast,this_size]=np.nanstd(diffs)
                    covariance_difference_loco[this_celltype,this_contrast,this_size]=np.around(np.cov(this_stat,diffs)[0,1]/np.nanstd(this_stat)**2,6)


                    axs[this_celltype,this_contrast].scatter(this_stat,diffs,s=1,color=self.mycolor[this_celltype+1])

                    axs[-1,this_contrast].set_xlabel("Stat",fontsize=8)
                    axs[this_celltype,this_contrast].set_ylabel("Loco-Stat " + self.celltypes_names[this_celltype+1],fontsize=8)

                    if this_celltype==0:
                        axs[0,this_contrast].set_title("Contrast " + str(self.ucontrast[this_contrast]) +'\n Cov/Var='+str(covariance_difference_loco[this_celltype,this_contrast,this_size]),fontsize=8)
                    else:
                        axs[this_celltype,this_contrast].set_title('Cov/Var='+str(covariance_difference_loco[this_celltype,this_contrast,this_size]),fontsize=8)


            fig.tight_layout()

            nameout='Differences_and_Covs_Due_to_Locomotion-size_'+str(self.usize[this_size])+'-expttype='+expttype+ '-datafield='+datafield+'-running_pct_cutoff='+ str(running_pct_cutoff)+ '-speed_thr='+ str(running_speed_threshold) + '-normalization=' + which_normalization+ '-cell_selec='+ cell_selec
            fig.savefig(self.dir_2_output+nameout+'.pdf',dpi=300)





        fig, axs = plt.subplots(self.nsize, 3, figsize=(12, 12),dpi= 250, facecolor='w', edgecolor='k',sharex=True)
        fig.subplots_adjust(hspace = .1, wspace=.3)

        for this_size in range(self.nsize):


            for this_celltype in range(self.npops-1):
                axs[this_size,0].plot(self.ucontrast,mean_difference_loco[this_celltype,:,this_size],color=self.mycolor[this_celltype+1],linewidth=1)
                axs[this_size,1].plot(self.ucontrast,std_difference_loco[this_celltype,:,this_size],color=self.mycolor[this_celltype+1],linewidth=1)
                axs[this_size,2].plot(self.ucontrast,covariance_difference_loco[this_celltype,:,this_size],color=self.mycolor[this_celltype+1],linewidth=1)



            axs[-1,0].set_xlabel("Contrast ",fontsize=8)
            axs[-1,1].set_xlabel("Contrast ",fontsize=8)
            axs[-1,2].set_xlabel("Contrast ",fontsize=8)



            axs[this_size,0].set_ylabel(" Mean diff, size= " + str(self.usize[this_size]),fontsize=8)
            axs[this_size,1].set_ylabel(" Std diff, size= " + str(self.usize[this_size]),fontsize=8)
            axs[this_size,2].set_ylabel(" Cov diff- Stat, size= " + str(self.usize[this_size]),fontsize=8)


            axs[this_size,0].locator_params(nbins=4)
            axs[this_size,1].locator_params(nbins=4)
            axs[this_size,2].locator_params(nbins=4)





        fig.tight_layout()

        nameout='Mean_Std_Cov-Locomotion_Stationary_'+str(self.usize[this_size])+'-expttype='+expttype+'-datafield='+datafield +'-running_pct_cutoff='+ str(running_pct_cutoff)+ '-speed_thr='+ str(running_speed_threshold) + '-normalization=' + which_normalization+ '-cell_selec='+ cell_selec
        fig.savefig(self.dir_2_output+nameout+'.pdf',dpi=300)




    def plot_Distrubitions_Activity_Vs_Contrast(self, expttype='size_contrast_0',datafield='F',running_pct_cutoff=default_running_pct_cutoff, running_speed_threshold=default_running_speed_threshold, which_normalization='individual_sum', cell_selec='all'):


        stat_loco_data_responsive, stat_loco_data_all=  self.get_size_contrast_centered_ori_averaged_locomotion_data_all_celltypes(expttype,datafield,running_pct_cutoff, running_speed_threshold, which_normalization)

        if cell_selec=='all':
            stat_loco_data=stat_loco_data_all
        elif cell_selec=='responsive':
            stat_loco_data=stat_loco_data_responsive
        else:
            print('your options are all cells or responsive')



        matplotlib.rc('xtick', labelsize=8)
        matplotlib.rc('ytick', labelsize=8)

        mean_stationary=np.zeros((self.npops-1, self.ncontrast,self.nsize))
        std_stationary=np.zeros((self.npops-1, self.ncontrast,self.nsize))

        mean_locomotion=np.zeros((self.npops-1, self.ncontrast,self.nsize))
        std_locomotion=np.zeros((self.npops-1, self.ncontrast,self.nsize))

        for this_size in range(self.nsize):
            #####################################################################################################
            #####################################################################################################
            #####################################################################################################
            # Stationary

            fig, axs = plt.subplots(self.npops-1, self.ncontrast, figsize=(12, 6),dpi= 250, facecolor='w', edgecolor='k',sharex=True,sharey='row')
            fig.subplots_adjust(hspace = .1, wspace=.3)

            for this_celltype in range(self.npops-1):
                for this_contrast in range(self.ncontrast):

                    nexpt=len(stat_loco_data[this_celltype])

                    this_stat=np.array([])
                    for iexpt in range(nexpt):
                        try:
                            this_stat=np.concatenate((this_stat, stat_loco_data[this_celltype][iexpt][0,:,this_size,this_contrast]))
                        except:
                            pass
                    this_stat=this_stat[this_stat>0]
                    bins=np.linspace(np.percentile(this_stat,1),np.percentile(this_stat,95),int(len(this_stat)/10));
                    mean_stationary[this_celltype,this_contrast,this_size]=np.nanmean(this_stat)
                    std_stationary[this_celltype,this_contrast,this_size]=np.nanstd(this_stat)


                    axs[this_celltype,this_contrast].hist(this_stat,density=True,bins=bins,color=self.mycolor[this_celltype+1],edgecolor='None',alpha = 0.8)
                    axs[-1,this_contrast].set_xlabel("Activity",fontsize=8)
                    axs[this_celltype,this_contrast].set_ylabel(" Distribution " + self.celltypes_names[this_celltype+1],fontsize=8)
                    if this_celltype==0:
                        axs[0,this_contrast].set_title("Contrast " + str(self.ucontrast[this_contrast]),fontsize=8)

            fig.tight_layout()

            nameout='Stationary_Distribution-size_'+str(self.usize[this_size])+'-expttype='+expttype+'-datafield='+datafield+ '-running_pct_cutoff='+ str(running_pct_cutoff)+ '-speed_thr='+ str(running_speed_threshold) + '-normalization=' + which_normalization
            fig.savefig(self.dir_2_output+nameout+'.pdf',dpi=300)

            #####################################################################################################
            #####################################################################################################
            #####################################################################################################
            # Locomotion


            fig, axs = plt.subplots(self.npops-1, self.ncontrast, figsize=(12, 6),dpi= 250, facecolor='w', edgecolor='k',sharex=True,sharey='row')
            fig.subplots_adjust(hspace = .1, wspace=.3)

            for this_celltype in range(self.npops-1):
                for this_contrast in range(self.ncontrast):

                    nexpt=len(stat_loco_data[this_celltype])

                    this_loco=np.array([])
                    for iexpt in range(nexpt):
                        try:
                            this_loco=np.concatenate((this_loco, stat_loco_data[this_celltype][iexpt][1,:,this_size,this_contrast]))
                        except:
                            pass
                    this_loco=this_loco[this_loco>0]
                    bins=np.linspace(np.percentile(this_loco,1),np.percentile(this_loco,95),int(len(this_loco)/10));
                    mean_locomotion[this_celltype,this_contrast,this_size]=np.nanmean(this_loco)
                    std_locomotion[this_celltype,this_contrast,this_size]=np.nanstd(this_loco)


                    axs[this_celltype,this_contrast].hist(this_loco,density=True,bins=bins,color=self.mycolor[this_celltype+1],edgecolor='None',alpha = 0.8)
                    axs[-1,this_contrast].set_xlabel("Activity",fontsize=8)
                    axs[this_celltype,this_contrast].set_ylabel(" Distribution " + self.celltypes_names[this_celltype+1],fontsize=8)
                    if this_celltype==0:
                        axs[0,this_contrast].set_title("Contrast " + str(self.ucontrast[this_contrast]),fontsize=8)

            fig.tight_layout()

            nameout='Locomotion_Distribution-size_'+str(self.usize[this_size])+'-expttype='+expttype+'-datafield='+datafield+ '-running_pct_cutoff='+ str(running_pct_cutoff)+ '-speed_thr='+ str(running_speed_threshold) + '-normalization=' + which_normalization+ '-cell_selec='+ cell_selec
            fig.savefig(self.dir_2_output+nameout+'.pdf',dpi=300)




        fig, axs = plt.subplots(self.nsize, 4, figsize=(12, 12),dpi= 250, facecolor='w', edgecolor='k',sharex=True)
        fig.subplots_adjust(hspace = .1, wspace=.3)

        for this_size in range(self.nsize):


            for this_celltype in range(self.npops-1):
                axs[this_size,0].plot(self.ucontrast,mean_stationary[this_celltype,:,this_size],color=self.mycolor[this_celltype+1],linewidth=1)
                axs[this_size,1].plot(self.ucontrast,std_stationary[this_celltype,:,this_size],color=self.mycolor[this_celltype+1],linewidth=1)
                axs[this_size,2].plot(self.ucontrast,mean_locomotion[this_celltype,:,this_size],color=self.mycolor[this_celltype+1],linewidth=1)
                axs[this_size,3].plot(self.ucontrast,std_locomotion[this_celltype,:,this_size],color=self.mycolor[this_celltype+1],linewidth=1)

            axs[-1,0].set_xlabel("Contrast ",fontsize=8)
            axs[-1,1].set_xlabel("Contrast ",fontsize=8)
            axs[-1,2].set_xlabel("Contrast ",fontsize=8)
            axs[-1,3].set_xlabel("Contrast ",fontsize=8)

            axs[this_size,0].set_ylabel(" Mean Stationary, size= " + str(self.usize[this_size]),fontsize=8)
            axs[this_size,1].set_ylabel(" Std Stationary, size= " + str(self.usize[this_size]),fontsize=8)
            axs[this_size,2].set_ylabel(" Mean Locomotion, size= " + str(self.usize[this_size]),fontsize=8)
            axs[this_size,3].set_ylabel(" Std Locomotion, size= " + str(self.usize[this_size]),fontsize=8)
            
            axs[0,0].set_title(" Mean Stationary Distribution, size= " + str(self.usize[this_size]),fontsize=8)
            axs[0,1].set_title(" Std Stationary Distribution, size= " + str(self.usize[this_size]),fontsize=8)
            axs[0,2].set_title(" Mean Locomotion Distribution, size= " + str(self.usize[this_size]),fontsize=8)
            axs[0,3].set_title(" Std Locomotion Distribution, size= " + str(self.usize[this_size]),fontsize=8)
            


            axs[this_size,0].locator_params(nbins=4)
            axs[this_size,1].locator_params(nbins=4)
            axs[this_size,2].locator_params(nbins=4)
            axs[this_size,3].locator_params(nbins=4)


        fig.tight_layout()

        nameout='Mean_Std_Activity-Locomotion_Stationary_'+str(self.usize[this_size])+'-expttype='+expttype+'-datafield='+datafield+ '-running_pct_cutoff='+ str(running_pct_cutoff)+ '-speed_thr='+ str(running_speed_threshold) + '-normalization=' + which_normalization + '-cell_selec='+ cell_selec
        fig.savefig(self.dir_2_output+nameout+'.pdf',dpi=300)





    def plot_Differences_and_Covs_Due_to_Contrast(self, expttype='size_contrast_0',datafield='F',running_pct_cutoff=default_running_pct_cutoff, running_speed_threshold=default_running_speed_threshold, which_normalization='individual_sum', cell_selec='all'):

        stat_loco_data_responsive, stat_loco_data_all=  self.get_size_contrast_centered_ori_averaged_locomotion_data_all_celltypes(expttype,datafield,running_pct_cutoff, running_speed_threshold, which_normalization)

        if cell_selec=='all':
            stat_loco_data=stat_loco_data_all
        elif cell_selec=='responsive':
            stat_loco_data=stat_loco_data_responsive
        else:
            print('your options are all cells or responsive')

        mean_difference=np.zeros((self.npops-1, self.ncontrast-1,self.nsize))
        std_difference=np.zeros((self.npops-1, self.ncontrast-1,self.nsize))
        covariance_difference=np.zeros((self.npops-1, self.ncontrast-1,self.nsize))

        for irun in [0,1]:

            for this_size in range(self.nsize):


                ######################


                matplotlib.rc('xtick', labelsize=8)
                matplotlib.rc('ytick', labelsize=8)

                fig, axs = plt.subplots(self.npops-1, self.ncontrast-1, figsize=(12, 6),dpi= 250, facecolor='w', edgecolor='k',sharex=True,sharey='row')
                fig.subplots_adjust(hspace = .1, wspace=.3)

                for this_celltype in range(self.npops-1):
                    for this_contrast in range(self.ncontrast-1):

                        nexpt=len(stat_loco_data[this_celltype])

                        this_activity_this_c=np.array([])
                        this_activity_this_c_plus_1=np.array([])


                        for iexpt in range(nexpt):
                            try:
                                this_activity_this_c=np.concatenate((this_activity_this_c, stat_loco_data[this_celltype][iexpt][irun,:,this_size,this_contrast]))
                                this_activity_this_c_plus_1=np.concatenate((this_activity_this_c_plus_1, stat_loco_data[this_celltype][iexpt][irun,:,this_size,this_contrast+1]))

                            except:
                                pass

                        diffs=this_activity_this_c_plus_1-this_activity_this_c;
                        this_activity_this_c_plus_1=this_activity_this_c_plus_1[~np.isnan(diffs)]
                        this_activity_this_c=this_activity_this_c[~np.isnan(diffs)]
                        diffs=diffs[~np.isnan(diffs)]



                        #bins=np.linspace(-max_min,max_min,nbins_data[k]);
                        bins=np.linspace(np.percentile(diffs,1),np.percentile(diffs,99),int(len(diffs)/10));

                        mean_difference[this_celltype,this_contrast,this_size]=np.nanmean(diffs)
                        std_difference[this_celltype,this_contrast,this_size]=np.nanstd(diffs)
                        covariance_difference[this_celltype,this_contrast,this_size]=np.around(np.cov(this_activity_this_c,diffs)[0,1]/np.nanstd(this_activity_this_c)**2,6)


                        axs[this_celltype,this_contrast].scatter(this_activity_this_c,diffs,s=1,color=self.mycolor[this_celltype+1])

                        axs[-1,this_contrast].set_xlabel("'Baseline' Contrast ="+str(self.ucontrast[this_contrast]),fontsize=8)
                        axs[this_celltype,this_contrast].set_ylabel("C="+str(self.ucontrast[this_contrast+1])+ " - C="+str(self.ucontrast[this_contrast])+' \n ' + self.celltypes_names[this_celltype+1],fontsize=8)

                        if this_celltype==0:
                            axs[0,this_contrast].set_title("Contrast " + str(self.ucontrast[this_contrast]) +'\n Cov/Var='+str(covariance_difference[this_celltype,this_contrast,this_size]),fontsize=8)
                        else:
                            axs[this_celltype,this_contrast].set_title('Cov/Var='+str(covariance_difference[this_celltype,this_contrast,this_size]),fontsize=8)


                fig.tight_layout()
                if irun==0:
                    beh_stat='Stationary'
                elif irun==1:
                    beh_state='Locomotion'


                nameout='Differences_and_Covs_Due_to_Contrast-'+beh_stat+'_Data-size_'+str(self.usize[this_size])+'-expttype='+expttype+'-datafield='+datafield+ '-running_pct_cutoff='+ str(running_pct_cutoff)+ '-speed_thr='+ str(running_speed_threshold) + '-normalization=' + which_normalization+ '-cell_selec='+ cell_selec
                fig.savefig(self.dir_2_output+nameout+'.pdf',dpi=300)


            fig, axs = plt.subplots(self.nsize, 3, figsize=(12, 12),dpi= 250, facecolor='w', edgecolor='k',sharex=True)
            fig.subplots_adjust(hspace = .1, wspace=.3)

            for this_size in range(self.nsize):


                for this_celltype in range(self.npops-1):
                    axs[this_size,0].plot(self.ucontrast[:-1],mean_difference[this_celltype,:,this_size],color=self.mycolor[this_celltype+1],linewidth=1)
                    axs[this_size,1].plot(self.ucontrast[:-1],std_difference[this_celltype,:,this_size],color=self.mycolor[this_celltype+1],linewidth=1)
                    axs[this_size,2].plot(self.ucontrast[:-1],covariance_difference[this_celltype,:,this_size],color=self.mycolor[this_celltype+1],linewidth=1)



                axs[-1,0].set_xlabel("Contrast ",fontsize=8)
                axs[-1,1].set_xlabel("Contrast ",fontsize=8)
                axs[-1,2].set_xlabel("Contrast ",fontsize=8)



                axs[this_size,0].set_ylabel(" Mean diff, size= " + str(self.usize[this_size]),fontsize=8)
                axs[this_size,1].set_ylabel(" Std diff, size= " + str(self.usize[this_size]),fontsize=8)
                axs[this_size,2].set_ylabel(" Cov diff- Stat, size= " + str(self.usize[this_size]),fontsize=8)


                axs[0,0].set_title(" Mean diff, size= " + str(self.usize[this_size]),fontsize=8)
                axs[0,1].set_title(" Std diff, size= " + str(self.usize[this_size]),fontsize=8)
                axs[0,2].set_title(" Cov diff- Stat, size= " + str(self.usize[this_size]),fontsize=8)



                axs[this_size,0].locator_params(nbins=4)
                axs[this_size,1].locator_params(nbins=4)
                axs[this_size,2].locator_params(nbins=4)


            fig.tight_layout()

            nameout='Mean_Std_Cov_Due_to_Contrast-'+beh_stat+'_Data-size_'+str(self.usize[this_size])+'-expttype='+expttype+'-datafield='+datafield+ '-running_pct_cutoff='+ str(running_pct_cutoff)+ '-speed_thr='+ str(running_speed_threshold) + '-normalization=' + which_normalization+ '-cell_selec='+ cell_selec
            fig.savefig(self.dir_2_output+nameout+'.pdf',dpi=300)


