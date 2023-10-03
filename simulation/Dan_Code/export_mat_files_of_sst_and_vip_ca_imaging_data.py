# %%
import pyute as ut
import numpy as np
import calnet.utils
import scipy.io as sio

CELLTYPE_LIST = ['sst_l23','vip_l23']
DSBASE = '/Users/dan/Documents/notebooks/mossing-PC/shared_data/' # folder containing imaging data
# ^ can be any of ['pyr_l4','pyr_l23','sst_l23','vip_l23','pv_l23']
UNSUCCESSFUL_EXPTS = ['session_'+exptname for exptname in ['180714_M9053','180321_M7955', '180519_M8959', '180531_M8961', '180618_M8956','190202_M10075', '190620_M10619']]
RUNNING_PCT_CUTOFF = 0.2
RET_PVAL_CUTOFF = 0.05 # p-value cutoff for neurons significantly visually driven during retinotopy expt
RF_DIST_CUTOFF = 10. # max degrees of retinotopic map distance from stimulus center to include
RUN_SPEED_CUTOFF_CM_S = 1.
SST_ITYPE = 0#2
VIP_ITYPE = 1#3
IALIGN_FOR_MAT_EXPORT = 0 # only export data for neurons within RF_DIST_CUTOFF
EXPORT_SUFFIX = "test"#"_210418"

def construct_dsnames(
        dsbase=DSBASE, 
        celltype_list=CELLTYPE_LIST,
    ):
    dsnames = [dsbase+x+'_data_struct.hdf5' for x in celltype_list]
    return dsnames

def print_sc_and_fg_expts(dsnames):
    for dsname in dsnames:
        print(dsname)
        with ut.hdf5read(dsname) as ds:
            keylist = list(ds.keys())
            for key in keylist:
                print(key)
                if 'size_contrast_0' in ds[key].keys():
                    print('SC')
                if 'figure_ground_0' in ds[key].keys():
                    print('FG')

def get_modal_stimulus_params():
    # the stimulus parameters most commonly used in each experiment type
    modal_uparam_ret = [
        np.array([-20., -15., -10.,  -5.,   0.,   5.,  10.,  15.,  20.]),
        np.array([-20., -15., -10.,  -5.,   0.,   5.,  10.,  15.,  20.]),
        ]

    modal_uparam = [
        np.array([ 5.,  8.21865442, 13.50925609, 22.20558144, 36.5, 60.]),
        np.array([0.  , 0.06, 0.12, 0.25, 0.5 , 1.  ]),
        np.array([  0,  45,  90, 135, 180, 225, 270, 315], dtype=np.uint16)
        ]

    modal_uparam_fg = [
        np.array([b'ctrl', b'fig', b'grnd', b'iso', b'cross'], dtype='|S5'),
        np.array([  0.,  45.,  90., 135., 180., 225., 270., 315.])
        ]

    return modal_uparam,modal_uparam_fg,modal_uparam_ret

def construct_selection(dsnames, to_exclude=UNSUCCESSFUL_EXPTS):
    ncelltypes = len(dsnames)
    selection = [None for itype in range(ncelltypes)]
    for itype in range(ncelltypes):
        with ut.hdf5read(dsnames[itype]) as ds:
            keylist = list(ds.keys())
            nexpt = len(keylist)
            to_keep = np.array([k not in to_exclude for k in keylist])
            selection[itype] = np.arange(nexpt)[to_keep]
    return selection

dsnames = construct_dsnames()

print_sc_and_fg_expts(dsnames)

SELECTION = construct_selection(dsnames, to_exclude=UNSUCCESSFUL_EXPTS)

def preprocess_ca_data(dsnames, expttype='', modal_uparam=None):
    rs,rs_sem,expt_ids,roi_ids, disps = [[None for irun in range(2)] for ivar in range(5)]
    for irun in range(2):
        rs[irun],rs_sem[irun],expt_ids[irun],roi_ids[irun],disps[irun] = calnet.utils.gen_rs_modal_uparam_expt_with_sem(
            dsnames=dsnames,selection=SELECTION,running=irun,modal_uparam=modal_uparam,
            expttype=expttype,pval_cutoff=RET_PVAL_CUTOFF,average_ori=False,dcutoff=RF_DIST_CUTOFF,
            run_cutoff=RUN_SPEED_CUTOFF_CM_S,running_pct_cutoff=RUNNING_PCT_CUTOFF
            )
    return rs,rs_sem,expt_ids,roi_ids,disps

modal_uparam, modal_uparam_fg, modal_uparam_ret = get_modal_stimulus_params()

rs_ret, rs_sem_ret, expt_ids_ret, roi_ids_ret, disps_ret = preprocess_ca_data(dsnames, expttype='retinotopy_0', modal_uparam=modal_uparam_ret)

rs, rs_sem, expt_ids, roi_ids, disps = preprocess_ca_data(dsnames, expttype='size_contrast_0', modal_uparam=modal_uparam)

rs_fg, rs_sem_fg, expt_ids_fg, roi_ids_fg, disps_fg = preprocess_ca_data(dsnames, expttype='figure_ground_0', modal_uparam=modal_uparam_fg)

def norm_to_exptwise_mean_non_running(vip_sc_event_rate,vip_sc_expt_ids):
    nexpt = int(vip_sc_expt_ids.max()+1)
    mn = np.ones((nexpt,))
    for iexpt in range(nexpt):
        mn[iexpt] = np.nanmean(vip_sc_event_rate[vip_sc_expt_ids==iexpt,0])
    print(mn)
    try:
        return vip_sc_event_rate/mn[vip_sc_expt_ids.astype('int')][:,np.newaxis,np.newaxis]
    except:
        return vip_sc_event_rate/mn[vip_sc_expt_ids.astype('int')][:,np.newaxis,np.newaxis,np.newaxis]

def gen_vip_sst_matdict(rs,expt_ids,lbl='sc'):
    if lbl=='sc':
        ori_axis = 3
    else:
        ori_axis = 2

    ialign = IALIGN_FOR_MAT_EXPORT # neurons within 10 degrees of retinotopic space of stimulus center

    itype = VIP_ITYPE
    lkat = [(expt_ids[irun][itype][ialign]!=10) for irun in range(2)]
    vip_fg_event_rate = [np.nanmean(rs[irun][itype][ialign][lkat[irun]],ori_axis) for irun in range(2)]
    vip_fg_expt_ids = [expt_ids[irun][itype][ialign][lkat[irun]] for irun in range(2)]
    vip_fg_event_rate,vip_fg_expt_ids,vip_fg_neuron_ids = calnet.utils.merge_by_neuron(*vip_fg_event_rate,*vip_fg_expt_ids)
    vip_fg_event_rate_norm = norm_to_exptwise_mean_non_running(vip_fg_event_rate,vip_fg_expt_ids)

    itype = SST_ITYPE
    lkat = [(expt_ids[irun][itype][ialign]!=10) for irun in range(2)]
    sst_fg_event_rate = [np.nanmean(rs[irun][itype][ialign][lkat[irun]],ori_axis) for irun in range(2)]
    sst_fg_expt_ids = [expt_ids[irun][itype][ialign][lkat[irun]] for irun in range(2)]
    sst_fg_event_rate,sst_fg_expt_ids,sst_fg_neuron_ids = calnet.utils.merge_by_neuron(*sst_fg_event_rate,*sst_fg_expt_ids)
    sst_fg_event_rate_norm = norm_to_exptwise_mean_non_running(sst_fg_event_rate,sst_fg_expt_ids)

    if lbl=='sc':
        array_dims = 'Neuron_x_Running_x_Size_x_Contrast'
        usize = np.array((5,8,13,22,36,60))
        ucontrast = np.array((0,6,12,25,50,100))
    else:
        array_dims = 'Neuron_x_Running_x_Stim'
        stim_lbls = ['ctrl','figure','ground','iso','cross']

    running_lbls = ['non_running','running']

    matdict = {'vip_%s_event_rate'%lbl:vip_fg_event_rate,               'vip_%s_event_rate_norm'%lbl:vip_fg_event_rate_norm,               'sst_%s_event_rate'%lbl:sst_fg_event_rate,               'sst_%s_event_rate_norm'%lbl:sst_fg_event_rate_norm,               'array_dims':array_dims,               'running_lbls':running_lbls,               'vip_%s_expt_ids'%lbl:vip_fg_expt_ids,               'sst_%s_expt_ids'%lbl:sst_fg_expt_ids,               'vip_%s_neuron_ids'%lbl:vip_fg_neuron_ids,               'sst_%s_neuron_ids'%lbl:sst_fg_neuron_ids}
    
    if lbl=='sc':
        matdict['sizes'] = usize
        matdict['contrasts'] = ucontrast
    else:
        matdict['stim_lbls'] = stim_lbls
    
    return matdict

def save_vip_sst_matfiles(rs, expt_ids, rs_fg, expt_ids_fg, suffix=EXPORT_SUFFIX):

    matdict_sc = gen_vip_sst_matdict(rs,expt_ids,lbl='sc')
    matdict_fg = gen_vip_sst_matdict(rs_fg,expt_ids_fg,lbl='fg')

    sio.savemat(f'vip_sst_fg_responses_by_running_{suffix}.mat',matdict_fg)
    sio.savemat(f'vip_sst_sc_responses_by_running_{suffix}.mat',matdict_sc)

matfile_fg = sio.loadmat('vip_sst_fg_responses_by_running.mat')
matfile_sc = sio.loadmat('vip_sst_sc_responses_by_running.mat')

def print_neuron_nos(matfile_fg,matfile_sc):
    for i in range(int(matfile_fg['vip_fg_expt_ids'].max()+1)):
        print(np.sum(matfile_fg['vip_fg_expt_ids']==i))
    print('\n')
    for i in range(int(matfile_sc['vip_sc_expt_ids'].max()+1)):
        print(np.sum(matfile_sc['vip_sc_expt_ids']==i))
    print('\n')
    for i in range(int(matfile_fg['sst_fg_expt_ids'].max()+1)):
        print(np.sum(matfile_fg['sst_fg_expt_ids']==i))
    print('\n')
    for i in range(int(matfile_sc['sst_sc_expt_ids'].max()+1)):
        print(np.sum(matfile_sc['sst_sc_expt_ids']==i))

print_neuron_nos(matfile_fg,matfile_sc)
# %%
