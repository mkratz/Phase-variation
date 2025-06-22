from random import sample
from scipy.signal import argrelextrema
from scipy.signal import find_peaks_cwt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_style('darkgrid') 
import tqdm 
import pandas as pd
from itertools import chain, product
import re
import datetime


def time_convert(x):
    h,m,s = map(int,x.split(':'))
    return (h*60+m)*60+s


def retrieve_dfs(file,dims,n_tps,csv=False):
    J = dict()
    for part in dims:
        if csv==True:
            df = pd.read_csv(file,header=part[1],index_col=0,nrows=n_tps,)
            df.reset_index(inplace=True)
            df_time = df['Time'].apply(time_convert)
            df = df.apply(pd.to_numeric, errors='coerce')

        else:
            df = pd.read_excel(file,header=part[1],index_col=0,nrows=n_tps,)
            # df.drop(0)
            df = df.reset_index()
            df = df.apply(pd.to_numeric, errors='coerce')

        J[part[0]] = df
        
    if csv==True:
        return J,df_time
    else:
        return J

def retrieve_samples(df_dict,well_start,multi=True):
    spl_list = []
    for part in well_start:
        splits = re.split('(\d+)', part)
        if multi == True:
            n = [ord(splits[0]) + i for i in [0,1,2]]
        else: 
            n = [ord(splits[0]) + i for i in [0,]]
        spl_list.append([chr(x) + splits[1] for x in n])
    J = dict()
    for k in list(df_dict.keys()):
        trimmed = df_dict[k][list(chain.from_iterable(spl_list))]
        J[k] = trimmed
    return J,spl_list
        
def pool_trips(df_dict,spl_list,r_names = True,name_map = dict(),multi=True):
    J = dict()
    for k in list(df_dict.keys()):
        if multi==True:
            df = df_dict[k].groupby(np.arange(len(df_dict[k].columns))//3, axis=1).mean()
        else:
            df = df_dict[k].groupby(np.arange(len(df_dict[k].columns))//1, axis=1).mean()
        for i,part in enumerate(spl_list):
            if r_names is True:
                name = name_map[part[0]]
            else:
                name = part[0]+ '-' + part[-1]
            df.rename(columns={df.columns[i]: name},inplace=True)
        J[k] = df
    return J

def process_dataset(indA_concs,indB_concs,rows,cols,path,nts,channels):
    indA_strs = [str(x) + ' uM IPTG' for x in indA_concs]
    indB_strs = [str(x) + ' uM Sal' for x in indB_concs]
    
    IxS_names=list(product(indA_strs,indB_strs))
    IxS_coords = list(product(indA_concs,indB_concs))
    IxS_wells = [r+c for r,c in product(rows,cols)]
    coords_to_names = dict(zip(IxS_coords,IxS_names))

    spl = IxS_wells
    spl_names = IxS_names
    spl_map = dict(zip(spl,spl_names))

    dic_full = retrieve_dfs(path,channels,nts)
    dic_cut,spl_list = retrieve_samples(dic_full,spl,multi=False)
    dic_pool = pool_trips(dic_cut,spl_list,name_map=spl_map,multi=False)

    spl_array = [spl,spl_names,spl_map]
    IxS_array = [IxS_coords,IxS_names,IxS_wells,coords_to_names]

    return dic_pool,spl_array,IxS_array

def find_peaks(dic_pool,spl_array,IxS_array,channel_name,dims,validate=True,search_params=[0,100,10,30],double_peak=False):
    peaks = []
    wndw_strt,wndw_end,wave_min,wave_max = search_params

    if channel_name == 'mTuq':
        ref = pd.read_csv('../Data/mTuq_ref.csv',index_col=0).values.flatten()

    elif channel_name == 'YFP':
        ref = pd.read_csv('../Data/YFP_ref.csv',index_col=0).values.flatten()

    else:
        ref = pd.read_csv('../Data/YFP_ref.csv',index_col=0).values.flatten()

    spl_names = spl_array[1]
    IxS_coords = IxS_array[0]
    coords_to_names = IxS_array[3]
    size= dims[0]*dims[1]
    

    for coord in IxS_coords:
        name = coords_to_names[coord]
        loc_max = find_peaks_cwt(dic_pool[channel_name][name][wndw_strt:wndw_end].values-ref[wndw_strt:wndw_end],np.arange(wave_min,wave_max))
        if len(loc_max) == 0:
            peaks.append(0)
            continue

        if double_peak == True:
            if len(loc_max) > 1:
                maxi_sub_list= []
                for loc in loc_max:
                    maxi_sub = dic_pool[channel_name][name].iloc[loc]
                    maxi_sub_list.append(maxi_sub)
                maxi= np.argmax(maxi_sub_list)
                peaks.append(dic_pool[channel_name][name].iloc[loc_max[maxi]])

            else:
                maxi=dic_pool[channel_name][name].iloc[loc_max].values[0]
                peaks.append(maxi)

        else:
            maxi=dic_pool[channel_name][name].iloc[loc_max].values[0]
            peaks.append(maxi)
    
    peaks = np.array(peaks)
    peaks = peaks.reshape(dims)

    if validate == True:
        rand_samples =np.arange(0,size)
        fig,axs = plt.subplots(dims[0],dims[1],figsize=(dims[1]*3,dims[1]*3))

        for i,ax in enumerate(axs.flatten()):
            ax.title.set_text(str(i//dims[0])+str(i%dims[0]))
            ax.plot(dic_pool[channel_name][spl_names[rand_samples[i]]][0:240])
            ax.hlines(y=peaks.flatten()[rand_samples[i]],color='r',xmin=0,xmax=150)

        plt.tight_layout()

    return peaks

def plot_heatmap(peaks,indA_concs,indB_concs,labels,vmin=None,vmax=None):
    peaks = peaks
    cmap='plasma'
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,8))
    obj=ax.pcolor(peaks, edgecolors='k', linewidths=1,cmap=cmap,vmin=vmin,vmax=vmax)
    C_range = indA_concs
    gamma_range = indB_concs
    title,x,y = labels

    plt.xticks(np.arange(0,len(indB_concs)),indB_concs,size=15)
    plt.yticks(np.arange(0,len(indA_concs)),indA_concs,size=15)
    # plt.yticks(np.arange(len(gamma_range))+0.5, gamma_range, rotation=45,)
    # plt.xticks(np.arange(len(C_range))+0.5, C_range, rotation=45)

    plt.xlabel(x,size=18,labelpad=10)
    plt.ylabel(y,size=18,labelpad=10)
    cbar = plt.colorbar(obj)
    cbar.ax.tick_params(labelsize=15)
    plt.title(title,size=25,pad=25,fontweight='bold')

    for i in range(len(indA_concs)):
        for j in range(len(indB_concs)):
            ax.text(j, i, np.round(peaks[i,j],2),
                        ha="left", va="bottom",ma='center', color="w",size=15,fontweight='bold')
    
    plt.tight_layout()