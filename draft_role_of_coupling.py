#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 10:12:42 2022

@author: abhishek
"""

# import plotly.graph_objects as go
%matplotlib qt5
# %matplotlib inline
import copy
import geopandas as gpd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import random
import cmath
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from scipy.linalg import block_diag
# import seaborn as sns
from my_functions import *
# from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)




import matplotlib.ticker

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format



#################################################################### Parameters

save_results = '/home/abhishek/Desktop/Data_where2test/modeling_border_control/output/output_draft/'
# df_old= pd.read_csv('/home/abhishek/Desktop/Data_where2test/modeling_border_control/data/sx_pl_cz_v3.csv',sep=',',dayfirst=True, parse_dates=[0])

df= pd.read_csv('/home/abhishek/Desktop/Data_where2test/modeling_border_control/data/cz-pl-de-daily.csv',sep=',',dayfirst=True, parse_dates=[0])

i = df[df.population==0].index    ### Remove the rows with zero/missing population
df=df.drop(i)
df = df.reset_index()

missing_data=df.isnull().sum().sort_values(ascending=False)  ### check whether is any misssing data or not
total_number_of_missing_data=missing_data.sum()

df.fillna(0, inplace=True)  ### Fill the missing values with zeros

df['centroid']=gpd.GeoSeries.from_wkt(df['centroid'])
df['geometry']=gpd.GeoSeries.from_wkt(df['geometry'])
geo_df=gpd.GeoDataFrame(df, geometry='geometry', crs='epsg:4326')

# load dataset in coordinate system EPSG: 4326
# geo_df.set_geometry('geometry', inplace=True)
# # convert the data to metric coordinate system EPSG: 8857
# geo_df.to_crs('epsg:8857', inplace=True)
# # calculate real areas
# geo_df['area_meters'] = geo_df['geometry'].area
# # convert data to WGS back
# geo_df.to_crs('epsg:4326', inplace=True)
geo_df['area_sq_km'] = (1/1000000)*geo_df['area_meters']   ### Area in square km

#### Extract Countrywise data
df_CZ=df.loc[df['country']=='cz']  ### Czechia
df_PL=df.loc[df['country']=='pl']   ### Poland
df_DE=df.loc[df['country']=='de']   ### Germany
###############  ARea 
area_CZ=np.asarray(df_CZ['area_sq_km'])
area_PL=np.asarray(df_PL['area_sq_km'])
area_DE=np.asarray(df_DE['area_sq_km'])
area_ALL=np.hstack((area_CZ,area_PL,area_DE))
### Extract poulation in each country
pop_CZ=np.asarray(df_CZ['population'])
pop_PL=np.asarray(df_PL['population'])
pop_DE=np.asarray(df_DE['population'])
pop_ALL=np.hstack((pop_CZ,pop_PL,pop_DE))
#######  Distance matrix all 3 countries together
centroid=geo_df['centroid']
centroid_coord= [(x,y) for x,y in zip(geo_df['centroid'].x , geo_df['centroid'].y)]
centroid_coord=np.asarray(centroid_coord)
centroid_coord=centroid_coord.transpose()
centroid_coord_radians = np.radians(centroid_coord)   ### transform degrre to radians
dist_matrix_all=6371*haversine_distances(centroid_coord_radians.transpose())  ##multiply by Earth radius to get kilometers
####################  Extract distance between the regions within a country
dist_matrix_CZ= dist_matrix_all[0:len(df_CZ),0:len(df_CZ) ]
dist_matrix_PL= dist_matrix_all[len(df_CZ):len(df_CZ)+len(df_PL),len(df_CZ):len(df_CZ)+len(df_PL) ]
dist_matrix_DE= dist_matrix_all[len(df_CZ)+len(df_PL):len(df_CZ)+len(df_PL)+len(df_DE),len(df_CZ)+len(df_PL):len(df_CZ)+len(df_PL)+len(df_DE) ]
############  Only inter-country distances  (Note the difference between CZ_PL and CZ_PL_inter )
dist_matrix_CZ_PL_inter= dist_matrix_all[0:len(df_CZ),len(df_CZ):len(df_CZ)+len(df_PL)]
dist_matrix_PL_DE_inter= dist_matrix_all[len(df_CZ):len(df_CZ)+len(df_PL),len(df_CZ)+len(df_PL):len(df_CZ)+len(df_PL)+len(df_DE) ]
dist_matrix_CZ_DE_inter=dist_matrix_all[0:len(df_CZ), len(df_CZ)+len(df_PL):len(df_CZ)+len(df_PL)+len(df_DE)]


################  Time series
# cols = np.arange(7,473)  ## Time series
# cols = np.arange(7,157)  ## Time series

# df_only_timeseries=df[df.columns[cols]].transpose()
# df_only_timeseries_numpy=df_only_timeseries.to_numpy()

# CZ_incidence=df_only_timeseries_numpy[:,0:np.shape(pop_CZ)[0]]
# PL_incidence=df_only_timeseries_numpy[:,np.shape(pop_CZ)[0]:np.shape(pop_CZ)[0]+np.shape(pop_PL)[0]]
# DE_incidence=df_only_timeseries_numpy[:,np.shape(pop_CZ)[0]+np.shape(pop_PL)[0]:np.shape(pop_CZ)[0]+np.shape(pop_PL)[0]+np.shape(pop_DE)[0]]      


#####################   Create Migration Model Gravity Model
tau_1=0.46   ## Assumed from previous literature
tau_2=0.64
rho=2  ## spatial dependence

def create_mobility_matrix(dist_matrix,area,pop):
    
    dist_matrix[np.diag_indices_from(dist_matrix)] =np.sqrt((area/np.pi)) ### diagonals are approximated as refenece from a paper in Royal Society Interface       
    M=np.divide(np.outer(pop**tau_1,pop**tau_2), dist_matrix**rho)   ### formula without population density    
    M=M/M.sum(axis=1, keepdims=True)
    
    return M

###  mobility matrix of each country

M_CZ=create_mobility_matrix(dist_matrix_CZ, area_CZ,pop_CZ)
M_PL=create_mobility_matrix(dist_matrix_PL, area_PL,pop_PL)
M_DE=create_mobility_matrix(dist_matrix_DE, area_DE,pop_DE)

n_CZ=len(M_CZ)
n_PL=len(M_PL)
n_DE=len(M_DE)
n_M= n_CZ + n_PL + n_DE

M_CZ_PL_DE=block_diag(M_CZ,M_PL,M_DE)  ## This is for isolated countries/ baseline case


# ### plot     
# from matplotlib.colors import LogNorm
# from pylab import figure, cm
# from matplotlib import colors
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # ax.tick_params(axis=u'both', which=u'both',length=0)
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# cax = ax.matshow(M_CZ_PL_DE, cmap=cm.viridis, norm=colors.LogNorm(vmin=None, vmax=None))
# cb=fig.colorbar(cax)





# M_PL_DE_inter=inter_country_matrix_uniform( dist_matrix=dist_matrix_PL_DE_inter,
#                                 dist_threshold_low=300, 
#                                 theta=0.7, M_country=M_PL)


# M_inter_country= np.block([ 
#                           [M_PL, M_PL_DE_inter ],  
#                             [M_PL_DE_inter.transpose(), M_DE ]  ])


# ### plot     
# from matplotlib.colors import LogNorm
# from pylab import figure, cm
# from matplotlib import colors


# plt.rcParams['xtick.major.width'] = 3
# plt.rcParams['ytick.major.width'] = 3
# plt.rcParams['axes.linewidth'] = 3
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# # ax.tick_params(axis=u'both', which=u'both',length=0)
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# cax = ax.matshow(M_inter_country, cmap=cm.viridis, norm=colors.LogNorm(vmin=None, vmax=None))
# cb=fig.colorbar(cax)
# fig.tight_layout



################# Inter-country mobility matrix uniform mobility

def mobility_matrix_uniform(dist_PL_DE,dist_DE_PL, theta_PL_DE, theta_DE_PL, M_PL1,M_DE1):
        
### PL_DE
    M_PL_DE_inter=inter_country_matrix_uniform( dist_matrix=dist_matrix_PL_DE_inter,
                                    dist_threshold_low=dist_PL_DE, 
                                    theta=theta_PL_DE, M_country=M_PL1)      
### DE_PL
    M_DE_PL_inter=inter_country_matrix_uniform( dist_matrix=dist_matrix_PL_DE_inter.transpose(),
                                    dist_threshold_low=dist_DE_PL, 
                                    theta=theta_DE_PL, M_country=M_DE1)    
    find_PL_DE = border_non_border_wo_distinct_id(dist_matrix=dist_matrix_PL_DE_inter,
                                                      dist_threshold_low=dist_PL_DE)
    find_DE_PL = border_non_border_wo_distinct_id(dist_matrix=dist_matrix_PL_DE_inter.transpose(),
                                                      dist_threshold_low=dist_DE_PL) 
    X_PL_DE=np.zeros(n_PL)    
    X_PL_DE[find_PL_DE[0]]=theta_PL_DE
    M_PL2=copy.copy(M_PL1)
    M_PL2[np.diag_indices_from(M_PL2)]=(1 - X_PL_DE)*M_PL2[np.diag_indices_from(M_PL2)]    
    X_DE_PL=np.zeros(n_DE)    
    X_DE_PL[find_DE_PL[0]]=theta_DE_PL
    M_DE2=copy.copy(M_DE1)
    M_DE2[np.diag_indices_from(M_DE2)]=(1 - X_DE_PL)*M_DE2[np.diag_indices_from(M_DE2)]            
    M_inter_country= np.block([ [ M_CZ, np.zeros((n_CZ,n_PL)), np.zeros((n_CZ,n_DE))],
                          [ np.zeros((n_PL,n_CZ)), M_PL2, M_PL_DE_inter ],  
                          [ np.zeros((n_DE,n_CZ)), M_DE_PL_inter, M_DE2 ]  ])
    
    return M_inter_country




rho_grav=2
def mobility_matrix_gravity(dist_PL_DE,dist_DE_PL, theta_PL_DE, theta_DE_PL, M_PL1,M_DE1):
    
### PL_DE
    M_PL_DE_inter=inter_country_matrix_gravity( dist_matrix=dist_matrix_PL_DE_inter,
                                    dist_threshold_low=dist_PL_DE, 
                                    theta=theta_PL_DE, M_country=M_PL1, pop1=pop_PL,pop2=pop_DE, tau_1=tau_1, tau_2=tau_2, rho=rho_grav )
### DE_PL
    M_DE_PL_inter=inter_country_matrix_gravity( dist_matrix=dist_matrix_PL_DE_inter.transpose(),
                                    dist_threshold_low=dist_DE_PL, 
                                    theta=theta_DE_PL, M_country=M_DE1, pop1=pop_DE, pop2=pop_PL, tau_1=tau_1, tau_2=tau_2, rho=rho_grav)    
    find_PL_DE = border_non_border_wo_distinct_id(dist_matrix=dist_matrix_PL_DE_inter,
                                                      dist_threshold_low=dist_PL_DE)
    find_DE_PL = border_non_border_wo_distinct_id(dist_matrix=dist_matrix_PL_DE_inter.transpose(),
                                                      dist_threshold_low=dist_DE_PL)        
    X_PL_DE=np.zeros(n_PL)    
    X_PL_DE[find_PL_DE[0]]=theta_PL_DE
    M_PL2=copy.copy(M_PL1)
    M_PL2[np.diag_indices_from(M_PL2)]=(1 - X_PL_DE)*M_PL2[np.diag_indices_from(M_PL2)]    
    X_DE_PL=np.zeros(n_DE)    
    X_DE_PL[find_DE_PL[0]]=theta_DE_PL
    M_DE2=copy.copy(M_DE1)
    M_DE2[np.diag_indices_from(M_DE2)]=(1 - X_DE_PL)*M_DE2[np.diag_indices_from(M_DE2)]            
    M_inter_country= np.block([ [ M_CZ, np.zeros((n_CZ,n_PL)), np.zeros((n_CZ,n_DE))],
                          [ np.zeros((n_PL,n_CZ)), M_PL2, M_PL_DE_inter ],  
                          [ np.zeros((n_DE,n_CZ)), M_DE_PL_inter, M_DE2 ]  ])
    
    return M_inter_country



  ### CZ_PL for visualizing the border region
  
# find_PL_DE = border_non_border_wo_distinct_id(dist_matrix=dist_matrix_PL_DE_inter,
#                                                    dist_threshold_low=120)  

# id_PL=np.array(list(set(find_PL_DE[0])))
# id_DE=np.array(list(set(find_PL_DE[1])))

# id_neighbour_PL=id_PL
# id_neighbour_DE=id_DE
# ind_country1=n_CZ  
# ind_country2=n_CZ + n_PL

# ind_col_PL=ind_country1+id_neighbour_PL
# ind_col_DE=ind_country2+id_neighbour_DE
# fig = plt.figure(figsize=(5, 5), dpi=300)
  
# geo_df['color']= '#16777215'   ### color code for white
# geo_df['color'][ ind_col_PL]= 'blue'
# geo_df['color'][ ind_col_DE]= 'orange'

# ax = fig.add_subplot(111)
# geo_df.plot(color=geo_df['color'],ax=ax)
# ax.set_xlabel('Latitude', fontsize=12)
# ax.set_ylabel('Longitude', fontsize=12)

# # plt.xlim([11.9, 19.6])
# # plt.ylim([48.4, 52.7])
# fig.tight_layout()
# # plt.savefig(save_results + 'Map_dist_PL_DE_50.png',bbox_inches='tight', dpi=400)
# plt.show()

# def beta1(t):
    
   
#     if t <= 30:
#         beta_DE=0.0*np.ones(n_DE)
        
#     else:
#         beta_DE=0.6*np.ones(n_DE)
        
#     if t <= 0:
#         beta_PL=0.0*np.ones(n_PL)
        
#     else:
#         beta_PL=0.4*np.ones(n_PL)        
        
#     beta_CZ=0.0*np.ones(n_CZ)
#     # beta_PL=0.4*np.ones(n_PL)
    
#     beta=np.hstack((beta_CZ,beta_PL,beta_DE))
            
    
#     return beta
  


beta_CZ=0.0*np.ones(n_CZ)
beta_PL=0.4*np.ones(n_PL)
beta_DE=0.6*np.ones(n_DE)
beta=np.hstack((beta_CZ,beta_PL,beta_DE))

gamma_CZ=0.2*np.ones(n_CZ)
gamma_PL=0.2*np.ones(n_PL)
gamma_DE=0.2*np.ones(n_DE)
gamma=np.hstack((gamma_CZ,gamma_PL,gamma_DE))

###################    INitial conditions    ## TO DO different types of initial conditions

### only one infection in each of the country

i0=np.zeros(n_M)
s0=pop_ALL
r0=np.zeros(n_M)

random.seed(123440)
i0_CZ=random.randint(0,n_CZ)
random.seed(3096843)
i0_PL=random.randint(n_CZ+1, n_CZ+n_PL)
random.seed(550674)
i0_DE=random.randint(n_CZ+n_PL+1,n_CZ+n_PL+n_DE)

i0[i0_CZ]=0
i0[i0_PL]=1
i0[i0_DE]=1
s0[i0_CZ]=s0[i0_CZ]-i0[i0_CZ]
s0[i0_PL]=s0[i0_PL]-i0[i0_PL]
s0[i0_DE]=s0[i0_DE]-i0[i0_DE]

init_cond_network=np.hstack((s0,i0,r0))



def ode_model_network_commuting(y,t,M_coup):
    
    dpop=np.zeros(3*n_M,float)   
    s=y[0:n_M]
    i=y[n_M:2*n_M]
    r=y[2*n_M:3*n_M]    
    N=s+i+r   
    # M=Mig(t,M_coup, M_CZ_PL_DE)    
    M=M_coup    
    
    M_t=M.transpose()    
    # beta=beta1(t)
    N_pre=np.dot(M_t,N)   ### Present population 
    
    
    ########  THIS is the spatially coupled SIR model
    dpop[0:n_M]= -s*(np.dot(M,(beta/N_pre)*(np.dot(M_t,i))))
    dpop[n_M:2*n_M]=s*(np.dot(M,(beta/N_pre)*(np.dot(M_t,i)))) -gamma*i
    dpop[2*n_M:3*n_M]=gamma*i 
    
    return dpop


#### Sort out the solutions according to the country


def extract_sol_countrywise(sol,country):
    sol_ode_model_network_commuting=sol
######### CZECHIA    
    if country=='CZ':    
        sol_CZ=np.empty((len(time_span), 3*n_CZ))   ### 3 is for number of compartments
        for i in range (0,3):
            sol_CZ[:,i*(n_CZ):i*(n_CZ)+n_CZ]=sol_ode_model_network_commuting[:,i*(n_M):i*(n_M)+n_CZ]   
    ##sol_CZ_incidence=100000*(np.divide(sol_CZ[:,len(M_CZ):2*len(M_CZ)],pop_CZ) )
        sol=sol_CZ    
##### Poland
    elif country=='PL':
        sol_PL=np.empty((len(time_span), 3*n_PL))   ### 3 is for number of compartments
        for i in range (0,3):    
            sol_PL[:,i*(n_PL):i*(n_PL)+n_PL]=sol_ode_model_network_commuting[:,i*(n_M)+n_CZ:i*(n_M)+n_CZ+n_PL]                        
        sol=sol_PL    
####################   Saxony  ######
    elif country=='DE':
        sol_DE=np.empty((len(time_span), 3*n_DE))   ### 3 is for number of compartments
        for i in range (0,3):        
            sol_DE[:,i*(n_DE):i*(n_DE)+n_DE]=sol_ode_model_network_commuting[:,i*(n_M)+n_CZ+n_PL:i*(n_M)+n_CZ+n_PL+n_DE]

        sol=sol_DE

    return sol

final_time=200
time_span=np.linspace(0,final_time, final_time*1)
time_span=list(time_span)


#### array for theta values to compare the reslts from baseline values
theta_PL_DE_temp= np.array([0,0.05,0.1,0.3,0.6])
theta_DE_PL_temp= np.array([0,0.05,0.1,0.3,0.6])

# theta_PL_DE_temp= np.linspace(0,0.6, 13)
# theta_DE_PL_temp= np.linspace(0,0.6, 13)



theta1=theta_PL_DE_temp

### Border
mean_max_incidence_PL_border=np.empty(np.shape(theta_PL_DE_temp)[0])
mean_max_time_PL_border=np.empty(np.shape(theta_PL_DE_temp)[0])

mean_max_incidence_DE_border=np.empty(np.shape(theta_DE_PL_temp)[0])
mean_max_time_DE_border=np.empty(np.shape(theta_DE_PL_temp)[0])

FOS_PL_border=np.empty(np.shape(theta_PL_DE_temp)[0])
FOS_DE_border=np.empty(np.shape(theta_DE_PL_temp)[0])

### non-Border
mean_max_incidence_PL_nonborder=np.empty(np.shape(theta_PL_DE_temp)[0])
mean_max_time_PL_nonborder=np.empty(np.shape(theta_PL_DE_temp)[0])

mean_max_incidence_DE_nonborder=np.empty(np.shape(theta_DE_PL_temp)[0])
mean_max_time_DE_nonborder=np.empty(np.shape(theta_DE_PL_temp)[0])

FOS_PL_nonborder=np.empty(np.shape(theta_PL_DE_temp)[0])
FOS_DE_nonborder=np.empty(np.shape(theta_DE_PL_temp)[0])




##### ALL
mean_max_incidence_PL_all=np.empty(np.shape(theta_PL_DE_temp)[0])
mean_max_time_PL_all=np.empty(np.shape(theta_PL_DE_temp)[0])

mean_max_incidence_DE_all=np.empty(np.shape(theta_DE_PL_temp)[0])
mean_max_time_DE_all=np.empty(np.shape(theta_DE_PL_temp)[0])

FOS_PL_all=np.empty(np.shape(theta_PL_DE_temp)[0])
FOS_DE_all=np.empty(np.shape(theta_DE_PL_temp)[0])


# fig1 = plt.figure(figsize=(15, 4), dpi=300) 
# fig2 = plt.figure(figsize=(15, 4), dpi=300) 
# plt.rcParams['xtick.major.width'] = 2
# plt.rcParams['ytick.major.width'] = 2
# plt.rcParams['axes.linewidth'] = 2



dist_PL_DE=120
find_PL_DE = border_non_border_wo_distinct_id(dist_matrix=dist_matrix_PL_DE_inter,
                                                    dist_threshold_low=dist_PL_DE)  

id_PL=np.array(list(set(find_PL_DE[0])))
id_DE=np.array(list(set(find_PL_DE[1])))
id_PL_nonborder=np.setdiff1d(np.array(range(0,n_PL)) ,id_PL)
id_DE_nonborder=np.setdiff1d(np.array(range(0,n_DE)) ,id_DE)

for k in range (0,np.shape(theta_PL_DE_temp)[0]):
    
    M_coup=[]
    sol_ode_model_network_commuting=[]
    sol_PL=[]
    sol_PL_infection=[]
    sol_PL_infection_incidence=[]
    sol_DE=[]
    sol_DE_infection=[]
    sol_DE_infection_incidence=[]
    
    theta_PL_DE=theta_PL_DE_temp[k] 
    theta_DE_PL=theta_DE_PL_temp[k]
    
   
    
    ##### Uniform inter-country mobility
    
    # M_coup=mobility_matrix_uniform(dist_PL_DE=dist_PL_DE,
    #                                 dist_DE_PL=dist_PL_DE,
    #                                 theta_PL_DE=theta_PL_DE, 
    #                                 theta_DE_PL=theta_DE_PL, M_PL1=M_PL,M_DE1=M_DE)
    
    
    
    ####   Garvity Inter-country mobility
    
    M_coup=mobility_matrix_gravity(dist_PL_DE=dist_PL_DE,
                                    dist_DE_PL=dist_PL_DE,
                                    theta_PL_DE=theta_PL_DE, 
                                    theta_DE_PL=theta_DE_PL, M_PL1=M_PL,M_DE1=M_DE)
    
    
    
    
   
    sol_ode_model_network_commuting = odeint(ode_model_network_commuting,init_cond_network,time_span, args=(M_coup,))
    sol_PL=extract_sol_countrywise(sol=sol_ode_model_network_commuting,country="PL")
    sol_PL_infection= sol_PL[:,n_PL:2*n_PL]
    sol_PL_infection_incidence=100000*np.divide(sol_PL_infection,pop_PL)
    sol_DE=extract_sol_countrywise(sol=sol_ode_model_network_commuting, country="DE")
    sol_DE_infection=sol_DE[:,n_DE:2*n_DE]
    sol_DE_infection_incidence=100000*np.divide(sol_DE_infection,pop_DE)
        
    

    
    ########### border region
     
    mean_max_incidence_PL_border[k]=(1/np.shape(id_PL)[0])*np.sum(np.amax(sol_PL_infection_incidence[:,id_PL], axis=0))
    mean_max_time_PL_border[k]=(1/np.shape(id_PL)[0])*np.sum(np.argmax(sol_PL_infection_incidence[:,id_PL], axis=0))
    FOS_PL_border[k]=(1/np.sum(pop_PL[id_PL]))*np.sum(sol_PL[-1,2*n_PL:3*n_PL][id_PL])
    
    mean_max_incidence_DE_border[k]=(1/np.shape(id_DE)[0])*np.sum(np.amax(sol_DE_infection_incidence[:,id_DE], axis=0))
    mean_max_time_DE_border[k]=(1/np.shape(id_DE)[0])*np.sum(np.argmax(sol_DE_infection_incidence[:,id_DE], axis=0))
    FOS_DE_border[k]=(1/np.sum(pop_DE[id_DE]))*np.sum(sol_DE[-1,2*n_DE:3*n_DE][id_DE])
    
###### Plotting  

    
    # label1=['(A)','(B)','(C)','(D)','(E)']
    # ax1 = fig1.add_subplot(1,np.shape(theta_PL_DE_temp)[0],k+1)
    # ax1.plot(time_span,sol_PL_infection_incidence[:,id_PL],'tab:blue', linewidth=2)
    # ax1.set_xlabel(r"$\bf{Time}$",  color = 'k', fontsize='20')
    # ax1.set_xlim([0, 180])
    # ax1.set_xticks([0, 90,180])
    # ax1.minorticks_off()
    # ax1.set_xticklabels([r'$\bf{0}$',r'$\bf{90}$',r'$\bf{180}$'], fontsize='20')
    # ax1.set_title(r'$\bf{\theta}$='+ r"$\bf{"+str(theta1[k])+"}$", fontsize='20')
    # ax1.set_ylim([-300, 24000])
    # if k==0:

    #     ax1.set_yticks([0, 8000, 16000, 24000])
    #     ax1.set_yticklabels([r'$\bf{0}$',r'$\bf{800}$',r'$\bf{16000}$', r'$\bf{24000}$'], fontsize='20')
    #     ax1.yaxis.set_major_formatter(OOMFormatter(4, "%1.1f"))
    #     ax1.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    #     plt.setp(ax1.get_yticklabels(), fontsize=20, fontweight="bold")

    # else:
    #     ax1.tick_params(labelleft=False) 
             
    # fig1.supylabel(r"$\bf{PL-Incidence}$",  color = 'k', fontsize='20')
    # ax1.text(0.08, 1.2, label1[k], transform=ax1.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
    # fig1.tight_layout()
    # # fig1.savefig(save_results + 'PL_sym_time_dist_120_rho_2_unif_scenaio2.png', dpi=300)
        
    
    

    # label2=['(F)','(G)','(H)','(I)','(J)']
    # ax2 = fig2.add_subplot(1,np.shape(theta_PL_DE_temp)[0],k+1)
    # ax2.plot(time_span,sol_DE_infection_incidence[:,id_DE], 'tab:green', linewidth=2)
    # ax2.set_xlabel(r"$\bf{Time}$",  color = 'k', fontsize='20')
    # ax2.set_xlim([0, 180])
    # ax2.set_xticks([0, 90, 180])
    # ax2.minorticks_off()
    # ax2.set_xticklabels([r'$\bf{0}$',r'$\bf{90}$',r'$\bf{180}$'], fontsize='20')
    # ax2.set_title(r'$\bf{\theta}$='+ r"$\bf{"+str(theta1[k])+"}$", fontsize='20')
    # ax2.set_ylim([-300, 32000])
    # if k==0:

    #     ax2.set_yticks([0, 10000, 20000, 32000])
    #     ax2.set_yticklabels(['0','10000','20000', '32000'], fontsize='20')
    #     ax2.yaxis.set_major_formatter(OOMFormatter(4, "%1.1f"))
    #     ax2.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    #     plt.setp(ax2.get_yticklabels(), fontsize=20, fontweight="bold")


    # else:
    #     ax2.tick_params(labelleft=False)
        
        
        
    # fig2.supylabel(r"$\bf{DE-Incidence}$",  color = 'k', fontsize='20')
    # ax2.text(0.08, 1.2, label2[k], transform=ax2.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
    # fig2.tight_layout()
    # # fig2.savefig(save_results + 'DE_sym_time_dist_120_rho_2_unif_scenario2.png', dpi=300)
    
    ########### non-border region
   
   

    # mean_max_incidence_PL_nonborder[k]=(1/np.shape(id_PL_nonborder)[0])*np.sum(np.amax(sol_PL_infection_incidence[:,id_PL_nonborder], axis=0))
    # mean_max_time_PL_nonborder[k]=(1/np.shape(id_PL_nonborder)[0])*np.sum(np.argmax(sol_PL_infection_incidence[:,id_PL_nonborder], axis=0))
    # FOS_PL_nonborder[k]=(1/np.sum(pop_PL[id_PL_nonborder]))*np.sum(sol_PL[-1,2*n_PL:3*n_PL][id_PL_nonborder])
    
    # mean_max_incidence_DE_nonborder[k]=(1/np.shape(id_DE_nonborder)[0])*np.sum(np.amax(sol_DE_infection_incidence[:,id_DE_nonborder], axis=0))
    # mean_max_time_DE_nonborder[k]=(1/np.shape(id_DE_nonborder)[0])*np.sum(np.argmax(sol_DE_infection_incidence[:,id_DE_nonborder], axis=0))
    # FOS_DE_nonborder[k]=(1/np.sum(pop_DE[id_DE_nonborder]))*np.sum(sol_DE[-1,2*n_DE:3*n_DE][id_DE_nonborder])
    
    # ax1 = fig1.add_subplot(1,np.shape(theta_PL_DE_temp)[0],k+1)
    # ax1.plot(time_span,sol_PL_infection_incidence[:,id_PL_nonborder],'tab:blue')
    # ax1.set_xlabel("Time (Days)",  color = 'k', fontsize='15')
    # ax1.set_ylabel("PL-Incidence",  color = 'k', fontsize='15')
    # ax1.set_title(r'$\theta$='+ str(theta1[k]), fontsize='15')
    # ax1.set_ylim([0, 20000])
    # ax1.set_xticks([0, 100, 200])
    # ax1.set_xticklabels(['0','100','200'], fontsize='15')
    # ax1.set_yticks([0, 4000, 8000, 12000])
    # ax1.set_yticklabels(['0','4000','8000','12000'], fontsize='15')
    # fig1.tight_layout()
    # # fig1.savefig(save_results + 'PL_sym_time_dist_120_rho_2.png', dpi=300)
        
    
    # ax2 = fig2.add_subplot(1,np.shape(theta_PL_DE_temp)[0],k+1)
    # ax2.plot(time_span,sol_DE_infection_incidence[:,id_PL_nonborder], 'tab:green')
    # ax2.set_xlabel("Time (Days)",  color = 'k', fontsize='15')
    # ax2.set_ylabel("DE-Incidence",  color = 'k', fontsize='15')
    # ax2.set_title(r'$\theta$='+ str(theta1[k]), fontsize='15')
    # ax2.set_ylim([0, 30000])
    # ax2.set_xticks([0, 100, 200])
    # ax2.set_xticklabels(['0','100','200'], fontsize='15')
    # ax2.set_yticks([0, 4000, 8000, 12000, 16000])
    # ax2.set_yticklabels(['0','4000','8000','12000', '16000'], fontsize='15')
    # fig2.tight_layout()
    # # fig2.savefig(save_results + 'DE_sym_time_dist_120_rho_2.tiff', dpi=300)
    
    
    
    
    
#####  All regions    
    
    # mean_max_incidence_PL_all[k]=(1/n_PL)*np.sum(np.amax(sol_PL_infection_incidence, axis=0))
    # mean_max_time_PL_all[k]=(1/n_PL)*np.sum(np.argmax(sol_PL_infection_incidence, axis=0))
    # FOS_PL_all[k]=(1/np.sum(pop_PL))*np.sum(sol_PL[-1,2*n_PL:3*n_PL])
    
    # mean_max_incidence_DE_all[k]=(1/n_DE)*np.sum(np.amax(sol_DE_infection_incidence, axis=0))
    # mean_max_time_DE_all[k]=(1/n_DE)*np.sum(np.argmax(sol_DE_infection_incidence, axis=0))
    # FOS_DE_all[k]=(1/np.sum(pop_DE))*np.sum(sol_DE[-1,2*n_DE:3*n_DE])
    
    
    # ax1 = fig1.add_subplot(1,np.shape(theta_PL_DE_temp)[0],k+1)
    # ax1.plot(time_span,sol_PL_infection_incidence[:,:], 'tab:blue')
    # ax1.set_xlabel("Time",  color = 'k', fontsize='15')
    # ax1.set_ylabel("PL-Incidence",  color = 'k', fontsize='15')
    # ax1.set_title(r'$\theta$='+ str(theta1[k]))
    # ax1.set_ylim([0, 25000])
    # plt.tight_layout()
        
    
    # ax2 = fig2.add_subplot(1,np.shape(theta_PL_DE_temp)[0],k+1)
    # ax2.plot(time_span,sol_DE_infection_incidence[:,:], 'tab:green')
    # ax2.set_xlabel("Time",  color = 'k', fontsize='15')
    # ax2.set_ylabel("DE-Incidence",  color = 'k', fontsize='15')
    # ax2.set_title(r'$\theta$='+ str(theta1[k]))
    # ax2.set_ylim([0, 17000])
    # plt.tight_layout()
        
    
   
    
    
    
    

    
    

#####  relative Change for

##### border

RC_max_inc_PL_border=[]
RC_max_time_PL_border=[]
RC_FOS_PL_border=[]
RC_max_inc_DE_border=[]
RC_max_time_DE_border=[]
RC_FOS_DE_border=[]


### Non-border
# RC_max_inc_PL_nonborder=[]
# RC_max_time_PL_nonborder=[]
# RC_FOS_PL_nonborder=[]
# RC_max_inc_DE_nonborder=[]
# RC_max_time_DE_nonborder=[]
# RC_FOS_DE_nonborder=[]


# ### All
# RC_max_inc_PL_all=[]
# RC_max_time_PL_all=[]
# RC_FOS_PL_all=[]
# RC_max_inc_DE_all=[]
# RC_max_time_DE_all=[]
# RC_FOS_DE_all=[]


#### Relative change in diffrence 

### border
# Diff_max_inc_border=[]
# Diff_max_time_border=[]
# Diff_FOS_border=[]

# ## non-border
# Diff_max_inc_nonborder=[]
# Diff_max_time_nonborder=[]
# Diff_FOS_nonborder=[]

# ### All

# Diff_max_inc_all=[]
# Diff_max_time_all=[]
# Diff_FOS_all=[]


for i in range(0,np.shape(theta_PL_DE_temp)[0]):

## Border    
    RC_max_inc_PL_border.append(100*(1 - (mean_max_incidence_PL_border[i])/(mean_max_incidence_PL_border[0])))
    
    RC_max_time_PL_border.append(100*(1 - mean_max_time_PL_border[i]/mean_max_time_PL_border[0]))
    
    RC_FOS_PL_border.append(100*(1 - FOS_PL_border[i]/FOS_PL_border[0]))
    
    RC_max_inc_DE_border.append(100*(1 -mean_max_incidence_DE_border[i]/mean_max_incidence_DE_border[0]))
    
    RC_max_time_DE_border.append(100*(1 - mean_max_time_DE_border[i]/mean_max_time_DE_border[0]))
    
    RC_FOS_DE_border.append( 100*(1 - FOS_DE_border[i]/FOS_DE_border[0]))
    
        
    
    
#     ### Non-Border
    
#     RC_max_inc_PL_nonborder.append(100*(1 -mean_max_incidence_PL_nonborder[i]/mean_max_incidence_PL_nonborder[0]))
    
#     RC_max_time_PL_nonborder.append(100*(1 - mean_max_time_PL_nonborder[i]/mean_max_time_PL_nonborder[0]))
    
#     RC_FOS_PL_nonborder.append(100*(1 - FOS_PL_nonborder[i]/FOS_PL_nonborder[0]))
    
#     RC_max_inc_DE_nonborder.append(100*(1 -mean_max_incidence_DE_nonborder[i]/mean_max_incidence_DE_nonborder[0]))
    
#     RC_max_time_DE_nonborder.append(100*(1 - mean_max_time_DE_nonborder[i]/mean_max_time_DE_nonborder[0]))
    
#     RC_FOS_DE_nonborder.append( 100*(1 - FOS_DE_nonborder[i]/FOS_DE_nonborder[0]))
    


# ## ALL
#     RC_max_inc_PL_all.append(100*(1 -mean_max_incidence_PL_all[i]/mean_max_incidence_PL_all[0]))
    
#     RC_max_time_PL_all.append(100*(1 - mean_max_time_PL_all[i]/mean_max_time_PL_all[0]))
    
#     RC_FOS_PL_all.append(100*(1 - FOS_PL_all[i]/FOS_PL_all[0]))
    
#     RC_max_inc_DE_all.append(100*(1 -mean_max_incidence_DE_all[i]/mean_max_incidence_DE_all[0]))
    
#     RC_max_time_DE_all.append(100*(1 - mean_max_time_DE_all[i]/mean_max_time_DE_all[0]))
    
#     RC_FOS_DE_all.append( 100*(1 - FOS_DE_all[i]/FOS_DE_all[0]))  
    
    
    
    #### Difference
    
    ## Border
    
    
    
    # Diff_max_inc_border.append(100*((abs(mean_max_incidence_PL_border[i]-mean_max_incidence_DE_border[i]))/(abs(mean_max_incidence_PL_border[0]-mean_max_incidence_DE_border[0])) ) )
    # Diff_max_time_border.append(100*((abs(mean_max_time_PL_border[i]-mean_max_time_DE_border[i]))/(abs(mean_max_time_PL_border[0]-mean_max_time_DE_border[0])) ) )
    # Diff_FOS_border.append( 100*((abs(FOS_PL_border[i]-FOS_DE_border[i]))/(abs(FOS_PL_border[0]-FOS_DE_border[0])) ) )
    
    # ### NOn -border
    
    # Diff_max_inc_nonborder.append(100*((abs(mean_max_incidence_PL_nonborder[i]-mean_max_incidence_DE_nonborder[i]))/(abs(mean_max_incidence_PL_nonborder[0]-mean_max_incidence_DE_nonborder[0])) ) )
    # Diff_max_time_nonborder.append(100*((abs(mean_max_time_PL_nonborder[i]-mean_max_time_DE_nonborder[i]))/(abs(mean_max_time_PL_nonborder[0]-mean_max_time_DE_nonborder[0])) ) )
    # Diff_FOS_nonborder.append( 100*((abs(FOS_PL_nonborder[i]-FOS_DE_nonborder[i]))/(abs(FOS_PL_nonborder[0]-FOS_DE_nonborder[0]) )) )
    
    
    # ### All
    
    # Diff_max_inc_all.append(100*((abs(mean_max_incidence_PL_all[i]-mean_max_incidence_DE_all[i]))/(abs(mean_max_incidence_PL_all[0]-mean_max_incidence_DE_all[0])) ) )
    # Diff_max_time_all.append(100*((abs(mean_max_time_PL_all[i]-mean_max_time_DE_all[i]))/(abs(mean_max_time_PL_all[0]-mean_max_time_DE_all[0])) ) )
    # Diff_FOS_all.append( 100*((abs(FOS_PL_all[i]-FOS_DE_all[i]))/(abs(FOS_PL_all[0]-FOS_DE_all[0])) ) )
    
    
    


##### Plotting  Difference

# ### Border
# plt.figure(figsize=(6,4), dpi=400)
# plt.plot(theta1,Diff_max_inc_border,linewidth=2,label="Peak incidence" )
# plt.plot(theta1,Diff_max_time_border,linewidth=2, label="Peak Time")
# plt.plot(theta1,Diff_FOS_border,linewidth=2,label="FOS")
# plt.xlabel(r"$\theta$", color = 'k', fontsize = '20')
# plt.ylabel('Relative change(%)', color = 'k',fontsize = '20')
# plt.legend(loc="upper right",fontsize = '10')
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.tight_layout()



# #### NON-Border
    
# plt.figure(figsize=(6,4), dpi=400)
# plt.plot(theta1,Diff_max_inc_nonborder,linewidth=2,label="Peak incidence" )
# plt.plot(theta1,Diff_max_time_nonborder,linewidth=2, label="Peak Time")
# plt.plot(theta1,Diff_FOS_nonborder,linewidth=2,label="FOS")
# plt.xlabel(r"$\theta$", color = 'k', fontsize = '20')
# plt.ylabel('Relative change(%)', color = 'k',fontsize = '20')
# plt.legend(loc="upper right",fontsize = '10')
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.tight_layout()



# ### All
    
# plt.figure(figsize=(6,4), dpi=400)
# plt.plot(theta1,Diff_max_inc_all,linewidth=2,label="Peak incidence" )
# plt.plot(theta1,Diff_max_time_all,linewidth=2, label="Peak Time")
# plt.plot(theta1,Diff_FOS_all,linewidth=2,label="FOS")
# plt.xlabel(r"$\theta$", color = 'k', fontsize = '20')
# plt.ylabel('Relative change(%)', color = 'k',fontsize = '20')
# plt.legend(loc="upper right",fontsize = '10')
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.tight_layout()
    
 ################   Relative Change
    
 
######   Border    
 
fig,axs=plt.subplots(1,2,figsize=(10,4), dpi=300)
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['axes.linewidth'] = 2
markers = ['s', '^', 'o']   
linecolor = [plt.cm.PuBuGn(0.5), plt.cm.OrRd(0.7), plt.cm.PuBuGn(0.8)]
lables = [r'$\bf{I_{peak}}$',r'$\bf{I_{peak}^{\tau}}$', r'$\bf{FOS}$']
markersizes = [10,10,10]

## PL
axs[0].axhline(y=0.0, color='gray', linestyle='--')
axs[0].plot(theta1,RC_max_inc_PL_border,color=linecolor[0],marker=markers[0],
              lw=3, markersize=markersizes[0], label=lables[0] )
axs[0].plot(theta1,RC_max_time_PL_border,color=linecolor[1],marker=markers[1],
            lw=3,markersize=markersizes[1], label=lables[1])
axs[0].plot(theta1,RC_FOS_PL_border,color=linecolor[2],marker=markers[2],
            lw=3,markersize=markersizes[2], label=lables[2])

axs[0].set_xlabel(r"$\bf{\theta}$", color = 'k', fontsize = '20')
axs[0].set_ylabel(r'$\bf{RC_{PL}(\%)}$', color = 'k',fontsize = '20')
axs[0].set_xlim([-0.018,0.62])
axs[0].set_xticks([0, 0.2,0.4,0.6])
axs[0].minorticks_off()
axs[0].set_xticklabels([r"$\bf{0}$",  r"$\bf{0.2}$",r"$\bf{0.4}$", r"$\bf{0.6}$"],fontsize=20)
axs[0].set_ylim([-60, 50])
axs[0].set_yticks([-60,-30,0,30, 50])
axs[0].set_yticklabels([r"$\bf{-60}$",r"$\bf{-30}$", r"$\bf{0}$", r"$\bf{30}$",  r"$\bf{50}$"],fontsize=20)
axs[0].legend(fontsize=12, frameon=True,loc=(0.72, 0.51), handlelength=1.5)  
axs[0].text(-0.2, 1.2, '(A)', transform=axs[0].transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
# fig.suptitle(r'$\bf{\rho=1.5}$', fontsize=20)

## DE
axs[1].axhline(y=0.0, color='gray', linestyle='--')
axs[1].plot(theta1,RC_max_inc_DE_border,color=linecolor[0],marker=markers[0],
            lw=3,markersize=markersizes[0], label=lables[0])
axs[1].plot(theta1,RC_max_time_DE_border,color=linecolor[1],marker=markers[1],
            lw=3,markersize=markersizes[1], label=lables[1])
axs[1].plot(theta1,RC_FOS_DE_border,color=linecolor[2],marker=markers[2],
            lw=3,markersize=markersizes[2], label=lables[2])

axs[1].set_xlim([-0.018,0.62])
axs[1].set_xlabel(r"$\bf{\theta}$", color = 'k', fontsize = '20')
axs[1].set_ylabel(r'$\bf{RC_{DE}(\%)}$', color = 'k',fontsize = '20')
axs[1].set_xticks([0, 0.2,0.4,0.6])
axs[1].minorticks_off()
axs[1].set_xticklabels([r"$\bf{0}$",  r"$\bf{0.2}$",r"$\bf{0.4}$", r"$\bf{0.6}$"],fontsize=20)
axs[1].set_ylim([-10, 20])
axs[1].set_yticks([ -10,0,10,20])
axs[1].set_yticklabels([r"$\bf{-10}$",r"$\bf{0}$",r"$\bf{10}$",r"$\bf{20}$"],fontsize=20)
axs[1].legend(fontsize=12, frameon=True,loc=(0.03, 0.61), handlelength=1.5)  
axs[1].text(-0.2, 1.2, '(B)', transform=axs[1].transAxes,fontsize=16, fontweight='bold', va='top', ha='right')


fig.tight_layout()
# fig.subplots_adjust(top=1)
# 
# fig.savefig(save_results + 'PL_DE_sym_theta_dist_120_rho_3_unif.png', dpi=400)







    
#### Non-border    
    
    
# plt.figure(figsize=(6,4), dpi=400)
# plt.plot(theta1,RC_max_inc_DE_nonborder,linewidth=2,label="Peak incidence" )
# plt.plot(theta1,RC_max_time_DE_nonborder,linewidth=2, label="Peak Time")
# plt.plot(theta1,RC_FOS_DE_nonborder,linewidth=2,label="FOS")
# plt.xlabel(r"$\theta$", color = 'k', fontsize = '20')
# plt.ylabel('Relative change(%)', color = 'k',fontsize = '20')
# plt.legend(loc="upper right",fontsize = '10')
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.tight_layout()

# # # plt.savefig(save_results + 'RC_PL_CZ_CZ.png', dpi=400)



# plt.figure(figsize=(6,4), dpi=400)
# plt.plot(theta1,RC_max_inc_PL_nonborder,linewidth=2,label="Peak incidence" )
# plt.plot(theta1,RC_max_time_PL_nonborder,linewidth=2, label="Peak Time")
# plt.plot(theta1,RC_FOS_PL_nonborder,linewidth=2,label="FOS")
# plt.xlabel(r"$\theta$", color = 'k', fontsize = '20')
# plt.ylabel('Relative change(%)', color = 'k',fontsize = '20')
# plt.legend(loc="upper right",fontsize = '10')
# # pl.xticks(np.arange(0, final_time+1, 20))
# # pl.yticks(np.arange(0, 4000, 1000))  
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.tight_layout()    
    



    
### All    
     
# plt.figure(figsize=(6,4), dpi=400)
# plt.plot(theta1,RC_max_inc_DE_all,linewidth=2,label="Peak incidence" )
# plt.plot(theta1,RC_max_time_DE_all,linewidth=2, label="Peak Time")
# plt.plot(theta1,RC_FOS_DE_all,linewidth=2,label="FOS")
# plt.xlabel(r"$\theta$", color = 'k', fontsize = '20')
# plt.ylabel('Relative change(%)', color = 'k',fontsize = '20')
# plt.legend(loc="upper right",fontsize = '10')
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.tight_layout()

# # # plt.savefig(save_results + 'RC_PL_CZ_CZ.png', dpi=400)



# plt.figure(figsize=(6,4), dpi=400)
# plt.plot(theta1,RC_max_inc_PL_all,linewidth=2,label="Peak incidence" )
# plt.plot(theta1,RC_max_time_PL_all,linewidth=2, label="Peak Time")
# plt.plot(theta1,RC_FOS_PL_all,linewidth=2,label="FOS")
# plt.xlabel(r"$\theta$", color = 'k', fontsize = '20')
# plt.ylabel('Relative change(%)', color = 'k',fontsize = '20')
# plt.legend(loc="upper right",fontsize = '10')
# # pl.xticks(np.arange(0, final_time+1, 20))
# # pl.yticks(np.arange(0, 4000, 1000))  
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.tight_layout()













    
    
