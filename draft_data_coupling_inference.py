#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:24:10 2022

@author: abhishek
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:16:57 2022

@author: abhishek
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:59:13 2022

@author: abhishek
"""

%matplotlib qt5
# %matplotlib inline
import numpy as np
from my_functions import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import random
import cmath
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from scipy.linalg import block_diag
from scipy.signal import savgol_filter
# import seaborn as sns

from itertools import product
from multiprocessing import Pool

import timeit

start = timeit.default_timer()

#Your statements here




#################################################################### Parameters

save_results = '/home/abhishek/Desktop/Data_where2test/modeling_border_control/output/output_draft'
df= pd.read_csv('/home/abhishek/Desktop/Data_where2test/modeling_border_control/data/cz-pl-de-daily.csv',sep=',',dayfirst=True, parse_dates=[0])
ind = df[df.population==0].index    ### Remove the rows with zero/missing population
df=df.drop(ind)
df=df.drop('Unnamed: 0', axis=1)

df_time=df.iloc[:,np.arange(5,471)]
df_time_interp=df_time.transpose().interpolate(method='linear',axis=0).transpose()
df.iloc[:,np.arange(5,471)]=df_time_interp
missing_data=df.isnull().sum().sort_values(ascending=False)  ### check whether is any misssing data or not
total_missing=missing_data.sum()
df['centroid']=gpd.GeoSeries.from_wkt(df['centroid'])
df['geometry']=gpd.GeoSeries.from_wkt(df['geometry'])
geo_df=gpd.GeoDataFrame(df, geometry='geometry', crs='epsg:4326')
geo_df['area_sq_km'] = (1/1000000)*geo_df['area_meters']   ### Area in square km
#### Extract Countrywise data
df_CZ=df.loc[df['country']=='cz']  ### Czechia
df_PL=df.loc[df['country']=='pl']   ### Poland
df_PL = df_PL.reset_index()
df_PL=df_PL.drop('index', axis=1)
df_DE=df.loc[df['country']=='de']   ### Germany
df_DE = df_DE.reset_index()
df_DE=df_DE.drop('index', axis=1)

###############  AREA 
area_CZ=np.asarray(df_CZ['area_sq_km'])
area_PL=np.asarray(df_PL['area_sq_km'])
area_DE=np.asarray(df_DE['area_sq_km'])
area_ALL=np.hstack((area_CZ,area_PL,area_DE))
### Extract poulation in each country
pop_CZ=np.asarray(df_CZ['population'])
pop_PL=np.asarray(df_PL['population'])
pop_DE=np.asarray(df_DE['population'])
pop_ALL=np.hstack((pop_CZ,pop_PL,pop_DE))
n_CZ=np.shape(pop_CZ)[0]
n_PL=np.shape(pop_PL)[0]
n_DE=np.shape(pop_DE)[0]

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



cols = np.arange(5,471)
df_PL_time=df_PL[df_PL.columns[cols]].transpose().to_numpy()
df_PL_time_smooth=savgol_filter(df_PL_time, 51,3, axis=0)
df_DE_time=df_DE[df_DE.columns[cols]].transpose().to_numpy()
df_DE_time_smooth=savgol_filter(df_DE_time, 51,3, axis=0)



###  time 50 -170 for both PL and DE is taken as common time window
###     TIME 370 TO 460 for the next wave of infection


find_PL_DE = border_non_border_wo_distinct_id(dist_matrix=dist_matrix_PL_DE_inter,
                                                  dist_threshold_low=150)


id_PL=np.array(list(set(find_PL_DE[0])))
id_DE=np.array(list(set(find_PL_DE[1])))


t_begin=50
t_end=170
first_wave=np.arange(t_begin,t_end)

# plt.plot(df_PL_time_smooth[t_begin:t_end, id_PL],'b')
# plt.plot(df_DE_time_smooth[t_begin:t_end,id_DE],'r')
names_PL=np.array(df_PL["name"][id_PL] )  ### names_PL[id_PL[6]]= 'zgorzelecki'
names_DE=np.array(df_DE["name"][id_DE] )  ####  names_DE[id_DE[2]]= 'Görlitz'

#####  indices for which we estimate m (PL,DE)
## (6,2)(+), m_log=0-0.2, max -0.012
## (19,2)(-), m_log: 0-0.05, max=0.008 
## (19,13),m_log:0-0.2, max-0.07
## (10,13)(+),m_log: 0-0.3, max: 0.1 
## (13,4)(+),(m_log:0-0.1) max-0.03
## (13,12)(+) , highly connected max-0.04 onwards
## (24,9)(+)(m_log_Range:0-0.3, max-0.07), 
## (24,7)(+)(m_log:0-0.6, highly connected), 
## (24,14)(+)(m_log: 0-0.1, max=0.038) 
### (27,15)(+)(Time shouls start from 60 to 170 otherwise it will give negative slope; m_log: 0-0.9, max-0.5) 
#####################

pair_PL=24
pair_DE=14

plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['axes.linewidth'] = 3


## Plot thetime series of chosen pair of region

fig1, ax1 = plt.subplots(figsize=(8,5))
ax1.plot(df_PL_time_smooth[t_begin:t_end, id_PL[pair_PL]], color='#104E8B', linewidth=4,linestyle='solid', label=names_PL[pair_PL])
ax1.plot(df_DE_time_smooth[t_begin:t_end, id_DE[pair_DE]], color='#458B00', linewidth=4,linestyle='solid', label=names_DE[pair_DE])
ax1.set_xlabel(r'$\bf{Time (days)}$', color = 'k', fontsize = '20')
# ax1.set_xlim(0,120)
# ax1.set_xticks([0,30,60,90,120])    
# ax1.set_xticklabels([r'$\bf{0}$',r'$\bf{30}$',r'$\bf{60}$',r'$\bf{90}$',r'$\bf{120}$'], fontsize='20')
# ax1.set_ylabel(r'$\bf{Infection}$', color = 'k',fontsize = '20')
# ax1.set_ylim(0,100)
# ax1.set_yticks([0,20,40, 60, 80, 100])
# ax1.set_yticklabels([r'$\bf{0}$',r'$\bf{20}$', r'$\bf{40}$', r'$\bf{60}$',r'$\bf{80}$',r'$\bf{100}$'], fontsize='20')
plt.legend(fontsize=15, frameon=True,loc=(0.65, 0.71), handlelength=1.5)  
ax1.minorticks_off()
fig1.tight_layout()
# fig1.savefig(save_results + 'ts_24_14.png', dpi=400)


### Estimate beta from early growth of epidemics

peaktime1=int(np.floor(np.argmax(df_PL_time_smooth[t_begin:t_end, id_PL[pair_PL]])/2))
peaktime2=int(np.floor(np.argmax(df_DE_time_smooth[t_begin:t_end, id_DE[pair_DE]])/2))
cum_PL=np.cumsum(df_PL_time_smooth[t_begin:t_end, id_PL[pair_PL]])
cum_DE=np.cumsum(df_DE_time_smooth[t_begin:t_end, id_DE[pair_DE]])

plt.figure()
plt.scatter(cum_PL[0:peaktime1], df_PL_time_smooth[t_begin:t_begin+peaktime1, id_PL[pair_PL]])

m1, b1 = np.polyfit(cum_PL[0:peaktime1],df_PL_time_smooth[t_begin:t_begin+peaktime1, id_PL[pair_PL]], deg=1)
plt.axline(xy1=(0, b1), slope=m1, label=f'$y = {m1:.3f}x {b1:+.3f}$')
plt.legend(loc='upper left')

plt.figure()
plt.scatter(cum_DE[0:peaktime2],df_DE_time_smooth[t_begin:t_begin+peaktime2, id_DE[pair_DE]] )
m2, b2 = np.polyfit(cum_DE[0:peaktime2], df_DE_time_smooth[t_begin:t_begin+peaktime2, id_DE[pair_DE]], deg=1)
plt.axline(xy1=(0, b2), slope=m2, label=f'$y = {m2:.3f}x {b2:+.3f}$')
plt.legend(loc='upper left')



#### first plot the border region with region id
# fig = plt.figure(figsize=(5, 5), dpi=300)
# ind_country1=n_CZ  
# ind_country2=n_CZ + n_PL
# geo_df['color']= '#16777215'   ### color code for white
# geo_df['color'][np.array(list(set(ind_country1+id_PL)))]= 'blue'
# geo_df['color'][ np.array(list(set(ind_country2+id_DE)))]= 'orange'
# geo_df['label']=''
# geo_df['label'][ind_country1+id_PL]= np.arange(0,np.shape(id_PL)[0]).tolist() ##[",".join(item) for item in ind_col_PL.astype(str)]
# geo_df['label'][ind_country2+id_DE]= np.arange(0,np.shape(id_DE)[0]).tolist() ##[",".join(item) for item in ind_col_PL.astype(str)]
# ax = fig.add_subplot(111)
# geo_df.plot(color=geo_df['color'],ax=ax)
# ax.set_xlabel('Latitude', fontsize=12)
# ax.set_ylabel('Longitude', fontsize=12)
# for x, y, label in zip(geo_df['centroid'].x, geo_df['centroid'].y, geo_df['label']):
#     ax.annotate(label, xy=(x,y), xytext=(-1,1), fontsize=2,textcoords='offset points')
    
    


###  On map
pair_PL=[24,13,6]
pair_DE=[14,4,2]


# pair_PL=19
# pair_DE=2

fig = plt.figure(figsize=(10, 5), dpi=300)
ind_country1=n_CZ  
ind_country2=n_CZ + n_PL
geo_df['color']= '#16777215'   ### color code for white
geo_df['color'][0:n_CZ]='#FFFFFF'

geo_df['color'][n_CZ:n_CZ+n_PL]='#B0E2FF'
geo_df['color'][n_CZ+n_PL:n_CZ+n_PL+n_DE]='#BDFCC9'
geo_df['color'][ind_country1+id_PL[pair_PL]]= '#104E8B' ## blue
geo_df['color'][ind_country2+id_DE[pair_DE]]= '#458B00' ## green
# geo_df['color'][ind_country1+id_PL]= '#104E8B' ## blue
# geo_df['color'][ind_country2+id_DE]= '#458B00' ## green

ax = fig.add_subplot(111)
geo_df.plot(color=geo_df['color'],ax=ax)
plt.xticks(color='w')
plt.yticks(color='w')
# fig.tight_layout()
ax.set_xlim([8,20])
ax.set_ylim([48,55])
# ax.set_xlabel('Latitude', fontsize=12)
# ax.set_ylabel('Longitude', fontsize=12)
fig.tight_layout()









######## for time series 

# def stoc_eqs_timeseries(INP):    
#     V=INP
#     S1=V[0]
#     I1=V[1]
#     R1=V[2]
#     S2=V[3]
#     I2=V[4]
#     R2=V[5]   
#     Rate=np.zeros((4))
#     Change=np.zeros((np.shape(Rate)[0],np.shape(V)[0]))  ### Dimension: (no of Events, Number of variables)
#     N1=N01  
#     N2=N02  
#     N_pre_1=m11*N1 + m21*N2
#     N_pre_2= m12*N1 + m22*N2
    
    
#     Rate[0]=beta1*m11*S1*(m11*I1+m21*I2)/N_pre_1 + beta2*m12*S1*(m12*I1+m22*I2)/N_pre_2
#     Change[0,:]=[-1, +1, 0, 0,0,0]
#     Rate[1]=gamma1*I1
#     Change[1,:]=[0, -1, +1,0,0,0]
#     Rate[2]= beta2*m22*S2*(m12*I1+m22*I2)/N_pre_2 +  beta1*m21*S2*(m11*I1+m21*I2)/N_pre_1
#     Change[2,:]=[0,0,0,-1, +1, 0]
#     Rate[3] = gamma2*I2
#     Change[3,:]=[0,0,0,0,-1, +1]
           
#     for i in range(np.shape(Rate)[0]):       
#         Num=np.random.poisson(Rate[i]*tau)
#         Use=min([Num, V[np.where(Change[i,:]<0)]])  ## Asure that no population is negative
#         V=V+Change[i,:]*Use                
#     return V       
     
# def Stoch_Iteration_timeseries(INPUT):
#     lop=0
#     S1=[0]
#     I1=[0]
#     R1=[0]
#     S2=[0]
#     I2=[0]
#     R2=[0]
    
    
#     while (INPUT[1]>=1) or (INPUT[4]>=1):
#     # for lop in T:
#         res=stoc_eqs_timeseries(INPUT)
#         S1.append(INPUT[0])
#         I1.append(INPUT[1])        
#         R1.append(INPUT[2])
#         S2.append(INPUT[3])
#         I2.append(INPUT[4])
#         R2.append(INPUT[5])
#         INPUT=res
#     return [S1,I1,R1,S2,I2,R2]     

#### for estimation

# def stoc_eqs_estimation(INP,MIG,BETA):   
def stoc_eqs_estimation(INP,MIG,BETA):   ###    
    V=INP
    S1=V[0]
    I1=V[1]
    R1=V[2]
    S2=V[3]
    I2=V[4]
    R2=V[5]
    m11=MIG[0]
    m12=MIG[1]
    m21=MIG[2]
    m22=MIG[3]    
    beta1=BETA[0]
    beta2=BETA[1]
    Rate=np.zeros((4))
    Change=np.zeros((np.shape(Rate)[0],np.shape(V)[0]))  ### Dimension: (no of Events, Number of variables)
    N1=N01  
    N2=N02  
    N_pre_1=m11*N1 + m21*N2
    N_pre_2= m12*N1 + m22*N2        
    Rate[0]=beta1*m11*S1*(m11*I1+m21*I2)/N_pre_1 + beta2*m12*S1*(m12*I1+m22*I2)/N_pre_2
    Change[0,:]=[-1, +1, 0, 0,0,0]
    Rate[1]=gamma1*I1
    Change[1,:]=[0, -1, +1,0,0,0]
    Rate[2]= beta2*m22*S2*(m12*I1+m22*I2)/N_pre_2 +  beta1*m21*S2*(m11*I1+m21*I2)/N_pre_1
    Change[2,:]=[0,0,0,-1, +1, 0]
    Rate[3] = gamma2*I2
    Change[3,:]=[0,0,0,0,-1, +1]
           
    for i in range(np.shape(Rate)[0]):       
        Num=np.random.poisson(Rate[i]*tau)
        Use=min([Num, V[np.where(Change[i,:]<0)]])  ## Assure that no population is negative
        V=V+Change[i,:]*Use                
    return V       
 
# def Stoch_Iteration_estimation(INPUT,MIG):
def Stoch_Iteration_estimation(INPUT,MIG,BETA):    
    lop=0
    S1=[0]
    I1=[0]
    R1=[0]
    S2=[0]
    I2=[0]
    R2=[0]
    
    while (INPUT[1]>=1) or (INPUT[4]>=1):
    # for lop in T:
        
           
        res=stoc_eqs_estimation(INPUT,MIG, BETA)  
        S1.append(INPUT[0])
        I1.append(INPUT[1])        
        R1.append(INPUT[2])
        S2.append(INPUT[3])
        I2.append(INPUT[4])
        R2.append(INPUT[5])
        INPUT=res
    return [S1,I1,R1,S2,I2,R2]  



### Solve ODE
# m12=0.4
# m11=1-m12
# m21=m12 
# m22=1-m21
ND=MaxTime=np.shape(first_wave)[0]
tau=1
T=np.arange(0.0, ND+tau, tau)

i0=np.hstack((df_PL_time_smooth[t_begin,id_PL[pair_PL]], df_DE_time_smooth[t_begin,id_DE[pair_DE]] ))

total=np.hstack( ( pop_PL[id_PL[pair_PL]], pop_DE[id_DE[pair_DE]] ) )
s0=total - i0
r0=np.zeros(2)
    
# beta1=0.15
# beta2=0.15
# beta=np.hstack((beta1,beta2))
gamma1=(1/14)
gamma2=(1/14)
gamma=np.hstack((gamma1,gamma2))


beta1=m1+gamma1
beta2=m2+gamma2
beta=np.hstack((beta1,beta2))


N01=total[0]
N02=total[1]

realization=1000
  
# peaktime_diff=[] 
# for i in range(realization):    
#     Y01=i0[0]    
#     X01=N01-Y01  
#     Z01=0
#     Y02=i0[1]  
#     X02=N02-Y02
#     Z02=0
#     INPUT = np.array((X01,Y01,Z01,X02,Y02,Z02))       
#     [S1,I1,R1,S2,I2,R2]=Stoch_Iteration_timeseries(INPUT)
#     INPUT=[]

#     # tT=np.array(T)
#     tS1=np.array(S1)[1:,]
#     tI1=np.array(I1)[1:,]
#     tR1=np.array(R1)[1:,]
    
#     tS2=np.array(S2)[1:,]
#     tI2=np.array(I2)[1:,]
#     tR2=np.array(R2)[1:,]
    
#     peaktime_diff.append(tau*abs(np.argmax(tI1[1:])-np.argmax(tI2[1:])))  
    
#     # plt.figure()
#     plt.plot(tI1/1,'r',alpha=0.05)
    


#     # plt.figure()
#     plt.plot(tI2/1,'b',alpha=0.05)
    

# plt.figure()
# plt.plot(peaktime_diff)
# plt.figure()
# plt.hist(peaktime_diff)
### Simulation for generating likelihood for different values of coupling parameters (\beta is given in this case)

start_m=0.000
end_m=0.05
step=0.001
m_log=np.arange(start_m,end_m+step,step)


###  Given quantities,
BETA_target=beta
I0_target=i0
given_diff_peak=abs(np.argmax(df_PL_time_smooth[t_begin:t_end, id_PL[pair_PL]]) - np.argmax(df_DE_time_smooth[t_begin:t_end, id_DE[pair_DE]]))
under_report=0.0


###############
# peak_diff_m=np.empty((np.shape(m_log)[0], realization))
# k=-1 
# for m in m_log:
#     k=k+1
#     print(k)    
#     peaktime_diff=[]   
#     BETA=BETA_target
#     I0=I0_target
#     for i in range(realization):
#         # print(i)
        
#         ### mobility flows are assumed to be equal
        
#         M12=m
#         M11=1-M12
#         M21=m 
#         M22=1-M21
        
        
#         ### Randomness in initial conditions
#         Y01=np.random.uniform(I0[0], (1+under_report)*I0[0])
#         Y02=np.random.uniform(I0[1], (1+under_report)*I0[1])
    
        
        
#         # Y01=I0[0]
#         X01=N01-Y01  
#         Z01=0
#         # Y02=I0[1]
#         X02=N02-Y02 ##np.floor(gamma*N0/beta)
#         Z02=0
#         INPUT = np.array((X01,Y01,Z01,X02,Y02,Z02))
#         MIG=np.array((M11,M12,M21,M22))
#         [S1,I1,R1,S2,I2,R2]=Stoch_Iteration_estimation(INPUT, MIG, BETA)
#         INPUT=[]

#         # tT=np.array(T)
#         tS1=np.array(S1)[1:,]
#         tI1=np.array(I1)[1:,]
#         tR1=np.array(R1)[1:,]            
#         tS2=np.array(S2)[1:,]
#         tI2=np.array(I2)[1:,]
#         tR2=np.array(R2)[1:,]           
        
#         peaktime_diff.append(tau*abs(np.argmax(tI1[1:])-np.argmax(tI2[1:]))) 
        

       
# ######################

#     peak_diff_m[k,:]=np.array(peaktime_diff)
    
    
##### using nultiprocess    
    
def run_mle(m,i):
    print(m)
      
    BETA=BETA_target
    I0=I0_target
        
    M12=m
    M11=1-m
    M21=m 
    M22=1-m
    
    # Y01=np.random.uniform(I0[0], (1+under_report)*I0[0])
    # Y02=np.random.uniform(I0[1], (1+under_report)*I0[1])
    
        
        
    Y01=I0[0]
    X01=N01-Y01  
    Z01=0
    Y02=I0[1]
    X02=N02-Y02 ##np.floor(gamma*N0/beta)
    Z02=0
    INPUT = np.array((X01,Y01,Z01,X02,Y02,Z02))
    MIG=np.array((M11,M12,M21,M22))
    [S1,I1,R1,S2,I2,R2]=Stoch_Iteration_estimation(INPUT, MIG, BETA)
    INPUT=[]

    tT=np.array(T)
    tS1=np.array(S1)[1:,]
    tI1=np.array(I1)[1:,]
    tR1=np.array(R1)[1:,]            
    tS2=np.array(S2)[1:,]
    tI2=np.array(I2)[1:,]
    tR2=np.array(R2)[1:,]           
        
    peaktime_diff=(tau*abs(np.argmax(tI1[1:])-np.argmax(tI2[1:]))) 
          
    return peaktime_diff



iterations=np.arange(1,realization+1)
na = len(m_log)
nb = len(iterations)

p = Pool(8)  # <== maximum number of simultaneous worker processes

peak_diff_m = np.array(p.starmap(run_mle, product(m_log, iterations))).reshape(na, nb)    

p.terminate()
p.close()
    
#################    
    

norm_freq=[]   ##np.empty(np.shape(m_log)[0])
for i in range (np.shape(m_log)[0]):
    
    bin_count=np.linspace(np.min(peak_diff_m), np.max(peak_diff_m) , 100)
    hist=np.histogram(peak_diff_m[i,:], bins=bin_count)    ## this will the frequncy of occurence of values from an interval 
    freq=hist[0]
    bin1=hist[1]
    index=np.where(bin1 > given_diff_peak)
    
    if len(index[0])==0:
        norm_freq1=0
        print(1)
        
    else:
        min_index=np.min(index)                
        norm_freq1=float(freq[min_index-1]/realization)        
    norm_freq.append(norm_freq1)
    


plt.figure()
plt.plot(m_log, np.log(norm_freq),'o') 
plt.figure()
plt.plot(m_log, (norm_freq),'o') 

# np.savetxt('m_vs_freq_i0_10_1_beta_0.30_td_5.txt',list(zip(m_log,norm_freq)), fmt='%.18g' )

##### Plotting
import scipy.stats
from scipy.interpolate import UnivariateSpline

cs = UnivariateSpline(m_log,norm_freq)
cs.set_smoothing_factor(.002)
cut=scipy.stats.chi2.ppf(0.70, 1)/2   ## half the critical valueof the chi-squared distribution ( x 21 (0:95)=2 ¼ 1:92)
with np.errstate(invalid='ignore'):
    loglik=np.log(cs(m_log))
max_loglik=np.nanmax(loglik)
upper_loglik=max_loglik - cut 
tolerance =np.inf
idx=np.argwhere((np.diff(np.sign(loglik-upper_loglik)) != 0) & (np.abs(np.diff(loglik - upper_loglik)) <= tolerance)).flatten()
ci_low=m_log[np.min(idx)]
ci_mean=m_log[np.nanargmax(loglik)]
ci_up=m_log[np.max(idx)]

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(m_log, np.log((norm_freq)),'o',markersize=15, color="k",markeredgecolor="None", alpha=0.5)
ax.plot(m_log,loglik, lw=4,color='k',alpha=1)
ax.set_xlabel(r'$\bf{m}$', color = 'k', fontsize = '20')
ax.set_xlim(-0.003,0.101)
ax.set_xticks([-0.003,0.02,0.04,0.06,0.08,0.1])    
ax.set_xticklabels([r'$\bf{0}$',r'$\bf{0.02}$',r'$\bf{0.04}$',r'$\bf{0.06}$',r'$\bf{0.08}$',r'$\bf{0.1}$' ], fontsize='20')
ax.set_ylabel(r'$\bf{LL}$', color = 'k',fontsize = '20')
ax.set_ylim(-5.1,-1)
ax.set_yticks([-5,-4,-3,-2,-1])
ax.set_yticklabels([r'$\bf{-5}$',r'$\bf{-4}$', r'$\bf{-3}$', r'$\bf{-2}$',r'$\bf{-1}$'], fontsize='20')
ax.minorticks_off()
plt.vlines(x=ci_mean,ymin=-7, ymax=max_loglik,  color='k',lw=4 , linestyle='--',alpha=1)
# plt.vlines(x=ci_low,ymin=-7, ymax=upper_loglik, color='k', alpha=0.9)
# plt.vlines(x=ci_up, ymin=-7, ymax=upper_loglik, color='k', alpha=0.9)
# ax.vlines(x=true_m[i],ymin=-12, ymax=max_loglik,color=colors[i],lw=4, alpha=1, linestyle='--')   
plt.axvspan(ci_low, ci_up,ymin=0, ymax=1,  facecolor='k',edgecolor="None", alpha=0.1)
fig.tight_layout()
# fig.savefig(save_results + 'mle_24_14.png', dpi=400)

