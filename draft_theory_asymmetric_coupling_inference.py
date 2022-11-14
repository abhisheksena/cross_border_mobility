#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:31:36 2022

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
from scipy.integrate import odeint
import matplotlib.pyplot as plt
# import geopandas as gpd
# import pandas as pd
# import random
# import cmath



save_results = '/home/abhishek/Desktop/Data_where2test/modeling_border_control/output/output_draft/'



# def two_patch_model(y, t):
def two_patch_model(y, t,mig_12,mig_21):    
    """
    Model System
    """
    dpop=np.zeros(3*2,float)   
    s=y[0:2]
    i=y[2:2*2]
    r=y[2*2:3*2]        
    N=s+i+r      
    m12=mig_12
    m11=1-mig_12
    m21=mig_21
    m22=1-mig_21
    M=np.array([[m11,m12],[m21,m22]])       
    # print(M)
    M_t=M.transpose()           
    N_pre=np.dot(M_t,N)   ### Present population     
    ########  THIS is the spatially coupled SIR model    
    dpop[0:2]= -s*(np.dot(M,(beta/N_pre)*(np.dot(M_t,i))))
    dpop[2:2*2]= s*(np.dot(M,(beta/N_pre)*(np.dot(M_t,i)))) - gamma*i
    dpop[2*2:3*2]=gamma*i    
    return dpop



######## for time series 

def stoc_eqs_timeseries(INP):    
    V=INP
    S1=V[0]
    I1=V[1]
    R1=V[2]
    S2=V[3]
    I2=V[4]
    R2=V[5]   
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
        Use=min([Num, V[np.where(Change[i,:]<0)]])  ## Asure that no population is negative
        V=V+Change[i,:]*Use                
    return V       
     
def Stoch_Iteration_timeseries(INPUT):
    lop=0
    S1=[0]
    I1=[0]
    R1=[0]
    S2=[0]
    I2=[0]
    R2=[0]
    
    for lop in T:
        res=stoc_eqs_timeseries(INPUT)
        S1.append(INPUT[0])
        I1.append(INPUT[1])        
        R1.append(INPUT[2])
        S2.append(INPUT[3])
        I2.append(INPUT[4])
        R2.append(INPUT[5])
        INPUT=res
    return [S1,I1,R1,S2,I2,R2]     

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
        Use=min([Num, V[np.where(Change[i,:]<0)]])  ## Asure that no population is negative
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
    
    for lop in T:
        # res=stoc_eqs_estimation(INPUT,MIG)     
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
# m12=0.01
# m11=1-m12
# m21=m12 
# m22=1-m21
ND=MaxTime=200
tau=1
T=np.arange(0.0, ND+tau, tau)

i0=np.hstack((10, 1))
total=np.hstack( ( 10000, 20000 ) )
s0=total - i0
r0=np.zeros(2)
    
beta1=0.2
beta2=0.2
beta=np.hstack((beta1,beta2))
gamma1=(1/14)
gamma2=(1/14)
gamma=np.hstack((gamma1,gamma2))

N01=total[0]
N02=total[1]

realization=5000
# init_noise=1




# ## Solve ODE once
# final_time=MaxTime
# time_span=np.arange(0,MaxTime+tau,tau)
# time_span=list(time_span)    
# y0=np.hstack((s0,i0,r0))
# mig_12=0.01
# mig_21=0.005

# sol = odeint(two_patch_model,y0,time_span, args=(mig_12,mig_21,))

# Sus=sol[:,0:2]
# Inf=sol[:,2:4]
# Rec=sol[:,4:6]    
        
# Diff_peak=tau*abs(np.argmax(Inf[:,0])-np.argmax(Inf[:,1]))
        
# plt.plot(Inf[:,0],'b')
# plt.plot(Inf[:,1],'r')
# print(Diff_peak)    




##########  Will give the m12, m21 vs peak difference from ODE

# final_time=MaxTime
# time_span=np.arange(0,MaxTime+tau,tau)
# time_span=list(time_span)    
# y0=np.hstack((s0,i0,r0))

# M0=np.arange(0,0.1,0.005)

# Diff_peak=np.zeros((np.shape(M0)[0], np.shape(M0)[0]))

# k12=-1
# for mig_12 in M0:
#     k12=k12+1
    
#     k21=-1
#     for mig_21 in M0:
#         k21=k21+1
        
       
    

#         sol = odeint(two_patch_model,y0,time_span, args=(mig_12,mig_21,))

#         Sus=sol[:,0:2]
#         Inf=sol[:,2:4]
#         Rec=sol[:,4:6]    
            
#         Diff_peak[k12,k21]=(tau*abs(np.argmax(Inf[:,0])-np.argmax(Inf[:,1])))
            
        
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow( Diff_peak, interpolation='nearest')
# fig.colorbar(cax)
        
    
    



## For Genarating time series for fixed beta and and coupling parameters and initial 
## conditions from stochastic simulation


# m12=0.05
# m11=1-m12
# m21=0.04 
# m22=1-m21

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

#     tT=np.array(T)
#     tS1=np.array(S1)[1:,]
#     tI1=np.array(I1)[1:,]
#     tR1=np.array(R1)[1:,]
    
#     tS2=np.array(S2)[1:,]
#     tI2=np.array(I2)[1:,]
#     tR2=np.array(R2)[1:,]
    
#     peaktime_diff.append(tau*abs(np.argmax(tI1[1:])-np.argmax(tI2[1:])))  
    
#     # plt.figure()
#     plt.plot(tI1/total[0],'r',alpha=0.05)
    


#     # plt.figure()
#     plt.plot(tI2/total[1],'b',alpha=0.05)


# plt.figure()
# plt.plot(peaktime_diff,'bo')   
# plt.axhline(y=0, xmin=0, xmax=realization,linestyle='--') 
# plt.figure()
# plt.hist(peaktime_diff,bins=np.linspace(np.min(peaktime_diff), np.max(peaktime_diff), np.floor(np.max(peaktime_diff)- np.min(peaktime_diff)/tau).astype(int)) ) 
# plt.figure()
# plt.hist(peaktime_diff,bins=np.linspace(np.min(peaktime_diff), np.max(peaktime_diff), 10 ) )
  

             

###################################################################################
### Simulation for generating likelihood for different values of coupling parameters (\beta is given in this case)

# start_m=0.0005
# end_m=0.02
# step=0.0005
# m_log=np.arange(start_m,end_m+step,step)

# # m_log=np.geomspace(start_m, end_m, np.floor((end_m - start_m)/step).astype(int), endpoint=True)





# ###  Given quantities
# BETA_target=beta
# I0_target=i0
# given_diff_peak=14

# peak_diff_m=np.empty((np.shape(m_log)[0], realization))
# peak_diff_final=np.empty(( np.shape(m_log)[0],np.shape(m_log)[0],realization))
# k12=-1 

# for mig_12 in m_log:
#     k12=k12+1
#     print(k12)    
#     k21=-1
#     for mig_21 in m_log:
#         k21=k21+1
        
#         peaktime_diff=[]   
#         BETA=BETA_target
#         I0=I0_target
#         for i in range(realization):
#         # print(i)
        
#         ### mobility flows are assumed to be unequal
        
#             M12=mig_12
#             M11=1-mig_12
#             M21=mig_21 
#             M22=1-mig_21
        
#             Y01=I0[0]
#             X01=N01-Y01  
#             Z01=0
#             Y02=I0[1]
#             X02=N02-Y02 ##np.floor(gamma*N0/beta)
#             Z02=0
#             INPUT = np.array((X01,Y01,Z01,X02,Y02,Z02))
#             MIG=np.array((M11,M12,M21,M22))
#             [S1,I1,R1,S2,I2,R2]=Stoch_Iteration_estimation(INPUT, MIG, BETA)
#             INPUT=[]

#             tT=np.array(T)
#             tS1=np.array(S1)[1:,]
#             tI1=np.array(I1)[1:,]
#             tR1=np.array(R1)[1:,]            
#             tS2=np.array(S2)[1:,]
#             tI2=np.array(I2)[1:,]
#             tR2=np.array(R2)[1:,]           
            
#             peaktime_diff.append(tau*abs(np.argmax(tI1[1:])-np.argmax(tI2[1:]))) 
        

       
# ######################

#         peak_diff_m[k21,:]=np.array(peaktime_diff)
        
#     peak_diff_final[k12,:]=peak_diff_m        
    


# norm_freq=np.empty((np.shape(m_log)[0],np.shape(m_log)[0]) )
# for i in range (np.shape(m_log)[0]):
    
#     for j in range (np.shape(m_log)[0]):
    
#         bin_count=np.linspace(np.min(peak_diff_final), np.max(peak_diff_final) , 100)
#         hist=np.histogram(peak_diff_final[i,j,:], bins=bin_count)    ## this will be the frequncy of occurence of values from an interval 
#         freq=hist[0]
#         bin1=hist[1]
#         index=np.where(bin1 > given_diff_peak)
    
#         if len(index[0])==0:
#             norm_freq1=0
#             print(1)
            
#         else:
#             min_index=np.min(index) 
#             norm_freq1=float(freq[min_index-1]/realization)
#             norm_freq[i,j]=(norm_freq1)


# from matplotlib.colors import LogNorm
# M12, M21 = np.mgrid[slice(start_m, end_m + step, step),
#                 slice(start_m, end_m + step, step)]            
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.pcolor(M12,M21, norm_freq,norm=LogNorm(vmin=norm_freq.min(), vmax=norm_freq.max()), cmap='PuBu_r' )
# fig.colorbar(cax)
             


############ fixing one of the value.. here mig_21

start_m=0.0001
end_m=0.05
step=0.001
m_log=np.arange(start_m,end_m+step,step)

# m_log=np.geomspace(start_m, end_m, np.floor((end_m - start_m)/step).astype(int), endpoint=True)





###  Given quantities
BETA_target=beta
I0_target=i0
given_diff_peak=15

peak_diff_m=np.empty((np.shape(m_log)[0], realization))
mig_21=0.005

k=-1 
for mig_12 in m_log:
    k=k+1
    print(k)    
        
    peaktime_diff=[]   
    BETA=BETA_target
    I0=I0_target
    for i in range(realization):
        # print(i)
        
        ###
        
        M12=mig_12
        M11=1-mig_12
        M21=mig_21 
        M22=1-mig_21
        
        Y01=I0[0]
        X01=N01-Y01  
        Z01=0
        Y02=I0[1]
        X02=N02-Y02 ##np.floor(gamma*N0/beta)
        Z02=0
        INPUT = np.array((X01,Y01,Z01,X02,Y02,Z02))
        MIG=np.array((M11,M12,M21,M22))
        # print(MIG)
        [S1,I1,R1,S2,I2,R2]=Stoch_Iteration_estimation(INPUT, MIG, BETA)
        INPUT=[]

        tT=np.array(T)
        tS1=np.array(S1)[1:,]
        tI1=np.array(I1)[1:,]
        tR1=np.array(R1)[1:,]            
        tS2=np.array(S2)[1:,]
        tI2=np.array(I2)[1:,]
        tR2=np.array(R2)[1:,]           
            
        peaktime_diff.append(tau*abs(np.argmax(tI1[1:])-np.argmax(tI2[1:]))) 
######################

    peak_diff_m[k,:]=np.array(peaktime_diff)
        
   

norm_freq=[] 
for i in range (np.shape(m_log)[0]):  
    bin_count=np.linspace(np.min(peak_diff_m), np.max(peak_diff_m) , 100)
    hist=np.histogram(peak_diff_m[i,:], bins=bin_count)    ## this will be the frequncy of occurence of values from an interval 
    freq=hist[0]
    bin1=hist[1]
    index=np.where(bin1 > given_diff_peak)
    
    if len(index[0])==0:
        norm_freq1=0
        print(1)            
    else:
        min_index=np.min(index) 
        norm_freq1=float(freq[min_index-1]/realization)
        print(min_index)
        
    norm_freq.append(norm_freq1)


    
# plt.plot(norm_freq)

plt.figure()
plt.plot(m_log, np.log10(norm_freq),'o') 
plt.figure()
plt.plot(m_log, (norm_freq),'o') 

# # np.savetxt('m_vs_freq_i0_10_1_beta_0.15_td_5.txt',list(zip(m_log,norm_freq)), fmt='%.18g' )

# ##### Plotting

# import scipy.stats
# from scipy.interpolate import UnivariateSpline
# cs = UnivariateSpline(m_log,(norm_freq))
# cs.set_smoothing_factor(.5)
# cut=scipy.stats.chi2.ppf(0.95, 1)/2   ## half the critical valueof the chi-squared distribution ( x 21 (0:95)=2 ¼ 1:92)
# loglik=np.log(cs(m_log))
# max_loglik=np.nanmax(np.log(cs(m_log)))
# # diff_log=-*(loglik - max_loglik)
# # upper_ind= np.where(diff_log< cut  )
# # upper_ind1=upper_ind[0][np.argmax(diff_log[upper_ind])]
# # upper_loglik=loglik[upper_ind1]  ##max_loglik - cut 

# upper_loglik=max_loglik - cut 
# tolerance = np.inf
# idx=np.argwhere((np.diff(np.sign(loglik-upper_loglik)) != 0) & (np.abs(np.diff(loglik - upper_loglik)) <= tolerance)).flatten()
# ci_low=m_log[np.min(idx)]
# ci_mean=m_log[np.nanargmax(loglik)]
# ci_up=m_log[np.max(idx)]

# # upper_ind= np.argmax(np.where( -2*( np.log(norm_freq)-np.nanmax(np.log(norm_freq)) )< cut   ))
# # upper=np.log(norm_freq[upper_ind])
# # tolerance = np.inf
# # idx=np.argwhere((np.diff(np.sign(np.log(norm_freq)-upper)) != 0) & (np.abs(np.diff(np.log(norm_freq)-upper)) <= tolerance)).flatten()
# # ci_low=m_log[np.min(idx)]
# # ci_mean=m_log[np.nanargmax(np.log(norm_freq))]
# # ci_up=m_log[np.max(idx)]



# fig, ax = plt.subplots(figsize=(5,5), dpi=400)
# plt.rcParams['xtick.major.width'] = 3
# plt.rcParams['ytick.major.width'] = 3
# plt.rcParams['axes.linewidth'] = 3
# ax.plot(m_log, np.log(cs(m_log)), 'k', lw=3)
# ax.plot(m_log, np.log((norm_freq)),'o', color='gray')


# ax.set_xlabel(r'$\bf{m}$', color = 'k', fontsize = '20')
# # ax.set_xlim([start_m,end_m])
# # ax.set_xticks([start_m,end_m])    
# # ax.set_xticklabels([r'$\bf{0}$',r'$\bf{5}$',r'$\bf{10}$',r'$\bf{15}$',r'$\bf{20}$'], fontsize='20')
# ax.set_ylabel(r'$\bf{LL}$', color = 'k',fontsize = '20')
# # ax.set_ylim(-5,-1)
# # ax.set_yticks([-5, -1])
# # ax.set_yticklabels([r'$\bf{0}$',r'$\bf{0.10}$', r'$\bf{0.20}$',r'$\bf{0.30}$'], fontsize='20')
# # ax.minorticks_off()
# ax.vlines(x=ci_mean,ymin=-5, ymax=max_loglik, color='b', linestyle='--')
# ax.vlines(x=ci_low,ymin=-5, ymax=upper_loglik, color='k', linestyle='--')
# ax.vlines(x=ci_up, ymin=-5, ymax=upper_loglik,color='k', linestyle='--')
# # plt.tight_layout()
# # fig2.savefig(save_results + 'mle_.png', dpi=400)




