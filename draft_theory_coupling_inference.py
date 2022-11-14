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
def two_patch_model(y, t,m0):    
    """
    Model System
    """
    dpop=np.zeros(3*2,float)   
    s=y[0:2]
    i=y[2:2*2]
    r=y[2*2:3*2]        
    N=s+i+r      
    m12=m0
    m11=1-m0
    m21=m0
    m22=1-m0
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
# m21=0.01 
# m22=1-m21
ND=MaxTime=200
tau=0.01
T=np.arange(0.0, ND+tau, tau)

i0=np.hstack((10, 1))
total=np.hstack( ( 10000, 20000 ) )
s0=total - i0
r0=np.zeros(2)
    
beta1=0.30
beta2=0.20
beta=np.hstack((beta1,beta2))
gamma1=(1/14)
gamma2=(1/14)
gamma=np.hstack((gamma1,gamma2))

N01=total[0]
N02=total[1]

realization=2000
# init_noise=1




##### Quantifying \beta from slope 
# final_time=MaxTime
# time_span=np.arange(0,MaxTime+tau,tau)
# time_span=list(time_span)    
# y0=np.hstack((s0,i0,r0))
# m0=0.05

# sol = odeint(two_patch_model,y0,time_span, args=(m0,))
# Sus=sol[:,0:2]
# Inf=sol[:,2:4]
# Rec=sol[:,4:6]  


# plt.figure()
# plt.plot(Inf)

# peaktime1=int(np.floor(np.argmax(Inf[:,0])/1.5))
# peaktime2=int(np.floor(np.argmax(Inf[:,1])/1.5))
# cum_Inf=np.cumsum(Inf, axis=0)




# plt.figure()
# plt.scatter(cum_Inf[0:peaktime1,0], Inf[0:peaktime1,0])
# m1, b1 = np.polyfit(cum_Inf[0:peaktime1,0],Inf[0:peaktime1,0], deg=1)
# plt.axline(xy1=(0, b1), slope=m1, label=f'$y = {m1:.1f}x {b1:+.1f}$')
# plt.legend(loc='upper left')

# plt.figure()
# plt.scatter(cum_Inf[0:peaktime2,1], Inf[0:peaktime2,1])
# m2, b2 = np.polyfit(cum_Inf[0:peaktime2,1],Inf[0:peaktime2,1], deg=1)
# plt.axline(xy1=(0, b2), slope=m2, label=f'$y = {m2:.1f}x {b2:+.1f}$')
# plt.legend(loc='upper left')


# beta1=m1+gamma1
# beta2=m2+gamma2
# beta=np.hstack((beta1,beta2))

##########  Will give the m vs peak difference from ODE

# final_time=MaxTime
# time_span=np.arange(0,MaxTime+tau,tau)
# time_span=list(time_span)    
# y0=np.hstack((s0,i0,r0))

# M0=np.arange(0,.2,0.0001)

# Diff_peak=[]
# peak1=[]
# peak2=[]

# k=-1
# for m0 in M0:
#     k=k+1
    

#     sol = odeint(two_patch_model,y0,time_span, args=(m0,))

#     Sus=sol[:,0:2]
#     Inf=sol[:,2:4]
#     Rec=sol[:,4:6]    
        
#     Diff_peak.append(tau*abs(np.argmax(Inf[:,0])-np.argmax(Inf[:,1])))
#     peak1.append(np.max(Inf[:,0]))
#     peak2.append(np.max(Inf[:,1]))
    
        
# plt.figure()
# plt.plot(M0,Diff_peak)      
# plt.figure()
# plt.plot(M0,peak1)  
# plt.figure()
# plt.plot(M0,peak2)  


# final_time=MaxTime
# time_span=np.arange(0,MaxTime+tau,tau)
# time_span=list(time_span)    
# y0=np.hstack((s0,i0,r0))
# m0=0.05


# sol = odeint(two_patch_model,y0,time_span, args=(m0,))

# Sus=sol[:,0:2]
# Inf=sol[:,2:4]
# Rec=sol[:,4:6]    




# N_pre_1=(1-m0)*N01 + m0*N02
# N_pre_2= m0*N01 + (1-m0)*N02

# Inc1=(beta1*(1-m0)*Sus[:,0]*((1-m0)*Inf[:,0]+m0*Inf[:,1])/N_pre_1 + beta2*m0*Sus[:,0]*(m0*Inf[:,0]+(1-m0)*Inf[:,1])/N_pre_2)
# Inc2=(beta2*(1-m0)*Sus[:,1]*(m0*Inf[:,0]+(1-m0)*Inf[:,1])/N_pre_2 + beta1*m0*Sus[:,1]*((1-m0)*Inf[:,0]+m0*Inf[:,1])/N_pre_1)



# Inc1_final=np.empty(np.shape(T)[0])
# Inc1_final[0]= Inf[0,0]  ##i0[0]
# Inc2_final=np.empty(np.shape(T)[0])
# Inc2_final[0]=Inf[0,1]  ##i0[1]


# for j in range(1,np.shape(T)[0]):

#     Inc1_final[j]=np.trapz(Inc1[j-1:j+1])
#     Inc2_final[j]=np.trapz(Inc2[j-1:j+1])
    



        
# Diff_peak1=tau*abs(np.argmax(Inf[:,0])-np.argmax(Inf[:,1]))

# Diff_peak_inc=tau*abs(np.argmax(Inc1_final)-np.argmax(Inc2_final))

# plt.figure()        
# plt.plot(Inf[:,0],'b')
# plt.plot(Inf[:,1],'r')
# print(Diff_peak1)   

# plt.figure()        
# plt.plot(Inc1,'b')
# plt.plot(Inc2,'r')
# print(Diff_peak_inc)     
  




## For Genarating time series for fixed beta and and coupling parameters and initial 
## conditions from stochastic simulation

# colors = ['tab:red', 'tab:blue', 'tab:gray']

# fig, ax = plt.subplots(figsize=(10, 8), dpi=400)
# plt.rcParams['xtick.major.width'] = 3
# plt.rcParams['ytick.major.width'] = 3
# plt.rcParams['axes.linewidth'] = 3

# peaktime_diff=[] 
# Inf1=[]
# Inf2=[]
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
#     Inf1.append(tI1)
#     Inf2.append(tI2)
    
#     peaktime_diff.append(tau*abs(np.argmax(tI1[1:])-np.argmax(tI2[1:])))  
    
    
    # plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(tI1,'r', lw=3,color=colors[0],alpha=0.8)
    # ax.plot(tI2,'b', lw=3,color=colors[1],alpha=0.8)
    
### for genrating time series    
# ax.set_xlabel(r'$\bf{Time}$', color = 'k', fontsize = '30')
# ax.set_xlim(0,1500)
# ax.set_xticks([0,250,500,750,1000,1250,1500])    
# ax.set_xticklabels([r'$\bf{0}$',r'$\bf{25}$',r'$\bf{50}$',r'$\bf{75}$',r'$\bf{100}$',r'$\bf{125}$',r'$\bf{150}$'], fontsize='30')
# ax.set_ylabel(r'$\bf{I_{1}(t), I_{2}(t)}$', color = 'k',fontsize = '30')
# ax.set_ylim(-100,6050)
# ax.set_yticks([0,1000,2000,3000,4000,5000, 6050])
# ax.set_yticklabels([r'$\bf{0}$',r'$\bf{1000}$',r'$\bf{2000}$',r'$\bf{3000}$', r'$\bf{4000}$',r'$\bf{5000}$',r'$\bf{6000}$'], fontsize='30')
# ax.minorticks_off()
# ax.vlines(x=np.argmax(Inf1[0]),ymin=-100, ymax=np.max(Inf1[0]),  color=colors[0],lw=4 ,alpha=0.6, linestyle='--')
# ax.vlines(x=np.argmax(Inf2[0]),ymin=-100, ymax=np.max(Inf2[0]),  color=colors[1],lw=4 ,alpha=0.6, linestyle='--')
# ax.hlines(y=np.argmax(Inf2[0]),xmin=np.argmax(Inf1[0]), xmax=np.argmax(Inf2[0]),  color='k',lw=4 ,alpha=0.6, linestyle='--')
# fig.tight_layout()
# fig.savefig(save_results + 'try_td.png', dpi=400)
    
        
    
### For Histogram of t_d      
# ax.hist(peaktime_diff,bins=np.linspace(np.min(peaktime_diff), np.max(peaktime_diff), np.floor(np.max(peaktime_diff)- np.min(peaktime_diff)/tau).astype(int)), 
#         density=True,stacked=False,  color=colors[2], alpha=1 ) 
# ax.set_xlim(0,25)
# ax.set_xticks([0,5,10,15,20,25])    
# ax.set_xticklabels([r'$\bf{0}$',r'$\bf{5}$',r'$\bf{10}$',r'$\bf{15}$',r'$\bf{20}$',r'$\bf{25}$'], fontsize='30')
# ax.set_xlabel(r'$\bf{t_{d}}$', color = 'k',fontsize = '30')
# ax.set_ylim(0,0.1)
# ax.set_yticks([0,0.025,0.05,0.075,0.1])
# ax.set_yticklabels([r'$\bf{0}$',r'$\bf{0.025}$',r'$\bf{0.05}$',r'$\bf{0.075}$',r'$\bf{0.1}$'], fontsize='30')
# ax.set_ylabel(r'$\bf{Density}$', color = 'k',fontsize = '30')
# ax.minorticks_off()
# fig.tight_layout()
# fig.savefig(save_results + 'try_td_hist.png', dpi=400)


      
  


     
### creating histogram and td_vs_m using for loop


# BETA_target=np.array([[0.15,0.15],[0.20,0.20],[0.3,0.3],[0.15,0.15],[0.2,0.2],[0.3,0.3]])
# I0_target=np.array([[1,10],[1,10],[1,10],[10,1],[10,1],[10,1]])



# fig1 = plt.figure(figsize=(15, 10), dpi=300) 
# fig2 = plt.figure(figsize=(15, 10), dpi=300) 
# plt.rcParams['xtick.major.width'] = 3
# plt.rcParams['ytick.major.width'] = 3
# plt.rcParams['axes.linewidth'] = 3
# colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:pink', 'tab:olive','tab:cyan']


# for l in range (0, np.shape(BETA_target)[0]):
#     print(l)
#     BETA=BETA_target[l]
#     i0=I0_target[l]
    
    
#     m_log=np.arange(0,0.1+0.005,0.005)


#     peak_diff_m=np.empty((np.shape(m_log)[0], realization))
#     k=-1 
#     for m in m_log:
#         k=k+1
#         # print(k)    
#         peaktime_diff=[]   
#         for i in range(realization):
#             # print(i)        
#             ### mobility flows are assumed to be equal
            
#             M12=m
#             M11=1-M12
#             M21=m 
#             M22=1-M21
    
        
        
#             Y01=np.random.uniform(i0[0], (1/init_noise)*i0[0])
#             X01=N01-Y01  
#             Z01=0
#             Y02=np.random.uniform(i0[1], (1/init_noise)*i0[1])
#             X02=N02-Y02 ##np.floor(gamma*N0/beta)
#             Z02=0
#             INPUT = np.array((X01,Y01,Z01,X02,Y02,Z02))
#             MIG=np.array((M11,M12,M21,M22))
#             [S1,I1,R1,S2,I2,R2]=Stoch_Iteration_estimation(INPUT, MIG,BETA)
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

#         peak_diff_m[k,:]=np.array(peaktime_diff)
    
#     mean_peak_diff_m=np.quantile(peak_diff_m,0.5, axis=1)
#     min_peak_diff_m=np.quantile(peak_diff_m,0.05, axis=1)
#     max_peak_diff_m=np.quantile(peak_diff_m,0.95, axis=1)
    
    
    
    
    
#     label1=['(A)','(B)','(C)','(D)','(E)', '(F)']
#     label_beta=[r'$\bf{\beta}$='+ r"$\bf{"+str(BETA[0])+"}$",r'$\bf{\beta}$='+ r"$\bf{"+str(BETA[0])+"}$",
#                  r'$\bf{\beta}$='+ r"$\bf{"+str(BETA[0])+"}$",r'$\bf{\beta}$='+ r"$\bf{"+str(BETA[0])+"}$",
#                  r'$\bf{\beta}$='+ r"$\bf{"+str(BETA[0])+"}$",r'$\bf{\beta}$='+ r"$\bf{"+str(BETA[0])+"}$"]
#     label_i10=[r'$\bf{I_{1}(0)}}$='+ r"$\bf{"+str(i0[0])+"}$",r'$\bf{I_{1}(0)}}$='+ r"$\bf{"+str(i0[0])+"}$",
#                  r'$\bf{I_{1}(0)}}$='+ r"$\bf{"+str(i0[0])+"}$",r'$\bf{I_{1}(0)}}$='+ r"$\bf{"+str(i0[0])+"}$",
#                 r'$\bf{I_{1}(0)}}$='+ r"$\bf{"+str(i0[0])+"}$",r'$\bf{I_{1}(0)}}$='+ r"$\bf{"+str(i0[0])+"}$"]
              
#     label_i20=[r'$\bf{I_{2}(0)}}$='+ r"$\bf{"+str(i0[1])+"}$",r'$\bf{I_{2}(0)}}$='+ r"$\bf{"+str(i0[1])+"}$",
#               r'$\bf{I_{2}(0)}}$='+ r"$\bf{"+str(i0[1])+"}$",r'$\bf{I_{2}(0)}}$='+ r"$\bf{"+str(i0[1])+"}$",
#              r'$\bf{I_{2}(0)}}$='+ r"$\bf{"+str(i0[1])+"}$",r'$\bf{I_{2}(0)}}$='+ r"$\bf{"+str(i0[1])+"}$"]
    
#     ax1 = fig1.add_subplot(2,np.shape(BETA_target)[0]/2,l+1)
#     ax1.plot(m_log,mean_peak_diff_m, color=colors[l],linewidth=2)
#     ax1.fill_between(m_log, (min_peak_diff_m), (max_peak_diff_m), color=colors[l], alpha=.1)
#     ax1.set_xlim([-0.0005, 0.1])
    
#     if l==3 or l==4 or l==5:
#         ax1.set_xlabel(r"$\bf{m}$",  color = 'k', fontsize='20')
#         ax1.set_xticks([0,0.025, 0.05,0.075, 0.1])    
#         ax1.set_xticklabels([r'$\bf{0}$',r'$\bf{0.025}$',r'$\bf{0.05}$',r'$\bf{0.075}$',r'$\bf{0.1}$'], fontsize='20')
        
#     else:
#         ax1.set_xticks([0,0.025, 0.05,0.075, 0.1])    
#         ax1.set_xticklabels([r'$\bf{0}$',r'$\bf{0.025}$',r'$\bf{0.05}$',r'$\bf{0.075}$',r'$\bf{0.1}$'], fontsize='20')
   
#     if l==0 or l==3:
#         print(l)
#         ax1.set_ylabel(r"$\bf{t_{d}}$",  color = 'k', fontsize='20')
#         ax1.set_ylim([-3, 60])
#         ax1.set_yticks([0, 20, 40, 60])
#         ax1.set_yticklabels([r'$\bf{0}$',r'$\bf{20}$',r'$\bf{40}$', r'$\bf{60}$'], fontsize='20')
#         ax1.minorticks_off()
#     else:
#         ax1.set_ylim([-5, 60])
#         ax1.set_yticks([0, 20, 40, 60])
#         ax1.tick_params(labelleft=False) 
#         ax1.minorticks_off()
        
#     ax1.text(0.07, 1.1, label1[l], transform=ax1.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
#     ax1.text(0.6,0.7, label_beta[l], transform=ax1.transAxes,fontsize=16, fontweight='bold', va='top', ha='left')
#     ax1.text(0.6,0.6, label_i10[l], transform=ax1.transAxes,fontsize=16, fontweight='bold', va='top', ha='left')
#     ax1.text(0.6,0.5, label_i20[l], transform=ax1.transAxes,fontsize=16, fontweight='bold', va='top', ha='left')
#     fig1.tight_layout()
    
#     fig1.savefig(save_results + 'm_vs_td_try.png', dpi=400)
        
    
    

#     label2=['(A)','(B)','(C)','(D)','(E)', '(F)']
#     ax2 = fig2.add_subplot(2,np.shape(BETA_target)[0]/2,l+1)
#     ind=np.asscalar(np.array(np.where(m_log==0.025)))
#     bins = np.arange(min(peak_diff_m[ind,:]), max(peak_diff_m[ind,:]) + 1, 1)
#     ax2.hist(peak_diff_m[ind,:], bins=bins,density=True,stacked=False, color=colors[l], alpha=1 )            
#     ax2.set_xlim(0, 20)
    
#     if l==3 or l==4 or l==5:
#         ax2.set_xlabel(r"$\bf{t_{d}}$",  color = 'k', fontsize='20')
#         ax2.set_xticks([0,5,10,15, 20])    
#         ax2.set_xticklabels([r'$\bf{0}$',r'$\bf{5}$',r'$\bf{10}$',r'$\bf{15}$',r'$\bf{20}$'], fontsize='20')
        
#     else:
#         ax2.set_xticks([0,5,10,15, 20])    
#         ax2.set_xticklabels([r'$\bf{0}$',r'$\bf{5}$',r'$\bf{10}$',r'$\bf{15}$',r'$\bf{20}$'], fontsize='20')

      
#     if l==0 or l==3:
#         # print(l)
#         ax2.set_ylabel(r"$\bf{Density}$",  color = 'k', fontsize='20')
#         ax2.set_ylim(0,0.3)
#         ax2.set_yticks([0, 0.1,0.20, 0.30])
#         ax2.set_yticklabels([r'$\bf{0}$',r'$\bf{0.10}$', r'$\bf{0.20}$',r'$\bf{0.30}$'], fontsize='20')
#         ax2.minorticks_off()
        
#     else:
#         ax2.set_ylim(0,0.3)
#         ax2.set_yticks([0, 0.1,0.20, 0.30])
#         ax2.tick_params(labelleft=False) 
#         ax2.minorticks_off()
       
#     ax2.text(0.07, 1.1, label2[l], transform=ax2.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
#     ax2.text(0.6,0.7, label_beta[l], transform=ax2.transAxes,fontsize=16, fontweight='bold', va='top', ha='left')
#     ax2.text(0.6,0.6, label_i10[l], transform=ax2.transAxes,fontsize=16, fontweight='bold', va='top', ha='left')
#     ax2.text(0.6,0.5, label_i20[l], transform=ax2.transAxes,fontsize=16, fontweight='bold', va='top', ha='left')
#     plt.tight_layout()
    
#     fig2.savefig(save_results + 'td_vs_density_try.png', dpi=400)
    

###################################################################################
### Simulation for generating likelihood for different values of coupling parameters (\beta is given in this case)

start_m=0.001
end_m=0.06
step=0.0003
m_log=np.arange(start_m,end_m+step,step)

# m_log=np.geomspace(start_m, end_m, np.floor((end_m - start_m)/step).astype(int), endpoint=True)


###  Given quantities
BETA_target=beta
I0_target=i0
given_diff_peak=5

peak_diff_m=np.empty((np.shape(m_log)[0], realization))
k=-1 
for m in m_log:
    k=k+1
    print(k)    
    peaktime_diff=[]   
    BETA=BETA_target
    I0=I0_target
    for i in range(realization):
        # print(i)
        
        ### mobility flows are assumed to be equal
        
        M12=m
        M11=1-m
        M21=m 
        M22=1-m
    
        
        
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
        
        peaktime_diff.append(tau*abs(np.argmax(tI1[1:])-np.argmax(tI2[1:]))) 
        

       
######################

    peak_diff_m[k,:]=np.array(peaktime_diff)
    





norm_freq=[]   ##np.empty(np.shape(m_log)[0])
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
        
        
    norm_freq.append(norm_freq1)
    


plt.figure()
plt.plot(m_log, np.log10(norm_freq),'o') 
plt.figure()
plt.plot(m_log, (norm_freq),'o') 

# np.savetxt('new_m_vs_freq_i0_10_1_beta_0.20_td_5.txt',list(zip(m_log,norm_freq)), fmt='%.18g' )

##### Plotting

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




    
    
# plt.figure()    
# q1 = np.quantile(peaktime_diff,0.25)
# q3 = np.quantile(peaktime_diff,0.75)
# iqr = q3 - q1
# bin_width = (2 * iqr) / (len(peaktime_diff) ** (1 / 3))
# bin_count = int(np.ceil((np.max(peaktime_diff) - np.min(peaktime_diff)) / bin_width))
# # hist=np.histogram(peaktime_diff, bins=bin_count) 
# plt.hist(peaktime_diff,bins=bin_count)    
# plt.figure()    
# plt.plot(peaktime_diff)   
# plt.figure()
# plt.hist(peaktime_diff)
    

#### Expression basic reproduction number
# # basic_repro_1=(beta1/gamma1)*(X01/N01)
# # basic_repro_2=(beta2/gamma2)*(X02/N02)
# # R01=(beta1/gamma1)*X01*( ( 1/((1-m)*N01 +m*N02) )*(1-m)**2 + ( 1/(m*N01 +(1-m)*N02) )*m**2 )
# # R02=(beta1/gamma1)*X01*m*(1-m)*( ( 1/((1-m)*N01 +m*N02) ) + ( 1/(m*N01 +(1-m)*N02) ) )
# # R03=(beta2/gamma2)*X02*m*(1-m)*( ( 1/(m*N01 +(1-m)*N02) ) + ( 1/((1-m)*N01 +m*N02) ) )
# # R04= (beta2/gamma2)*X02*( ( 1/(m*N01 +(1-m)*N02) )*(1-m)**2 + ( 1/((1-m)*N01 +m*N02) )*m**2 )
# # basic_repro_coupled=(R01+R04)/2 + (1/2)* np.sqrt((R01-R04)**2 + 4*R02*R03)
# # print(basic_repro_1)
# # print(basic_repro_2)
# # print(basic_repro_coupled)






