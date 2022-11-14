#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 10:15:53 2022

@author: abhishek
"""
import numpy as np
import copy


def order_parameter(data):

    time=np.shape(data)[0]
    space=np.shape(data)[1]
    slopes=np.zeros((time-1,space))
    for i in range(0,space):
        a=data[:,i]
        dy = np.diff(a)

        dt = np.diff(range (0,time))

        slope = (dy/dt)
    
        slopes[:,i]=np.arctan(slope)
    
    order=np.zeros(time-1, dtype=complex)     
    order=abs( (1/space)*np.sum(np.exp(1j*slopes), axis=1) )
    
    return order    


def border_non_border_distint_id(dist_matrix, dist_threshold_low,dist_threshold_up, country_pair, n_M,):
    find_border = np.where(dist_matrix <= dist_threshold_low)
    find_non_border = np.where((dist_matrix > dist_threshold_low) & (dist_matrix < dist_threshold_up))
    
    if country_pair== "CZ_PL":
        
        index_border_country1=list(set(find_border[0]))
        index_border_country2=list(set(find_border[1]))
        index_border_country2 = [x+ np.shape(dist_matrix)[0] for x in index_border_country2]
        
        index_non_border_country1=np.setdiff1d(list(set(find_non_border[0])),index_border_country1) 
        index_non_border_country2=np.setdiff1d(list(set(find_non_border[1])), index_border_country2)   
       
    elif country_pair== "PL_DE":
        index_border_country1=list(set(find_border[0]))
        index_border_country1 = [x+ (n_M - np.shape(dist_matrix)[0] + np.shape(dist_matrix)[1]) for x in index_border_country1]
        
        index_border_country2=list(set(find_border[1]))
        index_border_country2 = [x+ (n_M - np.shape(dist_matrix)[1]) for x in index_border_country2]
        
        index_non_border_country1=np.setdiff1d(list(set(find_non_border[0])),index_border_country1) 
        index_non_border_country2=np.setdiff1d(list(set(find_non_border[1])), index_border_country2)
        
    elif country_pair== "CZ_DE":
        
        index_border_country1=list(set(find_border[0]))
        
        index_border_country2=list(set(find_border[1]))
        index_border_country2 = [x+ (n_M - np.shape(dist_matrix)[1]) for x in index_border_country2]
        
        index_non_border_country1=np.setdiff1d(list(set(find_non_border[0])),index_border_country1) 
        index_non_border_country2=np.setdiff1d(list(set(find_non_border[1])), index_border_country2)
        
    
    return index_border_country1, index_border_country2, index_non_border_country1, index_non_border_country2 



def border_non_border_wo_distinct_id(dist_matrix, dist_threshold_low):
    
    find_border = np.where(dist_matrix <= dist_threshold_low)
    
    
    
    return find_border
    
    
    


def inter_country_matrix_uniform(dist_matrix,dist_threshold_low, theta, M_country):
     
    find_border=border_non_border_wo_distinct_id(dist_matrix=dist_matrix, dist_threshold_low=dist_threshold_low)

    M_inter=np.zeros((np.shape(dist_matrix)[0], np.shape(dist_matrix)[1]))
    
    
    for i in set(find_border[0]):
       
        for j in set(find_border[1]):
            M_inter[i,j]=M_country[i,i]
    
    

    # M_inter[find_border]=M_country.diagonal()[find_border[0]]

    n_neigh= np.count_nonzero(M_inter,axis=1)
    

    M_inter=copy.copy(theta*M_inter/n_neigh[:,np.newaxis])

    M_inter=np.nan_to_num(M_inter)
    
    # print(np.sum(np.sum(M_inter,axis=1)))
    
    
    # A=[]
    # for i in set(find_border[0]):
        
    #     # print(i)
    #     A.append(theta*M_country[i,i])
    
    
    # print(np.sum(A))

   
    return M_inter





def inter_country_matrix_gravity(dist_matrix,dist_threshold_low, theta, M_country, pop1,pop2, tau_1, tau_2, rho):
   
    find_border=border_non_border_wo_distinct_id(dist_matrix=dist_matrix, dist_threshold_low=dist_threshold_low)
    # print(find_border)

    M_inter=np.zeros((np.shape(dist_matrix)[0], np.shape(dist_matrix)[1]))
    
    for i in set(find_border[0]):
       
        for j in set(find_border[1]):
            M_inter[i,j]=(pop1[i]**tau_1)*(pop2[j]**tau_2)/(dist_matrix[i,j]**rho)
    
    

   
    M_inter=copy.copy((M_inter/M_inter.sum(axis=1, keepdims=True)))
    for i in set(find_border[0]):
        for j in set(find_border[1]):
            M_inter[i,j]= (theta*M_country[i,i])*M_inter[i,j]
    
   

    M_inter=np.nan_to_num(M_inter)
    # print(np.sum(np.sum(M_inter,axis=1)))

    

    return M_inter







def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))


   
    