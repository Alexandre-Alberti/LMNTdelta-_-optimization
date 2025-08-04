# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 06:01:56 2025

@author: alexa
"""

import numpy as np
from numpy import random as rd
from scipy.integrate import quad
from scipy.integrate import dblquad
import streamlit as st


st.title("Opportunistic Maintenance Policy - Analysis and Optimization")

st.markdown("Enter the parameters below and get the recommended maintenance policy:")

# Entrada de parâmetros
beta_x = st.number_input("Shape parameter (βx) for time to defect (Weibull distribution)")
eta_x = st.number_input("Scale parameter (ηx) for time to defect (Weibull distribution)")
beta_h = st.number_input("Shape parameter (βh) for delay-time (Weibull distribution)")
eta_h = st.number_input("Scale parameter (ηh) for delay-time (Weibull distribution)")
lbda = st.number_input("Rate of opportunities arrival (λ)")

Cp = st.number_input("Cost of pre-programmed preventive replacement (Cp)")
Cop = st.number_input("Cost of opportunistic preventive replacement (Cop)")
Ci = st.number_input("Cost of regular inspection (Ci)")
Coi = st.number_input("Cost of opportunistic inspection (Coi)")
Cf = st.number_input("Cost of failure (Cf)")

Cep_max = st.number_input("Cost of early preventive replacement with minimal waiting")

delta_min = st.number_input("Minimum wait for preventive maintenance after regular inspection")
delta_lim = st.number_input("Regular waiting time to provide resources for preventive maintenance")

Dp = st.number_input("Downtime for preventive (Dp)")
Df = st.number_input("Downtime for corrective (Df)")


def policy(L,M,N,T,delta,eta_x,beta_x,eta_h,beta_h,lbda,Cp,Cop,Ci,Coi,Cf,C1,C2,C3,Dp,Df,delta_lim):
    
    Z = int(delta / T)
    Y = max(0, N - Z - 1)
    
    # Cost of early preventive replacement after time-lag delta
    def Cep(time_lag):
        if time_lag <= delta_lim:
            Cep = C1*time_lag + C2
        else:
            Cep = C3
        return (Cep)
    
    
    # Functions for X (time to defect arrival)
    def fx(x):
        return (beta_x / eta_x) * ((x / eta_x) ** (beta_x - 1)) * np.exp(-((x / eta_x) ** beta_x))
    def Rx(x):
        return np.exp(-((x / eta_x) ** beta_x))
    def Fx(x):
        return 1 - np.exp(-((x / eta_x) ** beta_x))

    # Functions for H (delay-time)
    def fh(h):
        return (beta_h / eta_h) * ((h / eta_h) ** (beta_h - 1)) * np.exp(-((h / eta_h) ** beta_h))
    def Rh(h):
        return np.exp(-((h / eta_h) ** beta_h))
    def Fh(h):
        return 1 - np.exp(-((h / eta_h) ** beta_h))

    # Functions for W (time between two consecutive opportunities)
    def fw(w):
        return lbda * np.exp(- lbda * w)
    def Rw(w):
        return np.exp(- lbda * w)
    def Fw(w):
        return 1 - np.exp(- lbda * w)
    
    def scenario_1(): 
        # Preventive replacement at NT, with system in good state
        P1 = Rx(N*T)*Rw((N-M)*T)
        EC1 = (Y*Ci + (M-L)*T*lbda*Coi + Cp)*P1
        EV1 = (N*T + Dp)*P1
        ED1 = Dp*P1
        return (P1, EC1, EV1, ED1)
    
    def scenario_2():
        # Opportunistic preventive replacement between MT and NT, with system in good state
        if (M < N) and (M < Y):
            P2_1 = 0; EC2_1 = 0
            for i in range(1, Y-M+1):
                prob2_1 = quad(lambda w: fw(w)*Rx(M*T + w), (i-1)*T, i*T)[0] 
                P2_1 = P2_1 + prob2_1
                EC2_1 = EC2_1 + ((M+i-1)*Ci + (M-L)*T*lbda*Coi + Cop)*prob2_1
            
            P2_2 = quad(lambda w: fw(w)*Rx(M*T + w), (Y-M)*T, (N-M)*T)[0]
            EC2_2 = (Y*Ci + (M-L)*T*lbda*Coi + Cop)*P2_2
            
            P2 = P2_1 + P2_2
            EC2 = EC2_1 + EC2_2
            
            EV2 = quad(lambda w: (M*T + w + Dp)*fw(w)*Rx(M*T + w), 0, (N-M)*T)[0]
            ED2 = Dp*P2
            
        if (M < N) and (M >= Y):
            P2 = quad(lambda w: fw(w)*Rx(M*T + w), 0, (N-M)*T)[0]       
            EC2 = (Y*Ci + (M-L)*T*lbda*Coi + Cop)*P2
            EV2 = quad(lambda w: (M*T + w + Dp)*fw(w)*Rx(M*T + w), 0, (N-M)*T)[0]
            ED2 = Dp*P2
        
        if (M == N):
            P2 = 0; EC2 = 0; EV2 = 0; ED2 = 0
        
        return (P2, EC2, EV2, ED2)
    
    def scenario_3():
        # Early preventive replacement after a positive in-house inspection (time lag delta)
        if (L >= 0) and (L < M) and (M < N) and (M < Y):
            P3_1 = 0; EC3_1 = 0; EV3_1 = 0
            for i in range(1, L+1):
                prob3_1 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(delta), (i-1)*T, i*T)[0]
                P3_1 = P3_1 + prob3_1
                EC3_1 = EC3_1 + (i*Ci + Cep(delta))*prob3_1
                EV3_1 = EV3_1 + (i*T + delta + Dp)*prob3_1
                
            P3_2 = 0; EC3_2 = 0; EV3_2 = 0
            for i in range(L+1, M+1):
                prob3_2 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(i*T + delta - x), (i-1)*T, i*T)[0]
                P3_2 = P3_2 + prob3_2
                EC3_2 = EC3_2 + quad(lambda x: (i*Ci + (x-L*T)*lbda*Coi + Cep(delta))*fx(x)*Rh(i*T + delta - x)*Rw(i*T + delta - x), (i-1)*T, i*T)[0]
                EV3_2 = EV3_2 + (i*T + delta + Dp)*prob3_2
            
            P3_3 = 0; EC3_3 = 0; EV3_3 = 0
            for i in range(M+1, Y+1):
                prob3_3 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(i*T + delta - M*T), (i-1)*T, i*T)[0]
                P3_3 = P3_3 + prob3_3
                EC3_3 = EC3_3 + (i*Ci + (M-L)*T*lbda*Coi + Cep(delta))*prob3_3
                EV3_3 = EV3_3 + (i*T + delta + Dp)*prob3_3
            
            P3 = P3_1 + P3_2 + P3_3
            EC3 = EC3_1 + EC3_2 + EC3_3
            EV3 = EV3_1 + EV3_2 + EV3_3
            ED3 = Dp*P3
            
        if (L >= 0) and (L < M) and (M >= Y) and (L < Y):
            P3_1 = 0; EC3_1 = 0; EV3_1 = 0
            for i in range(1, L+1):
                prob3_1 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(delta), (i-1)*T, i*T)[0]
                P3_1 = P3_1 + prob3_1
                EC3_1 = EC3_1 + (i*Ci + Cep(delta))*prob3_1
                EV3_1 = EV3_1 + (i*T + delta + Dp)*prob3_1
                
            P3_2 = 0; EC3_2 = 0; EV3_2 = 0
            for i in range(L+1, Y+1):
                prob3_2 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(i*T + delta - x), (i-1)*T, i*T)[0]
                P3_2 = P3_2 + prob3_2
                EC3_2 = EC3_2 + quad(lambda x: (i*Ci + (x-L*T)*lbda*Coi + Cep(delta))*fx(x)*Rh(i*T + delta - x)*Rw(i*T + delta - x), (i-1)*T, i*T)[0]
                EV3_2 = EV3_2 + (i*T + delta + Dp)*prob3_2
            
            P3 = P3_1 + P3_2
            EC3 = EC3_1 + EC3_2
            EV3 = EV3_1 + EV3_2
            ED3 = Dp*P3
            
        if (L >= 0) and (L == M) and (M < Y):
            P3_1 = 0; EC3_1 = 0; EV3_1 = 0
            for i in range(1, L+1):
                prob3_1 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(delta), (i-1)*T, i*T)[0]
                P3_1 = P3_1 + prob3_1
                EC3_1 = EC3_1 + (i*Ci + Cep(delta))*prob3_1
                EV3_1 = EV3_1 + (i*T + delta + Dp)*prob3_1
            
            P3_3 = 0; EC3_3 = 0; EV3_3 = 0
            for i in range(M+1, Y+1):
                prob3_3 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(i*T + delta - M*T), (i-1)*T, i*T)[0]
                P3_3 = P3_3 + prob3_3
                EC3_3 = EC3_3 + (i*Ci + (M-L)*T*lbda*Coi + Cep(delta))*prob3_3
                EV3_3 = EV3_3 + (i*T + delta + Dp)*prob3_3
            
            P3 = P3_1 + P3_3
            EC3 = EC3_1 + EC3_3
            EV3 = EV3_1 + EV3_3
            ED3 = Dp*P3
            
        if (L >= Y) and (Y >= 1):
            P3 = 0; EC3 = 0; EV3 = 0
            for i in range(1, Y+1):
                prob3 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(delta), (i-1)*T, i*T)[0]
                P3 = P3 + prob3
                EC3 = EC3 + (i*Ci + Cep(delta))*prob3
                EV3 = EV3 + (i*T + delta + Dp)*prob3
            ED3 = Dp*P3
            
        if (Y == 0):
            P3 = 0
            EC3 = 0
            EV3 = 0
            ED3 = 0
        
        return (P3, EC3, EV3, ED3)
    
    def scenario_4():
        #Opportunistic preventive replacement of a defective system
        if (L >= 0) and (L < M) and (M < N) and (M < Y):
            P4_1 = 0; EC4_1 = 0; EV4_1 = 0
            P4_2 = 0; EC4_2 = 0; EV4_2 = 0
            for i in range(1, L+1):
                #prob4_1 = 0
                prob4_2 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                #P4_1 = P4_1 + prob4_1
                P4_2 = P4_2 + prob4_2
                #EC4_1 = EC4_1 + ((i-1)*Ci + Cop)*prob4_1
                EC4_2 = EC4_2 + (i*Ci + Cop)*prob4_2
                #EV4_1 = EV4_1 + dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV4_2 = EV4_2 + dblquad(lambda w, x: (i*T + w + Dp)*fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                      
            P4_3 = sum(
                dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,M+1))
            P4_4 = sum(
                dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,M+1))
            EC4_3 = sum(
                dblquad(lambda w, x: ((i-1)*Ci + (x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), 
                        (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,M+1))
            EC4_4 = sum(
                dblquad(lambda w, x: (i*Ci + (x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), 
                        (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,M+1))
            EV4_3 = sum(
                dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,M+1))
            EV4_4 = sum(
                dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,M+1))
            
            P4_5 = 0; EC4_5 = 0; EV4_5 = 0
            P4_6 = 0; EC4_6 = 0; EV4_6 = 0
            for i in range(M+1, Y+1):
                prob4_5 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: x-M*T, lambda x: (i-M)*T)[0]
                prob4_6 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: (i-M)*T, lambda x: (i-M)*T+delta)[0]
                P4_5 = P4_5 + prob4_5
                P4_6 = P4_6 + prob4_6
                EC4_5 = EC4_5 + ((i-1)*Ci + (M-L)*T*lbda*Coi + Cop)*prob4_5
                EC4_6 = EC4_6 + (i*Ci + (M-L)*T*lbda*Coi + Cop)*prob4_6
                EV4_5 = EV4_5 + dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: x-M*T, lambda x: (i-M)*T)[0]
                EV4_6 = EV4_6 + dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: (i-M)*T, lambda x: (i-M)*T+delta)[0]

            P4_7 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), Y*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            EC4_7 = (Y*Ci + (M-L)*T*lbda*Coi + Cop)*P4_7
            EV4_7 = dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), Y*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
                 
            P4 = P4_1 + P4_2 + P4_3 + P4_4 + P4_5 + P4_6 + P4_7
            EC4 = EC4_1 + EC4_2 + EC4_3 + EC4_4 + EC4_5 + EC4_6 + EC4_7
            EV4 = EV4_1 + EV4_2 + EV4_3 + EV4_4 + EV4_5 + EV4_6 + EV4_7
            ED4 = Dp*P4
            
        if (L >= 0) and (L < M) and (M >= Y) and (Y > L):
            P4_1 = 0; EC4_1 = 0; EV4_1 = 0
            P4_2 = 0; EC4_2 = 0; EV4_2 = 0
            for i in range(1, L+1):
                #prob4_1 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob4_2 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                #P4_1 = P4_1 + prob4_1
                P4_2 = P4_2 + prob4_2
                #EC4_1 = EC4_1 + ((i-1)*Ci + Cop)*prob4_1
                EC4_2 = EC4_2 + (i*Ci + Cop)*prob4_2
                #EV4_1 = EV4_1 + dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV4_2 = EV4_2 + dblquad(lambda w, x: (i*T + w + Dp)*fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                      
            P4_3 = sum(
                dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,Y+1))
            P4_4 = sum(
                dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,Y+1))
            EC4_3 = sum(
                dblquad(lambda w, x: ((i-1)*Ci + (x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), 
                        (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,Y+1))
            EC4_4 = sum(
                dblquad(lambda w, x: (i*Ci + (x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), 
                        (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,Y+1))
            EV4_3 = sum(
                dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,Y+1))
            EV4_4 = sum(
                dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,Y+1))
            
            
            P4_5 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), Y*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            P4_6 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), M*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            EC4_5 = dblquad(lambda w, x: (Y*Ci + (x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), Y*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EC4_6 = (Y*Ci + (M-L)*T*lbda*Coi + Cop)*P4_6
            EV4_5 = dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), Y*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EV4_6 = dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), M*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
                
            P4 = P4_1 + P4_2 + P4_3 + P4_4 + P4_5 + P4_6
            EC4 = EC4_1 + EC4_2 + EC4_3 + EC4_4 + EC4_5 + EC4_6
            EV4 = EV4_1 + EV4_2 + EV4_3 + EV4_4 + EV4_5 + EV4_6 
            ED4 = Dp*P4
            
        if (L >= 0) and (L == M) and (M < Y):
            P4_1 = 0; EC4_1 = 0; EV4_1 = 0
            P4_2 = 0; EC4_2 = 0; EV4_2 = 0
            for i in range(1, L+1):
                #prob4_1 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob4_2 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                #P4_1 = P4_1 + prob4_1
                P4_2 = P4_2 + prob4_2
                #EC4_1 = EC4_1 + ((i-1)*Ci + Cop)*prob4_1
                EC4_2 = EC4_2 + (i*Ci + Cop)*prob4_2
                #EV4_1 = EV4_1 + dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV4_2 = EV4_2 + dblquad(lambda w, x: (i*T + w + Dp)*fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
            
            P4_3 = 0; EC4_3 = 0; EV4_3 = 0
            P4_4 = 0; EC4_4 = 0; EV4_4 = 0
            for i in range(L+1, Y+1):
                prob4_3 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: x-M*T, lambda x: (i-M)*T)[0]
                prob4_4 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: (i-M)*T, lambda x: (i-M)*T+delta)[0]
                P4_3 = P4_3 + prob4_3
                P4_4 = P4_4 + prob4_4
                EC4_3 = EC4_3 + ((i-1)*Ci + Cop)*prob4_3
                EC4_4 = EC4_4 + (i*Ci + Cop)*prob4_4
                EV4_3 = EV4_3 + dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: x-M*T, lambda x: (i-M)*T)[0]
                EV4_4 = EV4_4 + dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: (i-M)*T, lambda x: (i-M)*T+delta)[0] 
                
            P4_5 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), Y*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            EC4_5 = (Y*Ci + Cop)*dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), Y*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            EV4_5 = dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), Y*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            
            P4 = P4_1 + P4_2 + P4_3 + P4_4 + P4_5
            EC4 = EC4_1 + EC4_2 + EC4_3 + EC4_4 + EC4_5
            EV4 = EV4_1 + EV4_2 + EV4_3 + EV4_4 + EV4_5
            ED4 = Dp*P4
           
        if (Y >= 1) and (Y <= L):
            P4_1 = 0; EC4_1 = 0; EV4_1 = 0
            P4_2 = 0; EC4_2 = 0; EV4_2 = 0
            for i in range(1, Y+1):
                #prob4_1 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob4_2 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                #P4_1 = P4_1 + prob4_1
                P4_2 = P4_2 + prob4_2
                #EC4_1 = EC4_1 + ((i-1)*Ci + Cop)*prob4_1
                EC4_2 = EC4_2 + (i*Ci + Cop)*prob4_2
                #EV4_1 = EV4_1 + dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV4_2 = EV4_2 + dblquad(lambda w, x: (i*T + w + Dp)*fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                
            P4_3 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(L*T+w-x), Y*T, L*T, lambda x: 0, lambda x: (N-L)*T)[0]
            P4_4 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            P4_5 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), M*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            P4 = P4_1 + P4_2 + P4_3 + P4_4 + P4_5
            
            EC4_3 = (Y*Ci + Cop)*P4_3
            EC4_4 = dblquad(lambda w, x: (Y*Ci + (x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EC4_5 = (Y*Ci + (M-L)*T*lbda*Coi + Cop)*P4_5
            EC4 = EC4_1 + EC4_2 + EC4_3 + EC4_4 + EC4_5
            
            EV4_3 = dblquad(lambda w, x: (L*T + w + Dp)*fx(x)*fw(w)*Rh(L*T+w-x), Y*T, L*T, lambda x: 0, lambda x: (N-L)*T)[0]
            EV4_4 = dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EV4_5 = dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), M*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            EV4 = EV4_1 + EV4_2 + EV4_3 + EV4_4 + EV4_5
            
            ED4 = Dp*P4
              
        if (Y == 0):
            P4_1 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(L*T+w-x), 0, L*T, lambda x: 0, lambda x: (N-L)*T)[0]
            P4_2 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            P4_3 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), M*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
                
            P4 = P4_1 + P4_2 + P4_3
            
            EC4_1 = Cop*P4_1
            EC4_2 = dblquad(lambda w, x: ((x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EC4_3 = ((M-L)*T*lbda*Coi + Cop)*P4_3
                
            EC4 = EC4_1 + EC4_2 + EC4_3
            
            EV4_1 = dblquad(lambda w, x: (L*T + w + Dp)*fx(x)*fw(w)*Rh(L*T+w-x), 0, L*T, lambda x: 0, lambda x: (N-L)*T)[0]
            EV4_2 = dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EV4_3 = dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), M*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
                
            EV4 = EV4_1 + EV4_2 + EV4_3
            
            ED4 = Dp*P4
        
        return (P4, EC4, EV4, ED4)
    
    def scenario_5():
        # Preventive replacement at N.T with system in defective state
        if (Y <= L):
            P5_1 = quad(lambda x: fx(x)*Rh(N*T-x)*Rw((N-L)*T), Y*T, L*T)[0]
            P5_2 = quad(lambda x: fx(x)*Rh(N*T-x)*Rw(N*T-x), L*T, M*T)[0]
            P5_3 = quad(lambda x: fx(x)*Rh(N*T-x)*Rw((N-M)*T), M*T, N*T)[0]
            
            P5 = P5_1 + P5_2 + P5_3
            
            EC5_1 = (Y*Ci + Cp)*P5_1
            EC5_2 = quad(lambda x: (Y*Ci + (x - L*T)*lbda*Coi + Cp)*fx(x)*Rh(N*T-x)*Rw(N*T-x), L*T, M*T)[0]
            EC5_3 = (Y*Ci + (M-L)*T*lbda*Coi + Cp)*P5_3
            
            EC5 = EC5_1 + EC5_2 + EC5_3
            
            EV5 = (N*T + Dp)*P5
            ED5 = Dp*P5
            
        if (L < Y) and (Y <= M):
            P5_1 = quad(lambda x: fx(x)*Rh(N*T-x)*Rw(N*T-x), Y*T, M*T)[0]
            P5_2 = quad(lambda x: fx(x)*Rh(N*T-x)*Rw((N-M)*T), M*T, N*T)[0]
            
            P5 = P5_1 + P5_2
            
            EC5_1 = quad(lambda x: (Y*Ci + (x - L*T)*lbda*Coi + Cp)*fx(x)*Rh(N*T-x)*Rw(N*T-x), Y*T, M*T)[0]
            EC5_2 = (Y*Ci + (M-L)*T*lbda*Coi + Cp)*P5_2
            
            EC5 = EC5_1 + EC5_2
            
            EV5 = (N*T + Dp)*P5
            ED5 = Dp*P5
            
        if (Y >= M):
            P5 = quad(lambda x: fx(x)*Rh(N*T-x)*Rw((N-M)*T), Y*T, N*T)[0]

            EC5 = (Y*Ci + (M-L)*T*lbda*Coi + Cp)*P5
            
            EV5 = (N*T + Dp)*P5
            ED5 = Dp*P5
            
        return(P5, EC5, EV5, ED5)
    
    def scenario_6():
        if (L >= 0) and (L < M) and (M < N) and (M < Y):
            P6_1 = 0; EC6_1 = 0; EV6_1 = 0
            P6_2 = 0; EC6_2 = 0; EV6_2 = 0
            for i in range(1, L+1):
                prob6_1 = dblquad(lambda h, x: fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob6_2 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                P6_1 = P6_1 + prob6_1
                P6_2 = P6_2 + prob6_2
                EC6_1 = EC6_1 + ((i-1)*Ci + Cf)*prob6_1
                EC6_2 = EC6_2 + (i*Ci + Cf)*prob6_2
                EV6_1 = EV6_1 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV6_2 = EV6_2 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
            
            P6_3 = sum(
                dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,M+1))
            P6_4 = sum(
                dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,M+1))
            EC6_3 = sum(
                dblquad(lambda h, x: ((i-1)*Ci + (x- L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), 
                        (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,M+1))
            EC6_4 = sum(
                dblquad(lambda h, x: (i*Ci + (x- L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), 
                        (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,M+1))
            EV6_3 = sum(
                dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,M+1))
            EV6_4 = sum(
                dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,M+1))
            
            P6_5 = 0; EC6_5 = 0; EV6_5 = 0
            P6_6 = 0; EC6_6 = 0; EV6_6 = 0
            for i in range(M+1, Y+1):
                prob6_5 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob6_6 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                P6_5 = P6_5 + prob6_5
                P6_6 = P6_6 + prob6_6
                EC6_5 = EC6_5 + ((i-1)*Ci + (M-L)*T*lbda*Coi + Cf)*prob6_5
                EC6_6 = EC6_6 + (i*Ci + (M-L)*T*lbda*Coi + Cf)*prob6_6
                EV6_5 = EV6_5 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV6_6 = EV6_6 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]

            P6_7 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), Y*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            P6 = P6_1 + P6_2 + P6_3 + P6_4 + P6_5 + P6_6 + P6_7
            
            EC6_7 = (Y*Ci + (M-L)*T*lbda*Coi + Cf)*P6_7
            EC6 = EC6_1 + EC6_2 + EC6_3 + EC6_4 + EC6_5 + EC6_6 + EC6_7
            
            EV6_7 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), Y*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6 = EV6_1 + EV6_2 + EV6_3 + EV6_4 + EV6_5 + EV6_6 + EV6_7
            
            ED6 = Df*P6
            
        if (L >= 0) and (L < M) and (M >= Y) and (Y > L):
            P6_1 = 0; EC6_1 = 0; EV6_1 = 0
            P6_2 = 0; EC6_2 = 0; EV6_2 = 0
            for i in range(1, L+1):
                prob6_1 = dblquad(lambda h, x: fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob6_2 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                P6_1 = P6_1 + prob6_1
                P6_2 = P6_2 + prob6_2
                EC6_1 = EC6_1 + ((i-1)*Ci + Cf)*prob6_1
                EC6_2 = EC6_2 + (i*Ci + Cf)*prob6_2
                EV6_1 = EV6_1 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV6_2 = EV6_2 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
            
            P6_3 = sum(
                dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,Y+1))
            P6_4 = sum(
                dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,Y+1))
            P6_5 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), Y*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            P6_6 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), M*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            P6 = P6_1 + P6_2 + P6_3 + P6_4 + P6_5 + P6_6
            
            EC6_3 = sum(
                dblquad(lambda h, x: ((i-1)*Ci + (x- L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), 
                        (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,Y+1))
            EC6_4 = sum(
                dblquad(lambda h, x: (i*Ci + (x- L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), 
                        (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,Y+1))
            EC6_5 = dblquad(lambda h, x: (Y*Ci + (x - L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), Y*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EC6_6 = (Y*Ci + (M-L)*T*lbda*Coi + Cf)*P6_6
            EC6 = EC6_1 + EC6_2 + EC6_3 + EC6_4 + EC6_5 + EC6_6
            
            EV6_3 = sum(
                dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,Y+1))
            EV6_4 = sum(
                dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,Y+1))
            EV6_5 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), Y*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6_6 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), M*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6 = EV6_1 + EV6_2 + EV6_3 + EV6_4 + EV6_5 + EV6_6
            
            ED6 = Df*P6
            
        if (L >= 0) and (L == M) and (M < Y):
            P6_1 = 0; EC6_1 = 0; EV6_1 = 0
            P6_2 = 0; EC6_2 = 0; EV6_2 = 0
            for i in range(1, L+1):
                prob6_1 = dblquad(lambda h, x: fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob6_2 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                P6_1 = P6_1 + prob6_1
                P6_2 = P6_2 + prob6_2
                EC6_1 = EC6_1 + ((i-1)*Ci + Cf)*prob6_1
                EC6_2 = EC6_2 + (i*Ci + Cf)*prob6_2
                EV6_1 = EV6_1 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV6_2 = EV6_2 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
            
            P6_5 = 0; EC6_5 = 0; EV6_5 = 0
            P6_6 = 0; EC6_6 = 0; EV6_6 = 0
            for i in range(M+1, Y+1):
                prob6_5 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob6_6 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                P6_5 = P6_5 + prob6_5
                P6_6 = P6_6 + prob6_6
                EC6_5 = EC6_5 + ((i-1)*Ci + (M-L)*T*lbda*Coi + Cf)*prob6_5
                EC6_6 = EC6_6 + (i*Ci + (M-L)*T*lbda*Coi + Cf)*prob6_6
                EV6_5 = EV6_5 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV6_6 = EV6_6 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]

            P6_7 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), Y*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            P6 = P6_1 + P6_2 + P6_5 + P6_6 + P6_7
            
            EC6_7 = (Y*Ci + (M-L)*T*lbda*Coi + Cf)*P6_7
            EC6 = EC6_1 + EC6_2 + EC6_5 + EC6_6 + EC6_7
            
            EV6_7 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), Y*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6 = EV6_1 + EV6_2 + EV6_5 + EV6_6 + EV6_7
            
            ED6 = Df*P6
            
        if (Y >= 1) and (Y <= L):
            P6_1 = 0; EC6_1 = 0; EV6_1 = 0
            P6_2 = 0; EC6_2 = 0; EV6_2 = 0
            for i in range(1,Y+1):
                prob6_1 = dblquad(lambda h, x: fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob6_2 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                P6_1 = P6_1 + prob6_1
                P6_2 = P6_2 + prob6_2
                EC6_1 = EC6_1 + ((i-1)*Ci + Cf)*prob6_1
                EC6_2 = EC6_2 + (i*Ci + Cf)*prob6_2
                EV6_1 = EV6_1 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV6_2 = EV6_2 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]

            P6_3 = dblquad(lambda h, x: fx(x)*fh(h), Y*T, L*T, lambda x: 0, lambda x: L*T-x)[0]
            P6_4 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-L*T), Y*T, L*T, lambda x: L*T-x, lambda x: N*T-x)[0]
            P6_5 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            P6_6 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), M*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            P6 = P6_1 + P6_2 + P6_3 + P6_4 + P6_5 + P6_6
            
            EC6_3 = (Y*Ci + Cf)*P6_3
            EC6_4 = (Y*Ci + Cf)*P6_4
            EC6_5 = dblquad(lambda h, x: (Y*Ci + (x-L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EC6_6 = (Y*Ci + (M-L)*T*lbda*Coi + Cf)*P6_6
            EC6 = EC6_1 + EC6_2 + EC6_3 + EC6_4 + EC6_5 + EC6_6
            
            EV6_3 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h), Y*T, L*T, lambda x: 0, lambda x: L*T-x)[0]
            EV6_4 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-L*T), Y*T, L*T, lambda x: L*T-x, lambda x: N*T-x)[0]
            EV6_5 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6_6 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), M*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6 = EV6_1 + EV6_2 + EV6_3 + EV6_4 + EV6_5 + EV6_6
            
            ED6 = Df*P6
            
        if (Y == 0):
            P6_1 = dblquad(lambda h, x: fx(x)*fh(h), 0, L*T, lambda x: 0, lambda x: L*T-x)[0]
            P6_2 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-L*T), 0, L*T, lambda x: L*T-x, lambda x: N*T-x)[0]
            P6_3 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            P6_4 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), M*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            P6 = P6_1 + P6_2 + P6_3 + P6_4
            
            EC6_1 = Cf*P6_1
            EC6_2 = Cf*P6_2
            EC6_3 = dblquad(lambda h, x: ((x-L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EC6_4 = ((M-L)*T*lbda*Coi + Cf)*P6_4
            EC6 = EC6_1 + EC6_2 + EC6_3 + EC6_4
            
            EV6_1 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h), 0, L*T, lambda x: 0, lambda x: L*T-x)[0]
            EV6_2 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-L*T), 0, L*T, lambda x: L*T-x, lambda x: N*T-x)[0]
            EV6_3 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6_4 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), M*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6 = EV6_1 + EV6_2 + EV6_3 + EV6_4
            
            ED6 = Df*P6
            
        return (P6, EC6, EV6, ED6)
    
    (P1, EC1, EV1, ED1) = scenario_1()
    (P2, EC2, EV2, ED2) = scenario_2()
    (P3, EC3, EV3, ED3) = scenario_3()        
    (P4, EC4, EV4, ED4) = scenario_4()        
    (P5, EC5, EV5, ED5) = scenario_5()        
    (P6, EC6, EV6, ED6) = scenario_6()
    
    P_total = P1 + P2 + P3 + P4 + P5 + P6
    EC = EC1 + EC2 + EC3 + EC4 + EC5 + EC6
    EV = EV1 + EV2 + EV3 + EV4 + EV5 + EV6
    ED = ED1 + ED2 + ED3 + ED4 + ED5 + ED6
    
    cost_rate = EC/EV
    MTBOF = EV/P6
    availability = 1 - (ED/EV)
    
    print('solution:', 'L:', L, 'M:', M, 'N:', N, 'T:', T, 'Delta:', delta)
    print('cost-rate:', cost_rate, 'MTBOF:', MTBOF)
    
    return (P_total, EC, EV, ED, cost_rate, MTBOF, availability, P1, P2, P3, P4, P5, P6)


def optimal_1(eta_x,beta_x,eta_h,beta_h,lbda,Cp,Cop,Ci,Coi,Cf,C1,C2,C3,Dp,Df,delta_lim,N_max,T_max,delta_min,pop,gen,mut,eli,mig,mov):
    
    def Cep(time_lag):
        if time_lag <= delta_lim:
            Cep = C1*time_lag + C2
        else:
            Cep = C3
        return (Cep)
    #this cost will be used to support solutions generation
    
    def new_solution():
        N = rd.randint(1,N_max+1)
        M = rd.randint(0,N+1)
        L = rd.randint(0,M+1)
        T = rd.uniform(delta_min/N,2*(T_max/N))
        #defining delta
        Cep_T = Cep(T)
        coin = rd.uniform(0,1)
        if abs((Cep_T-Cp)/Cp) <= 0.05:
            if (coin <= 0.5) and (T > delta_min):
                delta = rd.uniform(delta_min,T)
            else:
                delta = rd.uniform(delta_min,N*T)
        else:
            delta = rd.uniform(delta_min,N*T)
        
        cost_rate = policy(L,M,N,T,delta,eta_x,beta_x,eta_h,beta_h,lbda,Cp,Cop,Ci,Coi,Cf,C1,C2,C3,Dp,Df,delta_lim)[4]
        new_solution = [L,M,N,T,delta,cost_rate]
        return new_solution
    
    def son_(population, father_1, father_2):
        population_ = population.copy()
        pop_dvs = [line[:-1] for line in population_]
        f1 = population_[father_1].copy()
        f2 = population_[father_2].copy()
        stop = 0
        while stop == 0:
            
            #CROSSOVER
            cross_1 = rd.randint(0,5)
            cross_2 = rd.randint(0,5)
            f1[cross_1] = f2[cross_1]
            f1[cross_2] = f2[cross_2]
            
            #MUTATION
            coin = rd.uniform(0,1)
            if coin <= mut:  #mutation
                mutate = rd.randint(0,5)
                if mutate == 0:  #mutate L
                    f1[0] = rd.randint(0,f1[1]+1)
                elif mutate == 1:  #mutate M
                    f1[1] = rd.randint(0,f1[2]+1)
                elif mutate == 2:  #mutate N
                    f1[2] = rd.randint(1,N_max+1)
                elif mutate == 3:  #mutate T
                    f1[3] = rd.uniform(delta_min/f1[2],2*(T_max/f1[2]))
                else: #mutate delta
                    Cep_T = Cep(f1[3])
                    coin_delta = rd.uniform(0,1)
                    if abs((Cep_T-Cp)/Cp) <= 0.05:
                        if (coin_delta <= 0.5) and (f1[3] > delta_min):
                            delta = rd.uniform(delta_min,f1[3])
                        else:
                            delta = rd.uniform(delta_min,f1[2]*f1[3])
                    else:
                        delta = rd.uniform(delta_min,f1[2]*f1[3])

            #solution feasibility
            if f1[1] > f1[2]: #adapting M in order to ensure M <= N
                f1[1] = rd.randint(0,f1[2]+1)
            if f1[0] > f1[1]: #adapting L in order to ensure L <= M
                f1[0] = rd.randint(0,f1[1]+1)
            if f1[4] > f1[2]*f1[3]: #adapting delta in order to ensure delta <= N*T
                f1[4] = rd.uniform(delta_min,f1[2]*f1[3])
            
            L = f1[0]; M = f1[1]; N = f1[2]; T = f1[3]; delta = f1[4]
            if [L,M,N,T,delta] not in pop_dvs:
                cost_rate = policy(L,M,N,T,delta,eta_x,beta_x,eta_h,beta_h,lbda,Cp,Cop,Ci,Coi,Cf,C1,C2,C3,Dp,Df,delta_lim)[4]
                f1[5] = cost_rate
                stop = 1 #new son is defined
                
            print('tentativa son')
        return f1
            
    def iterate_ga(population):
        pop_ = population.copy()
        print('pop_',pop_)
        for i in range(0,mig):
            #migrant
            [L,M,N,T,delta,cost_rate] = new_solution()
            pop_.append([L,M,N,T,delta,cost_rate])
            
        n_pop_ = len(pop_)
        for i in range(0,pop-mig):
            #father 1 comes from the elite group
            father_1 = rd.randint(0,eli)
            father_2 = father_1
            while father_2 == father_1:
                father_2 = rd.randint(0,n_pop_)
            new_son = son_(pop_,father_1,father_2)
            pop_.append(new_son)
        pop_ = sorted(pop_, key=lambda x: x[5])
        pop_cut = []
        for i in range(0,eli):
            pop_cut.append(pop_[i])
        
        complete = 0
        verify = []
        while complete < pop-eli:
            j = rd.randint(eli,len(pop_))
            if j not in verify:
                pop_cut.append(pop_[j])
                verify.append(j)
                complete = complete + 1
            print('tentativa complete')

        pop_cut = sorted(pop_cut, key=lambda x: x[5])
        return pop_cut
    
    #initial population
    pop_ini = []
    for i in range(0,pop):
        (L,M,N,T,delta,cost_rate) = new_solution()
        pop_ini.append([L,M,N,T,delta,cost_rate])
    pop_ini = sorted(pop_ini, key=lambda x: x[5])
    
    for i in range(0,gen):
        pop_ini = iterate_ga(pop_ini)
        print('iteracao',i,'pop',pop_ini)
        
    def pso(mov):
        elite_x = []
        elite_h = []
        elite_v = []
        for i in range(0,eli):
            elite_x.append(list(pop_ini[i]))  
            elite_h.append(list(pop_ini[i])) #record the best position of each particule
            elite_v.append([0,0,0,0,0])
        for i in range(0,mov):
            #index_min = np.argmin(elite_x[:,5])
            #g_best = elite_x[index_min]
            g_best = min(elite_h, key=lambda x: x[5])
            print('mov', i, 'g_best', g_best)
            for j in range(0,eli):
                l_best = elite_h[j]
                for k in range(0,5):
                    r1 = rd.uniform(0,1)
                    r2 = rd.uniform(0,1)
                    elite_v[j][k] = 0.1*elite_v[j][k] + 1.2*r1*(g_best[k]-elite_x[j][k]) + 1.2*r2*(l_best[k]-elite_x[j][k])
                    elite_x[j][k] = max(elite_x[j][k] + elite_v[j][k], 0)
                #solution adjustement
                elite_x[j][0] = int(elite_x[j][0])
                elite_x[j][1] = int(elite_x[j][1])
                elite_x[j][2] = int(elite_x[j][2])
                if elite_x[j][1] > elite_x[j][2]:
                    elite_x[j][1] = rd.randint(0,elite_x[j][2]+1)
                if elite_x[j][0] > elite_x[j][1]:
                    elite_x[j][0] = rd.randint(0,elite_x[j][1]+1)
                if elite_x[j][3] <= 0:
                    elite_x[j][3] = rd.uniform(0,T_max)
                if (elite_x[j][4] > (elite_x[j][2]*elite_x[j][3])) or (elite_x[j][4] <= delta_min):
                    elite_x[j][4] = rd.uniform(delta_min,elite_x[j][2]*elite_x[j][3])
                L = elite_x[j][0]; M = elite_x[j][1]; N = elite_x[j][2]; T = elite_x[j][3]; delta = elite_x[j][4]
                cr_calc = policy(L, M, N, T, delta, eta_x, beta_x, eta_h, beta_h, lbda, Cp, Cop, Ci, Coi, Cf, C1, C2, C3, Dp, Df, delta_lim)[4]
                elite_x[j][5] = cr_calc
                if float(elite_x[j][5]) <= float(elite_h[j][5]):
                    elite_h[j] = list(elite_x[j])
        final_solution = min(elite_h, key=lambda x: x[5])
        return final_solution
    
    final_solution = pso(mov)
    
    return(final_solution)


#Cost of early replacement after time lag delta
# Cep(delta) = C1 + C2*exp(-C3*delta)
C1 = (Cp - Cep_max)/(delta_lim - delta_min)
C2 = Cep_max - C1*delta_min
C3 = Cp

# Solutions tested
N_max = 1*(Cf/Ci)
T_max = 2*(eta_x+eta_h)

#paremeters GA
pop = 50
gen = 10
mut = 0.1
eli = 15
mig = 5
mov = 10

recomendation = optimal_1(eta_x,beta_x,eta_h,beta_h,lbda,Cp,Cop,Ci,Coi,Cf,C1,C2,C3,Dp,Df,delta_lim,N_max,T_max,delta_min,pop,gen,mut,eli,mig,mov)


# Executar
if st.button("Get Recommendation"):
    with st.spinner('⏳ Running optimization...'):
        start = time.time()
        recommendation = optimal_1(eta_x, beta_x, eta_h, beta_h, lbda,
                                   Cp, Cop, Ci, Coi, Cf,
                                   C1, C2, C3,
                                   Dp, Df, delta_lim,
                                   N_max, T_max, delta_min,
                                   pop, gen, mut, eli, mig, mov)
        end = time.time()

    st.success("✅ Recommendation completed!")
    st.markdown(f"**Optimal Policy Parameters:**")
    st.write({
        "L": recommendation[0],
        "M": recommendation[1],
        "N": recommendation[2],
        "T": round(recommendation[3], 3),
        "Delta": round(recommendation[4], 3),
        "Cost Rate": round(recommendation[5], 5)
    })

        
    
                
    
    
    
    

