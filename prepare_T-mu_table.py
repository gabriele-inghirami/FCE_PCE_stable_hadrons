# Gabriele Inghirami - g.inghirami@gsi.de - (2020-2022) - License: GPLv.3

import fileinput
import math
import numpy as np
import sys
import os
import pickle
from scipy import optimize
from scipy import special
import scipy.integrate as integrate

boson_mu_limits=(-10,0.99) #min and max limits of mu computation for bosons: mu_limit*hadron_mass
fermion_mu_limits=(0,3) #min and max limits of mu computation for fermions: mu_limit*hadron_mass
p_limit=50 #momentum integration limit: plim*hadron_mass*mu (Gev)
one_over_twopihbarc3 = 1/(2*((np.pi)**2)*(0.197326**3))

mu_sp=[-10,0,2,10]
mu_p=[100,2000,81]
T_sp=[0.001,0.5,1,10]
T_p=[500,250,901]


def integrand_dens(k,T,mu,mass):
    #print("dens: "+str(k**2/(np.exp((np.sqrt(mass**2+k**2)-mu)/T) - s)))
    return k**2/(np.exp((np.sqrt(mass**2+k**2)-mu)/T)) 

def integrand_en(k,T,mu,mass):
    en=np.sqrt(mass**2+k**2)
    #print("ene: "+str(en))
    return en*k**2/(np.exp((en-mu)/T)) 

pnames=("pion-","pion+","pion0","kaon0","kaon+","kaon-","kaon0bar","Neutron","Proton","anti-Proton","anti-Neutron","eta","omega","eta1","phi","Lambda1116","anti-Lambda1116","Sigma1192-","Sigma1192+","Sigma1192","anti-Sigma1192-","anti-Sigma1192+","anti-Sigma1192","Xi1317-","Xi1317-0","anti-Xi1317-0","Xi1317+","Lambda1520","anti-Lambda1520","Xi1530-","Xi1530-0","anti-Xi1530-0","Xi1530+","Omega1672","anti-Omega1672","Others")
mass_urqmd=(0.138,0.138,0.138,0.494,0.494,0.494,0.494,0.938,0.938,0.938,0.938,0.547,0.782,0.958,1.019,1.116,1.116,1.192,1.192,1.192,1.192,1.192,1.192,1.317,1.317,1.317,1.317,1.520,1.520,1.530,1.530,1.530,1.530,1.672,1.672,0)
mass_pdg=(0.13957,0.13957,0.13498,0.49761,0.49368,0.49761,0.49368,0.93956,0.93827,0.93827,0.93956,0.54786,0.78265,0.9578,1.0195,1.1157,1.1157,1.1974,1.1894,1.1926,1.1894,1.974,1.1926,1.3217,1.3148,1.3148,1.3217,1.5195,1.5195,1.5350,1.5318,1.5318,1.5350,1.6724,1.6724,0)
spin=np.array((0,0,0,0,0,0,0,0.5,0.5,0.5,0.5,0,1,0,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,0),dtype=np.float64)
sfac=np.array((1,1,1,1,1,1,1,-1,-1, -1  ,-1 ,1,1,1,1,-1 ,-1 ,-1 ,-1 ,-1,-1,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,0),dtype=np.float64)                
gfac=2*spin+1
#we get the name of input and output files
N_input_args=len(sys.argv)-1

if(N_input_args!=1):
   print ('Syntax: ./prepare_T-mu_table.py <outputfile>')
   sys.exit(1)

outputfile=sys.argv[1]
tmp_arr=[]
for i in range(len(mu_p)-1):
    tmp_arr.append(np.linspace(mu_sp[i],mu_sp[i+1],mu_p[i],endpoint=False))
tmp_arr.append(np.linspace(mu_sp[-2],mu_sp[-1],mu_p[-1],endpoint=True))
mu_points=np.concatenate((tmp_arr[:]))
N_mu=len(mu_points)

tmp_arr=[]
for i in range(len(T_p)-1):
    tmp_arr.append(np.linspace(T_sp[i],T_sp[i+1],T_p[i],endpoint=False))
tmp_arr.append(np.linspace(T_sp[-2],T_sp[-1],T_p[-1],endpoint=True))
T_points=np.concatenate((tmp_arr[:]))
N_T=len(T_points)

print("List of particles that are considered:")
print("Name     mass[GeV](UrQMD)    mass[GeV](PDG)     J(sping)      g_s")
number_of_particles=len(pnames)-1
for i in range(number_of_particles):
    print(pnames[i]+"  "+str(mass_urqmd[i])+"  "+str(mass_pdg[i])+"  "+str(spin[i])+"  "+str(gfac[i]))

en_arr=np.zeros((number_of_particles,N_T,N_mu),dtype=np.float64)
rho_arr=np.zeros((number_of_particles,N_T,N_mu),dtype=np.float64)

for ff in range(len(pnames)-1):
    mass=mass_urqmd[ff]
    #statf=sfac[ff]
    gs=gfac[ff]
    if((mass==mass_urqmd[ff-1]) and (gs==gfac[ff-1])):
        en_arr[ff,:,:]=en_arr[ff-1,:,:]
        rho_arr[ff,:,:]=rho_arr[ff-1,:,:]
        print("Computations for "+pnames[ff]+" copied from "+pnames[ff-1])
    else:
      mu_min_boson=boson_mu_limits[0]*mass
      mu_max_boson=boson_mu_limits[1]*mass
      mu_min_fermion=fermion_mu_limits[0]*mass
      mu_max_fermion=fermion_mu_limits[1]*mass
      for i in range(1,N_T): #we skip T=0
          T=T_points[i]
          for j in range(N_mu):
              mu=mu_points[j]
              if(sfac[ff]>0): #it is a boson
                  if((mu<mu_min_boson) or (mu>mu_max_boson)):
                      continue
              else: #it is a fermion
                  if((mu<mu_min_fermion) or (mu>mu_max_fermion)):
                      continue
              #if((mu<mass) or (statf==-1)):
              plim=p_limit*mass*mu
              int1=integrate.quad(integrand_dens,0,plim,args=(T,mu,mass),limit=300)[0]
              int2=integrate.quad(integrand_en,0,plim,args=(T,mu,mass),limit=300)[0]
              en_arr[ff,i,j]=gs*one_over_twopihbarc3*int2
              rho_arr[ff,i,j]=gs*one_over_twopihbarc3*int1
#              print(str(T)+"    "+str(mu)+"   "+str(en_arr[ff,i,j])+"   "+str(en_arr[ff,i,j])+"   "+str(mass)+"   "+str(statf))
      print("Computations for "+pnames[ff]+" completed")
              
    outfile_hadron=outputfile+"_"+pnames[ff]+".dat"
    print("Writing the results on "+outfile_hadron)
    with open(outfile_hadron,"wb") as po:
       pickle.dump((T_points,mu_points,rho_arr[ff,:,:],en_arr[ff,:,:]),po)
    print("Done.") 
print("All done.") 
                         



