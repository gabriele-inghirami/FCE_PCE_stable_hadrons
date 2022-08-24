# Gabriele Inghirami - g.inghirami@gsi.de - (2020-2022) - License: GPLv.3

import fileinput
import math
import numpy as np
import sys
import os
import pickle
from scipy import optimize
from scipy import special
from scipy import interpolate
import scipy.integrate as integrate

plim=25 #momentum integration limit in GeV
one_over_twopihbarc3 = 1/(2*((np.pi)**2)*(0.197326**3))

#root finding tolerance error
root_tol=1.e-5

#if False it prints only error messages, if True it writes what it is doing at the moment and the intermediate results 
verbose=True

#by default this script assumes to work with UrQMD data, with smash=True it assumes to work with SMASH data, instead
smash=True

#must be set to False when using data from cg 1.5.4 or earlier
version_2_data_format=True

#tabulated data files prefix (if in a subdir of the current directory, it should include its name, as well, e.g., eos_data/eos if the files eos_pion0.dat, eos_neutron.dat... are inside eos_data
tabdata_prefix="run_table_python_20.7.20/tab_eos"

#it uses interpolation methods instead of computing the integrals whenever possible
use_interpolation=False

#minimum number of a particle species to accept a cell:
particle_count_for_acceptance=40

#cells_to_evaluate can be a tuple of tuples of coordinate points ((x1,y1,z1),(x2,y2,z2),...), a number identifying the z0 coordinate of the transverse plane (x,y,z=z0) or the string "all"
#cells_to_evaluate="all"
#cells_to_evaluate=0
#cells_to_evaluate=((0,0,0),(3,3,0),(-3,-3,0))
cells_to_evaluate=((0,0,0),)

#it allows to rearrange the list of hadrons so that the thermodynamic quantities of some of them are evaluated first (by default, hadrons are listed as a growing functions of mass)
#the output files will maintain the default order, the program takes care to properly rearrange the order again at the end
use_preferred=True
#list of hadrons to be evaluated first
preferred=("kaon-","kaon+","kaon0","Neutron","Proton")

#maximum number of failures in solving the PCE for hadron types in a row before giving up (if larger than 35 no limit is effectively in place)
max_failures=100

#we parse the command line arguments
N_input_args=len(sys.argv)-1

if(N_input_args<2):
   print ('Syntax: ./store_cg.py <coarse file data 1> [coarse file data 2] ... <outputfile>')
   print ("coarse file data 1,2,3...N are the density files produced by the coarse graining code")
   print ("outputfile is obviously the name of the output file with the results of the postprocessing")
   print ("all the necessary informations about the grid and the particles can be found in the info file produced by the coarse graining code")
   sys.exit(1)


ncomp_cells=len(cells_to_evaluate)
#here we check at runtime that cells_to_evaluate has been chosen correctly, but we do not check if the chosen values are inside the computational grid
if(isinstance(cells_to_evaluate, str)):
    if(cells_to_evaluate=="all"):
      comp_all=True
    else:
      print('If you set the parameter cells_to_evaluate as a string, it can be only "all", meaning that all cells of the grid will be evaluated.\nPlease, check and run again.\n')
      sys.exit(1)
else:
    comp_all=False
    if(isinstance(cells_to_evaluate, (int, float))):
       comp_trans=True
       zcoordinate=float(cells_to_evaluate) #it is more understandable later
    else:
       comp_trans=False
       if(isinstance(cells_to_evaluate, tuple)):
          xc_eval=np.zeros(ncomp_cells,dtype=np.float64)
          yc_eval=np.zeros(ncomp_cells,dtype=np.float64)
          zc_eval=np.zeros(ncomp_cells,dtype=np.float64)
          for i in range(len(cells_to_evaluate)):
              if(len(cells_to_evaluate[i])!=3):
                 print("Each tuple of coordinates must have three values (for x, y and z), but I read that the "+'{:3d}'.format(i)+"-th element is: "+str(cells_to_evaluate[i])+"\nPlease, check and run again.\n")
                 sys.exit(1)
              for j in range(3):
                  if(not(isinstance(cells_to_evaluate[i][j], (int, float)))):
                      print("Please, inside the code, the tuple inside cells_to_evaluate: "+str(cells_to_evaluate[i])+", because "+str(cells_to_evaluate[i][j])+" does not seem a valid coordinate...\n")
                      sys.exit(1)
              xc_eval[i]=np.float64(cells_to_evaluate[i][0])
              yc_eval[i]=np.float64(cells_to_evaluate[i][1])
              xc_eval[i]=np.float64(cells_to_evaluate[i][2])
       else:
          print('Sorry, but I am unable to understand what the parameter cells_to_evaluate (look into the code) is...\nIt can be the string "all", meaning that all cells of the grid will be evaluated,\nor a number with the z coordinate, meaning that only the cells on the corresponding transverse plane will be evaluated,\nor a tuple of tuples of coordinate points ((x1,y1,z1),(x2,y2,z2),...).\nPlease, check and run again.\n')
          sys.exit(1)

if((not comp_all) and (not comp_trans)):
    comp_points=True
else:
    comp_points=False



pnames=["pion-","pion+","pion0","kaon0","kaon+","kaon-","kaon0bar","Neutron","Proton","anti-Proton","anti-Neutron","eta","omega","eta1","phi","Lambda1116","anti-Lambda1116","Sigma1192-","Sigma1192+","Sigma1192","anti-Sigma1192-","anti-Sigma1192+","anti-Sigma1192","Xi1317-","Xi1317-0","anti-Xi1317-0","Xi1317+","Lambda1520","anti-Lambda1520","Xi1530-","Xi1530-0","anti-Xi1530-0","Xi1530+","Omega1672","anti-Omega1672","Others"]
mass_urqmd=np.array((0.138,0.138,0.138,0.494,0.494,0.494,0.494,0.938,0.938,0.938,0.938,0.547,0.782,0.958,1.019,1.116,1.116,1.192,1.192,1.192,1.192,1.192,1.192,1.317,1.317,1.317,1.317,1.520,1.520,1.530,1.530,1.530,1.530,1.672,1.672,0),dtype=np.float64)
mass_smash=np.array((0.138,0.138,0.138,0.494,0.494,0.494,0.494,0.938,0.938,0.938,0.938,0.548,0.783,0.958,1.019,1.116,1.116,1.189,1.189,1.189,1.189,1.189,1.189,1.318,1.318,1.318,1.318,1.520,1.520,1.533,1.533,1.533,1.533,1.672,1.672,0),dtype=np.float64)
if(smash):
    mass_had=mass_smash
else:
    mass_had=mass_urqmd
#mass_pdg=np.array((0.13957,0.13957,0.13498,0.49761,0.49368,0.49761,0.49368,0.93956,0.93827,0.93827,0.93956,0.54786,0.78265,0.9578,1.0195,1.1157,1.1157,1.1974,1.1894,1.1926,1.1894,1.974,1.1926,1.3217,1.3148,1.3148,1.3217,1.5195,1.5195,1.5350,1.5318,1.5318,1.5350,1.6724,1.6724,0),dtype=np.float64)
spin=np.array((0,0,0,0,0,0,0,0.5,0.5,0.5,0.5,0,1,0,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,0),dtype=np.float64)
sfac=np.array((1,1,1,1,1,1,1,-1,-1, -1  ,-1 ,1,1,1,1,-1 ,-1 ,-1 ,-1 ,-1,-1,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,0),dtype=np.float64)                
gfac=2*spin+1


baryon_rest_mass_urqmd=np.array((0.938,1.440,1.515,1.550,1.645,1.675,1.680,1.730,1.710,1.720,1.850,1.950,2.000, 2.150, 2.220,2.250,1.232,1.700,1.675,1.750,1.840,1.880, 1.900,1.920,1.970,1.990, 1.116,1.407,1.520,1.600,1.670,1.690, 1.800,1.810,1.820,1.830,1.890,2.100, 2.110, 1.192,1.384,1.660,1.670,1.750,1.775, 1.915,1.940,2.030, 1.315,1.532,1.700,1.823,1.950,2.025, 1.672),dtype=np.float64)

baryon_rest_mass_smash=np.array((0.938,1.440,1.515,1.530,1.650,1.675,1.685,1.720,1.710,1.720,1.875,1.880,1.895,1.900,1.990,2.100,2.000,2.100,2.120,2.180,2.220,2.250,   1.232,1.610,1.710,1.860,1.880,1.900,1.920,1.950,1.930,   1.116,1.405,1.520,1.600,1.670,1.690,1.800,1.810,1.820,1.830,1.890,2.100,2.110,2.350,   1.189,1.385,1.660,1.670,1.750,1.775,1.915,1.940,2.030,2.250,    1.318,1.533,1.690,1.823,1.950,2.025,   1.672,2.252),dtype=np.float64)

if(smash):
    baryon_rest_mass=baryon_rest_mass_smash
else:
    baryon_rest_mass=baryon_rest_mass_urqmd

num_baryons=len(baryon_rest_mass)
baryon_mult=np.zeros(num_baryons,dtype=np.float64)


if(smash):
  for i in range(0,num_baryons):
    if(i<22):
      baryon_mult[i]=2.
      continue
    elif(i<31):
      baryon_mult[i]=4.
      continue
    elif(i<45):
      baryon_mult[i]=1.
      continue
    elif(i<55):
      baryon_mult[i]=3.
      continue
    elif(i<61):
      baryon_mult[i]=2.
      continue
    else:
      baryon_mult[i]=1.
else:
  for i in range(0,num_baryons):
    if(i<16):
      baryon_mult[i]=2.
      continue
    elif(i<26):
      baryon_mult[i]=4.
      continue
    elif(i<39):
      baryon_mult[i]=1.
      continue
    elif(i<48):
      baryon_mult[i]=3.
      continue
    elif(i<54):
      baryon_mult[i]=2.
      continue
    else:
      baryon_mult[i]=1.

meson_rest_mass_urqmd=np.array((0.138, 0.547, 0.782, 0.769, 0.990, 0.494, 0.958, 0.893, 1.019, 1.429, 0.990, 1.370, 1.273, 1.230, 1.282, 1.426, 1.430, 1.318, 1.275, 1.525, 1.400, 1.235, 1.170, 1.386, 1.410,1.465,1.419,1.681, 1.680,1.720,1.649,1.910,1.866, 2.01, 3.097, 3.415, 3.686, 1.968, 2.112),dtype=np.float64)

meson_rest_mass_smash=np.array((0.138, 0.548, 0.800, 0.776, 0.783, 0.958, 0.990, 0.989, 1.019, 1.170, 1.2295, 1.23,  1.276, 1.2819, 1.294, 1.30, 1.3183, 1.35, 1.354, 1.409, 1.4264, 1.425, 1.474, 1.465, 1.476, 1.507, 1.525, 1.662, 1.617, 1.670, 1.667, 1.672, 1.680, 1.689, 1.720, 1.723, 1.812, 1.854, 1.944, 2.010, 1.995, 2.018, 2.297, 2.350,     0.494,0.892,1.272,1.403,1.421,1.453,1.429,1.718,1.773,1.776,1.819,2.045),dtype=np.float64)


if(smash):
    meson_rest_mass=meson_rest_mass_smash
else:
    meson_rest_mass=meson_rest_mass_urqmd

num_mesons=len(meson_rest_mass)

meson_mult=np.zeros(num_mesons,dtype=np.float64)
antimeson_mult=np.zeros(num_mesons,dtype=np.float64)

if(smash):
  meson_mult[0]=3. #π
  meson_mult[1]=1. #η
  meson_mult[2]=1. #σ
  meson_mult[3]=3. #ρ
  meson_mult[4:7]=1. #omega,eta',f₀(980)
  meson_mult[7]=3. #a₀(980)
  meson_mult[8:10]=1. #phi,h1
  meson_mult[10:12]=3. #b1,a1
  meson_mult[12:15]=1. #f2,f1,eta
  meson_mult[15:17]=3. #pi1300,a2
  meson_mult[17]=1. #f0
  meson_mult[18]=3. #pi1
  meson_mult[19:22]=1. #eta,f1,omega
  meson_mult[22:24]=3. #a0,rho
  meson_mult[24:27]=1. #eta,f0,f2'
  meson_mult[27]=3. #pi1
  meson_mult[28:31]=1. #eta2,omega,omega3
  meson_mult[31]=3. #pi2
  meson_mult[32]=1. #phi
  meson_mult[33:35]=3. #rho3,rho
  meson_mult[35]=1. #f0
  meson_mult[36]=3. #pi
  meson_mult[37:40]=1. #phi3,f2,f2
  meson_mult[40]=3. #a4
  meson_mult[41:44]=1. #f4,f2,f2

  meson_mult[44:]=3. #strange mesons
  antimeson_mult[44:]=3.

else:

  meson_mult[0]=3.
  meson_mult[1:3]=1.
  meson_mult[3]=3.
  meson_mult[4]=1.
  meson_mult[5]=2.
  meson_mult[6]=1.
  meson_mult[7]=2.
  meson_mult[8]=1.
  meson_mult[9]=2.
  meson_mult[10]=3.
  meson_mult[11]=1.
  meson_mult[12]=2.
  meson_mult[13]=3.
  meson_mult[14:16]=1.
  meson_mult[16]=2.
  meson_mult[17]=3.
  meson_mult[18:20]=1.
  meson_mult[20]=2.
  meson_mult[21]=3.
  meson_mult[22:24]=1.
  meson_mult[24]=2.
  meson_mult[25]=3.
  meson_mult[26:28]=1.
  meson_mult[28]=2.
  meson_mult[29]=3.
  meson_mult[30:32]=1.
  meson_mult[32:34]=2.
  meson_mult[34:39]=1.
  
  antimeson_mult[5]=2.
  antimeson_mult[7]=2.
  antimeson_mult[9]=2.
  antimeson_mult[12]=2.
  antimeson_mult[16]=2.
  antimeson_mult[20]=2.
  antimeson_mult[24]=2.
  antimeson_mult[28]=2.
  antimeson_mult[32:34]=2.
  antimeson_mult[37:39]=1.


strangeness_in_baryons=np.zeros(num_baryons,dtype=np.float64)

if(smash):
  for i in range(0,num_baryons):
    if(i<31):
      continue
    elif(i<55):
      strangeness_in_baryons[i]=-1.
      continue
    elif(i<61):
      strangeness_in_baryons[i]=-2.
      continue
    else:
      strangeness_in_baryons[i]=-3.
else:
  for i in range(0,num_baryons):
    if(i<26):
      continue
    elif(i<48):
      strangeness_in_baryons[i]=-1.
      continue
    elif(i<54):
      strangeness_in_baryons[i]=-2.
      continue
    else:
      strangeness_in_baryons[i]=-3.

strangeness_in_mesons=np.zeros(num_mesons,dtype=np.float64)
if(smash):
  strangeness_in_mesons[44:]=1.
else:
  strangeness_in_mesons[0:5]=0.
  strangeness_in_mesons[5]=1.
  strangeness_in_mesons[6]=0.
  strangeness_in_mesons[7]=1.
  strangeness_in_mesons[8]=0.
  strangeness_in_mesons[9]=1.
  strangeness_in_mesons[10:12]=0.
  strangeness_in_mesons[12]=1.
  strangeness_in_mesons[13:16]=0.
  strangeness_in_mesons[16]=1.
  strangeness_in_mesons[17:20]=0.
  strangeness_in_mesons[20]=1.
  strangeness_in_mesons[21:24]=0.
  strangeness_in_mesons[24]=1.
  strangeness_in_mesons[25:28]=0.
  strangeness_in_mesons[28]=1.
  strangeness_in_mesons[29:37]=0.
  strangeness_in_mesons[37:39]=1.

gs_baryon=np.zeros(num_baryons,dtype=np.float64)
if(smash):
  gs_baryon[0:2]=2.
  gs_baryon[2]=4.
  gs_baryon[3:5]=2.
  gs_baryon[5:7]=6.
  gs_baryon[7]=4.
  gs_baryon[8]=2.
  gs_baryon[9:11]=4.
  gs_baryon[11:13]=2.
  gs_baryon[13]=4.
  gs_baryon[14]=8.
  gs_baryon[15]=6.
  gs_baryon[16]=4.
  gs_baryon[17]=2.
  gs_baryon[18]=4.
  gs_baryon[19]=8.
  gs_baryon[20:22]=10.
  gs_baryon[22]=4.
  gs_baryon[23]=2.
  gs_baryon[24]=4.
  gs_baryon[25]=2.
  gs_baryon[26]=6.
  gs_baryon[27]=2.
  gs_baryon[28]=4.
  gs_baryon[29]=6.
  gs_baryon[30]=8.
  gs_baryon[31:33]=2.
  gs_baryon[33]=4.
  gs_baryon[34:36]=2.
  gs_baryon[36]=4.
  gs_baryon[37:39]=2.
  gs_baryon[39:41]=6.
  gs_baryon[41]=4.
  gs_baryon[42]=8.
  gs_baryon[43]=6.
  gs_baryon[44]=10.
  gs_baryon[45]=2.
  gs_baryon[46]=4.
  gs_baryon[47]=2.
  gs_baryon[48]=4.
  gs_baryon[49]=2.
  gs_baryon[50:52]=6.
  gs_baryon[52]=4.
  gs_baryon[53:55]=8.
  gs_baryon[55]=2.
  gs_baryon[56]=4.
  gs_baryon[57]=2.
  gs_baryon[58]=4.
  gs_baryon[59:61]=6.
  gs_baryon[61]=4.
  gs_baryon[62]=8.

else:
  gs_baryon[0:2]=2.
  gs_baryon[2]=4.
  gs_baryon[3:5]=2.
  gs_baryon[5:7]=6.
  gs_baryon[7]=4.
  gs_baryon[8]=2.
  gs_baryon[9:11]=4.
  gs_baryon[11]=8.
  gs_baryon[12]=4.
  gs_baryon[13]=8.
  gs_baryon[14:16]=10.
  gs_baryon[16:18]=4.
  gs_baryon[18]=2.
  gs_baryon[19]=4.
  gs_baryon[20]=2.
  gs_baryon[21]=6.
  gs_baryon[22]=2.
  gs_baryon[23]=4.
  gs_baryon[24]=6.
  gs_baryon[25]=8.
  gs_baryon[26:28]=2.
  gs_baryon[28]=4.
  gs_baryon[29:31]=2.
  gs_baryon[31]=4.
  gs_baryon[32:34]=2.
  gs_baryon[34:36]=6.
  gs_baryon[36]=4.
  gs_baryon[37]=8.
  gs_baryon[38]=6.
  gs_baryon[39]=2.
  gs_baryon[40]=4.
  gs_baryon[41]=2.
  gs_baryon[42]=4.
  gs_baryon[43]=2.
  gs_baryon[44:46]=6.
  gs_baryon[46]=4.
  gs_baryon[47]=8.
  gs_baryon[48]=2.
  gs_baryon[49:53]=4.
  gs_baryon[53]=6.
  gs_baryon[54]=4.

gs_meson=np.zeros(num_mesons,dtype=np.float64)
if(smash):
  gs_meson[0:3]=1
  gs_meson[3:5]=3
  gs_meson[5:8]=1
  gs_meson[8:12]=3
  gs_meson[12]=5
  gs_meson[13]=3
  gs_meson[14:16]=1
  gs_meson[16]=5
  gs_meson[17]=1
  gs_meson[18]=3
  gs_meson[19]=1
  gs_meson[20:22]=3
  gs_meson[22]=1
  gs_meson[23]=3
  gs_meson[24:26]=1
  gs_meson[26]=5
  gs_meson[27]=3
  gs_meson[28]=5
  gs_meson[29]=3
  gs_meson[30]=7
  gs_meson[31]=5
  gs_meson[32]=3
  gs_meson[33]=7
  gs_meson[34]=3
  gs_meson[35:37]=1
  gs_meson[37]=7
  gs_meson[38:40]=5
  gs_meson[40:42]=9
  gs_meson[42:44]=5
  gs_meson[44]=1 #K
  gs_meson[45:49]=3
  gs_meson[49]=1
  gs_meson[50]=5
  gs_meson[51]=3
  gs_meson[52]=5
  gs_meson[53]=7
  gs_meson[54]=5
  gs_meson[55]=9

else:
  gs_meson[0:2]=1
  gs_meson[2:4]=3
  gs_meson[4:7]=1
  gs_meson[7:9]=3
  gs_meson[9:12]=1
  gs_meson[12:16]=3
  gs_meson[16:20]=5
  gs_meson[20:32]=3
  gs_meson[32]=1
  gs_meson[33:37]=3
  gs_meson[37]=1
  gs_meson[38]=3


#we take care of the operations to rearrange the order of the hadron if a preferred order is expressed
if(use_preferred):
    new_order=[]
    for i in range(len(preferred)):
        for j in range(len(pnames)):
            if(preferred[i]==pnames[j]):
                new_order.append(j)
                break

    if(len(new_order)!=len(preferred)):
        print("Mismatch between the lenght of the dictionary with the new order of hadrons and the lenght of the list with the preferred order")
        print("Maybe there are typos in the hadron names in the 'preferred' tuple. Please, check. I quit.")
        sys.exit(2)


    for i in range(len(pnames)):
        not_inside=True
        for j in preferred:
            if(pnames[i]==j): #the hadron is in the preferred list, therefore is already included
                not_inside=False
                break
        if(not_inside):
            new_order.append(i)

    if(verbose):
        print("Hadrons according to the new order:")
        for i in range(len(pnames)):
            print(pnames[new_order[i]])

def f_to_be_zeroed(x,m,z):
    return m*(special.kn(1,m/x)/special.kn(2,m/x)+3*x/m)-z

def jacobian(x,m,z):
    r=m/x
    return -((m * special.kn(1,r))**2-special.kn(2,r)*(m**2 * special.kn(0,r) + (m**2+5*x**2) * special.kn(2,r)) + m**2 * special.kn(1,r) * special.kn(3,r))/(2*(x*special.kn(2,r))**2)

def integrand_dens(k,T,mu,mass,s):
    #print("dens: "+str(k**2/(np.exp((np.sqrt(mass**2+k**2)-mu)/T) - s)))
    return k**2/(np.exp((np.sqrt(mass**2+k**2)-mu)/T) - s) 

def integrand_en(k,T,mu,mass,s):
    en=np.sqrt(mass**2+k**2)
    #print("ene: "+str(en))
    return en*k**2/(np.exp((en-mu)/T) - s) 

def integrand_s(k,T,mu,m,s):
    en=np.sqrt(m**2+k**2)
    #f_dist=1./(np.exp((en-mu)/T) - s)
    f_dist=1./(np.exp((en-mu)/T))
    #if((f_dist<0) and verbose):
    #    print("Warning, in entropy density computation f_dist="+str(f_dist)+"!!")
    #else:
    #    return f_dist*(math.log(f_dist)-1.)*k**2
    if f_dist > 0:
        return f_dist*(math.log(f_dist)-1.)*k**2
    else:
        return 0

def check_limits(a,b): #this is just a temporary definition for the function f_system here below, we'll define properly this function later
    return False

def f_system(x,mass,energy_val,density_val,gs,statf,had_id):
    T=x[0]
    mu=x[1]   
    if(use_interpolation and check_inside(T,mu)):
            return [rho_interp[had_id](T,mu)[0]-density_val,en_interp[had_id](T,mu)[0]-energy_val]
    else:
            int1=integrate.quad(integrand_dens,0,plim,args=(T,mu,mass,statf),limit=300)[0]
            int2=integrate.quad(integrand_en,0,plim,args=(T,mu,mass,statf),limit=300)[0]
            return [gs*one_over_twopihbarc3*int1-density_val,gs*one_over_twopihbarc3*int2-energy_val]

def integrand_dens_hd_fer(k,T,mu,m):
    return k**2/(np.exp((np.sqrt(m**2+k**2)-mu)/T)+1)

def integrand_en_hd_fer(k,T,mu,m):
    en=np.sqrt(m**2+k**2)
    return en*k**2/(np.exp((en-mu)/T)+1) 

def integrand_dens_hd_bos(k,T,mu,m):
    return k**2/(np.exp((np.sqrt(m**2+k**2)-mu)/T)-1)

def integrand_en_hd_bos(k,T,mu,m):
    en=np.sqrt(m**2+k**2)
    return en*k**2/(np.exp((en-mu)/T)-1) 

def f_system_hd(x,inargs):
    T=x[0]
    muarr=x[1:]
    nv=len(muarr)
    energy_val=inargs[0]
    rho_arr=inargs[1]
    m_arr=inargs[2]
    g_arr=inargs[3]
    sfac_arr=inargs[4]
    int1=0
    int2=[]
    for i in range(nv):
        int1=int1+g_arr[i]*integrate.quad(integrand_en,0,plim,args=(T,muarr[i],m_arr[i],sfac_arr[i]),limit=300)[0]
        int2.append(one_over_twopihbarc3*g_arr[i]*integrate.quad(integrand_dens,0,plim,args=(T,muarr[i],m_arr[i],sfac_arr[i]),limit=300)[0]-rho_arr[i])
    int1=int1*one_over_twopihbarc3-energy_val
    return [int1]+int2

def f_system_fce(x,inargs):
    T_fce=x[0]
    muB_fce=x[1]
    muS_fce=x[2]
    energy_fce=inargs[0]
    rhoB_fce=inargs[1]
    rhoS_fce=inargs[2]
    int_energy=0
    int_rhoB=0
    int_rhoS=0
    if(verbose):
        print("Entering f_system_fce. Guess values for T, mu_B and mu_S: "+str(T_fce)+",  "+str(muB_fce)+",  "+str(muS_fce))
        print("Values of energy density, rhoB and rhoS to be matched: "+str(energy_fce)+",  "+str(rhoB_fce)+",  "+str(rhoS_fce))
    for i in range(num_baryons):
        chempot=muB_fce+strangeness_in_baryons[i]*muS_fce
#        print("Baryon "+str(i)+", chempot "+str(chempot))
        int_energy=int_energy+baryon_mult[i]*gs_baryon[i]*(integrate.quad(integrand_en_hd_fer,0,plim,args=(T_fce,chempot,baryon_rest_mass[i]),limit=300)[0]+integrate.quad(integrand_en_hd_fer,0,plim,args=(T_fce,-chempot,baryon_rest_mass[i]),limit=300)[0])
        density_integral=(integrate.quad(integrand_dens_hd_fer,0,plim,args=(T_fce,chempot,baryon_rest_mass[i]),limit=300)[0]-integrate.quad(integrand_dens_hd_fer,0,plim,args=(T_fce,-chempot,baryon_rest_mass[i]),limit=300)[0])
        int_rhoB=int_rhoB+baryon_mult[i]*gs_baryon[i]*density_integral
        int_rhoS=int_rhoS+baryon_mult[i]*gs_baryon[i]*strangeness_in_baryons[i]*density_integral

    for i in range(num_mesons):
        chempot=strangeness_in_mesons[i]*muS_fce
#        print("Mesons "+str(i)+", chempot "+str(chempot))
        int_energy=int_energy+gs_meson[i]*(meson_mult[i]*integrate.quad(integrand_en_hd_bos,0,plim,args=(T_fce,chempot,meson_rest_mass[i]),limit=300)[0]+antimeson_mult[i]*integrate.quad(integrand_en_hd_bos,0,plim,args=(T_fce,-chempot,meson_rest_mass[i]),limit=300)[0])
        density_integral=(meson_mult[i]*integrate.quad(integrand_dens_hd_bos,0,plim,args=(T_fce,chempot,meson_rest_mass[i]),limit=300)[0]-antimeson_mult[i]*integrate.quad(integrand_en_hd_bos,0,plim,args=(T_fce,-chempot,meson_rest_mass[i]),limit=300)[0])
        int_rhoS=int_rhoS+gs_meson[i]*strangeness_in_mesons[i]*density_integral
    if(verbose):
        print("Values computed after integration: "+str(int_energy*one_over_twopihbarc3)+",  "+str(int_rhoB*one_over_twopihbarc3)+",  "+str(int_rhoS*one_over_twopihbarc3))
    int_energy=int_energy*one_over_twopihbarc3-energy_fce
    int_rhoB=int_rhoB*one_over_twopihbarc3-rhoB_fce
    int_rhoS=int_rhoS*one_over_twopihbarc3-rhoS_fce
    return [int_energy,int_rhoB,int_rhoS]

def get_s_fce(temp,muB,muS):
    integral_of_s=0.
    for i in range(num_baryons):
        chempot=muB+strangeness_in_baryons[i]*muS
        integral_of_s=integral_of_s+baryon_mult[i]*gs_baryon[i]*(integrate.quad(integrand_s,0,plim,args=(temp,chempot,baryon_rest_mass[i],-1),limit=300)[0]+integrate.quad(integrand_s,0,plim,args=(temp,-chempot,baryon_rest_mass[i],-1),limit=300)[0])

    for i in range(num_mesons):
        chempot=strangeness_in_mesons[i]*muS
        integral_of_s=integral_of_s+meson_mult[i]*gs_meson[i]*integrate.quad(integrand_s,0,plim,args=(temp,chempot,meson_rest_mass[i],+1),limit=300)[0]+antimeson_mult[i]*gs_meson[i]*integrate.quad(integrand_s,0,plim,args=(temp,-chempot,meson_rest_mass[i],+1),limit=300)[0]
    return -integral_of_s*one_over_twopihbarc3

def try_with_tab(rho_val,en_val,p):
    #first we check if we are inside the borders of the auxiliary tabulated EoS
    if((rho_val>np.amax(rho_arr[p])) or (rho_val<np.amin(rho_arr[p])) or (en_val>np.amax(en_arr[p])) or (en_val<np.amin(en_arr[p]))):
      if(verbose):
         print("Values outside the auxiliary tabulated EoS")
      return False,(0.,0.)

    #here we first try to determine the best guess from the tabulated values
    min_rho_ratio=np.amin(np.abs(rho_arr[p]-rho_val)/rho_val)
    min_en_ratio=np.amin(np.abs(en_arr[p]-en_val)/en_val)
    if(verbose):
        print("Trying to exploit the tabulated values")
        print("Best tabulated relative errors: "+str(min_rho_ratio)+", "+str(min_en_ratio))
    cycles=1
    rlim_min=max(min_rho_ratio,min_en_ratio)
    rlim_max=2
    rlim=(rlim_min+rlim_max)/2.
    if(rlim_min>rlim_max):
      if(verbose):
         print("Values outside the auxiliary tabulated EoS")
      return False,(0.,0.)

    cycles_max=25
    go_on=True
    len_outcomes=0
    while(go_on):
        if(verbose):
           print("Cycle: "+str(cycles)+",  rlim="+str(rlim)+", rlim_min="+str(rlim_min)+", rlim_max="+str(rlim_max))
        cycles=cycles+1
        if(cycles>cycles_max):
          return False,(0.,0.) 
        outcomes=[]
        for aa in range(num_T_points):
            for bb in range(num_mu_points):
                if((np.abs(rho_arr[p][aa][bb]-rho_val)/rho_val<rlim) and (np.abs(en_arr[p][aa][bb]-en_val)/en_val<rlim)):
                   outcomes.append((T_points[aa],mu_points[bb]))
        len_outcomes=len(outcomes)
        if(len_outcomes==0):
            rlim_min=rlim
            rlim=(rlim+rlim_max)/2.
            continue
        if(len_outcomes>20):
            rlim_max=rlim
            rlim=(rlim+rlim_min)/2.
            continue
        go_on=False
    for q in range(len_outcomes):
        if(verbose):
           print("Tryng to solve BE/FD equations with: "+str(outcomes[q][0])+", "+str(outcomes[q][1]))
        TMUQS = optimize.root(f_system, [outcomes[q][0],outcomes[q][1]], args=(mass_had[p],en_val,rho_val,gfac[p],sfac[p],p), tol=root_tol)
        if(TMUQS.success):
           #we check the result
           rho_test_diff,en_test_diff = f_system(TMUQS.x,mass_had[p],en_val,rho_val,gfac[p],sfac[p],p)
           if((abs(rho_test_diff/rho_val)<1.e-3) and (abs(en_test_diff/en_val)<1.e-3)):
               if(verbose):
                   print("Test of TMUQS in try_with_tab successful")
               return True,TMUQS.x
    return False,(0.,0.)

#auxiliary funcion kronecker's delta
def kron(ii,jj):
    if(ii==jj):
      return 1
    return 0


#we get the name of input and output files

coarsefiles=sys.argv[1:N_input_args]
nt=len(coarsefiles)
outputfile=sys.argv[N_input_args]

tt=np.zeros(nt,dtype=np.float64)

number_of_particles=len(pnames)
if(verbose):
    print("List of particles that are considered:")
    print("Name     mass[GeV](UrQMD)    mass[GeV](PDG)     J(sping)      g_s      BE/FD")
    for i in range(number_of_particles):
        print(pnames[i]+"  "+str(mass_had[i])+"  "+str(spin[i])+"  "+str(gfac[i])+"  "+str(sfac[i]))

#now we read the tabulated EoS to get good initial guesses if the first attempt fails
rho_arr=[]
en_arr=[]
rho_interp=[]
en_interp=[]
if(verbose):
    print("Reading auxiliary EoS data files")
for i in range(number_of_particles-1):
    infile_hadron=tabdata_prefix+"_"+pnames[i]+".dat"
    if(not os.path.isfile(infile_hadron)):
        if((i>0) and (mass_had[i]==mass_had[i-1]) and (spin[i]==spin[i-1])): #we reuse the informations of the previous file
            if(verbose):
                print("Inserting "+infile_hadron+" by reusing "+tabdata_prefix+"_"+pnames[i-1]+".dat")
            rho_arr.append(rho_arr[i-1])
            en_arr.append(en_arr[i-1])
            rho_interp.append(rho_interp[i-1])
            en_interp.append(en_interp[i-1])
            continue
        else:
            print("Error, "+infile_hadron+" does not exist!")
            sys.exit(1)
    if(verbose):
        print("Reading "+infile_hadron)
    with open(infile_hadron,"rb") as po:
         data_tabulated=pickle.load(po)
    T_points_tmp,mu_points_tmp,rho_arr_tmp,en_arr_tmp=data_tabulated[:]
    if(i==0):
         T_points=T_points_tmp.copy()
         mu_points=mu_points_tmp.copy()
         def check_inside(T_test,mu_test):
             if((T_test>T_points[-1]) or (T_test<T_points[0])):
                 return False
             if((mu_test>mu_points[-1]) or (mu_test<mu_points[0])):
                 return False
             return True
         
    else:
        if(not np.array_equal(T_points,T_points_tmp)):
            print("Error, the T_points array in file "+infile_hadron+" is different from the previous ones...")
            sys.exit(1)
        if(not np.array_equal(mu_points,mu_points_tmp)):
            print("Error, the mu_points array in file "+infile_hadron+" is different from the previous ones...")
            sys.exit(1)
    rho_arr.append(rho_arr_tmp.copy())
    en_arr.append(en_arr_tmp.copy())
    rho_interp.append(interpolate.interp2d(T_points,mu_points,rho_arr_tmp.copy().transpose(),kind='linear'))
    en_interp.append(interpolate.interp2d(T_points,mu_points,en_arr_tmp.copy().transpose(),kind='linear'))
    data_tabulated=None
    T_points_tmp=None
    mu_points_tmp=None
    rho_arr_tmp=None
    en_arr_tmp=None

num_T_points=len(T_points)
num_mu_points=len(mu_points)
number_of_calls_to_aux_eos=0

#here we read the tabulated UrQMD HG EoS
#these are the unit values of energy density and pressure
e0=0.14651751415742
#these are the unit values of the net baryon density and entropy density
n0=0.15891

datadir="EOS_HG_UrQMD/"

fstd=datadir+"hadgas_eos.dat"
Ne_std=2001
Nn_std=401
en_std_max=1000.
rho_std_max=40.
enarr_std=np.linspace(0.,en_std_max,num=Ne_std)
rhoarr_std=np.linspace(0.,rho_std_max,num=Nn_std)

fmed=datadir+"hg_eos_small.dat"
Ne_med=201
Nn_med=201
en_med_max=10.
rho_med_max=2.
enarr_med=np.linspace(0.,en_med_max,num=Ne_med)
rhoarr_med=np.linspace(0.,rho_med_max,num=Nn_med)

fmin=datadir+"hg_eos_mini.dat"
Ne_min=201
Nn_min=201
en_min_max=0.1
rho_min_max=0.02
enarr_min=np.linspace(0.,en_min_max,num=Ne_min)
rhoarr_min=np.linspace(0.,rho_min_max,num=Nn_min)

temparr_std=np.zeros((Ne_std,Nn_std),dtype=np.float64)
muarr_std=np.zeros((Ne_std,Nn_std),dtype=np.float64)
sarr_std=np.zeros((Ne_std,Nn_std),dtype=np.float64)
parr_std=np.zeros((Ne_std,Nn_std),dtype=np.float64)

temparr_med=np.zeros((Ne_med,Nn_med),dtype=np.float64)
muarr_med=np.zeros((Ne_med,Nn_med),dtype=np.float64)
sarr_med=np.zeros((Ne_med,Nn_med),dtype=np.float64)
parr_med=np.zeros((Ne_med,Nn_med),dtype=np.float64)

temparr_min=np.zeros((Ne_min,Nn_min),dtype=np.float64)
muarr_min=np.zeros((Ne_min,Nn_min),dtype=np.float64)
sarr_min=np.zeros((Ne_min,Nn_min),dtype=np.float64)
parr_min=np.zeros((Ne_min,Nn_min),dtype=np.float64)

def readeos(ff,tarr,marr,parr,sarr,nx,ny):
    for j in range(ny):
        for i in range(nx):
            stuff=ff.readline().split()
            tarr[i,j],marr[i,j],parr[i,j],sarr[i,j]=np.float64(stuff[0]),np.float64(stuff[1]),np.float64(stuff[3]),np.float64(stuff[5])

if(verbose):
  print("Reading the UrQMD tabulated EoS from the files")

with open(fstd,"r") as infile:
     readeos(infile,temparr_std,muarr_std,parr_std,sarr_std,Ne_std,Nn_std)
     temp_interp_std=interpolate.interp2d(enarr_std, rhoarr_std, temparr_std.transpose(), kind='linear')
     muB_interp_std=interpolate.interp2d(enarr_std, rhoarr_std, muarr_std.transpose(), kind='linear')
     p_interp_std=interpolate.interp2d(enarr_std, rhoarr_std, parr_std.transpose(), kind='linear')
     s_interp_std=interpolate.interp2d(enarr_std, rhoarr_std, sarr_std.transpose(), kind='linear')

with open(fmed,"r") as infile:
     readeos(infile,temparr_med,muarr_med,parr_med,sarr_med,Ne_med,Nn_med)
     temp_interp_med=interpolate.interp2d(enarr_med, rhoarr_med, temparr_med.transpose(), kind='linear')
     muB_interp_med=interpolate.interp2d(enarr_med, rhoarr_med, muarr_med.transpose(), kind='linear')
     p_interp_med=interpolate.interp2d(enarr_med, rhoarr_med, parr_med.transpose(), kind='linear')
     s_interp_med=interpolate.interp2d(enarr_med, rhoarr_med, sarr_med.transpose(), kind='linear')

with open(fmin,"r") as infile:
     readeos(infile,temparr_min,muarr_min,parr_min,sarr_min,Ne_min,Nn_min)
     temp_interp_min=interpolate.interp2d(enarr_min, rhoarr_min, temparr_min.transpose(), kind='linear')
     muB_interp_min=interpolate.interp2d(enarr_min, rhoarr_min, muarr_min.transpose(), kind='linear')
     p_interp_min=interpolate.interp2d(enarr_min, rhoarr_min, parr_min.transpose(), kind='linear')
     s_interp_min=interpolate.interp2d(enarr_min, rhoarr_min, sarr_min.transpose(), kind='linear')

if(verbose):
  print("Done.\n")

def get_T_mub(rhoB_input_w_sign,edens):
     #before callin this function we already checked that both arguments are > 0
     compute=True
     rhoB=np.abs(rhoB_input_w_sign)
     edens=edens/e0
     rhoB=rhoB/n0
     if(edens<=en_std_max):
       if((edens<en_min_max ) and (rhoB<rho_min_max)):
         ftemp=temp_interp_min
         fmuB=muB_interp_min
         fpr=p_interp_min
         fs=s_interp_min
       if(edens<en_med_max ) and (rhoB<rho_med_max) and ((edens>=en_min_max ) or (rhoB>=rho_min_max)):     
         ftemp=temp_interp_med
         fmuB=muB_interp_med
         fpr=p_interp_med
         fs=s_interp_med
       if((edens>=en_med_max ) or (rhoB>=rho_med_max)):     
         if(rhoB>rho_std_max):
           print("Net baryon density exceeding the maximum of the table. Changed from "+str(rhoB)+" to "+str(rho_std_max*0.999999))
           rhoB=rho_std_max*0.999999
         ftemp=temp_interp_std
         fmuB=muB_interp_std
         fpr=p_interp_std
         fs=s_interp_std
     elif(edens>en_std_max):    
         compute=False
#         temperature=350./1000.
#         muB=3./1000.
         temperature=0.
         muB=0.
         pressure=0.
         entr_dens=0.
     else:        
         compute=False
         temperature=0.
         muB=0.
         pressure=0.
         entr_dens=0.
     
     if(compute): #all is expressed in GeV
       temperature=ftemp(edens,rhoB)[0]/1000.
       muB=3*fmuB(edens,rhoB)[0]/1000. 
#       pressure=fpr(edens,rhoB)[0]*e0
       entr_dens=fs(edens,rhoB)[0]*n0

#     return muB, temperature, pressure, entr_dens
     return temperature, muB, entr_dens 


#in this section we deal with the tabulated EoS by Monnai, Schenke and Chun Shen - Phys. Rev. C 100, 024907
prefix_dir="EOS_schenke/"
set_of_files=(prefix_dir+"neosB-v0.11",prefix_dir+"neosBS-v0.11",prefix_dir+"neosBQS-v0.11") #directories containing the various kind of tabulated EoS
suffix=("","s","qs") #suffixes of the EoS data filenames, depending on their content
ntables=7 #number of tables for each kind of EoS

data_big_array=[]

def check_vals(v1,v2,fn,val):
    if(v1 != v2): 
      print("Error in file "+fn)
      print("Mismatching between current minimum "+val+" value "+str(v1)+" and previous value "+str(v2))
      print("I quit.")
      sys.exit(2)
     
if(verbose):
  print("Reading the tabulated HG EoS with mu_B, mu_S and mu_Q")
for h in range(len(set_of_files)): 
    press_arrays=[]
    temp_arrays=[]
    muB_arrays=[]
    points_rhoB=[]
    points_edens=[]
    n_points_rhoB=[]
    n_points_edens=[]
    delta_rhoB=[]
    delta_edens=[]
    if(h>0):
      muS_arrays=[]
    if(h>1):
      muQ_arrays=[]
    for i in range(ntables):
        idf='{:1d}'.format(i+1) #the files are numbered starting from 1
        #here we read the pressure 
        filename=set_of_files[h]+"/"+"neos"+idf+suffix[h]+"_p.dat"
        try:
          datafile=open(filename,"r")
        except IOError:
          print("I have problems in opening "+filename)
          print("Plese, check that the variable prefix_dir inside the python script points to the parent directory of the three directories containing the tabulated files of the EoS by Monnai, Schenke and Chun Shen")
          sys.exit(3)
        rhoB_min,edens_min=np.float64(datafile.readline().split())
        stuff=datafile.readline().split()
        dx_rhoB=np.float64(stuff[0])
        dx_edens=np.float64(stuff[1])
        delta_rhoB.append(dx_rhoB)
        delta_edens.append(dx_edens)
        nB=int(stuff[2])+1
        nE=int(stuff[3])+1
        n_points_rhoB.append(nB)
        n_points_edens.append(nE)
        points_rhoB.append(np.linspace(rhoB_min,rhoB_min+(nB-1)*dx_rhoB, nB))
        points_edens.append(np.linspace(edens_min,edens_min+(nE-1)*dx_edens, nE))
        raw_arr=np.loadtxt(datafile,dtype=np.float64)         
        datafile.close()
        press_arrays.append(raw_arr.reshape(nE,nB)) 

        #here we read the temperature 
        #we could skip the first two lines, but we take the chance to recheck that everything is read correctly (hopefully)
        filename=set_of_files[h]+"/"+"neos"+idf+suffix[h]+"_t.dat"
        datafile=open(filename,"r")
        rhoB_min,edens_min=np.float64(datafile.readline().split())
        check_vals(rhoB_min, points_rhoB[i][0],filename,"rhoB") #we check that the grid min values have not changed within the same set of tables
        check_vals(edens_min, points_edens[i][0],filename,"edens") 
        stuff=datafile.readline().split()
        check_vals(float(stuff[0]),delta_rhoB[i],filename,"delta_rhoB") #we check that the grid resolution has not changed within the same set of tables
        check_vals(float(stuff[1]),delta_edens[i],filename,"delta_edens")
        check_vals(float(stuff[2])+1,n_points_rhoB[i],filename,"nB")
        check_vals(float(stuff[3])+1,n_points_edens[i],filename,"nE")
        raw_arr=np.loadtxt(datafile,dtype=np.float64)         
        datafile.close()
        temp_arrays.append(raw_arr.reshape(nE,nB)) 

        #here we read the baryon chemical potential 
        #we could skip the first two lines, but we take the chance to recheck that everything is read correctly (hopefully)
        filename=set_of_files[h]+"/"+"neos"+idf+suffix[h]+"_mub.dat"
        datafile=open(filename,"r")
        rhoB_min,edens_min=np.float64(datafile.readline().split())
        check_vals(rhoB_min, points_rhoB[i][0],filename,"rhoB") #we check that the grid min values have not changed within the same set of tables
        check_vals(edens_min, points_edens[i][0],filename,"edens") 
        stuff=datafile.readline().split()
        check_vals(float(stuff[0]),delta_rhoB[i],filename,"delta_rhoB") #we check that the grid resolution has not changed within the same set of tables
        check_vals(float(stuff[1]),delta_edens[i],filename,"delta_edens")
        check_vals(float(stuff[2])+1,n_points_rhoB[i],filename,"nB")
        check_vals(float(stuff[3])+1,n_points_edens[i],filename,"nE")
        raw_arr=np.loadtxt(datafile,dtype=np.float64)         
        datafile.close()
        muB_arrays.append(raw_arr.reshape(nE,nB)) 

        if(h>0):
          #here we read the strangeness chemical potential 
          #we could skip the first two lines, but we take the chance to recheck that everything is read correctly (hopefully)
          filename=set_of_files[h]+"/"+"neos"+idf+suffix[h]+"_mus.dat"
          datafile=open(filename,"r")
          rhoB_min,edens_min=np.float64(datafile.readline().split())
          check_vals(rhoB_min, points_rhoB[i][0],filename,"rhoB") #we check that the grid min values have not changed within the same set of tables
          check_vals(edens_min, points_edens[i][0],filename,"edens") 
          stuff=datafile.readline().split()
          check_vals(float(stuff[0]),delta_rhoB[i],filename,"delta_rhoB") #we check that the grid resolution has not changed within the same set of tables
          check_vals(float(stuff[1]),delta_edens[i],filename,"delta_edens")
          check_vals(float(stuff[2])+1,n_points_rhoB[i],filename,"nB")
          check_vals(float(stuff[3])+1,n_points_edens[i],filename,"nE")
          raw_arr=np.loadtxt(datafile,dtype=np.float64)         
          datafile.close()
          muS_arrays.append(raw_arr.reshape(nE,nB)) 

        if(h>1):
          #here we read the electric charge chemical potential 
          #we could skip the first two lines, but we take the chance to recheck that everything is read correctly (hopefully)
          filename=set_of_files[h]+"/"+"neos"+idf+suffix[h]+"_muq.dat"
          datafile=open(filename,"r")
          rhoB_min,edens_min=np.float64(datafile.readline().split())
          check_vals(rhoB_min, points_rhoB[i][0],filename,"rhoB") #we check that the grid min values have not changed within the same set of tables
          check_vals(edens_min, points_edens[i][0],filename,"edens") 
          stuff=datafile.readline().split()
          check_vals(float(stuff[0]),delta_rhoB[i],filename,"delta_rhoB") #we check that the grid resolution has not changed within the same set of tables
          check_vals(float(stuff[1]),delta_edens[i],filename,"delta_edens")
          check_vals(float(stuff[2])+1,n_points_rhoB[i],filename,"nB")
          check_vals(float(stuff[3])+1,n_points_edens[i],filename,"nE")
          raw_arr=np.loadtxt(datafile,dtype=np.float64)         
          datafile.close()
          muQ_arrays.append(raw_arr.reshape(nE,nB)) 


    if(h==0):
      data_big_array.append((points_rhoB,points_edens,n_points_rhoB,n_points_edens,delta_rhoB,delta_edens,press_arrays,temp_arrays,muB_arrays))
    if(h==1):
      data_big_array.append((points_rhoB,points_edens,n_points_rhoB,n_points_edens,delta_rhoB,delta_edens,press_arrays,temp_arrays,muB_arrays,muS_arrays))
    if(h==2):
      data_big_array.append((points_rhoB,points_edens,n_points_rhoB,n_points_edens,delta_rhoB,delta_edens,press_arrays,temp_arrays,muB_arrays,muS_arrays,muQ_arrays))

    #first index of data_big_array: the kind of tabulated EoS
    #second index: the kind of data: 0=points_rhoB, 1=points_edens,...
    #third index: the table
    #fourth index(-exes if 2D): position inside the array
if(verbose):
    print("Done")

def get_T_muBSQ(input_rhoB,input_edens):
  T_HBQS=np.zeros(3,dtype=np.float64)
  mu_HBQS=np.zeros(6,dtype=np.float64)
  mu_index=0
  for h in range(len(set_of_files)):
    if(input_edens > data_big_array[h][1][-1][-1]):
      print("Energy density outside of the tabulated range.")
      sys.exit(3)
    if(input_rhoB > data_big_array[h][0][-1][-1]):
      print("Baryon density outside of the tabulated range.")
      sys.exit(3)
    if(input_edens > data_big_array[h][1][-1][0]):
      ntable_index=ntables-1
    else:
      for i in range(ntables-1):
        if(input_edens <= data_big_array[h][1][i+1][0]): #first entry of points_edens array of table h+1
          ntable_index=i
          break
    if(verbose):
      print("Eos of kind "+str(h))
      print("Selected table index: "+str(ntable_index))
    f = interpolate.interp2d(data_big_array[h][1][ntable_index], data_big_array[h][0][ntable_index], data_big_array[h][7][ntable_index].transpose(), kind='linear')
    T_HBQS[h]=f(input_edens,input_rhoB)
    if(verbose):
      print("Temperature: "+str(T_HBQS[h]))
    f = interpolate.interp2d(data_big_array[h][1][ntable_index], data_big_array[h][0][ntable_index], data_big_array[h][8][ntable_index].transpose(), kind='linear')
    mu_HBQS[mu_index]=f(input_edens,input_rhoB)
    if(verbose):
       print("mu_B: "+str(mu_HBQS[mu_index]))
    mu_index=mu_index+1
    if(h>0):
      f = interpolate.interp2d(data_big_array[h][1][ntable_index], data_big_array[h][0][ntable_index], data_big_array[h][9][ntable_index].transpose(), kind='linear')
      mu_HBQS[mu_index]=f(input_edens,input_rhoB)
      if(verbose):
         print("mu_S: "+str(mu_HBQS[mu_index]))
      mu_index=mu_index+1
    if(h>1):
      f = interpolate.interp2d(data_big_array[h][1][ntable_index], data_big_array[h][0][ntable_index], data_big_array[h][10][ntable_index].transpose(), kind='linear')
      mu_HBQS[mu_index]=f(input_edens,input_rhoB)
      if(verbose):
         print("mu_Q: "+str(mu_HBQS[mu_index]))
      mu_index=mu_index+1
 
  return T_HBQS,mu_HBQS
        

if(verbose):
    print("Reading coarse data files... ")
#we open the coarse graining data files
first_time=True
for ff in range(nt):
    Tmunufile=coarsefiles[ff].replace("densities","Tmunu")
    if(not(os.path.isfile(coarsefiles[ff]))):
            print("Skipping "+coarsefiles[ff]+" because it seems that it doesn't exist...")
            continue
    if(not(os.path.isfile(Tmunufile))):
            print("Skipping "+Tmunufile+" because it seems that it doesn't exist...")
            continue
        
    cdata = open(coarsefiles[ff],"rb")
    Tdata = open(Tmunufile,"rb")
    
    if(verbose):
        print("\nWorking on file: "+coarsefiles[ff])
    nevents=np.fromfile(cdata,dtype=np.int64,count=1)[0]
    nevents_T=np.fromfile(Tdata,dtype=np.int64,count=1)[0]
    if(nevents != nevents_T):
      print("Error: different number of events in "+Tmunufile+" ("+str(nevents_T)+") and "+coarsefiles[ff]+" ("+str(nevents)+")")
      sys.exit(4)
    if(verbose):
        print("Number of events: "+str(nevents))

    time=np.fromfile(cdata,dtype=np.float64,count=1)[0]
    tt[ff]=time
    time_Tmunu=np.fromfile(Tdata,dtype=np.float64,count=1)[0]
    if(time_Tmunu != tt[ff]):
      print("FATAL ERROR: different times in "+Tmunufile+" ("+str(time_Tmunu)+") and "+coarsefiles[ff]+" ("+str(tt[ff])+")")
      sys.exit(4)
    if(verbose):
        print("Time: "+str(time))

    num_of_hadrons=np.fromfile(cdata,dtype=np.int32,count=1)[0]
    num_part_Tmunu=np.fromfile(Tdata,dtype=np.int32,count=1)[0]
    if(num_part_Tmunu != num_of_hadrons):
       print("FATAL ERROR: different number of hadrons!!! "+Tmunufile+" ("+str(num_part_Tmunu)+") and "+coarsefiles[ff]+" ("+str(num_of_hadrons)+")")
       sys.exit(4)


    nx=np.fromfile(cdata,dtype=np.int32,count=1)[0]
    ny=np.fromfile(cdata,dtype=np.int32,count=1)[0]
    nz=np.fromfile(cdata,dtype=np.int32,count=1)[0]
    nx_T,ny_T,nz_T=np.fromfile(Tdata,dtype=np.int32,count=3)
    if((nx_T != nx) or (ny_T != ny) or (nz_T != nz)):
      print("FATAL ERROR: different grid structure between energy-momentum and densities tensors!!! Expected: "+str(nx)+", "+str(ny)+", "+str(nz)+", read now: "+str(nx_T)+", "+str(ny_T)+", "+str(nz_T))
      sys.exit(4)
    if(verbose):
        print("Grid points along x, y and z: "+str(nx)+", "+str(ny)+", "+str(nz))

    dx=np.fromfile(cdata,dtype=np.float64,count=1)[0]
    dy=np.fromfile(cdata,dtype=np.float64,count=1)[0]
    dz=np.fromfile(cdata,dtype=np.float64,count=1)[0]
    dx_T,dy_T,dz_T=np.fromfile(Tdata,dtype=np.float64,count=3)
    if((dx_T != dx) or (dy_T != dy) or (dz_T != dz)):
      print("FATAL ERROR: different grid resolution between energy-momentum and densities tensors!!! Expected: "+str(dx)+", "+str(dy)+", "+str(dz)+", read now: "+str(dx_T)+", "+str(dy_T)+", "+str(dz_T))
      sys.exit(4)
    if(verbose):
        print("Cell widths along x, y and z: "+str(dx)+", "+str(dy)+", "+str(dz))

    if version_2_data_format or smash: #with smash we always use format 2
        xmin=np.fromfile(cdata,dtype=np.float64,count=1)[0]
        ymin=np.fromfile(cdata,dtype=np.float64,count=1)[0]
        zmin=np.fromfile(cdata,dtype=np.float64,count=1)[0]
        xmin_T,ymin_T,zmin_T=np.fromfile(Tdata,dtype=np.float64,count=3)
        if((xmin_T != xmin) or (ymin_T != ymin) or (zmin_T != zmin)):
            print("FATAL ERROR: different grid resolution between energy-momentum and densities tensors!!! Expected: "+str(xmin)+", "+str(ymin)+", "+str(zmin)+", read now: "+str(xmin_T)+", "+str(ymin_T)+", "+str(zmin_T))
            sys.exit(4)
        if(verbose):
            print("Cell widths along x, y and z: "+str(xmin)+", "+str(ymin)+", "+str(zmin))

    Tp=np.fromfile(Tdata,dtype=np.float64,count=nx*ny*nz*num_of_hadrons*10).reshape([nx,ny,nz,num_of_hadrons,10])
    Jb=np.fromfile(Tdata,dtype=np.float64,count=nx*ny*nz*4,offset=nx*ny*nz*num_of_hadrons*4*8).reshape([nx,ny,nz,4])
     #regarding the line above: offset option available only with numpy>=1.17, expressed in bytes (so, with float64, we must multiply by 8)
    Tdata.close()

    if(first_time):
      first_time=False
      nxref=nx
      nyref=ny
      nzref=nz
      dxref=dx
      dyref=dy
      dzref=dz
      if version_2_data_format or smash: #with smash we always use format 2
          xstart=xmin+dx/2.
          xend=xmin+(nx-1)*dx
          ystart=ymin+dy/2.
          yend=ymin+(ny-1)*dy
          zstart=zmin+dz/2.
          zend=zmin+(nz-1)*dz
      else:
          xstart=-nx*dx/2+dx/2
          xend=nx*dx/2-dx/2
          ystart=-ny*dy/2+dy/2
          yend=ny*dy/2-dy/2
          zstart=-nz*dz/2+dz/2
          zend=nz*dz/2-dz/2
          
      xx=np.linspace(xstart,xend,num=nx)
      yy=np.linspace(ystart,yend,num=ny)
      zz=np.linspace(zstart,zend,num=nz)
      nev_times_vol=nevents*dx*dy*dz
      Tmunu=np.zeros((4,4),dtype=np.float64)
      Lambda=np.zeros((4,4),dtype=np.float64)
      LambdaF=np.zeros((4,4),dtype=np.float64)
      uvel=np.zeros(4,dtype=np.float64)

      #arrays to store coarse graining data
      if(verbose):
          print("Allocating and initializing arrays... ")
      if(comp_all):
        temp=np.zeros((nt,nx,ny,nz,number_of_particles)) 
        tempQS=np.zeros((nt,nx,ny,nz,number_of_particles)) 
        muQS=np.zeros((nt,nx,ny,nz,number_of_particles)) 
        tempBZ=np.zeros((nt,nx,ny,nz,number_of_particles))
        muBZ=np.zeros((nt,nx,ny,nz,number_of_particles)) 
        tempPCE=np.zeros((nt,nx,ny,nz,number_of_particles-1))#the last entry of the particle array is for all unidentified particles and does not have a mu 
        muPCE=np.zeros((nt,nx,ny,nz,number_of_particles-1,number_of_particles-1))#we save also the partial results
        successPCE=np.zeros((nt,nx,ny,nz),dtype=np.int64)#counts the number of successful hadron solutions
        tempFCE=np.zeros((nt,nx,ny,nz))
        tempHGU=np.zeros((nt,nx,ny,nz))
        tempHBSQ=np.zeros((nt,nx,ny,nz,3))
        muHBSQ=np.zeros((nt,nx,ny,nz,6))#muB, muB and muS, muB+muS+muQ
        muHGU=np.zeros((nt,nx,ny,nz))
        sHGU=np.zeros((nt,nx,ny,nz))
        sFCE=np.zeros((nt,nx,ny,nz))
        rho_main=np.zeros((nt,nx,ny,nz,3)) #rho_B, rho_C and rho_S
        muFCE=np.zeros((nt,nx,ny,nz,2))#we have mu_B and mu_S 
        ene=np.zeros((nt,nx,ny,nz,number_of_particles+1)) #we include one additional entry for the sum of all particles
        ndens=np.zeros((nt,nx,ny,nz,number_of_particles+1)) #we include one additional entry for the sum of all particles
        pcomp=np.zeros((nt,nx,ny,nz,3))#components of the pressure from Tmunu 
      if(comp_trans):
        temp=np.zeros((nt,nx,ny,number_of_particles))
        tempQS=np.zeros((nt,nx,ny,number_of_particles))
        muQS=np.zeros((nt,nx,ny,number_of_particles)) 
        tempBZ=np.zeros((nt,nx,ny,number_of_particles)) 
        muBZ=np.zeros((nt,nx,ny,number_of_particles)) 
        tempPCE=np.zeros((nt,nx,ny,number_of_particles-1))#the last entry of the particle array is for all unidentified particles and does not have a mu 
        muPCE=np.zeros((nt,nx,ny,number_of_particles-1,number_of_particles-1))#we save also the partial results
        successPCE=np.zeros((nt,nx,ny),dtype=np.int64)#counts the number of successful hadron solutions
        tempFCE=np.zeros((nt,nx,ny))
        tempHGU=np.zeros((nt,nx,ny))
        sHGU=np.zeros((nt,nx,ny))
        sFCE=np.zeros((nt,nx,ny))
        rho_main=np.zeros((nt,nx,ny,3)) #rho_B, rho_C and rho_S
        muFCE=np.zeros((nt,nx,ny,2))#we have mu_B and mu_S 
        muHGU=np.zeros((nt,nx,ny)) 
        ene=np.zeros((nt,nx,ny,number_of_particles+1)) #we include one additional entry for the sum of all particles
        ndens=np.zeros((nt,nx,ny,number_of_particles+1)) #we include one additional entry for the sum of all particles
        tempHBSQ=np.zeros((nt,nx,ny,3))
        muHBSQ=np.zeros((nt,nx,ny,6))#muB, muB and muS, muB+muS+muQ
        pcomp=np.zeros((nt,nx,ny,3))#components of the pressure from Tmunu 
      if(comp_points):  
        temp=np.zeros((nt,ncomp_cells,number_of_particles))
        tempQS=np.zeros((nt,ncomp_cells,number_of_particles))
        muQS=np.zeros((nt,ncomp_cells,number_of_particles))
        tempBZ=np.zeros((nt,ncomp_cells,number_of_particles))
        muBZ=np.zeros((nt,ncomp_cells,number_of_particles))
        tempPCE=np.zeros((nt,ncomp_cells,number_of_particles-1))#the last entry of the particle array is for all unidentified particles and does not have a mu 
        muPCE=np.zeros((nt,ncomp_cells,number_of_particles-1,number_of_particles-1))#we save also the partial results
        successPCE=np.zeros((nt,ncomp_cells),dtype=np.int64)#counts the number of successful hadron solutions
        tempFCE=np.zeros((nt,ncomp_cells))
        tempHGU=np.zeros((nt,ncomp_cells))
        muHGU=np.zeros((nt,ncomp_cells))
        rho_main=np.zeros((nt,ncomp_cells,3)) #rho_B, rho_C and rho_S
        muFCE=np.zeros((nt,ncomp_cells,2))#the last entry of the particle array is for all unidentified particles and does not have a mu 
        ene=np.zeros((nt,ncomp_cells,number_of_particles+1))
        ndens=np.zeros((nt,ncomp_cells,number_of_particles+1))
        tempHBSQ=np.zeros((nt,ncomp_cells,3))
        muHBSQ=np.zeros((nt,ncomp_cells,6))#muB, muB and muS, muB+muS+muQ
        sHGU=np.zeros((nt,ncomp_cells))
        sFCE=np.zeros((nt,ncomp_cells))
        pcomp=np.zeros((nt,ncomp_cells,3))#components of the pressure from Tmunu 

      total_particles=np.zeros((nt,number_of_particles))
      if(verbose):
          print("Done.")
    else:
      if(nx != nxref):
        print("Error, the number of cells nx along x in file "+coarsefiles[ff]+" does not match with previously read values. I quit.")
        sys.exit(3)
      if(ny != nyref):
        print("Error, the number of cells ny along y in file "+coarsefiles[ff]+" does not match with previously read values. I quit.")
        sys.exit(3)
      if(nz != nzref):
        print("Error, the number of cells nz along z in file "+coarsefiles[ff]+" does not match with previously read values. I quit.")
        sys.exit(3)
      if(dx != dxref):
        print("Error, the cell width dx along x in file "+coarsefiles[ff]+" does not match with previously read values. I quit.")
        sys.exit(3)
      if(dy != dyref):
        print("Error, the cell width dy along y in file "+coarsefiles[ff]+" does not match with previously read values. I quit.")
        sys.exit(3)
      if(dz != dzref):
        print("Error, the cell width dz along z in file "+coarsefiles[ff]+" does not match with previously read values. I quit.")
        sys.exit(3)

    datas=np.fromfile(cdata,dtype=np.float64,count=nx*ny*nz*(15+3*number_of_particles)).reshape([nx,ny,nz,15+3*number_of_particles])
    cdata.close()
    #the loop with internal ifs is not very efficient, nevertheless it is clear, it allows a certain flexibility and it does not slow too much the program
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
#                rho[ff,i,j,k]=datas[i,j,k,0]
#                vx[ff,i,j,k]=datas[i,j,k,1]
#                vy[ff,i,j,k]=datas[i,j,k,2]
#                vz[ff,i,j,k]=datas[i,j,k,3]
                if(comp_trans):
                    if(zz[k]!=zcoordinate):
                         continue
                    else:
                         zk=k
                if(comp_points):
                     if (xx[i],yy[j],zz[k]) not in cells_to_evaluate:
                         continue
                     else:
                        indx_cell=cells_to_evaluate.index((xx[i],yy[j],zz[k]))
                if(verbose):
                    print("\n*** cell with coordinates: "+str(i)+"   "+str(j))
                edens_tot=0.
                mtot=0.
                mu_try_arr=[]
                mass_try_arr=[]
                gs_try_arr=[]
                rho_try_arr=[]
                sfac_try_arr=[]
                had_names=[]
                had_indx=[]
                T_try=0.
                T_last=0.
                rhoB_data=datas[i,j,k,0]
                rhoC_data=datas[i,j,k,5]
                rhoS_data=datas[i,j,k,6]
                at_least_one=False
                if(comp_all):
                    rho_main[ff,i,j,k,:]=rhoB_data,rhoC_data,rhoS_data
                elif(comp_trans):
                    rho_main[ff,i,j,:]=rhoB_data,rhoC_data,rhoS_data
                else:
                    rho_main[ff,indx_cell,:]=rhoB_data,rhoC_data,rhoS_data
                if(verbose):
                    print("Net baryon, electric and strange densities: "+str(rhoB_data)+",    "+str(rhoC_data)+",    "+str(rhoS_data))
                count_failures=0
                for hadron_item in range(number_of_particles-1):#we do not consider the last entry with unknown particles
                    if(use_preferred):
                      p=new_order[hadron_item]
                    else:
                      p=hadron_item
                    if(verbose):
                        print("Cell: "+str(i)+", "+str(j)+", hadron:"+str(p)+", i.e. "+pnames[p]+")")
                    particle_count=datas[i,j,k,15+3*p]
                    rho_val=datas[i,j,k,16+3*p]
                    en_val=datas[i,j,k,17+3*p]
                    mtot=mtot+mass_had[p]* rho_val
                    if(comp_all):
                        ene[ff,i,j,k,p]=en_val
                        ndens[ff,i,j,k,p]=rho_val
                        ene[ff,i,j,k,-1]=ene[ff,i,j,-1]+en_val
                        ndens[ff,i,j,k,-1]=ndens[ff,i,j,-1]+rho_val
                    elif(comp_trans):
                        ene[ff,i,j,p]=en_val
                        ndens[ff,i,j,p]=rho_val
                        ene[ff,i,j,-1]=ene[ff,i,j,-1]+en_val
                        ndens[ff,i,j,-1]=ndens[ff,i,j,-1]+rho_val
                    else:
                        ene[ff,indx_cell,p]=en_val
                        ndens[ff,indx_cell,p]=rho_val
                        ene[ff,indx_cell,-1]=ene[ff,indx_cell,-1]+en_val
                        ndens[ff,indx_cell,-1]=ndens[ff,indx_cell,-1]+rho_val
                    register_QS=False
                    register_BZ=False
                    #particle_count=rho_val*nev_times_vol
                    if(verbose):
                        print("Particle count: "+'{:12.7e}'.format(particle_count)+"( p is: "+str(p)+" )")
                    total_particles[ff,p]=total_particles[ff,p]+particle_count
                    if(particle_count>particle_count_for_acceptance):
                      if(verbose):
                          print("Sufficient particle number to proceed.")
                      if(p==7): #if we have the computation for neutrons, we compute also the full chem eos
                        at_least_one=True
                      z = en_val/rho_val
                      if(verbose):
                          print("Trying to solve for ratio: "+str(z))
                      Tsolve = optimize.root(f_to_be_zeroed, [0.1], args=(mass_had[p],z),jac=jacobian, tol=root_tol)
                      if(Tsolve.success):
                        T=Tsolve.x.item()
                        if(comp_all):
                          temp[ff,i,j,k,p]=T
                        elif(comp_trans):
                          temp[ff,i,j,p]=T
                        else:
                          temp[ff,indx_cell,p]=T
                          
                        if(verbose):
                             print("Solved obtaining T="+str(T))
                        T_try=T
                      else:
                        if(verbose):
                             print("Not solved")
                        T=0.

                      QS_unsolved=True
                      if(T_try>0.5):
                          T_try=T_try/3.
                      elif(T_try>0.25):
                         T_try=T_try/2.
                      TMUQS_success=False
                      if(Tsolve.success):
                        TMUQS = optimize.root(f_system, [T_try,mass_had[p]/2], args=(mass_had[p],en_val,rho_val,gfac[p],sfac[p],p), tol=root_tol)
                        if(TMUQS.success):
                           TMUQS_success=True
                      if(TMUQS_success):
                        #we check the result
                        rho_test_diff,en_test_diff = f_system(TMUQS.x,mass_had[p],en_val,rho_val,gfac[p],sfac[p],p)
                        if((abs(rho_test_diff)/rho_val<1.e-3) and (abs(en_test_diff)/en_val<1.e-3)):
                          register_QS=True
                          TMUQS_x=TMUQS.x
                          QS_unsolved=False
                        else:
                          if(verbose):
                            print("TMUQS checking failed")

                      if(QS_unsolved):
                         if(verbose):
                            number_of_calls_to_aux_eos=number_of_calls_to_aux_eos+1
                            print("Number of calls to try_with_tab (auxiliary EoS): "+str(number_of_calls_to_aux_eos))
                         register_QS,TMUQS_x=try_with_tab(rho_val,en_val,p)

                      if(register_QS):
                        if(comp_all):
                          tempQS[ff,i,j,k,p]=TMUQS_x[0]
                          muQS[ff,i,j,k,p]=TMUQS_x[1]
                        elif(comp_trans):
                          tempQS[ff,i,j,p]=TMUQS_x[0]
                          muQS[ff,i,j,p]=TMUQS_x[1]
                        else:
                          tempQS[ff,indx_cell,p]=TMUQS_x[0]
                          muQS[ff,indx_cell,p]=TMUQS_x[1]

                        T_try=TMUQS_x[0]
                        mu_try_arr.append(TMUQS_x[1])

                        TMUBZ = optimize.root(f_system, TMUQS_x, args=(mass_had[p],en_val,rho_val,gfac[p],0,p), tol=root_tol)
                      else:
                        TMUBZ = optimize.root(f_system, [T_try,mass_had[p]/2.], args=(mass_had[p],en_val,rho_val,gfac[p],0,p), tol=root_tol)
                      if(not TMUBZ.success):
                        if(tt[ff]>22):
                          for i_T in range(0,3):
                            T_try=0.001*10**i_T
                            for i_M in range(0,22):      
                                mu_try=mass_had[p]*(1.01-i_M/80)
                                TMUBZ = optimize.root(f_system, [T_try,mu_try], args=(mass_had[p],en_val,rho_val,gfac[p],sfac[p],p), tol=root_tol)
                                if(TMUBZ.success):
                                  break
                            if(TMUBZ.success):
                              break
                        elif(tt[ff]>5):
                          for i_T in range(0,4):
                            T_try=0.001*10**i_T
                            for i_M in range(0,12):      
                                mu_try=mass_had[p]*(1.101-i_M/10)
                                TMUBZ = optimize.root(f_system, [T_try,mu_try], args=(mass_had[p],en_val,rho_val,gfac[p],sfac[p],p), tol=root_tol)
                                if(TMUBZ.success):
                                  break
                            if(TMUBZ.success):
                              break
                        else:
                          for i_T in range(0,4):
                            T_try=1/10**i_T
                            for i_M in range(0,12):      
                                mu_try=mass_had[p]*(0.01+i_M/10)
                                TMUBZ = optimize.root(f_system, [T_try,mu_try], args=(mass_had[p],en_val,rho_val,gfac[p],sfac[p],p), tol=root_tol)
                                if(TMUBZ.success):
                                  break
                            if(TMUBZ.success):
                              break
                      if(TMUBZ.success):
                        #we check the result
                        rho_test_diff,en_test_diff = f_system(TMUBZ.x,mass_had[p],en_val,rho_val,gfac[p],0,p)
                        if((abs(rho_test_diff)/rho_val<1.e-3) and (abs(en_test_diff)/en_val<1.e-3)):
                          if(comp_all):
                            tempBZ[ff,i,j,p]=TMUBZ.x[0]
                            muBZ[ff,i,j,p]=TMUBZ.x[1]
                          elif(comp_trans):
                            tempBZ[ff,i,j,p]=TMUBZ.x[0]
                            muBZ[ff,i,j,p]=TMUBZ.x[1]
                          else:
                            tempBZ[ff,indx_cell,p]=TMUBZ.x[0]
                            muBZ[ff,indx_cell,p]=TMUBZ.x[1]

                          register_BZ=True
                      if(verbose and (register_QS or register_BZ or Tsolve.success)):
                        print(pnames[p]+":  en. dens."+str(en_val)+"  rho"+str(rho_val)+"  e/n"+str(en_val/rho_val)+"  T(e/n)"+str(T)+"  T(BZ)"+str(TMUBZ.x[0])+"  mu(BZ)"+str(TMUBZ.x[1])+"  "+str(register_BZ)+"  T(QS)"+str(TMUQS_x[0])+"   mu(QS)"+str(TMUQS_x[1])+"  "+str(register_QS))
                        if(count_failures <= max_failures):
                            edens_tot=edens_tot+en_val
                            mass_try_arr.append(mass_had[p])
                            gs_try_arr.append(gfac[p])
                            rho_try_arr.append(rho_val)
                            sfac_try_arr.append(sfac[p])
                            had_names.append(pnames[p])
                            had_indx.append(p)
                            if(T_last!=0):
                              #T_try=(T_try+T_last)/2.
                              T_try=T_last
                            alist=[]
                            alist.append(edens_tot)
                            alist.append(rho_try_arr)
                            alist.append(mass_try_arr)
                            alist.append(gs_try_arr)
                            alist.append(sfac_try_arr)
                            alist.append(had_indx)
                            if(verbose):
                                print("Trying to compute PCE with parameters:")
                                print("Guesses: "+str(T_try)+"  "+str(mu_try_arr))
                                print("Params: "+str(edens_tot)+"  "+str(rho_try_arr))
                                print("Had. prop: "+str(mass_try_arr)+"  "+str(gs_try_arr))
                            TMUPCE = optimize.root(f_system_hd, [T_try]+mu_try_arr, args=alist, tol=root_tol)
                            real_success = False
                            if(TMUPCE.success):
                              if(verbose):  
                                  results_check_pce=f_system_hd(TMUPCE.x,alist)
                                  if(abs(results_check_pce[0])/edens_tot<1.e-3):
                                     print("PCE_SOLVER_WORKED_FOR_EDENS  "+str(abs(results_check_pce[0])/edens_tot))
                                     real_success = True
                                  else:
                                     print("PCE_SOLVER_FAILED_FOR_EDENS  "+str(abs(results_check_pce[0])/edens_tot))
                                  print("Results from PCE:")
                                  print(str(TMUPCE.x[0]))
                            if(real_success):      
                              T_last=TMUPCE.x[0] 
                              count_failures=0
                              if(comp_all):
                                  tempPCE[ff,i,j,k,p]=T_last
                                  successPCE[ff,i,j,k]+=1
                              elif(comp_trans):
                                  tempPCE[ff,i,j,p]=T_last
                                  successPCE[ff,i,j]+=1
                              else:
                                  tempPCE[ff,indx_cell,p]=T_last
                                  successPCE[ff,indx_cell]+=1
                              for aa in range(len(mu_try_arr)):
                                  print(had_names[aa]+"    "+str(TMUPCE.x[aa+1]))
                                  mu_try_arr[aa]=TMUPCE.x[aa+1]
                                  if(comp_all):
                                     muPCE[ff,i,j,k,p,had_indx[aa]]=mu_try_arr[aa]
                                  elif(comp_trans):
                                     muPCE[ff,i,j,p,had_indx[aa]]=mu_try_arr[aa]
                                  else:
                                     muPCE[ff,indx_cell,p,had_indx[aa]]=mu_try_arr[aa]
                            else:
                               count_failures=count_failures+1
                               if(verbose):  
                                   print("PCE solution not found.")
                        else: #count_failures > max_failures
                            print("Search of PCE solution skipped because of already too many failed attempts")


                    else: #particle density is not sufficient
                      if(verbose):  
                          print("Insufficient particle number to proceed: "+'{:12.7e}'.format(particle_count))


                #p is equal to the last value of the previous loop, i.e. to number_of_particles-2 (the value before number_of_particles - 1),
                #therefore p+1 points to the last entry (arrays starts counting from 0) which contains the unidentified particles
                particle_count=datas[i,j,k,15+3*(p+1)]
                rho_val=datas[i,j,k,16+3*(p+1)]
                en_val=datas[i,j,k,17+3*(p+1)]
                if(comp_all):
                   ene[ff,i,j,k,-1]=ene[ff,i,j,k,-1]+en_val
                   ndens[ff,i,j,k,-1]=ndens[ff,i,j,k,-1]+rho_val
                elif(comp_trans):
                   ene[ff,i,j,-1]=ene[ff,i,j,-1]+en_val
                   ndens[ff,i,j,-1]=ndens[ff,i,j,-1]+rho_val
                else:
                   ene[ff,indx_cell,-1]=ene[ff,indx_cell,-1]+en_val
                   ndens[ff,indx_cell,-1]=ndens[ff,indx_cell,-1]+rho_val
                #particle_count=rho_val*nev_times_vol
                if(verbose):
                    print("Unidentified particles count: "+'{:12.7e}'.format(particle_count)+" with associated density: "+'{:12.7e}'.format(rho_val)+" and energy density: "+'{:12.7e}'.format(en_val))
                total_particles[ff,p+1]=total_particles[ff,p]+particle_count 

                if(at_least_one):
                  #here we use the UrQMD HG EoS
                  if(comp_all):
                     tempHGU[ff,i,j,k],muHGU[ff,i,j,k],sHGU[ff,i,j,k]=get_T_mub(rhoB_data,ene[ff,i,j,k,-1])
                  elif(comp_trans):
                     tempHGU[ff,i,j],muHGU[ff,i,j],sHGU[ff,i,j]=get_T_mub(rhoB_data,ene[ff,i,j,-1])
                  else:
                     tempHGU[ff,indx_cell],muHGU[ff,indx_cell],sHGU[ff,indx_cell]=get_T_mub(rhoB_data,ene[ff,indx_cell,-1])

                  #here we use the tabulated EoS by Monnai, Schenke and Chun Shen
                  if(comp_all):
                      tempHBSQ[ff,i,j,k,:],muHBSQ[ff,i,j,k,:]=get_T_muBSQ(rhoB_data,ene[ff,i,j,k,-1])
                  elif(comp_trans):
                      tempHBSQ[ff,i,j,:],muHBSQ[ff,i,j,:]=get_T_muBSQ(rhoB_data,ene[ff,i,j,-1])
                  else:
                      tempHBSQ[ff,indx_cell,:],muHBSQ[ff,indx_cell,:]=get_T_muBSQ(rhoB_data,ene[ff,indx_cell,-1])

                  if(comp_all):
                      alist=[ene[ff,i,j,k,-1],rhoB_data,rhoS_data]
                      if((ff>0) and (tempFCE[ff-1,i,j,k]!=0)):
                              T_last=(tempFCE[ff-1,i,j,k]+tempHGU[ff,i,j,k])/2
                              muB_guess_FCE,muS_guess_FCE=muFCE[ff-1,i,j,k,:]
                      else:
                              T_last=tempHGU[ff,i,j,k]
                              muB_guess_FCE=muHGU[ff,i,j,k]
                              muS_guess_FCE=muB_guess_FCE/2.5
                  elif(comp_trans):
                      alist=[ene[ff,i,j,-1],rhoB_data,rhoS_data]
                      if((ff>0) and (tempFCE[ff-1,i,j]!=0)):
                              T_last=(tempFCE[ff-1,i,j]+tempHGU[ff,i,j])/2
                              muB_guess_FCE,muS_guess_FCE=muFCE[ff-1,i,j,:]
                      else:
                              T_last=tempHGU[ff,i,j]
                              muB_guess_FCE=muHGU[ff,i,j]
                              muS_guess_FCE=muB_guess_FCE/2.5
                  else:
                      alist=[ene[ff,indx_cell,-1],rhoB_data,rhoS_data]
                      if((ff>0) and (tempFCE[ff-1,indx_cell]!=0)):
                              T_last=(tempFCE[ff-1,indx_cell]+tempHGU[ff,indx_cell])/2
                              muB_guess_FCE,muS_guess_FCE=muFCE[ff-1,indx_cell,:]
                      else:
                              T_last=tempHGU[ff,indx_cell]
                              muB_guess_FCE=muHGU[ff,indx_cell]
                              muS_guess_FCE=muB_guess_FCE/2.5

                  if(verbose):    
                      print("Arguments: "+str(alist[0])+"  "+str(alist[1])+"  "+str(alist[2]))
                  
                  if(verbose):  
                      print("Initial guesses: "+str(T_last)+"  "+str(muB_guess_FCE)+"  "+str(muS_guess_FCE))
                  if(comp_all):
                    TMUFCE = optimize.root(f_system_fce, [T_last,muB_guess_FCE,muS_guess_FCE], args=alist, tol=root_tol)
                  elif(comp_trans):
                    TMUFCE = optimize.root(f_system_fce, [T_last,muB_guess_FCE,muS_guess_FCE], args=alist, tol=root_tol)
                  else:
                    TMUFCE = optimize.root(f_system_fce, [T_last,muB_guess_FCE,muS_guess_FCE], args=alist, tol=root_tol)
                  if(TMUFCE.success):
                    if(verbose):
                        print("Results from successful FCE:")
                        print(str(TMUFCE.x[0])+"   "+str(TMUFCE.x[1])+"   "+str(TMUFCE.x[2]))
                    if(comp_all):
                        tempFCE[ff,i,j,k]=TMUFCE.x[0]
                        muFCE[ff,i,j,k,:]=TMUFCE.x[1:3]
                        entropy_density_fce=get_s_fce(TMUFCE.x[0],TMUFCE.x[1],TMUFCE.x[2])
                        sFCE[ff,i,j,k,:]=entropy_density_fce
                    elif(comp_trans):
                        tempFCE[ff,i,j]=TMUFCE.x[0]
                        muFCE[ff,i,j,:]=TMUFCE.x[1:3]
                        entropy_density_fce=get_s_fce(TMUFCE.x[0],TMUFCE.x[1],TMUFCE.x[2])
                        sFCE[ff,i,j,:]=entropy_density_fce
                    else:
                        tempFCE[ff,indx_cell]=TMUFCE.x[0]
                        muFCE[ff,indx_cell,:]=TMUFCE.x[1:3]
                        entropy_density_fce=get_s_fce(TMUFCE.x[0],TMUFCE.x[1],TMUFCE.x[2])
                        sFCE[ff,indx_cell]=entropy_density_fce
                    if(verbose):  
                      print("Entropy density accordinig to FCE: "+str(entropy_density_fce))
                  else:
                    if(verbose):  
                        print("FCE failed")

                  #now we turn our attention to the pressure components
                  glf_arg=Jb[i,j,k,0]*Jb[i,j,k,0]-Jb[i,j,k,1]*Jb[i,j,k,1]-Jb[i,j,k,2]*Jb[i,j,k,2]-Jb[i,j,k,3]*Jb[i,j,k,3]
                  if(glf_arg<=0):
                    continue
                  glf=np.sqrt(glf_arg)
                  uvel[:]=Jb[i,j,k,:]/glf
                  Lambda[0,0]=uvel[0]
                  Lambda[0,1:]=-uvel[1:]
                  Lambda[1:,0]=-uvel[1:]
                  for aa in range(1,4):
                      for bb in range(1,4):
                          Lambda[aa,bb]=kron(aa,bb)+uvel[aa]*uvel[bb]/(1+uvel[0])
                  LambdaF[0,0]=uvel[0]
                  LambdaF[0,1:]=uvel[1:]
                  LambdaF[1:,0]=uvel[1:]
                  for aa in range(1,4):
                      for bb in range(1,4):
                          LambdaF[aa,bb]=kron(aa,bb)+uvel[aa]*uvel[bb]/(1+uvel[0])
                  Tmunu[0,0]=np.sum(Tp[i,j,k,:,0])
                  for aa in range(1,4):
                      Tmunu[0,aa]=np.sum(Tp[i,j,k,:,aa])
                      Tmunu[aa,0]=Tmunu[0,aa]
                  Tmunu[1,1]=np.sum(Tp[i,j,k,:,4])
                  Tmunu[1,2]=np.sum(Tp[i,j,k,:,5])
                  Tmunu[2,1]=Tmunu[1,2]
                  Tmunu[1,3]=np.sum(Tp[i,j,k,:,6])
                  Tmunu[3,1]=Tmunu[1,3]
                  Tmunu[2,2]=np.sum(Tp[i,j,k,:,7])
                  Tmunu[2,3]=np.sum(Tp[i,j,k,:,8])
                  Tmunu[3,2]=Tmunu[2,3]
                  Tmunu[3,3]=np.sum(Tp[i,j,k,:,9])
                  Tmunu_lrf=np.matmul(Lambda.transpose(),np.matmul(Tmunu,Lambda))/nev_times_vol
                  if(verbose):
                      print("Pressure along x, y and z: "+str(Tmunu_lrf[1,1])+",  "+str(Tmunu_lrf[2,2])+",  "+str(Tmunu_lrf[3,3]))
                  if(comp_all):
                      pcomp[ff,i,j,k,:]=Tmunu_lrf[1,1],Tmunu_lrf[2,2],Tmunu_lrf[3,3]
                  elif(comp_trans):
                      pcomp[ff,i,j,:]=Tmunu_lrf[1,1],Tmunu_lrf[2,2],Tmunu_lrf[3,3]
                  else:
                      pcomp[ff,indx_cell,:]=Tmunu_lrf[1,1],Tmunu_lrf[2,2],Tmunu_lrf[3,3]
                          
    if(verbose):                   
        print("Input file processed.")
        print("Hadron counting:")
        for p in range(number_of_particles):
            print(pnames[p]+"    "+'{:12.7e}'.format(total_particles[ff,p]))

with open(outputfile,"wb") as po:
    if(comp_all):
        if(verbose):
             print("Pickling tt,xx,yy,zz,temp,tempBZ,muBZ,tempQS,muQS,tempPCE,muPCE,successPCE,tempFCE,muFCE,sFCE,rho_main,ndens,ene,total_particles,tempHGU,muHGU,sHGU,tempHBSQ,muHBSQ,pcomp")
        pickle.dump(("all_grid",tt[0:nt],xx,yy,zz,temp,tempBZ,muBZ,tempQS,muQS,tempPCE,muPCE,successPCE,tempFCE,muFCE,sFCE,rho_main,ndens,ene,total_particles,tempHGU,muHGU,sHGU,tempHBSQ,muHBSQ,pcomp),po)
    elif comp_trans:
        if(verbose):
            print("Pickling tt,xx,yy,z=z0,temp,tempBZ,muBZ,tempQS,muQS,tempPCE,muPCE,successPCE,tempFCE,muFCE,rho_main,ndens,ene,total_particles,tempHGU,muHGU,sHGU,tempHBSQ,muHBSQ,pcomp")
        pickle.dump(("only_tranverse_plane",tt[0:nt],xx,yy,zcoordinate,temp,tempBZ,muBZ,tempQS,muQS,tempPCE,muPCE,successPCE,tempFCE,muFCE,rho_main,ndens,ene,total_particles,tempHGU,muHGU,sHGU,tempHBSQ,muHBSQ,pcomp),po)
    else:
        if(verbose):
             print("Pickling tt,coordinate_list,temp,tempBZ,muBZ,tempQS,muQS,tempPCE,muPCE,successPCE,tempFCE,muFCE,sFCE,rho_main,ndens,ene,total_particles,tempHGU,muHGU,sHGU,tempHBSQ,muHBSQ,pcomp),po)")
        pickle.dump(("coordinate_list",tt[0:nt],cells_to_evaluate,temp,tempBZ,muBZ,tempQS,muQS,tempPCE,muPCE,successPCE,tempFCE,muFCE,sFCE,rho_main,ndens,ene,total_particles,tempHGU,muHGU,sHGU,tempHBSQ,muHBSQ,pcomp),po)

if(verbose):
    print("All done.")


