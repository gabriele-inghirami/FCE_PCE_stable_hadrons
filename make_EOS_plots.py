#version 6.5.0, 15/03/2022
#it plots also the energy density and the baryon density
#it plots also nh/nb and pt/pl
#it goes with store_temp v 6.4
#it plots also the entropy density
#a maximum can be set for the temperature to display
import matplotlib.pyplot as plt 
import pickle
import numpy as np
import sys
import os.path

temp_max=250 #maximum temperature to plot
chem_min=-100 #minimum chemical potential to plot
s_max=15 #maximum s to plot

linestyles = ['-', '--', '-.', ':', '--']


if(len(sys.argv) != 4):
        print("Syntax: python3 make_EOS_plots.py <inputfile> <index_in_the_list_of_cells_from_0> <outputfile>")
        sys.exit(1)

inputfile=sys.argv[1]
index_cell=int(sys.argv[2])
outputfile=sys.argv[3]

if(not os.path.isfile(inputfile)):
    print(inputfile+" not found. Exiting.")
    sys.exit(2)

with open(inputfile,"rb") as infi:
    data=pickle.load(infi)

tipo,tt,cells_to_evaluate,temp,tempBZ,muBZ,tempQS,muQS,tempPCE,muPCE,tempFCE,muFCE,sFCE,rho_main,ndens,ene,total_particles,tempHGU,muHGU,sHGU,tempHBSQ,muHBSQ,pcomp=data[:]


if(not type(cells_to_evaluate) is tuple):
    cells_to_evaluate=(cells_to_evaluate,)

if(index_cell>len(cells_to_evaluate)-1):
    print("Error, you have chosen an index larger then the maximum of the list of cells: "+str(index_cell)+" vs "+str(len(cells_to_evaluate)-1))
    sys.exit(2)
else:
    pos_string="x="+'{:6.2f}'.format(cells_to_evaluate[index_cell][0])+"fm, y="+'{:6.2f}'.format(cells_to_evaluate[index_cell][1])+"fm, z="+'{:6.2f}'.format(cells_to_evaluate[index_cell][2])+"fm"


#dictionary with hadrons to be plotted (True) or not (False)
#the index is probably useless, as dictionaries should currently preserve the order of the entries, but better safe than sorry considering also the minimum overhead
#hadrons=[]
hadrons = {
        'pion_minus' : (True,r"$\pi^-$",0),
        'pion_plus' : (False,r"$\pi^+$",1),
        'pion_zero' : (False,r"$\pi^0$",2),
        'kaon_zero' : (False,r"$K^0$",3),
        'kaon_plus' : (False,r"$K^+$",4),
        'kaon_minus' : (True,r"$K^-$",5),
        'kaon_zero_bar' : (False,r"$\bar K^-$",6),
        'Neutron' : (True,r"$n$",7),
        'Proton' : (False,r"$p$",8),
        'anti-Proton' : (False,r"$\bar p$",9),
        'anti-Neutron' : (False, r"$\bar n$",10),
        'eta' : (True, r"$\eta$",11),
        'omega_meson' : (False, r"$\omega$",12),
        'eta_prime' : (False, r"$\eta'$",13),
        'phi' : (False,r"$\phi$",14),
        'Lambda1116' : (True,r"$\Lambda_{1116}$",15),
        'anti-Lambda1116' : (False,r"$\bar \Lambda_{1116}$",16),
        'Sigma1192_minus' : (False,r"$\Sigma_{1192}^-$",17),
        'Sigma1192_plus' : (False,r"$\Sigma_{1192}^+$",18),
        'Sigma1192' : (True,r"$\Sigma_{1192}$",19),
        'anti-Sigma1192_minus' : (False,r"$\bar \Sigma_{1192}^-$",20),
        'anti-Sigma1192_plus' : (False,r"$\bar \Sigma_{1192}^+$",21),
        'anti-Sigma1192' : (False,r"$\Sigma_{1192}$",22),
        'Xi1317_minus' : (False,r"$\Xi_{1317}^-$",23),
        'Xi1317_zero' : (False,r"$\Xi_{1317}^0$",24),
        'anti-Xi1317_zero' : (False,r"$\bar \Xi_{1317}^-$",25),
        'Xi1317_plus' : (False,r"$\Xi_{1317}^+$",26),
        'Lambda1520' : (False,r"$\Lambda_{1520}$",27),
        'anti-Lambda1520' : (False,r"$\bar \Lambda_{1520}$",28),
        'Xi1530_minus' : (False,r"$\Xi_{1530}^-$",29),
        'Xi1530_zero' : (False,r"$\Xi_{1530}^0$",30),
        'anti-Xi1530_zero' : (False,r"$\bar \Xi_{1530}^0$",31),
        'Xi1530_plus' : (False,r"$\bar \Xi_{1530}^+$",32),
        'Omega1672' : (False,r"$\Omega_{1672}$",33),
        'anti-Omega1672' : (False,r"$\bar \Omega_{1672}$",34)
          }

print("\nPlotting temperature from e/n at "+pos_string)
plt.title("Temperature from e/n at "+pos_string)
plt.xlabel('time [fm]')
plt.ylabel('T [MeV]')
for vals in hadrons.values():
    if vals[0]==True:
        index_hadron=vals[2]
        tmax=1000*np.amax(temp[:,index_cell,index_hadron])
        if(tmax>temp_max):
            plt.ylim(top=temp_max)
            break
for vals in hadrons.values():
    if vals[0]==True:
        index_hadron=vals[2]
        hadron_name=vals[1]
        print("Plotting "+hadron_name)
        plt.plot(tt,1000*temp[:,index_cell,index_hadron],label=hadron_name)
plt.legend()
plt.tight_layout()
plt.savefig(outputfile+"_temperature_e_over_n.png",dpi=150,pad_inches=0.)
plt.close('all')

print("\nPlotting temp. from Boltzmann distr. at "+pos_string+"\n")
plt.title("Temp. from Boltzmann at "+pos_string)
plt.xlabel('time [fm]')
plt.ylabel('T [MeV]')
for vals in hadrons.values():
    if vals[0]==True:
        index_hadron=vals[2]
        tmax=1000*np.amax(tempBZ[:,index_cell,index_hadron])
        if(tmax>temp_max):
            plt.ylim(top=temp_max)
            break
for vals in hadrons.values():
    if vals[0]==True:
        index_hadron=vals[2]
        hadron_name=vals[1]
        plt.plot(tt,1000*tempBZ[:,index_cell,index_hadron],label=hadron_name)
plt.legend()
plt.tight_layout()
plt.savefig(outputfile+"_temperature_BZ.png",dpi=150,pad_inches=0.)
plt.close('all')

print("\nPlotting "+r"$\mu$"+" from Boltzmann distr. at "+pos_string+"\n")
plt.title(r"$\mu$"+" from Boltzmann at "+pos_string)
plt.xlabel('time [fm]')
plt.ylabel(r"$\mu$"+ ' [MeV]')
for vals in hadrons.values():
    if vals[0]==True:
        index_hadron=vals[2]
        chmin=1000*np.amin(muBZ[:,index_cell,index_hadron])
        if(chmin<chem_min):
            plt.ylim(chem_min,1000)
            break
for vals in hadrons.values():
    if vals[0]==True:
        index_hadron=vals[2]
        hadron_name=vals[1]
        plt.plot(tt,1000*muBZ[:,index_cell,index_hadron],label=hadron_name)
plt.legend()
plt.tight_layout()
plt.savefig(outputfile+"_chem_pot_BZ.png",dpi=150,pad_inches=0.)
plt.close('all')

print("\nPlotting temp. from FD/BE distr. at "+pos_string+"\n")
plt.title("Temp. from FD/BE at "+pos_string)
plt.xlabel('time [fm]')
plt.ylabel('T [MeV]')
for vals in hadrons.values():
    if vals[0]==True:
        index_hadron=vals[2]
        tmax=1000*np.amax(tempQS[:,index_cell,index_hadron])
        if(tmax>temp_max):
            plt.ylim(top=temp_max)
            break
for vals in hadrons.values():
    if vals[0]==True:
        index_hadron=vals[2]
        hadron_name=vals[1]
        plt.plot(tt,1000*tempQS[:,index_cell,index_hadron],label=hadron_name)
plt.legend()
plt.tight_layout()
plt.savefig(outputfile+"_temperature_FD_BE.png",dpi=150,pad_inches=0.)
plt.close('all')

print("\nPlotting "+r"$\mu$"+" from FD/BE distr. at "+pos_string+"\n")
plt.title(r"$\mu$"+" from FD/BE at "+pos_string)
plt.xlabel('time [fm]')
plt.ylabel(r"$\mu$"+ ' [MeV]')
for vals in hadrons.values():
    if vals[0]==True:
        index_hadron=vals[2]
        chmin=1000*np.amin(muQS[:,index_cell,index_hadron])
        if(chmin<chem_min):
            plt.ylim(chem_min,1000)
            break
for vals in hadrons.values():
    if vals[0]==True:
        index_hadron=vals[2]
        hadron_name=vals[1]
        plt.plot(tt,1000*muQS[:,index_cell,index_hadron],label=hadron_name)
plt.legend()
plt.tight_layout()
plt.savefig(outputfile+"_chem_pot_FD_BE.png",dpi=150,pad_inches=0.)
plt.close('all')

print("\nPlotting "+r"$\mu$"+" from FCE, HG and BQS at "+pos_string+"\n")
plt.title(r"$\mu$"+" from FCE, HG and BQS at "+pos_string)
plt.xlabel('time [fm]')
plt.ylabel(r"$\mu$"+ ' [MeV]')
if((1000*np.amin(muHBSQ[:,index_cell,4])<chem_min) or (1000*np.amin(muHBSQ[:,index_cell,3])<chem_min) or (1000*np.amin(muHGU[:,index_cell])<chem_min) or (1000*np.amin(muFCE[:,index_cell,1])<chem_min) or (1000*np.amin(muFCE[:,index_cell,0])<chem_min)):
    plt.ylim(bottom=chem_min)
plt.plot(tt,1000*muFCE[:,index_cell,0],label=r"$\mu_B$ (FCE)")
plt.plot(tt,1000*muFCE[:,index_cell,1],label=r"$\mu_S$ (FCE)")
plt.plot(tt,1000*muHGU[:,index_cell],label=r"$\mu_B$ (HGU)")
plt.plot(tt,1000*muHBSQ[:,index_cell,3],label=r"$\mu_B$ (BSQ)")
plt.plot(tt,1000*muHBSQ[:,index_cell,4],label=r"$\mu_S$ (BSQ)")
plt.legend()
plt.tight_layout()
plt.savefig(outputfile+"_chem_pot_FCE.png",dpi=150,pad_inches=0.)
plt.close('all')

print("\nPlotting temp. from PCE, FCE, HG and BQS at "+pos_string+"\n")
plt.title("Temperature at "+pos_string)
plt.xlabel('time [fm]')
plt.ylabel('T [MeV]')
pce_vals_arr=np.zeros(len(tt),dtype=np.float64)
nump=len(tempPCE[0,0,:])#we obtain the number of particles
#we create an array with the results of the PCE EoS
for time_int in range(len(tt)):
    for ii in range(nump-1,-1,-1): #we select the last valid entry 
        if(tempPCE[time_int,index_cell,ii]!=0):
              pce_vals_arr[time_int]=tempPCE[time_int,index_cell,ii]
              break #once we have found the last filled value (i.e. the heaviest hadron for which PCE was possible), we move to the next timestep
if((1000*np.amax(tempHBSQ[:,index_cell,1])>temp_max) or (1000*np.amax(tempFCE[:,index_cell])>temp_max) or (1000*np.amax(tempHGU[:,index_cell])>temp_max) or(1000*np.amax(pce_vals_arr)>temp_max)):
    plt.ylim(top=temp_max)
plt.plot(tt,1000*pce_vals_arr,label="PCE")
plt.plot(tt,1000*tempFCE[:,index_cell],label="FCE")
plt.plot(tt,1000*tempHGU[:,index_cell],label="HGU",linestyle="dashed")
plt.plot(tt,1000*tempHBSQ[:,index_cell,1],label="BQS",linestyle="dotted")
plt.legend()
plt.tight_layout()
plt.savefig(outputfile+"_temperature_PCE_FCE_HG_BQS.png",dpi=150,pad_inches=0.)
plt.close('all')

print("\nPlotting the entropy density at "+pos_string+"\n")
plt.title("Entropy density from FCE and HG at "+pos_string)
plt.xlabel('time [fm]')
plt.ylabel(r"$s [fm^{-3}]$")
if((np.amax(sFCE[:,index_cell])>s_max) or (np.amax(sHGU[:,index_cell])>s_max)):
    plt.ylim(top=s_max)
plt.plot(tt,sFCE[:,index_cell],label="FCE")
plt.plot(tt,sHGU[:,index_cell],linestyle="dashed",label="HGU")
plt.legend()
plt.tight_layout()
plt.savefig(outputfile+"_entropy_density.png",dpi=150,pad_inches=0.)
plt.close('all')

print("\nPlotting temp. from QS, PCE and FCE at "+pos_string+"\n")
plt.title("Temperature at "+pos_string)
plt.xlabel('time [fm]')
plt.ylabel('T [MeV]')
pce_vals_arr=np.zeros(len(tt),dtype=np.float64)
nump=len(tempPCE[0,0,:])#we obtain the number of particles
#we create an array with the results of the PCE EoS
for time_int in range(len(tt)):
    for ii in range(nump-1,-1,-1): #we select the last valid entry 
        if(tempPCE[time_int,index_cell,ii]!=0):
              pce_vals_arr[time_int]=tempPCE[time_int,index_cell,ii]
              break #once we have found the last filled value (i.e. the heaviest hadron for which PCE was possible), we move to the next timestep
if((1000*np.amax(tempQS[:,index_cell,hadrons['pion_plus'][2]])>temp_max) or (1000*np.amax(tempFCE[:,index_cell])>temp_max) or (1000*np.amax(tempQS[:,index_cell,hadrons['kaon_plus'][2]])>temp_max) or (1000*np.amax(tempQS[:,index_cell,hadrons['Neutron'][2]])>temp_max) or (1000*np.amax(tempQS[:,index_cell,hadrons['eta'][2]])>temp_max) or (1000*np.amax(tempQS[:,index_cell,hadrons['Lambda1116'][2]])>temp_max) or(1000*np.amax(pce_vals_arr)>temp_max)):
    plt.ylim(top=temp_max)
plt.plot(tt,1000*tempQS[:,index_cell,hadrons['pion_plus'][2]],label=hadrons['pion_plus'][1])
plt.plot(tt,1000*tempQS[:,index_cell,hadrons['kaon_plus'][2]],label=hadrons['kaon_plus'][1])
plt.plot(tt,1000*tempQS[:,index_cell,hadrons['Neutron'][2]],label=hadrons['Neutron'][1])
plt.plot(tt,1000*tempQS[:,index_cell,hadrons['eta'][2]],label=hadrons['eta'][1])
plt.plot(tt,1000*tempQS[:,index_cell,hadrons['Lambda1116'][2]],label=hadrons['Lambda1116'][1])
plt.plot(tt,1000*pce_vals_arr,label="PCE")
plt.plot(tt,1000*tempFCE[:,index_cell],label="FCE")
plt.legend()
plt.tight_layout()
plt.savefig(outputfile+"_temperature_QS_PCE_FCE.png",dpi=150,pad_inches=0.)
plt.close('all')

print("\nPlotting the ratio between pions+ and the net baryon number at "+pos_string+"\n")
plt.title(r"$n_{\pi^+}/n_B$ at "+pos_string)
plt.xlabel('time [fm]')
plt.ylabel(r"$n_{\pi^+}/n_B$")
ratios=np.zeros(len(tt),dtype=np.float64)
for ii in range(len(tt)):
    if(rho_main[ii,index_cell,0]!=0):
        ratios[ii]=ndens[ii,index_cell,hadrons['pion_plus'][2]]/rho_main[ii,index_cell,0]

plt.plot(tt,ratios)
plt.tight_layout()
plt.savefig(outputfile+"_pion_plus_over_net_baryon_density.png",dpi=150,pad_inches=0.)
plt.close('all')

print("\nPlotting the ratio between the transverse and the longitudinal pressure at "+pos_string+"\n")
plt.title(r"$p_{\perp}/p_{\parallel}$ at "+pos_string)
plt.xlabel('time [fm]')
plt.ylabel(r"$p_{\perp}/p_{\parallel}$")
ratios=np.zeros(len(tt),dtype=np.float64)
for ii in range(len(tt)):
    if(pcomp[ii,index_cell,2]!=0):
        ratios[ii]=0.5*(pcomp[ii,index_cell,0]+pcomp[ii,index_cell,1])/pcomp[ii,index_cell,2]

plt.plot(tt,ratios)
plt.tight_layout()
plt.savefig(outputfile+"_pressure_ratio.png",dpi=150,pad_inches=0.)
plt.close('all')

print("\nPlotting temp. PCE at "+pos_string+"\n")
pce_vals_arr=np.zeros(len(tt),dtype=np.float64)
nump=len(tempPCE[0,0,:])#we obtain the number of particles
it=0
ic=0
#we create an array with the results of the PCE EoS
for k, v in hadrons.items():  
    plt.title("Temperature at "+pos_string)
    plt.xlabel('time [fm]')
    plt.ylabel('T [MeV]')
    plt.ylim(top=180)
    plt.plot(tt,1000*tempPCE[:,index_cell,v[2]],label=v[1],linestyle=linestyles[it])
    it=it+1
    if((it%4==0) or (it==nump-1)):
      it=0
      ic=ic+1
      plt.legend()
      plt.tight_layout()
      plt.savefig("check_pce_"+outputfile+"_part_"+'{:1d}'.format(ic)+".png",dpi=150,pad_inches=0.)
      plt.close('all')

plt.close('all')
print("\nPlotting the energy density at "+pos_string+"\n")
plt.title("Energy density at "+pos_string)
plt.xlabel('time [fm]')
plt.ylabel(r"$\varepsilon$ [GeV/fm$^3$]")
plt.ylim(auto=True)
edens=np.sum(ene,axis=2)
plt.plot(tt,edens[:,index_cell])
plt.tight_layout()
plt.savefig(outputfile+"_energy_density.png",dpi=150,pad_inches=0.)
plt.close('all')

print("\nPlotting the net baryon density at "+pos_string+"\n")
plt.title("Net baryon density at "+pos_string)
plt.xlabel('time [fm]')
plt.ylabel(r"$n_B$ [fm$^{-3}$]")
plt.ylim(auto=True)
plt.plot(tt,rho_main[:,index_cell,0])
plt.tight_layout()
plt.savefig(outputfile+"_baryon_density.png",dpi=150,pad_inches=0.)
plt.close('all')
