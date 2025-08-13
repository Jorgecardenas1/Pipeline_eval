%%% TRM CALIBRTION USING 7 ERRORS TERMS %%%

%% clearvars %% 
clear all
close all

%% S-Parameters for measured calibration stems %%


 S_meas_refl = sparameters('reflector2.s2p');                             
 s11_meas_refl = rfparam(S_meas_refl,2,2);


 f=S_meas_refl.Frequencies;
 N=length(f);
  
 
 %% correct S data
S_dut = sparameters('absorber2.s2p');
S11_dut_meas = rfparam(S_dut,2,2); %S21_dut_meas = rfparam(S_dut,2,1)
S11_real=abs(S11_dut_meas)./abs(s11_meas_refl);


fghz=f./1e9;
 figure(1)
 plot(fghz,S11_real);%,fghz,20*log10(abs(S12_dut_true)))
 ylim([-1 1])
