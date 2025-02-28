function rdot = filtered_spike_dyn_KF(r_KF,s_out_KF)
global landa_KF 
    rdot = -landa_KF*r_KF + s_out_KF;
    
end