function rdot = filtered_spike_dyn_SIF(r_SIF,s_out_SIF)
global landa_SIF 
    rdot = -landa_SIF*r_SIF + s_out_SIF;
    
end