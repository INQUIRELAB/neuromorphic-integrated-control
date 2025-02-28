function vdot = snn_dyn_sif(v,s_out,r,y,K_SIF,x_desired,xdot_desired)
global A B C D_SIF DT_SIF NON eta_sc_SIF landa_SIF D_bar Kcnt miu 

Wslow = DT_SIF*(A + landa_SIF*eye(2))*D_SIF;
Wfast = -DT_SIF*D_SIF + miu*(landa_SIF^2)*eye(NON);
Wk = -DT_SIF*(K_SIF)*C*D_SIF;
Fk = DT_SIF*(K_SIF);
Wcnt = -DT_SIF*B*Kcnt*D_SIF;
Wbarcnt = DT_SIF*B*Kcnt*D_bar;
vdot = -landa_SIF*v + Wslow*r + Wfast*s_out + Wcnt*r + Wbarcnt*r + Wk*r + Fk*y + D_bar'*(xdot_desired + landa_SIF*x_desired) - (D_bar'*D_bar + miu*(landa_SIF^2)*eye(NON))*s_out + randn(NON,1)/eta_sc_SIF;
end