function vdot = snn_dyn_kf(v,s_out,r,y,K_KF,x_desired,xdot_desired)
global A B C D_KF DT_KF NON eta_sc_KF landa_KF D_bar Kcnt miu

Wslow = DT_KF*(A + landa_KF*eye(2))*D_KF;
Wfast = -DT_KF*D_KF + miu*(landa_KF^2)*eye(NON);
Wk = -DT_KF*(K_KF)*C*D_KF;
Fk = DT_KF*(K_KF);
Wcnt = -DT_KF*B*Kcnt*D_KF;
Wbarcnt = DT_KF*B*Kcnt*D_bar;
vdot = -landa_KF*v + Wslow*r + Wfast*s_out + Wcnt*r + Wbarcnt*r + Wk*r + Fk*y + D_bar'*(xdot_desired + landa_KF*x_desired) - (D_bar'*D_bar  + miu*(landa_KF^2)*eye(NON))*s_out + randn(NON,1)/eta_sc_KF;
end