function Xhat_dot = KF_Dyn(Xhat,U,Innovation,K_KF)
global A B

Xhat_dot = A*Xhat + B*U + K_KF*(Innovation);
end