function Xhat_dot = SIF_Dyn(Xhat,U,Innovation,K_SIF)
global A B

Xhat_dot = A*Xhat + B*U + K_SIF*(Innovation);
end