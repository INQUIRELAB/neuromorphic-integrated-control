function Pxx_dot = Cov_dyn(Pxx)
global A C Q R

Pxx_dot = A*Pxx + Pxx*A' + Q - Pxx*C'*(R^(-1))*C*Pxx;

end