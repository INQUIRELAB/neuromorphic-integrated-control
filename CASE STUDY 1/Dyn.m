function xdot = Dyn(X,U)
global A B Q

w = 0 + sqrt(Q)*randn(2,1);
xdot = A*X + B*U + w;
end