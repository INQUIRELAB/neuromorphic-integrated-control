function y = Measurement(X)
global C R

v = 0 + sqrt(R)*randn(3,1);
y = C*X + v;
end