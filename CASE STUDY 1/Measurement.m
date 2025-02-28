function y = Measurement(X)
global C R

v = 0 + sqrt(R)*randn(1,1);
y = C*X + v;
end