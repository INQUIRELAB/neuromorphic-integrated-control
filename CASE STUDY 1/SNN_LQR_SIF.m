% In the name of GOD 
% Ya Hussain (pbuh) peace be upon Him
%=====================
% Preparing MATLAB

clc
clear
close all
%=====================
% Definiton of Parameters
global A B C Q R NON Dim eta_sc_SIF landa_SIF D_SIF DT_SIF delta
global eta_sc_KF landa_KF D_KF DT_KF ICUC D_bar Kcnt miu nou 



x0 = [10;2];
ICUC = 1;
xhat0 = ICUC*x0;
Dim = length(x0);

A = [0 1;0 0];
B = [0;1];
C = [1 0];
Q_cnt = eye(2);
R_cnt = eye(1);

x_desired = [0;0];
xdot_desired = [0;0];


Q = eye(2)/1000;
R = eye(1)/100;
p0 = diag([0.1 .2]);
delta = .005;

[Kcnt,S,e] = lqr(A,B,Q_cnt,R_cnt);  % LQR gain


% Network Parameters
NON = 250;               % Number of neurons 200 and 100 ok but 50 failed

landa_SIF = .01;
eta_sc_SIF = 3000;

landa_KF = .01;
eta_sc_KF = 3000;

miu = 0.005;
nou = 0.005;

v0 = zeros(NON,1);


D_SIF  = 0.25*randn(Dim,NON)/1;    % State decoding matrix
DT_SIF = D_SIF';
r0_SIF = ICUC*pinv(D_SIF)*x0;
s_out0_SIF = zeros(NON,1);

D_KF  = 0.25*randn(Dim,NON)/1;    % State decoding matrix
DT_KF = D_KF';
r0_KF = ICUC*pinv(D_KF)*x0;
s_out0_KF = zeros(NON,1);

D_bar = randn(Dim,NON)/300;
Du = -Kcnt*(D_SIF - D_bar);

% Neurons firing threshold 
Threshold_SIF = zeros(1,NON);
for i = 1:NON
   Threshold_SIF(i) = (D_SIF(:,i)'*D_SIF(:,i) + nou*landa_SIF + miu*landa_SIF^2)/6; 
end

Threshold_KF = zeros(1,NON);
for i = 1:NON
   Threshold_KF(i) = (D_KF(:,i)'*D_KF(:,i) + nou*landa_KF + miu*landa_KF^2)/6;
end

%======================
% Simulation Parameters

dt = 0.01;
tf = 10;
tspan = 0:dt:tf;
E = length(tspan);
nMC = 1;
%======================
% Prelocation
x1_LQR_SIF = zeros(nMC,E);
x2_LQR_SIF = zeros(nMC,E);
x1_LQR_KF = zeros(nMC,E);
x2_LQR_KF = zeros(nMC,E);
x1_SNN_LQR_SIF = zeros(nMC,E);
x2_SNN_LQR_SIF = zeros(nMC,E);
x1_SNN_LQR_KF = zeros(nMC,E);
x2_SNN_LQR_KF = zeros(nMC,E);

x1_KF = zeros(nMC,E);
x2_KF = zeros(nMC,E);
x1_SIF = zeros(nMC,E);
x2_SIF = zeros(nMC,E);
x1_KF_SNN = zeros(nMC,E);
x2_KF_SNN = zeros(nMC,E);
x1_SIF_SNN = zeros(nMC,E);
x2_SIF_SNN = zeros(nMC,E);

xtilda1_KF = zeros(nMC,E);
xtilda2_KF = zeros(nMC,E);
xtilda1_SIF = zeros(nMC,E);
xtilda2_SIF = zeros(nMC,E);
xtilda1_KF_SNN = zeros(nMC,E);
xtilda2_KF_SNN = zeros(nMC,E);
xtilda1_SIF_SNN = zeros(nMC,E);
xtilda2_SIF_SNN = zeros(nMC,E);

P1_KF = zeros(nMC,E);
P2_KF = zeros(nMC,E);
P1_SIF = zeros(nMC,E);
P2_SIF = zeros(nMC,E);
P1_KF_SNN = zeros(nMC,E);
P2_KF_SNN = zeros(nMC,E);
P1_SIF_SNN = zeros(nMC,E);
P2_SIF_SNN = zeros(nMC,E);

R1_KF = zeros(nMC,E);
R2_KF = zeros(nMC,E);
R1_SIF = zeros(nMC,E);
R2_SIF = zeros(nMC,E);
R1_KF_SNN = zeros(nMC,E);
R2_KF_SNN = zeros(nMC,E);
R1_SIF_SNN = zeros(nMC,E);
R2_SIF_SNN = zeros(nMC,E);

for kk = 1:nMC
% disp(kk)

Xt_LQR_SIF = zeros(Dim,E);
Xt_LQR_KF = zeros(Dim,E);
Xt_SNN_LQR_SIF = zeros(Dim,E);
Xt_SNN_LQR_KF = zeros(Dim,E);
   
Xhatt_SIF = zeros(Dim,E);
Xhatt_SNN_SIF = zeros(Dim,E);
s_outt_SIF = zeros(NON,E);
Cov_SIF = zeros(Dim,E);
Xtilda_SIF = zeros(Dim,E); 
Xtilda_SNN_SIF = zeros(Dim,E); 
RMSE_SIF = zeros(Dim,E);
RMSE_SNN_SIF = zeros(Dim,E);

Xhatt_KF = zeros(Dim,E);
Xhatt_SNN_KF = zeros(Dim,E);
s_outt_KF = zeros(NON,E);
Cov_KF = zeros(Dim,E);
Xtilda_KF = zeros(Dim,E); 
Xtilda_SNN_KF = zeros(Dim,E); 
RMSE_KF = zeros(Dim,E);
RMSE_SNN_KF = zeros(Dim,E);

% Initialization
X_LQR_SIF = x0;
X_LQR_KF = x0;
X_SNN_LQR_SIF = x0;
X_SNN_LQR_KF = x0;

Xhat_SIF = xhat0;
Xhat_SNN_SIF = xhat0;
Pxx_SIF = p0;
v_SIF = v0;
s_out_SIF = s_out0_SIF;
r_SIF = r0_SIF;

Xhat_KF = xhat0;
Xhat_SNN_KF = xhat0;
Pxx_KF = p0;
v_KF = v0;
s_out_KF = s_out0_KF;  
r_KF = r0_KF;

for i = 1:E
  disp(i)
   %==================
   % Data Saving
   
   Xt_LQR_SIF(:,i) = X_LQR_SIF;
   Xt_LQR_KF(:,i) = X_LQR_KF;
   Xt_SNN_LQR_SIF(:,i) = X_SNN_LQR_SIF;
   Xt_SNN_LQR_KF(:,i) = X_SNN_LQR_KF;
   
   
   Xhatt_SIF(:,i) = Xhat_SIF;
   Xhatt_SNN_SIF(:,i) = Xhat_SNN_SIF;
   s_outt_SIF(:,i) = s_out_SIF;
   Cov_SIF(:,i) = diag(Pxx_SIF);
   Xtilda_SIF(:,i) = X_LQR_SIF - Xhat_SIF;
   Xtilda_SNN_SIF(:,i) = X_SNN_LQR_SIF - Xhat_SNN_SIF;
   RMSE_SIF(:,i) = (X_LQR_SIF - Xhat_SIF).^2;
   RMSE_SNN_SIF(:,i) = (X_SNN_LQR_SIF - Xhat_SNN_SIF).^2;
   
   Xhatt_KF(:,i) = Xhat_KF ;
   Xhatt_SNN_KF(:,i) = Xhat_SNN_KF ;
   s_outt_KF(:,i) = s_out_KF ;
   Cov_KF(:,i) = diag(Pxx_KF );
   Xtilda_KF(:,i) = X_LQR_KF - Xhat_KF ;
   Xtilda_SNN_KF(:,i) = X_SNN_LQR_KF - Xhat_SNN_KF ;
   RMSE_KF(:,i) = (X_LQR_KF - Xhat_KF).^2;
   RMSE_SNN_KF(:,i) = (X_SNN_LQR_KF - Xhat_SNN_KF).^2;
   %==================
   % Different control inputs
   
   U_LQR_SIF = -Kcnt*X_LQR_SIF;
   U_LQR_KF = -Kcnt*X_LQR_KF;
   U_SNN_LQR_SIF = Du*r_SIF;
   U_SNN_LQR_KF = Du*r_KF;
   
   U_SNN(:,i) = U_SNN_LQR_SIF;
   U_ALG(:,i) = U_LQR_SIF;
   
   % Dynamic simulation based on different control inputs
   
   %LQR-SIF
   f1 = dt*Dyn(X_LQR_SIF,U_LQR_SIF);
   f2 = dt*Dyn(X_LQR_SIF + f1/2,U_LQR_SIF);
   f3 = dt*Dyn(X_LQR_SIF + f2/2,U_LQR_SIF);
   f4 = dt*Dyn(X_LQR_SIF + f3,U_LQR_SIF);
   X_LQR_SIF = X_LQR_SIF + (f1 + 2*f2 + 2*f3 + f4)/6;
   
   y_LQR_SIF = Measurement(X_LQR_SIF);
   
   %LQR-KF
   f1 = dt*Dyn(X_LQR_KF,U_LQR_KF);
   f2 = dt*Dyn(X_LQR_KF + f1/2,U_LQR_KF);
   f3 = dt*Dyn(X_LQR_KF + f2/2,U_LQR_KF);
   f4 = dt*Dyn(X_LQR_KF + f3,U_LQR_KF);
   X_LQR_KF = X_LQR_KF + (f1 + 2*f2 + 2*f3 + f4)/6;
   
   y_LQR_KF = Measurement(X_LQR_KF);
   
   %SNN_LQR-SIF
   f1 = dt*Dyn(X_SNN_LQR_SIF,U_SNN_LQR_SIF);
   f2 = dt*Dyn(X_SNN_LQR_SIF + f1/2,U_SNN_LQR_SIF);
   f3 = dt*Dyn(X_SNN_LQR_SIF + f2/2,U_SNN_LQR_SIF);
   f4 = dt*Dyn(X_SNN_LQR_SIF + f3,U_SNN_LQR_SIF);
   X_SNN_LQR_SIF = X_SNN_LQR_SIF + (f1 + 2*f2 + 2*f3 + f4)/6;
   
   y_SNN_LQR_SIF = Measurement(X_SNN_LQR_SIF);
   
   %SNN_LQR-KF
   f1 = dt*Dyn(X_SNN_LQR_KF,U_SNN_LQR_KF);
   f2 = dt*Dyn(X_SNN_LQR_KF + f1/2,U_SNN_LQR_KF);
   f3 = dt*Dyn(X_SNN_LQR_KF + f2/2,U_SNN_LQR_KF);
   f4 = dt*Dyn(X_SNN_LQR_KF + f3,U_SNN_LQR_KF);
   X_SNN_LQR_KF = X_SNN_LQR_KF + (f1 + 2*f2 + 2*f3 + f4)/6;
   
   y_SNN_LQR_KF = Measurement(X_SNN_LQR_KF);
   %=====================================
   % SIF Estimator Simulation 
   
   yhat_SIF = Measurement(Xhat_SIF);
   Innovation_SIF = y_LQR_SIF - yhat_SIF;

   Pzz_SIF = C*Pxx_SIF*C' + R;
   K_SIF = pinv(C)*Sat(diag(Pzz_SIF)/delta);    % SIF Filter Gain
   
   t1 = dt*SIF_Dyn(Xhat_SIF,U_LQR_SIF,Innovation_SIF,K_SIF);
   t2 = dt*SIF_Dyn(Xhat_SIF + t1/2,U_LQR_SIF,Innovation_SIF,K_SIF);
   t3 = dt*SIF_Dyn(Xhat_SIF + t2/2,U_LQR_SIF,Innovation_SIF,K_SIF);
   t4 = dt*SIF_Dyn(Xhat_SIF + t3,U_LQR_SIF,Innovation_SIF,K_SIF);
   Xhat_SIF = Xhat_SIF + (t1 + 2*t2 + 2*t3 + t4)/6;
   
   h1 = dt*Cov_dyn(Pxx_SIF);
   h2 = dt*Cov_dyn(Pxx_SIF + h1/2);
   h3 = dt*Cov_dyn(Pxx_SIF + h2/2);
   h4 = dt*Cov_dyn(Pxx_SIF + h3);
   Pxx_SIF = Pxx_SIF + (h1 + 2*h2 + 2*h3 + h4)/6;
   %==================================
   % Neuromorphic Estimator SIF 
    
   % Network mempot. integration
   tt1 = dt*snn_dyn_sif(v_SIF,s_out_SIF,r_SIF,y_SNN_LQR_SIF,K_SIF,x_desired,xdot_desired);
   tt2 = dt*snn_dyn_sif(v_SIF + 0.5*tt1,s_out_SIF,r_SIF,y_SNN_LQR_SIF,K_SIF,x_desired,xdot_desired);
   tt3 = dt*snn_dyn_sif(v_SIF + 0.5*tt2,s_out_SIF,r_SIF,y_SNN_LQR_SIF,K_SIF,x_desired,xdot_desired);
   tt4 = dt*snn_dyn_sif(v_SIF + tt3,s_out_SIF,r_SIF,y_SNN_LQR_SIF,K_SIF,x_desired,xdot_desired);
   v_SIF = v_SIF + (1/6)*(tt1 + 2*tt2 + 2*tt3 + tt4);
    
    % Threshold crossing check
    for j = 1:NON
        if v_SIF(j) >= Threshold_SIF(j)
            s_out_SIF(j) = 1;
        else
            s_out_SIF(j) = 0;
        end
    end
    
    %=====================
    % Filtered spike trains 
    
    hh1 = dt*filtered_spike_dyn_SIF(r_SIF, s_out_SIF);
    hh2 = dt*filtered_spike_dyn_SIF(r_SIF + 0.5*hh1, s_out_SIF);
    hh3 = dt*filtered_spike_dyn_SIF(r_SIF + 0.5*hh2, s_out_SIF);
    hh4 = dt*filtered_spike_dyn_SIF(r_SIF + hh3, s_out_SIF);
    r_SIF = r_SIF + (1/6)*(hh1 + 2*hh2 + 2*hh3 + hh4);
    
    Xhat_SNN_SIF = D_SIF*r_SIF;    % decoding network output
   %=====================================
   % KF Estimator  Simulation 
   
   yhat_KF = Measurement(Xhat_KF);
   Innovation_KF = y_LQR_KF - yhat_KF ;

   K_KF  = Pxx_KF*C'*((R)^-1);    % KF Filter Gain
   
   tT1 = dt*KF_Dyn(Xhat_KF,U_LQR_KF,Innovation_KF,K_KF );
   tT2 = dt*KF_Dyn(Xhat_KF + tT1/2,U_LQR_KF,Innovation_KF,K_KF );
   tT3 = dt*KF_Dyn(Xhat_KF + tT2/2,U_LQR_KF,Innovation_KF,K_KF );
   tT4 = dt*KF_Dyn(Xhat_KF + tT3,U_LQR_KF,Innovation_KF,K_KF );
   Xhat_KF = Xhat_KF + (tT1 + 2*tT2 + 2*tT3 + tT4)/6;
   
   hH1 = dt*Cov_dyn(Pxx_KF);
   hH2 = dt*Cov_dyn(Pxx_KF + hH1/2);
   hH3 = dt*Cov_dyn(Pxx_KF + hH2/2);
   hH4 = dt*Cov_dyn(Pxx_KF + hH3);
   Pxx_KF = Pxx_KF + (hH1 + 2*hH2 + 2*hH3 + hH4)/6;
   %==================================
   % Neuromorphic Estimator SIF 
    
   % Network mempot. integration
   tTT1 = dt*snn_dyn_kf(v_KF ,s_out_KF,r_KF,y_SNN_LQR_KF,K_KF ,x_desired,xdot_desired);
   tTT2 = dt*snn_dyn_kf(v_KF  + 0.5*tTT1,s_out_KF,r_KF,y_SNN_LQR_KF,K_KF ,x_desired,xdot_desired);
   tTT3 = dt*snn_dyn_kf(v_KF  + 0.5*tTT2,s_out_KF,r_KF,y_SNN_LQR_KF,K_KF ,x_desired,xdot_desired);
   tTT4 = dt*snn_dyn_kf(v_KF  + tTT3,s_out_KF,r_KF,y_SNN_LQR_KF,K_KF ,x_desired,xdot_desired);
   v_KF  = v_KF  + (1/6)*(tTT1 + 2*tTT2 + 2*tTT3 + tTT4);
    
    % Threshold crossing check
    for j = 1:NON
        if v_KF (j) >= Threshold_KF (j)
            s_out_KF (j) = 1;
        else
            s_out_KF (j) = 0;
        end
    end
    
    %=====================
    % Filtered spike trains 
    
    hhH1 = dt*filtered_spike_dyn_KF(r_KF , s_out_KF );
    hhH2 = dt*filtered_spike_dyn_KF(r_KF  + 0.5*hhH1, s_out_KF );
    hhH3 = dt*filtered_spike_dyn_KF(r_KF  + 0.5*hhH2, s_out_KF );
    hhH4 = dt*filtered_spike_dyn_KF(r_KF  + hhH3, s_out_KF );
    r_KF = r_KF + (1/6)*(hhH1 + 2*hhH2 + 2*hhH3 + hhH4);
    
    Xhat_SNN_KF  = D_KF*r_KF ;     % decoding network output
end

x1_LQR_SIF(kk,:) = Xt_LQR_SIF(1,:);
x2_LQR_SIF(kk,:) = Xt_LQR_SIF(2,:);
x1_LQR_KF(kk,:) = Xt_LQR_KF(1,:);
x2_LQR_KF(kk,:) = Xt_LQR_KF(2,:);
x1_SNN_LQR_SIF(kk,:) = Xt_SNN_LQR_SIF(1,:);
x2_SNN_LQR_SIF(kk,:) = Xt_SNN_LQR_SIF(2,:);
x1_SNN_LQR_KF(kk,:) = Xt_SNN_LQR_KF(1,:);
x2_SNN_LQR_KF(kk,:) = Xt_SNN_LQR_KF(2,:);

x1_KF(kk,:) = Xhatt_KF(1,:);
x2_KF(kk,:) = Xhatt_KF(2,:);
x1_SIF(kk,:) = Xhatt_SIF(1,:);
x2_SIF(kk,:) = Xhatt_SIF(2,:);
x1_KF_SNN(kk,:) = Xhatt_SNN_KF(1,:);
x2_KF_SNN(kk,:) = Xhatt_SNN_KF(2,:);
x1_SIF_SNN(kk,:) = Xhatt_SNN_SIF(1,:);
x2_SIF_SNN(kk,:) = Xhatt_SNN_SIF(2,:);

 
xtilda1_KF(kk,:) = Xtilda_KF(1,:);
xtilda2_KF(kk,:) = Xtilda_KF(2,:);
xtilda1_SIF(kk,:) = Xtilda_SIF(1,:);
xtilda2_SIF(kk,:) = Xtilda_SIF(2,:);
xtilda1_KF_SNN(kk,:) = Xtilda_SNN_KF(1,:);
xtilda2_KF_SNN(kk,:) = Xtilda_SNN_KF(2,:);
xtilda1_SIF_SNN(kk,:) = Xtilda_SNN_SIF(1,:);
xtilda2_SIF_SNN(kk,:) = Xtilda_SNN_SIF(2,:);

P1_KF(kk,:) = Cov_KF(1,:);
P2_KF(kk,:) = Cov_KF(2,:);
P1_SIF(kk,:) = Cov_SIF(1,:);
P2_SIF(kk,:) = Cov_SIF(2,:);
P1_KF_SNN(kk,:) = Cov_KF(1,:);
P2_KF_SNN(kk,:) = Cov_KF(2,:);
P1_SIF_SNN(kk,:) = Cov_SIF(1,:);
P2_SIF_SNN(kk,:) = Cov_SIF(2,:);

R1_KF(kk,:) = RMSE_KF(1,:);
R2_KF(kk,:) = RMSE_KF(2,:);
R1_SIF(kk,:) = RMSE_SIF(1,:);
R2_SIF(kk,:) = RMSE_SIF(2,:);
R1_KF_SNN(kk,:) = RMSE_SNN_KF(1,:);
R2_KF_SNN(kk,:) = RMSE_SNN_KF(2,:);
R1_SIF_SNN(kk,:) = RMSE_SNN_SIF(1,:);
R2_SIF_SNN(kk,:) = RMSE_SNN_SIF(2,:);
end
%% Monte-Carlo mean computation

x1_LQR_SIF_MC = zeros(1,E);
x2_LQR_SIF_MC = zeros(1,E);
x1_LQR_KF_MC = zeros(1,E);
x2_LQR_KF_MC = zeros(1,E);
x1_SNN_LQR_SIF_MC = zeros(1,E);
x2_SNN_LQR_SIF_MC = zeros(1,E);
x1_SNN_LQR_KF_MC = zeros(1,E);
x2_SNN_LQR_KF_MC = zeros(1,E);

x1_KF_MC = zeros(1,E);
x2_KF_MC = zeros(1,E);
x1_SIF_MC = zeros(1,E);
x2_SIF_MC = zeros(1,E);
x1_KF_SNN_MC = zeros(1,E);
x2_KF_SNN_MC = zeros(1,E);
x1_SIF_SNN_MC = zeros(1,E);
x2_SIF_SNN_MC = zeros(1,E);

xtilda1_KF_MC = zeros(1,E);
xtilda2_KF_MC = zeros(1,E);
xtilda1_SIF_MC = zeros(1,E);
xtilda2_SIF_MC = zeros(1,E);
xtilda1_KF_SNN_MC = zeros(1,E);
xtilda2_KF_SNN_MC = zeros(1,E);
xtilda1_SIF_SNN_MC = zeros(1,E);
xtilda2_SIF_SNN_MC = zeros(1,E);

P1_KF_MC = zeros(1,E);
P2_KF_MC = zeros(1,E);
P1_SIF_MC = zeros(1,E);
P2_SIF_MC = zeros(1,E);
P1_KF_SNN_MC = zeros(1,E);
P2_KF_SNN_MC = zeros(1,E);
P1_SIF_SNN_MC = zeros(1,E);
P2_SIF_SNN_MC = zeros(1,E);

R1_KF_MC = zeros(1,E);
R2_KF_MC = zeros(1,E);
R1_SIF_MC = zeros(1,E);
R2_SIF_MC = zeros(1,E);
R1_KF_SNN_MC = zeros(1,E);
R2_KF_SNN_MC = zeros(1,E);
R1_SIF_SNN_MC = zeros(1,E);
R2_SIF_SNN_MC = zeros(1,E);

for i = 1:E
    
x1_LQR_SIF_MC(i) = mean(x1_LQR_SIF(:,i));
x2_LQR_SIF_MC(i) = mean(x2_LQR_SIF(:,i));
x1_LQR_KF_MC(i) = mean(x1_LQR_KF(:,i));
x2_LQR_KF_MC(i) = mean(x2_LQR_KF(:,i));
x1_SNN_LQR_SIF_MC(i) = mean(x1_SNN_LQR_SIF(:,i));
x2_SNN_LQR_SIF_MC(i) = mean(x2_SNN_LQR_SIF(:,i));
x1_SNN_LQR_KF_MC(i) = mean(x1_SNN_LQR_KF(:,i));
x2_SNN_LQR_KF_MC(i) = mean(x2_SNN_LQR_KF(:,i));

x1_KF_MC(i) = mean(x1_KF(:,i));
x2_KF_MC(i) = mean(x2_KF(:,i));
x1_SIF_MC(i) = mean(x1_SIF(:,i));
x2_SIF_MC(i) = mean(x2_SIF(:,i));
x1_KF_SNN_MC(i) = mean(x1_KF_SNN(:,i));
x2_KF_SNN_MC(i) = mean(x2_KF_SNN(:,i));
x1_SIF_SNN_MC(i) = mean(x1_SIF_SNN(:,i));
x2_SIF_SNN_MC(i) = mean(x2_SIF_SNN(:,i));

xtilda1_KF_MC(i) = mean(xtilda1_KF(:,i));
xtilda2_KF_MC(i) = mean(xtilda2_KF(:,i));
xtilda1_SIF_MC(i) = mean(xtilda1_SIF(:,i));
xtilda2_SIF_MC(i) = mean(xtilda2_SIF(:,i));
xtilda1_KF_SNN_MC(i) = mean(xtilda1_KF_SNN(:,i));
xtilda2_KF_SNN_MC(i) = mean(xtilda2_KF_SNN(:,i));
xtilda1_SIF_SNN_MC(i) = mean(xtilda1_SIF_SNN(:,i));
xtilda2_SIF_SNN_MC(i) = mean(xtilda2_SIF_SNN(:,i));

P1_KF_MC(i) = mean(P1_KF(:,i));
P2_KF_MC(i) = mean(P2_KF(:,i));
P1_SIF_MC(i) = mean(P1_SIF(:,i));
P2_SIF_MC(i) = mean(P2_SIF(:,i));
P1_KF_SNN_MC(i) = mean(P1_KF_SNN(:,i));
P2_KF_SNN_MC(i) = mean(P2_KF_SNN(:,i));
P1_SIF_SNN_MC(i) = mean(P1_SIF_SNN(:,i));
P2_SIF_SNN_MC(i) = mean(P2_SIF_SNN(:,i));

R1_KF_MC(i) = mean(R1_KF(:,i));
R2_KF_MC(i) = mean(R2_KF(:,i));
R1_SIF_MC(i) = mean(R1_SIF(:,i));
R2_SIF_MC(i) = mean(R2_SIF(:,i));
R1_KF_SNN_MC(i) = mean(R1_KF_SNN(:,i));
R2_KF_SNN_MC(i) = mean(R2_KF_SNN(:,i));
R1_SIF_SNN_MC(i) = mean(R1_SIF_SNN(:,i));
R2_SIF_SNN_MC(i) = mean(R2_SIF_SNN(:,i));
end

%% calculatng te mean error
total_error = zeros(1,E);
for i = 1:E
   total_error(i) = sqrt(x1_SNN_LQR_SIF_MC(i)^2 + x2_SNN_LQR_SIF_MC(i)^2); 
end

mean_error = mean(total_error(601:end));
disp('Error=')
disp(mean_error)
%% ========== Plots
% Controlled states

figure
subplot(2,2,1)
plot(tspan,x1_LQR_KF_MC,'b','linewidth',1.5)
hold on 
plot(tspan,x1_LQR_SIF_MC,'r-.','linewidth',1.5)
% plot(tspan,x1_SNN_LQR_KF_MC,'g--','linewidth',1.5)
plot(tspan,x1_SNN_LQR_SIF_MC,'k:','linewidth',1.5)
grid on 
ylabel("x_1")
legend("LQG","LQR-MSIF","SNN-LQR-MSIF")
xlim([0,max(tspan)])

subplot(2,2,2)
plot(tspan,x2_LQR_KF_MC,'b','linewidth',1.5)
hold on 
plot(tspan,x2_LQR_SIF_MC,'r-.','linewidth',1.5)
% plot(tspan,x2_SNN_LQR_KF_MC,'g--','linewidth',1.5)
plot(tspan,x2_SNN_LQR_SIF_MC,'k:','linewidth',1.5)
grid on 
ylabel("x_2")
xlim([0,max(tspan)])

 %3sigma stability
subplot(2,2,3)
plot(tspan,xtilda1_KF_MC,'b','linewidth',2)
hold on
plot(tspan,xtilda1_SIF_MC,'r--','linewidth',2)
% plot(tspan,xtilda1_KF_SNN_MC,'g-.','linewidth',2)
plot(tspan,xtilda1_SIF_SNN_MC,'k:','linewidth',2)
plot(tspan,3*sqrt(P1_KF_MC),'r--',tspan,3*sqrt(P1_SIF_MC),'k:','linewidth',2)
plot(tspan,-3*sqrt(P1_KF_MC),'r--',tspan,-3*sqrt(P1_SIF_MC),'k:','linewidth',2)
grid on
xlabel("time (s)")
ylabel('\deltax_1')
xlim([0,max(tspan)])

subplot(2,2,4)
plot(tspan,xtilda2_KF_MC,'b','linewidth',2)
hold on
plot(tspan,xtilda2_SIF_MC,'r--','linewidth',2)
% plot(tspan,xtilda2_KF_SNN_MC,'g-.','linewidth',2)
plot(tspan,xtilda2_SIF_SNN_MC,'k:','linewidth',2)
plot(tspan,3*sqrt(P2_KF_MC),'r--',tspan,3*sqrt(P2_SIF_MC),'k:','linewidth',2)
plot(tspan,-3*sqrt(P2_KF_MC),'r--',tspan,-3*sqrt(P2_SIF_MC),'k:','linewidth',2)
grid on
xlabel('time (s)')
ylabel('\deltax_2')
xlim([0,max(tspan)])
% 
% %% RMSE
% 
% figure
% subplot(2,1,1)
% semilogy(tspan, sqrt(R1_KF_MC),'b','linewidth',2)
% hold on 
% semilogy(tspan, sqrt(R1_SIF_MC),'r-.','linewidth',2)
% semilogy(tspan, sqrt(R1_KF_SNN_MC),'g--','linewidth',2)
% semilogy(tspan, sqrt(R1_SIF_SNN_MC),'k:','linewidth',2)
% grid on
% legend('KF' ,'SIF','SNN-KF','SNN-SIF')
% ylabel('\deltax_1')
% xlim([0,max(tspan)])
% 
% subplot(2,1,2)
% semilogy(tspan, sqrt(R2_KF_MC),'b','linewidth',2)
% hold on 
% semilogy(tspan, sqrt(R2_SIF_MC),'r-.','linewidth',2)
% semilogy(tspan, sqrt(R2_KF_SNN_MC),'g--','linewidth',2)
% semilogy(tspan, sqrt(R2_SIF_SNN_MC),'k:','linewidth',2)
% grid on
% xlabel('time (s)')
% ylabel('\deltax_2')
% xlim([0,max(tspan)])

%% ===================

% figure
% 
% imagesc(s_outt_KF)
% title('Spiking Pattern SNN-KF')
% xlabel('time step')
% ylabel('Neuron index')

figure
imagesc(s_outt_SIF)
title('Spiking Pattern SNN-SIF')
xlabel('time step')
ylabel('Neuron index')

disp(sum(sum(s_outt_SIF)))


% MX1_SIF = [MRNX1_SIF_N50,MRNX1_SIF_N100,MRNX1_SIF_N150,MRNX1_SIF_N200,MRNX1_SIF_N250,MRNX1_SIF_N300,MRNX1_SIF_N350,MRNX1_SIF_N400,MRNX1_SIF_N450];
% MX2_SIF = [MRNX2_SIF_N50,MRNX2_SIF_N100,MRNX2_SIF_N150,MRNX2_SIF_N200,MRNX2_SIF_N250,MRNX2_SIF_N300,MRNX2_SIF_N350,MRNX2_SIF_N400,MRNX2_SIF_N450];
% 
% MX1_KF = [MRNX1_KF_N50,MRNX1_KF_N100,MRNX1_KF_N150,MRNX1_KF_N200,MRNX1_KF_N250,MRNX1_KF_N300,MRNX1_KF_N350,MRNX1_KF_N400,MRNX1_KF_N450];
% MX2_KF = [MRNX2_KF_N50,MRNX2_KF_N100,MRNX2_KF_N150,MRNX2_KF_N200,MRNX2_KF_N250,MRNX2_KF_N300,MRNX2_KF_N350,MRNX2_KF_N400,MRNX2_KF_N450];
%% RMSE Vs N plot
% 
% figure
% 
% plot(N,MX1_KF,'b','linewidth',2)
% hold on 
% plot(N,MX1_SIF,'k:','linewidth',2)
% plot(N,MX2_KF,'g--','linewidth',2)
% plot(N,MX2_SIF,'r-.','linewidth',2)
% grid on 
% xlabel("Number of neurons")
% ylabel('RMSE')
% legend('x_1 SNN-KF','x_1 SNN-SIF','x_2 SNN-KF','x_2 SNN-SIF')
% 
