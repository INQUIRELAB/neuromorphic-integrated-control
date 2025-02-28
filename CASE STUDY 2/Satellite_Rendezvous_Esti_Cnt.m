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
global eta_sc_KF landa_KF D_KF DT_KF ICUC miuu miu nou Kcnt D_bar

r0 = [70;30;-5];
v0 = 1*[-1.7;-0.9;0.25];
x0 = [r0;v0];

ICUC = 1;
xhat0 = ICUC*x0;
Dim = length(x0);

miuu = 398600; %km^3/s^2
Req = 6371 + 400; %km
n   = sqrt(miuu/Req^3); % Mean Motion

tic
% Define System Matrice: xdot = Ax + Bu
A11 = zeros(3,3);
A12 = eye(3);
A21 = zeros(3,3);
A22 = [0 0 2*n; 0 -n^2 0; -2*n 0 2*n^2];

B1 = zeros(3,3);
B2 = eye(3);

A = [A11, A12; A21, A22];
B = [B1;B2];
C = [eye(3),zeros(3,3)];

Q_cnt = (1e-6)*eye(6);
R_cnt = 1*eye(3)/1;


x_desired = [0;0;0;0;0;0];
xdot_desired = [0;0;0;0;0;0];

p0 = diag((1e-9)*[100000 100000 100000  .2 .2 .2]);
Q = (1e-12)*diag([ones(1,3) ones(1,3)/1]);
R = (1e-2)*eye(3);

delta = .0095;

[Kcnt,S,e] = lqr(A,B,Q_cnt,R_cnt);  % LQR gain




% Network Parameters
NON = 350;               % Number of neurons 200 and 100 ok  but 50 failed

landa_SIF = .023;
eta_sc_SIF = 3000;

landa_KF = .0001;
eta_sc_KF = 3000;

miu = 1;
nou = .0001;

v0 = zeros(NON,1);


D_SIF  = 1*randn(Dim,NON)/50;    % State decoding matrix
DT_SIF = D_SIF';
r0_SIF = ICUC*pinv(D_SIF)*x0;
s_out0_SIF = zeros(NON,1);

D_KF  = 1*randn(Dim,NON)/50;    % State decoding matrix
DT_KF = D_KF';
r0_KF = ICUC*pinv(D_KF)*x0;
s_out0_KF = zeros(NON,1);

D_bar = 1*randn(Dim,NON)/2000;
Du = -Kcnt*(D_SIF - D_bar);

% Neurons firing threshold 
Threshold_SIF = zeros(1,NON);
for i = 1:NON
   Threshold_SIF(i) = (D_SIF(:,i)'*D_SIF(:,i) + nou*landa_SIF + miu*landa_SIF^2)/2; 
end

Threshold_KF = zeros(1,NON);
for i = 1:NON
   Threshold_KF(i) = (D_KF(:,i)'*D_KF(:,i) + nou*landa_KF + miu*landa_KF^2)/2;
end


%======================
% Simulation Parameters

dt = 0.05;
tf = 360;
tspan = 0:dt:tf;
E = length(tspan);
nMC = 1;
%======================
% Prelocation

x1_LQR_SIF = zeros(nMC,E);
x2_LQR_SIF = zeros(nMC,E);
x3_LQR_SIF = zeros(nMC,E);
x4_LQR_SIF = zeros(nMC,E);
x5_LQR_SIF = zeros(nMC,E);
x6_LQR_SIF = zeros(nMC,E);

x1_LQR_KF = zeros(nMC,E);
x2_LQR_KF = zeros(nMC,E);
x3_LQR_KF = zeros(nMC,E);
x4_LQR_KF = zeros(nMC,E);
x5_LQR_KF = zeros(nMC,E);
x6_LQR_KF = zeros(nMC,E);

x1_SNN_LQR_SIF = zeros(nMC,E);
x2_SNN_LQR_SIF = zeros(nMC,E);
x3_SNN_LQR_SIF = zeros(nMC,E);
x4_SNN_LQR_SIF = zeros(nMC,E);
x5_SNN_LQR_SIF = zeros(nMC,E);
x6_SNN_LQR_SIF = zeros(nMC,E);

x1_SNN_LQR_KF = zeros(nMC,E);
x2_SNN_LQR_KF = zeros(nMC,E);
x3_SNN_LQR_KF = zeros(nMC,E);
x4_SNN_LQR_KF = zeros(nMC,E);
x5_SNN_LQR_KF = zeros(nMC,E);
x6_SNN_LQR_KF = zeros(nMC,E);

x1_KF = zeros(nMC,E);
x2_KF = zeros(nMC,E);
x3_KF = zeros(nMC,E);
x4_KF = zeros(nMC,E);
x5_KF = zeros(nMC,E);
x6_KF = zeros(nMC,E);

x1_SIF = zeros(nMC,E);
x2_SIF = zeros(nMC,E);
x3_SIF = zeros(nMC,E);
x4_SIF = zeros(nMC,E);
x5_SIF = zeros(nMC,E);
x6_SIF = zeros(nMC,E);

x1_KF_SNN = zeros(nMC,E);
x2_KF_SNN = zeros(nMC,E);
x3_KF_SNN = zeros(nMC,E);
x4_KF_SNN = zeros(nMC,E);
x5_KF_SNN = zeros(nMC,E);
x6_KF_SNN = zeros(nMC,E);

x1_SIF_SNN = zeros(nMC,E);
x2_SIF_SNN = zeros(nMC,E);
x3_SIF_SNN = zeros(nMC,E);
x4_SIF_SNN = zeros(nMC,E);
x5_SIF_SNN = zeros(nMC,E);
x6_SIF_SNN = zeros(nMC,E);

xtilda1_KF = zeros(nMC,E);
xtilda2_KF = zeros(nMC,E);
xtilda3_KF = zeros(nMC,E);
xtilda4_KF = zeros(nMC,E);
xtilda5_KF = zeros(nMC,E);
xtilda6_KF = zeros(nMC,E);

xtilda1_SIF = zeros(nMC,E);
xtilda2_SIF = zeros(nMC,E);
xtilda3_SIF = zeros(nMC,E);
xtilda4_SIF = zeros(nMC,E);
xtilda5_SIF = zeros(nMC,E);
xtilda6_SIF = zeros(nMC,E);

xtilda1_KF_SNN = zeros(nMC,E);
xtilda2_KF_SNN = zeros(nMC,E);
xtilda3_KF_SNN = zeros(nMC,E);
xtilda4_KF_SNN = zeros(nMC,E);
xtilda5_KF_SNN = zeros(nMC,E);
xtilda6_KF_SNN = zeros(nMC,E);

xtilda1_SIF_SNN = zeros(nMC,E);
xtilda2_SIF_SNN = zeros(nMC,E);
xtilda3_SIF_SNN = zeros(nMC,E);
xtilda4_SIF_SNN = zeros(nMC,E);
xtilda5_SIF_SNN = zeros(nMC,E);
xtilda6_SIF_SNN = zeros(nMC,E);

P1_KF = zeros(nMC,E);
P2_KF = zeros(nMC,E);
P3_KF = zeros(nMC,E);
P4_KF = zeros(nMC,E);
P5_KF = zeros(nMC,E);
P6_KF = zeros(nMC,E);

P1_SIF = zeros(nMC,E);
P2_SIF = zeros(nMC,E);
P3_SIF = zeros(nMC,E);
P4_SIF = zeros(nMC,E);
P5_SIF = zeros(nMC,E);
P6_SIF = zeros(nMC,E);

P1_KF_SNN = zeros(nMC,E);
P2_KF_SNN = zeros(nMC,E);
P3_KF_SNN = zeros(nMC,E);
P4_KF_SNN = zeros(nMC,E);
P5_KF_SNN = zeros(nMC,E);
P6_KF_SNN = zeros(nMC,E);

P1_SIF_SNN = zeros(nMC,E);
P2_SIF_SNN = zeros(nMC,E);
P3_SIF_SNN = zeros(nMC,E);
P4_SIF_SNN = zeros(nMC,E);
P5_SIF_SNN = zeros(nMC,E);
P6_SIF_SNN = zeros(nMC,E);

R1_KF = zeros(nMC,E);
R2_KF = zeros(nMC,E);
R3_KF = zeros(nMC,E);
R4_KF = zeros(nMC,E);
R5_KF = zeros(nMC,E);
R6_KF = zeros(nMC,E);

R1_SIF = zeros(nMC,E);
R2_SIF = zeros(nMC,E);
R3_SIF = zeros(nMC,E);
R4_SIF = zeros(nMC,E);
R5_SIF = zeros(nMC,E);
R6_SIF = zeros(nMC,E);

R1_KF_SNN = zeros(nMC,E);
R2_KF_SNN = zeros(nMC,E);
R3_KF_SNN = zeros(nMC,E);
R4_KF_SNN = zeros(nMC,E);
R5_KF_SNN = zeros(nMC,E);
R6_KF_SNN = zeros(nMC,E);

R1_SIF_SNN = zeros(nMC,E);
R2_SIF_SNN = zeros(nMC,E);
R3_SIF_SNN = zeros(nMC,E);
R4_SIF_SNN = zeros(nMC,E);
R5_SIF_SNN = zeros(nMC,E);
R6_SIF_SNN = zeros(nMC,E);

for kk = 1:nMC
% disp(kk)

Xt_LQR_SIF = zeros(Dim,E);
Xt_LQR_KF = zeros(Dim,E);
Xt_SNN_LQR_SIF = zeros(Dim,E);
Xt_SNN_LQR_KF = zeros(Dim,E);

Xhatt_SIF = zeros(6,E);
Xhatt_SNN_SIF = zeros(6,E);
s_outt_SIF = zeros(NON,E);
Cov_SIF = zeros(6,E);
Xtilda_SIF = zeros(6,E); 
Xtilda_SNN_SIF = zeros(6,E); 
RMSE_SIF = zeros(6,E);
RMSE_SNN_SIF = zeros(6,E);

Xhatt_KF = zeros(6,E);
Xhatt_SNN_KF = zeros(6,E);
s_outt_KF = zeros(NON,E);
Cov_KF = zeros(6,E);
Xtilda_KF = zeros(6,E); 
Xtilda_SNN_KF = zeros(6,E); 
RMSE_KF = zeros(6,E);
RMSE_SNN_KF = zeros(6,E);


Ut_LQG = zeros(3,E);
Ut_LQR_SIF = zeros(3,E);
Ut_SNN_LQR_SIF = zeros(3,E);

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


   Ut_LQG(:,i) = U_LQR_KF;
   Ut_LQR_SIF(:,i) = U_LQR_SIF;
   Ut_SNN_LQR_SIF(:,i) = U_SNN_LQR_SIF;

   
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
   K_SIF = pinv(C)*diag(Sat(diag(Pzz_SIF)/delta));    % SIF Filter Gain
   
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
x3_LQR_SIF(kk,:) = Xt_LQR_SIF(3,:);
x4_LQR_SIF(kk,:) = Xt_LQR_SIF(4,:);
x5_LQR_SIF(kk,:) = Xt_LQR_SIF(5,:);
x6_LQR_SIF(kk,:) = Xt_LQR_SIF(6,:);

x1_LQR_KF(kk,:) = Xt_LQR_KF(1,:);
x2_LQR_KF(kk,:) = Xt_LQR_KF(2,:);
x3_LQR_KF(kk,:) = Xt_LQR_KF(3,:);
x4_LQR_KF(kk,:) = Xt_LQR_KF(4,:);
x5_LQR_KF(kk,:) = Xt_LQR_KF(5,:);
x6_LQR_KF(kk,:) = Xt_LQR_KF(6,:);

x1_SNN_LQR_SIF(kk,:) = Xt_SNN_LQR_SIF(1,:);
x2_SNN_LQR_SIF(kk,:) = Xt_SNN_LQR_SIF(2,:);
x3_SNN_LQR_SIF(kk,:) = Xt_SNN_LQR_SIF(3,:);
x4_SNN_LQR_SIF(kk,:) = Xt_SNN_LQR_SIF(4,:);
x5_SNN_LQR_SIF(kk,:) = Xt_SNN_LQR_SIF(5,:);
x6_SNN_LQR_SIF(kk,:) = Xt_SNN_LQR_SIF(6,:);

x1_SNN_LQR_KF(kk,:) = Xt_SNN_LQR_KF(1,:);
x2_SNN_LQR_KF(kk,:) = Xt_SNN_LQR_KF(2,:);
x3_SNN_LQR_KF(kk,:) = Xt_SNN_LQR_KF(3,:);
x4_SNN_LQR_KF(kk,:) = Xt_SNN_LQR_KF(4,:);
x5_SNN_LQR_KF(kk,:) = Xt_SNN_LQR_KF(5,:);
x6_SNN_LQR_KF(kk,:) = Xt_SNN_LQR_KF(6,:);


x1_KF(kk,:) = Xhatt_KF(1,:);
x2_KF(kk,:) = Xhatt_KF(2,:);
x3_KF(kk,:) = Xhatt_KF(3,:);
x4_KF(kk,:) = Xhatt_KF(4,:);
x5_KF(kk,:) = Xhatt_KF(5,:);
x6_KF(kk,:) = Xhatt_KF(6,:);

x1_SIF(kk,:) = Xhatt_SIF(1,:);
x2_SIF(kk,:) = Xhatt_SIF(2,:);
x3_SIF(kk,:) = Xhatt_SIF(3,:);
x4_SIF(kk,:) = Xhatt_SIF(4,:);
x5_SIF(kk,:) = Xhatt_SIF(5,:);
x6_SIF(kk,:) = Xhatt_SIF(6,:);

x1_KF_SNN(kk,:) = Xhatt_SNN_KF(1,:);
x2_KF_SNN(kk,:) = Xhatt_SNN_KF(2,:);
x3_KF_SNN(kk,:) = Xhatt_SNN_KF(3,:);
x4_KF_SNN(kk,:) = Xhatt_SNN_KF(4,:);
x5_KF_SNN(kk,:) = Xhatt_SNN_KF(5,:);
x6_KF_SNN(kk,:) = Xhatt_SNN_KF(6,:);

x1_SIF_SNN(kk,:) = Xhatt_SNN_SIF(1,:);
x2_SIF_SNN(kk,:) = Xhatt_SNN_SIF(2,:);
x3_SIF_SNN(kk,:) = Xhatt_SNN_SIF(3,:);
x4_SIF_SNN(kk,:) = Xhatt_SNN_SIF(4,:);
x5_SIF_SNN(kk,:) = Xhatt_SNN_SIF(5,:);
x6_SIF_SNN(kk,:) = Xhatt_SNN_SIF(6,:);

xtilda1_KF(kk,:) = Xtilda_KF(1,:);
xtilda2_KF(kk,:) = Xtilda_KF(2,:);
xtilda3_KF(kk,:) = Xtilda_KF(3,:);
xtilda4_KF(kk,:) = Xtilda_KF(4,:);
xtilda5_KF(kk,:) = Xtilda_KF(5,:);
xtilda6_KF(kk,:) = Xtilda_KF(6,:);

xtilda1_SIF(kk,:) = Xtilda_SIF(1,:);
xtilda2_SIF(kk,:) = Xtilda_SIF(2,:);
xtilda3_SIF(kk,:) = Xtilda_SIF(3,:);
xtilda4_SIF(kk,:) = Xtilda_SIF(4,:);
xtilda5_SIF(kk,:) = Xtilda_SIF(5,:);
xtilda6_SIF(kk,:) = Xtilda_SIF(6,:);

xtilda1_KF_SNN(kk,:) = Xtilda_SNN_KF(1,:);
xtilda2_KF_SNN(kk,:) = Xtilda_SNN_KF(2,:);
xtilda3_KF_SNN(kk,:) = Xtilda_SNN_KF(3,:);
xtilda4_KF_SNN(kk,:) = Xtilda_SNN_KF(4,:);
xtilda5_KF_SNN(kk,:) = Xtilda_SNN_KF(5,:);
xtilda6_KF_SNN(kk,:) = Xtilda_SNN_KF(6,:);

xtilda1_SIF_SNN(kk,:) = Xtilda_SNN_SIF(1,:);
xtilda2_SIF_SNN(kk,:) = Xtilda_SNN_SIF(2,:);
xtilda3_SIF_SNN(kk,:) = Xtilda_SNN_SIF(3,:);
xtilda4_SIF_SNN(kk,:) = Xtilda_SNN_SIF(4,:);
xtilda5_SIF_SNN(kk,:) = Xtilda_SNN_SIF(5,:);
xtilda6_SIF_SNN(kk,:) = Xtilda_SNN_SIF(6,:);

P1_KF(kk,:) = Cov_KF(1,:);
P2_KF(kk,:) = Cov_KF(2,:);
P3_KF(kk,:) = Cov_KF(3,:);
P4_KF(kk,:) = Cov_KF(4,:);
P5_KF(kk,:) = Cov_KF(5,:);
P6_KF(kk,:) = Cov_KF(6,:);

P1_SIF(kk,:) = Cov_SIF(1,:);
P2_SIF(kk,:) = Cov_SIF(2,:);
P3_SIF(kk,:) = Cov_SIF(3,:);
P4_SIF(kk,:) = Cov_SIF(4,:);
P5_SIF(kk,:) = Cov_SIF(5,:);
P6_SIF(kk,:) = Cov_SIF(6,:);

P1_KF_SNN(kk,:) = Cov_KF(1,:);
P2_KF_SNN(kk,:) = Cov_KF(2,:);
P3_KF_SNN(kk,:) = Cov_KF(3,:);
P4_KF_SNN(kk,:) = Cov_KF(4,:);
P5_KF_SNN(kk,:) = Cov_KF(5,:);
P6_KF_SNN(kk,:) = Cov_KF(6,:);

P1_SIF_SNN(kk,:) = Cov_SIF(1,:);
P2_SIF_SNN(kk,:) = Cov_SIF(2,:);
P3_SIF_SNN(kk,:) = Cov_SIF(3,:);
P4_SIF_SNN(kk,:) = Cov_SIF(4,:);
P5_SIF_SNN(kk,:) = Cov_SIF(5,:);
P6_SIF_SNN(kk,:) = Cov_SIF(6,:);


R1_KF(kk,:) = RMSE_KF(1,:);
R2_KF(kk,:) = RMSE_KF(2,:);
R3_KF(kk,:) = RMSE_KF(3,:);
R4_KF(kk,:) = RMSE_KF(4,:);
R5_KF(kk,:) = RMSE_KF(5,:);
R6_KF(kk,:) = RMSE_KF(6,:);

R1_SIF(kk,:) = RMSE_SIF(1,:);
R2_SIF(kk,:) = RMSE_SIF(2,:);
R3_SIF(kk,:) = RMSE_SIF(3,:);
R4_SIF(kk,:) = RMSE_SIF(4,:);
R5_SIF(kk,:) = RMSE_SIF(5,:);
R6_SIF(kk,:) = RMSE_SIF(6,:);

R1_KF_SNN(kk,:) = RMSE_SNN_KF(1,:);
R2_KF_SNN(kk,:) = RMSE_SNN_KF(2,:);
R3_KF_SNN(kk,:) = RMSE_SNN_KF(3,:);
R4_KF_SNN(kk,:) = RMSE_SNN_KF(4,:);
R5_KF_SNN(kk,:) = RMSE_SNN_KF(5,:);
R6_KF_SNN(kk,:) = RMSE_SNN_KF(6,:);

R1_SIF_SNN(kk,:) = RMSE_SNN_SIF(1,:);
R2_SIF_SNN(kk,:) = RMSE_SNN_SIF(2,:);
R3_SIF_SNN(kk,:) = RMSE_SNN_SIF(3,:);
R4_SIF_SNN(kk,:) = RMSE_SNN_SIF(4,:);
R5_SIF_SNN(kk,:) = RMSE_SNN_SIF(5,:);
R6_SIF_SNN(kk,:) = RMSE_SNN_SIF(6,:);
end
%% Monte-Carlo mean computation


x1_LQR_SIF_MC = zeros(1,E);
x2_LQR_SIF_MC = zeros(1,E);
x3_LQR_SIF_MC = zeros(1,E);
x4_LQR_SIF_MC = zeros(1,E);
x5_LQR_SIF_MC = zeros(1,E);
x6_LQR_SIF_MC = zeros(1,E);

x1_LQR_KF_MC = zeros(1,E);
x2_LQR_KF_MC = zeros(1,E);
x3_LQR_KF_MC = zeros(1,E);
x4_LQR_KF_MC = zeros(1,E);
x5_LQR_KF_MC = zeros(1,E);
x6_LQR_KF_MC = zeros(1,E);

x1_SNN_LQR_SIF_MC = zeros(1,E);
x2_SNN_LQR_SIF_MC = zeros(1,E);
x3_SNN_LQR_SIF_MC = zeros(1,E);
x4_SNN_LQR_SIF_MC = zeros(1,E);
x5_SNN_LQR_SIF_MC = zeros(1,E);
x6_SNN_LQR_SIF_MC = zeros(1,E);

x1_SNN_LQR_KF_MC = zeros(1,E);
x2_SNN_LQR_KF_MC = zeros(1,E);
x3_SNN_LQR_KF_MC = zeros(1,E);
x4_SNN_LQR_KF_MC = zeros(1,E);
x5_SNN_LQR_KF_MC = zeros(1,E);
x6_SNN_LQR_KF_MC = zeros(1,E);

x1_KF_MC = zeros(1,E);
x2_KF_MC = zeros(1,E);
x3_KF_MC = zeros(1,E);
x4_KF_MC = zeros(1,E);
x5_KF_MC = zeros(1,E);
x6_KF_MC = zeros(1,E);

x1_SIF_MC = zeros(1,E);
x2_SIF_MC = zeros(1,E);
x3_SIF_MC = zeros(1,E);
x4_SIF_MC = zeros(1,E);
x5_SIF_MC = zeros(1,E);
x6_SIF_MC = zeros(1,E);

x1_KF_SNN_MC = zeros(1,E);
x2_KF_SNN_MC = zeros(1,E);
x3_KF_SNN_MC = zeros(1,E);
x4_KF_SNN_MC = zeros(1,E);
x5_KF_SNN_MC = zeros(1,E);
x6_KF_SNN_MC = zeros(1,E);

x1_SIF_SNN_MC = zeros(1,E);
x2_SIF_SNN_MC = zeros(1,E);
x3_SIF_SNN_MC = zeros(1,E);
x4_SIF_SNN_MC = zeros(1,E);
x5_SIF_SNN_MC = zeros(1,E);
x6_SIF_SNN_MC = zeros(1,E);

xtilda1_KF_MC = zeros(1,E);
xtilda2_KF_MC = zeros(1,E);
xtilda3_KF_MC = zeros(1,E);
xtilda4_KF_MC = zeros(1,E);
xtilda5_KF_MC = zeros(1,E);
xtilda6_KF_MC = zeros(1,E);

xtilda1_SIF_MC = zeros(1,E);
xtilda2_SIF_MC = zeros(1,E);
xtilda3_SIF_MC = zeros(1,E);
xtilda4_SIF_MC = zeros(1,E);
xtilda5_SIF_MC = zeros(1,E);
xtilda6_SIF_MC = zeros(1,E);

xtilda1_KF_SNN_MC = zeros(1,E);
xtilda2_KF_SNN_MC = zeros(1,E);
xtilda3_KF_SNN_MC = zeros(1,E);
xtilda4_KF_SNN_MC = zeros(1,E);
xtilda5_KF_SNN_MC = zeros(1,E);
xtilda6_KF_SNN_MC = zeros(1,E);

xtilda1_SIF_SNN_MC = zeros(1,E);
xtilda2_SIF_SNN_MC = zeros(1,E);
xtilda3_SIF_SNN_MC = zeros(1,E);
xtilda4_SIF_SNN_MC = zeros(1,E);
xtilda5_SIF_SNN_MC = zeros(1,E);
xtilda6_SIF_SNN_MC = zeros(1,E);

P1_KF_MC = zeros(1,E);
P2_KF_MC = zeros(1,E);
P3_KF_MC = zeros(1,E);
P4_KF_MC = zeros(1,E);
P5_KF_MC = zeros(1,E);
P6_KF_MC = zeros(1,E);

P1_SIF_MC = zeros(1,E);
P2_SIF_MC = zeros(1,E);
P3_SIF_MC = zeros(1,E);
P4_SIF_MC = zeros(1,E);
P5_SIF_MC = zeros(1,E);
P6_SIF_MC = zeros(1,E);

P1_KF_SNN_MC = zeros(1,E);
P2_KF_SNN_MC = zeros(1,E);
P3_KF_SNN_MC = zeros(1,E);
P4_KF_SNN_MC = zeros(1,E);
P5_KF_SNN_MC = zeros(1,E);
P6_KF_SNN_MC = zeros(1,E);

P1_SIF_SNN_MC = zeros(1,E);
P2_SIF_SNN_MC = zeros(1,E);
P3_SIF_SNN_MC = zeros(1,E);
P4_SIF_SNN_MC = zeros(1,E);
P5_SIF_SNN_MC = zeros(1,E);
P6_SIF_SNN_MC = zeros(1,E);

R1_KF_MC = zeros(1,E);
R2_KF_MC = zeros(1,E);
R3_KF_MC = zeros(1,E);
R4_KF_MC = zeros(1,E);
R5_KF_MC = zeros(1,E);
R6_KF_MC = zeros(1,E);

R1_SIF_MC = zeros(1,E);
R2_SIF_MC = zeros(1,E);
R3_SIF_MC = zeros(1,E);
R4_SIF_MC = zeros(1,E);
R5_SIF_MC = zeros(1,E);
R6_SIF_MC = zeros(1,E);

R1_KF_SNN_MC = zeros(1,E);
R2_KF_SNN_MC = zeros(1,E);
R3_KF_SNN_MC = zeros(1,E);
R4_KF_SNN_MC = zeros(1,E);
R5_KF_SNN_MC = zeros(1,E);
R6_KF_SNN_MC = zeros(1,E);

R1_SIF_SNN_MC = zeros(1,E);
R2_SIF_SNN_MC = zeros(1,E);
R3_SIF_SNN_MC = zeros(1,E);
R4_SIF_SNN_MC = zeros(1,E);
R5_SIF_SNN_MC = zeros(1,E);
R6_SIF_SNN_MC = zeros(1,E);

for i = 1:E
    
x1_LQR_SIF_MC(i) = mean(x1_LQR_SIF(:,i));
x2_LQR_SIF_MC(i) = mean(x2_LQR_SIF(:,i));
x3_LQR_SIF_MC(i) = mean(x3_LQR_SIF(:,i));
x4_LQR_SIF_MC(i) = mean(x4_LQR_SIF(:,i));
x5_LQR_SIF_MC(i) = mean(x5_LQR_SIF(:,i));
x6_LQR_SIF_MC(i) = mean(x6_LQR_SIF(:,i));

x1_LQR_KF_MC(i) = mean(x1_LQR_KF(:,i));
x2_LQR_KF_MC(i) = mean(x2_LQR_KF(:,i));
x3_LQR_KF_MC(i) = mean(x3_LQR_KF(:,i));
x4_LQR_KF_MC(i) = mean(x4_LQR_KF(:,i));
x5_LQR_KF_MC(i) = mean(x5_LQR_KF(:,i));
x6_LQR_KF_MC(i) = mean(x6_LQR_KF(:,i));

x1_SNN_LQR_SIF_MC(i) = mean(x1_SNN_LQR_SIF(:,i));
x2_SNN_LQR_SIF_MC(i) = mean(x2_SNN_LQR_SIF(:,i));
x3_SNN_LQR_SIF_MC(i) = mean(x3_SNN_LQR_SIF(:,i));
x4_SNN_LQR_SIF_MC(i) = mean(x4_SNN_LQR_SIF(:,i));
x5_SNN_LQR_SIF_MC(i) = mean(x5_SNN_LQR_SIF(:,i));
x6_SNN_LQR_SIF_MC(i) = mean(x6_SNN_LQR_SIF(:,i));

x1_SNN_LQR_KF_MC(i) = mean(x1_SNN_LQR_KF(:,i));
x2_SNN_LQR_KF_MC(i) = mean(x2_SNN_LQR_KF(:,i));
x3_SNN_LQR_KF_MC(i) = mean(x3_SNN_LQR_KF(:,i));
x4_SNN_LQR_KF_MC(i) = mean(x4_SNN_LQR_KF(:,i));
x5_SNN_LQR_KF_MC(i) = mean(x5_SNN_LQR_KF(:,i));
x6_SNN_LQR_KF_MC(i) = mean(x6_SNN_LQR_KF(:,i));

x1_KF_MC(i) = mean(x1_KF(:,i));
x2_KF_MC(i) = mean(x2_KF(:,i));
x3_KF_MC(i) = mean(x3_KF(:,i));
x4_KF_MC(i) = mean(x4_KF(:,i));
x5_KF_MC(i) = mean(x5_KF(:,i));
x6_KF_MC(i) = mean(x6_KF(:,i));

x1_SIF_MC(i) = mean(x1_SIF(:,i));
x2_SIF_MC(i) = mean(x2_SIF(:,i));
x3_SIF_MC(i) = mean(x3_SIF(:,i));
x4_SIF_MC(i) = mean(x4_SIF(:,i));
x5_SIF_MC(i) = mean(x5_SIF(:,i));
x6_SIF_MC(i) = mean(x6_SIF(:,i));

x1_KF_SNN_MC(i) = mean(x1_KF_SNN(:,i));
x2_KF_SNN_MC(i) = mean(x2_KF_SNN(:,i));
x3_KF_SNN_MC(i) = mean(x3_KF_SNN(:,i));
x4_KF_SNN_MC(i) = mean(x4_KF_SNN(:,i));
x5_KF_SNN_MC(i) = mean(x5_KF_SNN(:,i));
x6_KF_SNN_MC(i) = mean(x6_KF_SNN(:,i));

x1_SIF_SNN_MC(i) = mean(x1_SIF_SNN(:,i));
x2_SIF_SNN_MC(i) = mean(x2_SIF_SNN(:,i));
x3_SIF_SNN_MC(i) = mean(x3_SIF_SNN(:,i));
x4_SIF_SNN_MC(i) = mean(x4_SIF_SNN(:,i));
x5_SIF_SNN_MC(i) = mean(x5_SIF_SNN(:,i));
x6_SIF_SNN_MC(i) = mean(x6_SIF_SNN(:,i));

xtilda1_KF_MC(i) = mean(xtilda1_KF(:,i));
xtilda2_KF_MC(i) = mean(xtilda2_KF(:,i));
xtilda3_KF_MC(i) = mean(xtilda3_KF(:,i));
xtilda4_KF_MC(i) = mean(xtilda4_KF(:,i));
xtilda5_KF_MC(i) = mean(xtilda5_KF(:,i));
xtilda6_KF_MC(i) = mean(xtilda6_KF(:,i));

xtilda1_SIF_MC(i) = mean(xtilda1_SIF(:,i));
xtilda2_SIF_MC(i) = mean(xtilda2_SIF(:,i));
xtilda3_SIF_MC(i) = mean(xtilda3_SIF(:,i));
xtilda4_SIF_MC(i) = mean(xtilda4_SIF(:,i));
xtilda5_SIF_MC(i) = mean(xtilda5_SIF(:,i));
xtilda6_SIF_MC(i) = mean(xtilda6_SIF(:,i));

xtilda1_KF_SNN_MC(i) = mean(xtilda1_KF_SNN(:,i));
xtilda2_KF_SNN_MC(i) = mean(xtilda2_KF_SNN(:,i));
xtilda3_KF_SNN_MC(i) = mean(xtilda3_KF_SNN(:,i));
xtilda4_KF_SNN_MC(i) = mean(xtilda4_KF_SNN(:,i));
xtilda5_KF_SNN_MC(i) = mean(xtilda5_KF_SNN(:,i));
xtilda6_KF_SNN_MC(i) = mean(xtilda6_KF_SNN(:,i));

xtilda1_SIF_SNN_MC(i) = mean(xtilda1_SIF_SNN(:,i));
xtilda2_SIF_SNN_MC(i) = mean(xtilda2_SIF_SNN(:,i));
xtilda3_SIF_SNN_MC(i) = mean(xtilda3_SIF_SNN(:,i));
xtilda4_SIF_SNN_MC(i) = mean(xtilda4_SIF_SNN(:,i));
xtilda5_SIF_SNN_MC(i) = mean(xtilda5_SIF_SNN(:,i));
xtilda6_SIF_SNN_MC(i) = mean(xtilda6_SIF_SNN(:,i));

P1_KF_MC(i) = mean(P1_KF(:,i));
P2_KF_MC(i) = mean(P2_KF(:,i));
P3_KF_MC(i) = mean(P3_KF(:,i));
P4_KF_MC(i) = mean(P4_KF(:,i));
P5_KF_MC(i) = mean(P5_KF(:,i));
P6_KF_MC(i) = mean(P6_KF(:,i));

P1_SIF_MC(i) = mean(P1_SIF(:,i));
P2_SIF_MC(i) = mean(P2_SIF(:,i));
P3_SIF_MC(i) = mean(P3_SIF(:,i));
P4_SIF_MC(i) = mean(P4_SIF(:,i));
P5_SIF_MC(i) = mean(P5_SIF(:,i));
P6_SIF_MC(i) = mean(P6_SIF(:,i));

P1_KF_SNN_MC(i) = mean(P1_KF_SNN(:,i));
P2_KF_SNN_MC(i) = mean(P2_KF_SNN(:,i));
P3_KF_SNN_MC(i) = mean(P3_KF_SNN(:,i));
P4_KF_SNN_MC(i) = mean(P4_KF_SNN(:,i));
P5_KF_SNN_MC(i) = mean(P5_KF_SNN(:,i));
P6_KF_SNN_MC(i) = mean(P6_KF_SNN(:,i));

P1_SIF_SNN_MC(i) = mean(P1_SIF_SNN(:,i));
P2_SIF_SNN_MC(i) = mean(P2_SIF_SNN(:,i));
P3_SIF_SNN_MC(i) = mean(P3_SIF_SNN(:,i));
P4_SIF_SNN_MC(i) = mean(P4_SIF_SNN(:,i));
P5_SIF_SNN_MC(i) = mean(P5_SIF_SNN(:,i));
P6_SIF_SNN_MC(i) = mean(P6_SIF_SNN(:,i));

R1_KF_MC(i) = mean(R1_KF(:,i));
R2_KF_MC(i) = mean(R2_KF(:,i));
R3_KF_MC(i) = mean(R3_KF(:,i));
R4_KF_MC(i) = mean(R4_KF(:,i));
R5_KF_MC(i) = mean(R5_KF(:,i));
R6_KF_MC(i) = mean(R6_KF(:,i));

R1_SIF_MC(i) = mean(R1_SIF(:,i));
R2_SIF_MC(i) = mean(R2_SIF(:,i));
R3_SIF_MC(i) = mean(R3_SIF(:,i));
R4_SIF_MC(i) = mean(R4_SIF(:,i));
R5_SIF_MC(i) = mean(R5_SIF(:,i));
R6_SIF_MC(i) = mean(R6_SIF(:,i));

R1_KF_SNN_MC(i) = mean(R1_KF_SNN(:,i));
R2_KF_SNN_MC(i) = mean(R2_KF_SNN(:,i));
R3_KF_SNN_MC(i) = mean(R3_KF_SNN(:,i));
R4_KF_SNN_MC(i) = mean(R4_KF_SNN(:,i));
R5_KF_SNN_MC(i) = mean(R5_KF_SNN(:,i));
R6_KF_SNN_MC(i) = mean(R6_KF_SNN(:,i));

R1_SIF_SNN_MC(i) = mean(R1_SIF_SNN(:,i));
R2_SIF_SNN_MC(i) = mean(R2_SIF_SNN(:,i));
R3_SIF_SNN_MC(i) = mean(R3_SIF_SNN(:,i));
R4_SIF_SNN_MC(i) = mean(R4_SIF_SNN(:,i));
R5_SIF_SNN_MC(i) = mean(R5_SIF_SNN(:,i));
R6_SIF_SNN_MC(i) = mean(R6_SIF_SNN(:,i));
end
toc
%% ========== Plots
% state trajectory
 
figure
subplot(3,2,1)
plot(tspan,x1_LQR_KF_MC,'b','linewidth',2)
hold on 
plot(tspan,x1_LQR_SIF_MC,'r-.','linewidth',2)
% plot(tspan,x1_SNN_LQR_KF_MC,'g--','linewidth',1.5)
plot(tspan,x1_SNN_LQR_SIF_MC,'k:','linewidth',2)
grid on 
ylabel("x (m)")
legend("LQG","LQR-SIF","SNN-LQR-SIF")
xlim([0,max(tspan)])

subplot(3,2,3)
plot(tspan,x2_LQR_KF_MC,'b','linewidth',2)
hold on 
plot(tspan,x2_LQR_SIF_MC,'r-.','linewidth',2)
% plot(tspan,x2_SNN_LQR_KF_MC,'g--','linewidth',1.5)
plot(tspan,x2_SNN_LQR_SIF_MC,'k:','linewidth',2)
grid on 
ylabel("y (m)")
xlim([0,max(tspan)])

subplot(3,2,5)
plot(tspan,x3_LQR_KF_MC,'b','linewidth',2)
hold on 
plot(tspan,x3_LQR_SIF_MC,'r-.','linewidth',2)
% plot(tspan,x3_SNN_LQR_KF_MC,'g--','linewidth',1.5)
plot(tspan,x3_SNN_LQR_SIF_MC,'k:','linewidth',2)
grid on 
xlabel("time (s)")
ylabel("z (m)")
xlim([0,max(tspan)])

subplot(3,2,2)
plot(tspan,x4_LQR_KF_MC,'b','linewidth',2)
hold on 
plot(tspan,x4_LQR_SIF_MC,'r-.','linewidth',2)
% plot(tspan,x4_SNN_LQR_KF_MC,'g--','linewidth',1.5)
plot(tspan,x4_SNN_LQR_SIF_MC,'k:','linewidth',2)
grid on 
ylabel("v_x (m/s)")
xlim([0,max(tspan)])

subplot(3,2,4)
plot(tspan,x5_LQR_KF_MC,'b','linewidth',2)
hold on 
plot(tspan,x5_LQR_SIF_MC,'r-.','linewidth',2)
% plot(tspan,x5_SNN_LQR_KF_MC,'g--','linewidth',1.5)
plot(tspan,x5_SNN_LQR_SIF_MC,'k:','linewidth',2)
grid on 
ylabel("v_y (m/s)")
xlim([0,max(tspan)])

subplot(3,2,6)
plot(tspan,x6_LQR_KF_MC,'b','linewidth',2)
hold on 
plot(tspan,x6_LQR_SIF_MC,'r-.','linewidth',2)
% plot(tspan,x6_SNN_LQR_KF_MC,'g--','linewidth',1.5)
plot(tspan,x6_SNN_LQR_SIF_MC,'k:','linewidth',2)
grid on 
xlabel("time (s)")
ylabel("v_z (m/s)")
xlim([0,max(tspan)])

% %%
% figure
% subplot(3,2,1)
% plot(tspan,x1_MC,'b','linewidth',1.5)
% hold on 
% plot(tspan,x1_KF_MC,'r-.','linewidth',2) 
% plot(tspan,x1_SIF_MC,'c:','linewidth',2)
% plot(tspan,x1_KF_SNN_MC,'g--','linewidth',2)
% plot(tspan,x1_SIF_SNN_MC,'k:','linewidth',2);
% grid on 
% ylabel("x_1")
% legend("True","KF","SIF","SNN-KF","SNN-SIF")
% xlim([0,max(tspan)])
% 
% subplot(3,2,3)
% plot(tspan,x2_MC,'b','linewidth',1.5)
% hold on 
% plot(tspan,x2_KF_MC,'r-.','linewidth',2)
% plot(tspan,x2_SIF_MC,'c:','linewidth',2)
% plot(tspan,x2_KF_SNN_MC,'g--','linewidth',2)
% plot(tspan,x2_SIF_SNN_MC,'k:','linewidth',2)
% grid on 
% xlabel("time (s)")
% ylabel("x_2")
% xlim([0,max(tspan)])
% 
% subplot(3,2,5)
% plot(tspan,x3_MC,'b','linewidth',1.5)
% hold on 
% plot(tspan,x3_KF_MC,'r-.','linewidth',2)
% plot(tspan,x3_SIF_MC,'c:','linewidth',2)
% plot(tspan,x3_KF_SNN_MC,'g--','linewidth',2)
% plot(tspan,x3_SIF_SNN_MC,'k:','linewidth',2)
% grid on 
% xlabel("time (s)")
% ylabel("x_3")
% xlim([0,max(tspan)])
% 
% subplot(3,2,2)
% plot(tspan,x4_MC,'b','linewidth',1.5)
% hold on 
% plot(tspan,x4_KF_MC,'r-.','linewidth',2)
% plot(tspan,x4_SIF_MC,'c:','linewidth',2)
% plot(tspan,x4_KF_SNN_MC,'g--','linewidth',2)
% plot(tspan,x4_SIF_SNN_MC,'k:','linewidth',2)
% grid on 
% xlabel("time (s)")
% ylabel("x_4")
% xlim([0,max(tspan)])
% 
% subplot(3,2,4)
% plot(tspan,x5_MC,'b','linewidth',1.5)
% hold on 
% plot(tspan,x5_KF_MC,'r-.','linewidth',2)
% plot(tspan,x5_SIF_MC,'c:','linewidth',2)
% plot(tspan,x5_KF_SNN_MC,'g--','linewidth',2)
% plot(tspan,x5_SIF_SNN_MC,'k:','linewidth',2)
% grid on 
% xlabel("time (s)")
% ylabel("x_5")
% xlim([0,max(tspan)])
% 
% subplot(3,2,6)
% plot(tspan,x6_MC,'b','linewidth',1.5)
% hold on 
% plot(tspan,x6_KF_MC,'r-.','linewidth',2)
% plot(tspan,x6_SIF_MC,'c:','linewidth',2)
% plot(tspan,x6_KF_SNN_MC,'g--','linewidth',2)
% plot(tspan,x6_SIF_SNN_MC,'k:','linewidth',2)
% grid on 
% xlabel("time (s)")
% ylabel("x_6")
% xlim([0,max(tspan)])
%% 3sigma stability
figure
subplot(3,2,1)
plot(tspan,xtilda1_KF_MC,'b','linewidth',2)
hold on
plot(tspan,xtilda1_SIF_MC,'r--','linewidth',2)
% plot(tspan,xtilda1_KF_SNN_MC,'g-.','linewidth',2)
plot(tspan,xtilda1_SIF_SNN_MC,'k:','linewidth',2)
plot(tspan,3*sqrt(P1_KF_MC),'r--',tspan,3*sqrt(P1_SIF_SNN_MC),'k:','linewidth',2)
plot(tspan,-3*sqrt(P1_KF_MC),'r--',tspan,-3*sqrt(P1_SIF_SNN_MC),'k:','linewidth',2)
grid on
ylabel('\deltax_1')
legend('KF','SIF','SNN-SIF')
xlim([0,max(tspan)])

subplot(3,2,3)
plot(tspan,xtilda2_KF_MC,'b','linewidth',2)
hold on
plot(tspan,xtilda2_SIF_MC,'r--','linewidth',2)
% plot(tspan,xtilda2_KF_SNN_MC,'g-.','linewidth',2)
plot(tspan,xtilda2_SIF_SNN_MC,'k:','linewidth',2)
plot(tspan,3*sqrt(P2_KF_MC),'r--',tspan,3*sqrt(P2_SIF_SNN_MC),'k:','linewidth',2)
plot(tspan,-3*sqrt(P2_KF_MC),'r--',tspan,-3*sqrt(P2_SIF_SNN_MC),'k:','linewidth',2)
grid on
xlabel('time (s)')
ylabel('\deltax_2')
xlim([0,max(tspan)])

subplot(3,2,5)
plot(tspan,xtilda3_KF_MC,'b','linewidth',2)
hold on
plot(tspan,xtilda3_SIF_MC,'r--','linewidth',2)
% plot(tspan,xtilda2_KF_SNN_MC,'g-.','linewidth',2)
plot(tspan,xtilda3_SIF_SNN_MC,'k:','linewidth',2)
plot(tspan,3*sqrt(P3_KF_MC),'r--',tspan,3*sqrt(P3_SIF_SNN_MC),'k:','linewidth',2)
plot(tspan,-3*sqrt(P3_KF_MC),'r--',tspan,-3*sqrt(P3_SIF_SNN_MC),'k:','linewidth',2)
grid on
xlabel('time (s)')
ylabel('\deltax_3')
xlim([0,max(tspan)])

subplot(3,2,2)
plot(tspan,xtilda4_KF_MC,'b','linewidth',2)
hold on
plot(tspan,xtilda4_SIF_MC,'r--','linewidth',2)
% plot(tspan,xtilda2_KF_SNN_MC,'g-.','linewidth',2)
plot(tspan,xtilda4_SIF_SNN_MC,'k:','linewidth',2)
plot(tspan,3*sqrt(P4_KF_MC),'r--',tspan,3*sqrt(P4_SIF_SNN_MC),'k:','linewidth',2)
plot(tspan,-3*sqrt(P4_KF_MC),'r--',tspan,-3*sqrt(P4_SIF_SNN_MC),'k:','linewidth',2)
grid on
xlabel('time (s)')
ylabel('\deltax_4')
xlim([0,max(tspan)])

subplot(3,2,4)
plot(tspan,xtilda5_KF_MC,'b','linewidth',2)
hold on
plot(tspan,xtilda5_SIF_MC,'r--','linewidth',2)
% plot(tspan,xtilda2_KF_SNN_MC,'g-.','linewidth',2)
plot(tspan,xtilda5_SIF_SNN_MC,'k:','linewidth',2)
plot(tspan,3*sqrt(P5_KF_MC),'r--',tspan,3*sqrt(P5_SIF_SNN_MC),'k:','linewidth',2)
plot(tspan,-3*sqrt(P5_KF_MC),'r--',tspan,-3*sqrt(P5_SIF_SNN_MC),'k:','linewidth',2)
grid on
xlabel('time (s)')
ylabel('\deltax_2')

xlim([0,max(tspan)])

subplot(3,2,6)
plot(tspan,xtilda6_KF_MC,'b','linewidth',2)
hold on
plot(tspan,xtilda6_SIF_MC,'r--','linewidth',2)
% plot(tspan,xtilda2_KF_SNN_MC,'g-.','linewidth',2)
plot(tspan,xtilda6_SIF_SNN_MC,'k:','linewidth',2)
plot(tspan,3*sqrt(P6_KF_MC),'r--',tspan,3*sqrt(P6_SIF_SNN_MC),'k:','linewidth',2)
plot(tspan,-3*sqrt(P6_KF_MC),'r--',tspan,-3*sqrt(P6_SIF_SNN_MC),'k:','linewidth',2)
grid on
xlabel('time (s)')
ylabel('\deltax_2')
xlim([0,max(tspan)])

% %% RMSE
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
NOSP = 100*(sum(s_outt_SIF)/NON);

figure
subplot(2,1,1)
imagesc(s_outt_SIF)
title('Spiking Pattern SNN-SIF')
xlabel('time step')
ylabel('Neuron index')

subplot(2,1,2)
plot(tspan, NOSP, 'b', 'linewidth',1.5)
grid on
xlabel('time')
ylabel('Active neurons in %')
xlim([0,360])
ylim([0,100])



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
% %%
% figure
% subplot(3,1,1)
% imagesc(s_outt_SIF_300)
% title('N = 300')
% ylabel('Neuron index')
% 
% subplot(3,1,2)
% imagesc(s_outt_SIF_450)
% title('N = 450')
% ylabel('Neuron index')
% 
% subplot(3,1,3)
% imagesc(s_outt_SIF_50)
% title('N = 50')
% xlabel('time step')
% ylabel('Neuron index')
% 
%% control input plots 

figure
subplot(3,1,1)
plot(tspan, Ut_LQG(1,:),'b','LineWidth',2)
hold on 
grid on 
plot(tspan, Ut_LQR_SIF(1,:),'r--','LineWidth',2)
plot(tspan, Ut_SNN_LQR_SIF(1,:),'k:','LineWidth',2)
ylabel('f_x (N)')
legend('LQG','LQR-MSIF','SNN-LQR-MSIF')
xlim([0,360])

subplot(3,1,2)
plot(tspan, Ut_LQG(2,:),'b','LineWidth',2)
hold on 
grid on 
plot(tspan, Ut_LQR_SIF(2,:),'r--','LineWidth',2)
plot(tspan, Ut_SNN_LQR_SIF(2,:),'k:','LineWidth',2)
ylabel('f_y (N)')
xlim([0,360])

subplot(3,1,3)
plot(tspan, Ut_LQG(3,:),'b','LineWidth',2)
hold on 
grid on 
plot(tspan, Ut_LQR_SIF(3,:),'r--','LineWidth',2)
plot(tspan, Ut_SNN_LQR_SIF(3,:),'k:','LineWidth',2)
ylabel('f_z (N)')
xlim([0,360])
xlabel('time (s)')
