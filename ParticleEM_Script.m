% Script for Inferring TAP Inference from neural activity using Particle EM
addpath FastICA_25/
close all;

% Set noise seed
noise_seed = randi(100000);
rng(noise_seed); 

% ----------------------- Initialize parameters ---------------------------

Nx  = 5;    % No. of variables
Nr  = 2*Nx;  % No. of neurons
Nh  = 1 + Nx;  % No. of external input variables

T   = 10000;  % No. of time steps
Th  = 2;     % No. of time steps for which h is the same
Ns  = 1;     % No. of batches
lam = 0.25;  % low pass filtering constant for the TAP dynamics

nltype = 'sigmoid'; % external nonlinearity in TAP dynamics

% Noise covariances
Qpr     = 1e-5*eye(Nx); % process noise
Qobs    = 4e-4*eye(Nr); % measurement noise

% True TAP model parameters
Jtype   = 'nonferr';
sc_J    = 1; % 1 : self coupling ON, 0: OFF
if Nx <= 4
    sp_J = 0.1; % sparsity in J
else
    sp_J = 0.3;
end
J       = 3*Create_J(Nx, sp_J, Jtype, sc_J); % Coupling matrix
G       = [2,4,-4,-8,8]';   % reduced parameter set  % G = [0,2,0,0,0,0,0,0,0,0,4,-4,0,-8,8,0,0,0]';
U       = 1*randn(Nr,Nx); % Embedding matrix
A       = randn(Nx,Nh); [~,~,V] = svd(A); clear A;
V       = V(1:Nx,:); % input embedding matrix

lG      = length(G);
if lG == 5
    RG = 1; %indicates whether we are using a reduced size for G or not
else
    RG = 0;
end

% ---------- Generate the latent dynamics and observations ----------------

% Inputs
gh          = 25/sqrt(Nx); % gain for h
hMatFull    = zeros(Nh,T,Ns);
x0Full      = rand(Nx,Ns);   

xMatFull    = zeros(Nx,T,Ns);
rMatFull    = zeros(Nr,T,Ns);
xMat0Full   = zeros(Nx,T,Ns);
b = hamming(5); b = b/sum(b);

for s = 1:Ns
    hMatFull(:,:,s) = filtfilt(b,1,generateBroadH(Nh,T,Th,gh)')';
    [xMat, rMat] = runTAP(x0Full(:,s), hMatFull(:,:,s), lam, Qpr, Qobs, U, V, J, G, nltype); % with coupling
    xMatFull(:,:,s) = xMat;
    rMatFull(:,:,s) = rMat;
    xMat_zeroJ   = runTAP(x0Full(:,s), hMatFull(:,:,s), lam, Qpr, Qobs, U, V, J*0, G, nltype); % without coupling
    xMat0Full(:,:,s) = xMat_zeroJ;
end

% Compare the dynamics with and without the coupling J
figure; 
subplot(1,2,1); plot(xMat(:,1:50)','b.-'); hold on; plot(xMat_zeroJ(:,1:50)','r.-'); 
axis([0,50,0,1]); title('blue with J, red: J=0'); xlabel('t'); ylabel('x_i(t)')

subplot(1,2,2);
plot(xMatFull(:),xMat0Full(:),'b.'); hold on; plot([0,1],[0,1],'k')
axis([0,1,0,1]); xlabel('x with J ~= 0'); ylabel('x with J = 0');

clear xMat0Full xMat_zeroJ xMat rMat h

temp = zeros(Nx,T+1,Ns);
temp(:,1,:)     = x0Full;
temp(:,2:end,:) = xMatFull;
xMatFull = temp; clear temp;


% ----------- Run ICA to get initial estimate of Uhat ---------------------
[Xe, ~, W] = fastica (reshape(rMatFull,[Nr, T*Ns]),'approach','defl','numOfIC',Nx,'g','pow3');

minx    = min(Xe,[],2);
maxx    = max(Xe,[],2);
DW      = zeros(Nx,1);

for ii = 1:Nx
    if abs(minx(ii)) > abs(maxx(ii))
        DW(ii) = minx(ii);
    else
        DW(ii) = maxx(ii);
    end
end

DW  = mean(abs(DW))*sign(DW);
W   = W./DW;

Xe = Xe./DW;

U_1 = reshape(rMatFull,[Nr, T*Ns])*pinv(Xe);

P = round(W*U);

if det(P) == 0
    keyboard;
end

clear Xe W minx maxx DW ii;

% Pick T such that we have a total of 500 samples
TTotal = 500;
T   = floor(TTotal/Ns);  % No. of time steps

rMatFull = rMatFull(:,1:T,:);
hMatFull = hMatFull(:,1:T,:);
xMatFull = xMatFull(:,1:T+1,:);


% ---------  Run the particle filter with true values of (U, J, G) --------

K = 100; % No. of particles

x_truedec = zeros(Nx,T+1,Ns);
r_truedec = zeros(Nr,T,Ns);
P_truedec = zeros(Nx,K,T+1,Ns);
WMat      = zeros(K,Ns);
LL        = zeros(Ns,1); % Log likelihood 

theta = [lam; G; JMatToVec(J); U(:); V(:)];

tic;
for s = 1:Ns
    [LL(s),x_truedec(:,:,s), P_truedec(:,:,:,s), WMat(:,s)] ...
        = particlefilter(rMatFull(:,:,s), hMatFull(:,:,s), K, Qpr, Qobs, RG, theta, nltype);
    r_truedec(:,:,s) = U*x_truedec(:,2:end,s);
end
toc;

% ----------- Now we try to learn the parameters from data using PF-EM ----

A   = randn(Nx,Nh); [~,~,V_1] = svd(A); clear A;
V_1 = V_1(1:Nx,:); % input embedding matrix
G_1 = 0.1*randn(5,1);
J_1 = Create_J(Nx, sp_J, Jtype, sc_J);
lam_1 = 0.25; 

x_1 = zeros(Nx,T+1,Ns);
r_1 = zeros(Nr,T,Ns);
P_1 = zeros(Nx,K,T+1,Ns);
W_1 = zeros(K,Ns);
L_1 = zeros(Ns,1);

theta_1 = [lam_1; G_1; JMatToVec(J_1); U_1(:); V_1(:)];

tic;
for s = 1:Ns
    [L_1(s),x_1(:,:,s), P_1(:,:,:,s), W_1(:,s)] ...
        = particlefilter(rMatFull(:,:,s), hMatFull(:,:,s), K, Qpr, Qobs, RG, theta_1, nltype);
    r_1(:,:,s) = U_1*x_1(:,2:end,s);
end
toc;

% Plot the true vs decoded latents and neural responses
st = 6; % start plotting from this time
figure; plot(mv(xMatFull(:,st:end,:)),mv(x_truedec(:,st:end,:)),'k.')
hold on; plot(mv(xMatFull(:,st:end,:)),mv(P'*reshape(x_1(:,st:end,:),[Nx,Ns*(T+2-st)])),'b.'); grid on

figure; plot(mv(rMatFull(:,st:end,:)),mv(r_truedec(:,st:end,:)),'k.')
hold on; plot(mv(rMatFull(:,st:end,:)),mv(r_1(:,st:end,:)),'b.'); grid on

xinit = x_1;
rinit = r_1;

Jinit = J_1;
Ginit = G_1;
Uinit = U_1;
Vinit = V_1;
lam_init = lam_1;


clear x_truedec r_truedec P_truedec
% ---------------------------- EM iterations ------------------------------

% options for unconstrained minimization
options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton','tolX',1e-4,'MaxFunEvals',200,'GradObj','on','TolFun',1e-4,'MaxIter',200);
NJ      = Nx*(Nx+1)/2;

% Initialize the batch
idx     = randi(Ns);
rB      = rMatFull(:,:,idx); % pick the observations for the mini batch
hB      = hMatFull(:,:,idx); 
P_B     = P_1(:,:,:,idx);
WB      = W_1(:,idx);

EMIters = 60;
LMat   = zeros(EMIters,1);

computegrad = [1, 1, 0, 1, 0];


for iterem = 1:EMIters
    
    if iterem == floor(EMIters/5)
        computegrad = [1, 1, 0, 1, 0];
    end

    if mod(iterem,Ns) ~= 0
        fprintf('%3d ',iterem);
    else
        fprintf('%3d\n',iterem);
    end
    
    fun     = @(theta_1)NegLL(rB, hB, P_B, WB, Qpr, Qobs, RG, nltype, computegrad, theta_1);
    
    theta_1 = fminunc(fun,theta_1,options);
        
    % Pick a new batch and run the particle filter with the updated parameters
    
    idx = randi(Ns);
    rB  = rMatFull(:,:,idx); 
    hB  = hMatFull(:,:,idx);
  
    [LMat(iterem),~, P_B, WB] = particlefilter(rB, hB, K, Qpr, Qobs,  RG, theta_1, nltype);
    
end

lam_1   = theta_1(1); 
theta_1 = theta_1(2:end);
G_1     = theta_1(1:lG);
J_1     = JVecToMat(theta_1(lG+1:lG+NJ));
U_1     = reshape(theta_1(lG+1+NJ:lG+NJ+Nr*Nx),Nr,Nx);
V_1     = reshape(theta_1(lG+NJ+Nr*Nx+1:end),Nx,Nh);
theta_1 = [lam_1; theta_1];

tic;
for s = 1:Ns
    [L_1(s),x_1(:,:,s), P_1(:,:,:,s), W_1(:,s)] ...
        = particlefilter(rMatFull(:,:,s), hMatFull(:,:,s), K, Qpr, Qobs, RG, theta_1, nltype);
    r_1(:,:,s) = U_1*x_1(:,2:end,s);
end
toc;


% plot the result
figure(2); plot(mv(xMatFull(:,st:end,:)),mv(P'*reshape(x_1(:,st:end,:),[Nx,Ns*(T+2-st)])),'r.');
xlabel('xtrue'); ylabel('xdecoded'); legend('True','init','After EM')

figure(3); plot(mv(rMatFull(:,st:end,:)),mv(r_1(:,st:end,:)),'r.')
xlabel('rtrue'); ylabel('rdecoded'); legend('True','init','After EM')

figure; plot(LMat,'b.-'); hold on; plot([0,EMIters],mean(LL)*ones(1,2),'r')
xlabel('iterations'); ylabel('Log likelihood');

figure; plot(U(:),mv(Uinit*P),'b*'); hold on; plot(U(:),mv(U_1*P),'r*');
xlabel('U true'); ylabel('U hat'); legend('init','After EM')

figure; plot(V(:),mv(P'*Vinit),'b*'); hold on; plot(V(:),mv(P'*V_1),'r*');
xlabel('V true'); ylabel('V hat'); legend('init','After EM')

Jhat = zeros(Nx,Nx);
x_idx = P'*(1:Nx)';
for ii = 1:Nx
    for jj = 1:Nx
        Jhat(ii,jj) = J_1(x_idx(ii),x_idx(jj));
    end
end

figure; plot(J(:),Jinit(:),'b*'); hold on; plot(J(:),Jhat(:),'r*');
xlabel('J true'); ylabel('J hat'); legend('init','After EM')


% Generate dynamics with the new set of parameters
xMat2 = zeros(Nx,T,Ns);
for s = 1:Ns
    xMat2(:,:,s) = runTAP(x0Full(:,s), hMatFull(:,:,s), lam, Qpr, Qobs, U_1, V_1, J_1, G_1, nltype);
end

Xtrue   = xMatFull(:,7:end,:);
Xinf    = xMat2(:,6:end,:);
Xinf    = P'*reshape(Xinf,[Nx,(T-5)*Ns]);

figure; plot(Xtrue(:),Xinf(:),'k.');
xlabel('x with true params'); ylabel('x with learnt params');
