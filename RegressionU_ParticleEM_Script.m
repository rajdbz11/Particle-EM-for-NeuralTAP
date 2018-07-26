% Script for Inferring TAP Inference from neural activity using Particle EM
close all;

% Set noise seed
noise_seed = randi(2000);
rng(noise_seed); 

% ----------------------- Initialize parameters ---------------------------

Nx  = 5;    % No. of variables
Nr  = 2*Nx;  % No. of neurons
T   = 50;    % No. of time steps
Nh  = 10;    % No. of time steps for which h is the same
Ns  = 1000;    % No. of batches
lam = 0.25;  % low pass filtering constant for the TAP dynamics
nltype = 'sigmoid'; % external nonlinearity in TAP dynamics

% Noise covariances
Qpr     = 1e-5*eye(Nx); % process noise
Qobs    = 4e-4*eye(Nr); % measurement noise

% True TAP model parameters
Jtype   = 'nonferr';
sc_J    = 1; % self coupling for J
J       = 3*Create_J(Nx, 0.25, Jtype, sc_J); % Coupling matrix
G       = [2,4,-4,-8,8]';   % reduced parameter set  % G = [0,2,0,0,0,0,0,0,0,0,4,-4,0,-8,8,0,0,0]';
U       = 2*randn(Nr,Nx);     % Embedding matrix

lG      = length(G);
if lG == 5
    RG = 1; %indicates whether we are using a reduced size for G or not
else
    RG = 0;
end

% ---------- Generate the latent dynamics and observations ----------------

% Inputs
% First use inputs with high gain to compute Uhat using regression

gh          = 200/sqrt(Nx); % gain for h
hMatFull    = zeros(Nx,T,Ns);
x0Full      = rand(Nx,Ns);  % initial values of x

xMatFull    = zeros(Nx,T,Ns);
rMatFull    = zeros(Nr,T,Ns);

for s = 1:Ns
    hMatFull(:,:,s) = generateH(Nx,T,Nh,gh);
    [xMat, rMat] = runTAP(x0Full(:,s), hMatFull(:,:,s), lam, Qpr, Qobs, U, J, G, nltype); % with coupling
    xMatFull(:,:,s) = xMat;
    rMatFull(:,:,s) = rMat;
end

U_1 = reshape(rMatFull,Nr,T*Ns)*pinv(sigmoid(reshape(hMatFull,Nx,T*Ns))); % obtain first estimate of Uhat using linear regression


% Generate inputs with low gain. This will be used to infer the parameters
% of the TAP dynamics

T   = 1000;    % No. of time steps
Nh  = 10;    % No. of time steps for which h is the same

Ns          = 1;    % No. of batches
gh          = 50/sqrt(Nx); % gain for h
hMatFull    = zeros(Nx,T,Ns);
x0Full      = rand(Nx,Ns);   

xMatFull    = zeros(Nx,T,Ns);
rMatFull    = zeros(Nr,T,Ns);
xMat0Full   = zeros(Nx,T,Ns);

for s = 1:Ns
    % hMatFull(:,:,s) = generateH(Nx,T,Nh,gh);
    hMatFull(:,:,s) = generateBroadH(Nx,T,Nh,gh);
    [xMat, rMat] = runTAP(x0Full(:,s), hMatFull(:,:,s), lam, Qpr, Qobs, U, J, G, nltype); % with coupling
    xMatFull(:,:,s) = xMat;
    rMatFull(:,:,s) = rMat;
    xMat_zeroJ   = runTAP(x0Full(:,s), hMatFull(:,:,s), lam, Qpr, Qobs, U, J*0, G, nltype); % without coupling
    xMat0Full(:,:,s) = xMat_zeroJ;
end

% Compare the dynamics with and without the coupling J
figure; 
subplot(1,2,1); plot(xMat','b.-'); hold on; plot(xMat_zeroJ','r.-'); 
axis([0,T,0,1]); title('blue with J, red: J=0'); xlabel('t'); ylabel('x_i(t)')

subplot(1,2,2);
plot(xMatFull(:),xMat0Full(:),'b.'); hold on; plot([0,1],[0,1],'k')
axis([0,1,0,1]); xlabel('x with J ~= 0'); ylabel('x with J = 0'); 

temp = zeros(Nx,T+1,Ns);
temp(:,1,:)     = x0Full;
temp(:,2:end,:) = xMatFull;
xMatFull = temp; clear temp;


% ---------  Run the particle filter with true values of (U, J, G) --------

K = 100; % No. of particles

x_truedec = zeros(Nx,T+1,Ns);
r_truedec = zeros(Nr,T,Ns);
P_truedec = zeros(Nx,K,T+1,Ns);
WMat      = zeros(K,Ns);
L         = zeros(Ns,1);

theta = [G; JMatToVec(J); U(:)];

tic;
for s = 1:Ns
    [L(s),x_truedec(:,:,s), P_truedec(:,:,:,s), WMat(:,s)] ...
        = particlefilter(rMatFull(:,:,s), hMatFull(:,:,s), K, lam, Qpr, Qobs, RG, theta, nltype);
    r_truedec(:,:,s) = U*x_truedec(:,2:end,s);
end
toc;


% Compute the negative log likelihood cost using these particles
CtrueVec = zeros(Ns,1);

% the cost is computed separately for each batch
tic;
for s = 1:Ns
    CtrueVec(s) = NegLL(rMatFull(:,:,s), hMatFull(:,:,s), P_truedec(:,:,:,s), WMat(:,s), lam, Qpr, Qobs, RG, nltype, theta);
end
toc;

% ----------- Now we try to learn the parameters from data using PF-EM ----

% Now use the Uhat using regression and intial values of G
% and J and run the Particle Filter to get an initial estimate of the
% latent variable dynamics

G_1 = 0.01*randn(5,1);
J_1 = Create_J(Nx, 0.05, Jtype, sc_J);

x_1 = zeros(Nx,T+1,Ns);
r_1 = zeros(Nr,T,Ns);
P_1 = zeros(Nx,K,T+1,Ns);
W_1 = zeros(K,Ns);
L_1 = zeros(Ns,1);

theta_1 = [G_1; JMatToVec(J_1); U_1(:)];

tic;
for s = 1:Ns
    [L_1(s),x_1(:,:,s), P_1(:,:,:,s), W_1(:,s)] ...
        = particlefilter(rMatFull(:,:,s), hMatFull(:,:,s), K, lam, Qpr, Qobs, RG, theta_1, nltype);
    r_1(:,:,s) = U_1*x_1(:,2:end,s);
end
toc;

% Compute negative log likelihood cost
C_1Vec = zeros(Ns,1);

tic;
for s = 1:Ns
    C_1Vec(s) = NegLL(rMatFull(:,:,s), hMatFull(:,:,s), P_1(:,:,:,s), W_1(:,s), lam, Qpr, Qobs, RG, nltype, theta_1);
end
toc;

% Plot the true vs decoded latents and neural responses
st = 6; % start plotting from this time
figure; plot(mv(xMatFull(:,st:end,:)),mv(x_truedec(:,st:end,:)),'k.')
hold on; plot(mv(xMatFull(:,st:end,:)),mv(x_1(:,st:end,:)),'b.')
% hold on; plot(mv(xMatFull(:,st:end,:)),mv(X(:,st:end,:)),'b.')

figure; plot(mv(rMatFull(:,st:end,:)),mv(r_truedec(:,st:end,:)),'k.')
hold on; plot(mv(rMatFull(:,st:end,:)),mv(r_1(:,st:end,:)),'b.')

Cinit = C_1Vec;
xinit = x_1;
rinit = r_1;

Jinit = J_1;
Ginit = G_1;
Uinit = U_1;

% ---------------------------- EM iterations ------------------------------

% EMIters = 10*Ns;
% CostEM  = zeros(EMIters,1);
% CMat    = zeros(Ns,ceil(EMIters/10));
% LMat    = zeros(Ns,ceil(EMIters/10));
% NJ      = Nx*(Nx+1)/2;


EMIters = 100;
CostEM  = zeros(EMIters,1);
CMat    = zeros(Ns,EMIters);
LMat    = zeros(Ns,EMIters);
NJ      = Nx*(Nx+1)/2;

% options for unconstrained minimization
options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton','tolX',1e-4,'MaxFunEvals',200,'GradObj','on','TolFun',1e-4,'MaxIter',200);
% options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton','tolX',1e-4,'MaxFunEvals',300,'GradObj','on','TolFun',1e-4,'MaxIter',300);

% % options for constrained minimization
% options = optimoptions(@fmincon,'Display','off','tolX',1e-4,'MaxFunEvals',200,'SpecifyObjectiveGradient',true,'TolFun',1e-4,'MaxIter',100);
% % set the bounds and constraints
% lb = zeros(length(theta),1); lb(1:5) = -9; lb(6:5+NJ) = -2; lb(6+NJ:end) = -6;
% ub = zeros(length(theta),1); ub(1:5) = 9;  ub(6:5+NJ) = 2; ub(6+NJ:end) = 6;


% Initialize the batch
idx     = randi(Ns);
rB      = rMatFull(:,:,idx); % pick the observations for the mini batch
hB      = hMatFull(:,:,idx); 
P_B     = P_1(:,:,:,idx);
WB      = W_1(:,idx);

fprintf('\nStarting EM iterations\n')

for iterem = 1:EMIters

    if mod(iterem,Ns) ~= 0
        fprintf('%3d ',iterem);
    else
        fprintf('%3d\n',iterem);
    end
    
    theta_1 = [G_1; JMatToVec(J_1); U_1(:)];
    
    fun     = @(theta_1)NegLL(rB, hB, P_B, WB, lam, Qpr, Qobs, RG, nltype, theta_1);
    
    [theta_1, CostEM(iterem)] = fminunc(fun,theta_1,options);
    % [theta_1, CostEM(iterem)] = fmincon(fun,theta_1,[],[],[],[],lb,ub,[],options);
    
    G_1     = theta_1(1:lG);
    J_1     = JVecToMat(theta_1(lG+1:lG+NJ));
    U_1     = reshape(theta_1(lG+1+NJ:end),Nr,Nx);
    
    
%     if mod(iterem,10) == 0
%         for s = 1:Ns
%             [LMat(s,iterem/10),x_1(:,:,s), P_1(:,:,:,s), W_1(:,s)] ...
%                 = particlefilter(rMatFull(:,:,s), hMatFull(:,:,s), K, lam, Qpr, Qobs, RG, theta_1, nltype);
%             r_1(:,:,s) = U_1*x_1(:,2:end,s);
%             CMat(s,iterem/10) = NegLL(rMatFull(:,:,s), hMatFull(:,:,s), P_1(:,:,:,s), W_1(:,s), lam, Qpr, Qobs, RG, nltype, theta_1);
%         end
%     end
    
    for s = 1:Ns
        [LMat(s,iterem),x_1(:,:,s), P_1(:,:,:,s), W_1(:,s)] ...
            = particlefilter(rMatFull(:,:,s), hMatFull(:,:,s), K, lam, Qpr, Qobs, RG, theta_1, nltype);
        r_1(:,:,s) = U_1*x_1(:,2:end,s);
        CMat(s,iterem) = NegLL(rMatFull(:,:,s), hMatFull(:,:,s), P_1(:,:,:,s), W_1(:,s), lam, Qpr, Qobs, RG, nltype, theta_1);
    end
    
    % Pick a new batch and run the particle filter with the updated parameters
    
    idx = randi(Ns);
    rB  = rMatFull(:,:,idx); 
    hB  = hMatFull(:,:,idx);

    [~, ~, P_B, WB] = particlefilter(rB, hB, K, lam, Qpr, Qobs, RG, theta_1, nltype);   
end

figure; 
plot(Cinit,'bx-')
hold on
plot(CMat,'bx-')
plot(CMat(:,end),'cx-')
plot(CtrueVec,'kx-')

% plot the result
figure(2); plot(mv(xMatFull(:,st:end,:)),mv(x_1(:,st:end,:)),'c.')
figure(3); plot(mv(rMatFull(:,st:end,:)),mv(r_1(:,st:end,:)),'c.')
