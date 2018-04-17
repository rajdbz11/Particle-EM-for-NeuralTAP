% Script for Inferring TAP Inference from neural activity using Particle EM

% Set noise seed
noise_seed = randi(2000);
rng(noise_seed); 

% ----------------------- Initialize parameters ---------------------------

Nx  = 3;     % No. of variables
Nr  = 4;    % No. of neurons
T   = 50;    % No. of time steps
Nh  = 10;    % No. of time steps for which h is the same
Ns  = 10;    % No. of batches
lam = 0.25;  % low pass filtering constant for the TAP dynamics

% Noise covariances
Qpr     = 1e-5*eye(Nx); % process noise
Qobs    = 4e-4*eye(Nr); % measurement noise

% True TAP brain parameters
Jtype   = 'ferr';
J       = 3*Create_J(Nx, 0.1, Jtype, 0); % Coupling matrix
G       = [0,2,0,0,0,0,0,0,0,0,4,-4,0,-8,8,0,0,0]'; % G_TAP
U       = randn(Nr,Nx); % Embedding matrix


% ---------- Generate the latent dynamics and observations ----------------

% Inputs
gh      = 2/sqrt(Nx); % gain for h

hMatFull = zeros(Nx,T,Ns);
x0Full   = rand(Nx,Ns);   

xMatFull = zeros(Nx,T,Ns);
rMatFull = zeros(Nr,T,Ns);
xMat0Full = zeros(Nx,T,Ns);

for s = 1:Ns
    hMatFull(:,:,s) = generateH(Nx,T,Nh,gh);
    [xMat, rMat] = runTAP(x0Full(:,s), hMatFull(:,:,s), lam, Qpr, Qobs, U, J, G); % with coupling
    xMatFull(:,:,s) = xMat;
    rMatFull(:,:,s) = rMat;
    xMat_zeroJ   = runTAP(x0Full(:,s), hMatFull(:,:,s), lam, Qpr, Qobs, U, J*0, G); % without coupling
    xMat0Full(:,:,s) = xMat_zeroJ;
end

% Compare the dynamics with and without the coupling J
figure; 
subplot(1,2,1); plot(xMat','b'); hold on; plot(xMat_zeroJ','r'); 
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

Qpr     = 2e-4*eye(Nx); % assumed process noise
Qobs    = 1e-3*eye(Nr); % assumed measurement noise

x_truedec = zeros(Nx,T+1,Ns);
r_truedec = zeros(Nr,T,Ns);
P_truedec = zeros(Nx,K,T+1,Ns);
WMat      = zeros(K,Ns);

for s = 1:Ns
    [x_truedec(:,:,s), P_truedec(:,:,:,s), WMat(:,s)] ...
        = particlefilter(rMatFull(:,:,s), hMatFull(:,:,s), K, lam, Qpr, Qobs, U, J, G);
    r_truedec(:,:,s) = U*x_truedec(:,2:end,s);
end

% Compute the negative log likelihood cost using these particles
theta           = [G; JMatToVec(J); U(:)];

CtrueVec = zeros(Ns,1);

for s = 1:Ns
    CtrueVec(s) = NegLL(rMatFull(:,:,s), hMatFull(:,:,s), P_truedec(:,:,:,s), WMat(:,s), lam, Qpr, Qobs, theta);
end


% ----------- Now we try to learn the parameters from data using PF-EM ----

U_1 = reshape(rMatFull,Nr,T*Ns)*pinv(sigmoid(reshape(hMatFull,Nx,T*Ns)));
G_1 = randn(18,1);
J_1 = Create_J(Nx, 0.05, Jtype, 0);


x_1 = zeros(Nx,T+1,Ns);
r_1 = zeros(Nr,T,Ns);
P_1 = zeros(Nx,K,T+1,Ns);
W_1 = zeros(K,Ns);

for s = 1:Ns
    [x_1(:,:,s), P_1(:,:,:,s), W_1(:,s), temp] ...
        = particlefilter(rMatFull(:,:,s), hMatFull(:,:,s), K, lam, Qpr, Qobs, U_1, J_1, G_1);
    r_1(:,:,s) = U_1*x_1(:,2:end,s);
end

% Compute negative log likelihood cost
theta_1 = [G_1; JMatToVec(J_1); U_1(:)];
C_1Vec = zeros(Ns,1);

tic;
for s = 1:Ns
    C_1Vec(s) = NegLL(rMatFull(:,:,s), hMatFull(:,:,s), P_1(:,:,:,s), W_1(:,s), lam, Qpr, Qobs, theta_1);
end
toc;

% Plot the true vs decoded latents and neural responses
figure; plot(mv(xMatFull(:,2:end,:)),mv(x_truedec(:,2:end,:)),'k.')
hold on; plot(mv(xMatFull(:,2:end,:)),mv(x_1(:,2:end,:)),'b.')

figure; plot(mv(rMatFull(:,2:end,:)),mv(r_truedec(:,2:end,:)),'k.')
hold on; plot(mv(rMatFull(:,2:end,:)),mv(r_1(:,2:end,:)),'b.')


Cinit = C_1Vec;
xinit = x_1;
rinit = r_1;

Jinit = J_1;
Ginit = G_1;
Uinit = U_1;

% ---------------------------- EM iterations ------------------------------

EMIters = 100;
CostEM = zeros(EMIters,1);

% options for unconstrained minimization
options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','tolX',1e-3,'MaxFunEvals',100,'GradObj','on','TolFun',1e-3,'MaxIter',50);
NJ      = Nx*(Nx+1)/2;

% % options for constrained minimization
% options = optimoptions(@fmincon,'Display','iter','tolX',1e-4,'MaxFunEvals',300,'SpecifyObjectiveGradient',true,'TolFun',1e-4,'MaxIter',200);
% % set the bounds and constraints
% NJ      = Nx*(Nx+1)/2;
% lb = zeros(length(theta),1); lb(1:18) = -9; lb(19:18+NJ) = 0; lb(19+NJ:end) = -3;
% ub = zeros(length(theta),1); ub(1:18) = 9;  ub(19:18+NJ) = 1; ub(19+NJ:end) = 3;
% Aeq = zeros(length(theta)); beq = zeros(length(theta),1);
% idx = find(JMatToVec(J) == 0); 
% % These equality constraints are  imposing the diagonal elements of J to be zero
% for k = 1:length(idx)
%     Aeq(idx(k)+18,idx(k)+18) = 1; 
% end

% Initialize the batch
idx     = randi(Ns);
rB      = rMatFull(:,:,idx); % pick the observations for the mini batch
hB      = hMatFull(:,:,idx); 
P_B     = P_1(:,:,:,idx);
WB      = W_1(:,idx);


for iterem = 1:EMIters

    disp(iterem);
    
    theta_1 = [G_1; JMatToVec(J_1); U_1(:)];
    
    fun     = @(theta_1)NegLL(rB, hB, P_B, WB, lam, Qpr, Qobs, theta_1);
    
    [theta_1, CostEM(iterem)] = fminunc(fun,theta_1,options);
    % [theta_1, CostEM(iterem)] = fmincon(fun,theta_1,[],[],Aeq,beq,lb,ub,[],options);
    
    G_1     = theta_1(1:18);
    J_1     = JVecToMat(theta_1(19:18+NJ));
    U_1     = reshape(theta_1(19+NJ:end),Nr,Nx);
    
    % Pick a new batch and run the particle filter with the updated parameters
    
    idx = randi(Ns);
    rB  = rMatFull(:,:,idx); 
    hB  = hMatFull(:,:,idx);

    [~, P_B, WB] = particlefilter(rB, hB, K, lam, Qpr, Qobs, U_1, J_1, G_1);   
end

% Run the PF on the full data set using the estimated parameters
for s = 1:Ns
    [x_1(:,:,s), P_1(:,:,:,s), W_1(:,s), temp] ...
        = particlefilter(rMatFull(:,:,s), hMatFull(:,:,s), K, lam, Qpr, Qobs, U_1, J_1, G_1);
    r_1(:,:,s) = U_1*x_1(:,2:end,s);
    C_1Vec(s) = NegLL(rMatFull(:,:,s), hMatFull(:,:,s), P_1(:,:,:,s), W_1(:,s), lam, Qpr, Qobs, theta_1);
end


% plot the result
figure(2); plot(mv(xMatFull(:,2:end,:)),mv(x_1(:,2:end,:)),'c.')
figure(3); plot(mv(rMatFull(:,2:end,:)),mv(r_1(:,2:end,:)),'c.')


% % compute the hessian
% % compute using only one batch for now
% 
% % first using the true parameters
% % Pick the batch
% idx     = 1;
% rB      = rMatFull(:,:,idx); % pick the observations for the mini batch
% hB      = hMatFull(:,:,idx); 
% 
% P_B     = P_truedec(:,:,:,idx);
% WB      = WMat(:,idx);
% fun     = @(theta)NegLL(rB, hB, P_B, WB, lam, Qpr, Qobs, theta);
% Htrue   = finitediffHess(fun,theta);
% 
% 
% % compute hessian using parameters obtained after running EM
% P_B     = P_1(:,:,:,idx);
% WB      = W_1(:,idx);
% fun     = @(theta_1)NegLL(rB, hB, P_B, WB, lam, Qpr, Qobs, theta_1);
% H_1     = finitediffHess(fun,theta_1);
