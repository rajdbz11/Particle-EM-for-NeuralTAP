Nx  = 3;    % No. of variables
Nr  = 5;    % No. of neurons
T   = 100;  % No. of time steps
Nh  = 2;   % No. of time steps for which h is the same
lam = 0.5;  % low pass filtering constant for the TAP dynamics

% True values of the representation (U), graphical model parameters (J) and
% global hyperparameters (G)

sp  = 0.5;  % fraction of zero entries in the coupling matrix 
J   = sparsePDMatrix(Nx,sp)/2; % Generate coupling matrix
G   = [0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,4,-4,0,-8,8,0,0,0]'; % These are the global hyperparameters for the true TAP model dynamics 
U   = randn(Nr,Nx); % Matrix for embedding the TAP dynamics into neural activity

% Noise covariances
Qpr     = 1e-4*eye(Nx); % process noise
Qobs    = 4e-4*eye(Nr); % observation  noise

hMat = generateH(Nx,T,Nh,0.25);


% Initial values for the TAP dynamics
x0      = rand(Nx,1); % This is drawn from the prior distribution on x0
 
% Generate the latent dynamics and observations for one experimental trial!
[xMat, rMat] = runTAP(x0, hMat, lam, Qpr, Qobs, U, J, G);

% Run the particle filter with true values for all the parameters of interest (U, J, G)
useprior = 0;
K = 90; % No. of particles

[x_truedec, P_AS_truedec, P_BS_truedec] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, U, J, G,useprior);
r_truedec = U*x_truedec;

% Compute the log likelihood cost
theta = [G; JMatToVec(J)];
[C_truedec, dtheta_truedec] = NegLogLikelihoodCost(rMat, hMat, P_AS_truedec, P_BS_truedec, lam, Qpr, Qobs, U, theta);


% Run the PF with true U and zero J and G

H   = sigmoid(hMat);
U_1 = rMat*pinv(H);
G_1 = randn(27,1);
J_1 = sparsePDMatrix(Nx,sp)/2;

tic;
[x_1, P_AS_1, P_BS_1] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, U_1, J_1, G_1,useprior);
r_1 = U_1*x_1;
toc;

tic;
theta_1 = [G_1; JMatToVec(J_1)];
[C_1, ~]     = NegLogLikelihoodCost(rMat, hMat, P_AS_1, P_BS_1, lam, Qpr, Qobs, U_1, theta_1);
toc;

figure; plot(xMat(:),x_truedec(:),'k.'); hold on; plot(xMat(:),x_1(:),'b.'); 
figure; plot(rMat(:),r_truedec(:),'k.'); hold on; plot(rMat(:),r_1(:),'b.'); 

Cinit = C_1;
xinit = x_1;
rinit = r_1;


% EM iterations

EMIters = 100;

xRecord = zeros(Nx,T,EMIters/10);
rRecord = zeros(Nr,T,EMIters/10);
CostVec = zeros(EMIters/10,1);

options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','tolX',1e-4,'MaxFunEvals',100,'GradObj','on','TolFun',1e-4,'MaxIter',50);


BS = 20; % batch size

% Initialize the batch
% pick the batch
si = randi(T-BS+1);
idx = si:si+BS-1;
rB = rMat(:,idx); % pick the observations for the mini batch
hB = hMat(:,idx);
P_AS_B = P_AS_1(:,:,idx);
P_BS_B = P_BS_1(:,:,idx);

for iterem = 1:EMIters
    
    disp(iterem);
    
    theta_1 = [G_1; JMatToVec(J_1)];

    fun     = @(theta_1)NegLogLikelihoodCost(rB, hB, P_AS_B, P_BS_B, lam, Qpr, Qobs, U_1, theta_1);
    
    theta_1 = fminunc(fun,theta_1,options);

    XMat    = reshape(P_AS_B, Nx,K*BS);
    XSum    = reshape(sum(P_AS_B, 2),Nx,BS);
    xcov    = XMat*XMat';
    rxcov   = rB*XSum';
    
    U_1     = rxcov/xcov;
    G_1     = theta_1(1:27);
    J_1     = JVecToMat(theta_1(28:end));
    
    % Pick a new batch and run the particle filter with the updated parameters
    
    si = randi(T-BS+1);
    idx = si:si+BS-1;
    rB = rMat(:,idx); % pick the observations for the mini batch
    hB = hMat(:,idx);

    [x_B, P_AS_B, P_BS_B] = particlefilter(rB, hB, K, lam, Qpr, Qobs, U_1, J_1, G_1,useprior);
    
    % Evaluate the cost every 10 iterations on all the observations
%     if mod(iterem,10) == 0
%         % First run the PF on all the observations
%         [x_1, P_AS_1, P_BS_1] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, U_1, J_1, G_1,useprior);
%         [C_1, dtheta_1]     = NegLogLikelihoodCost(rMat, hMat, P_AS_1, P_BS_1, lam, Qpr, Qobs, U_1, theta_1);
%         disp(C_1);
%         xRecord(:,:,iterem/10) = x_1;
%         rRecord(:,:,iterem/10) = r_1;
%         CostVec(iterem/10)     = C_1;
%     
%     end
    
end

[x_1, P_AS_1, P_BS_1] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, U_1, J_1, G_1,useprior);
[C_1, dtheta_1]     = NegLogLikelihoodCost(rMat, hMat, P_AS_1, P_BS_1, lam, Qpr, Qobs, U_1, theta_1);
r_1 = U_1*x_1;


