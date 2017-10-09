Nx  = 3;    % No. of variables
Nr  = 5;    % No. of neurons
T   = 500;  % No. of time steps
Nh  = 10;   % No. of time steps for which h is the same
lam = 0.2;  % low pass filtering constant for the TAP dynamics

% True values of the representation (U), graphical model parameters (J) and
% global hyperparameters (G)

sp  = 0.5;  % fraction of zero entries in the coupling matrix 
J   = sparsePDMatrix(Nx,sp)/2; % Generate coupling matrix
G   = [0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,4,-4,0,-8,8,0,0,0]'; % These are the global hyperparameters for the true TAP model dynamics 
U   = randn(Nr,Nx); % Matrix for embedding the TAP dynamics into neural activity

% Noise covariances
Qpr     = 1e-3*eye(Nx); % process noise
Qobs    = 4e-3*eye(Nr); % observation  noise

hMat = generateH(Nx,T,Nh,0.25);


% Initial values for the TAP dynamics
x0      = rand(Nx,1); % This is drawn from the prior distribution on x0
 
% Generate the latent dynamics and observations for one experimental trial!
[xMat, rMat] = runTAP(x0, hMat, lam, Qpr, Qobs, U, J, G);

% Run the particle filter with true values for all the parameters of interest (U, J, G)
useprior = 0;
K = 200; % No. of particles

[x_truedec, P_AS_truedec, P_BS_truedec] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, U, J, G,useprior);

% Compute the log likelihood cost
[C_truedec, dG_truedec] = NegLogLikelihoodCost(rMat, hMat, P_AS_truedec, P_BS_truedec, lam, Qpr, Qobs, J, G, U);
% fun             = @(G)NegLogLikelihoodCost(rMat, hMat, P_AS_truedec, P_BS_truedec, lam, Qpr, Qobs, J, G, U);
% dGfd_truedec    = finitediffGrad(fun,G);


% Run the PF with true U and zero J and G
U_1 = randn(Nr,Nx); % U + 0.05*randn(Nr,Nx);
G_1 = randn(27,1);
J_1 = J;

[x_1, P_AS_1, P_BS_1] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, U_1, J_1, G_1,useprior);

[C_1, dG_1]     = NegLogLikelihoodCost(rMat, hMat, P_AS_1, P_BS_1, lam, Qpr, Qobs, J_1, G_1, U_1);
% fun     = @(G_1)NegLogLikelihoodCost(rMat, hMat, P_AS_1, P_BS_1, lam, Qpr, Qobs, J_1, G_1, U_1);
% dGfd_1    = finitediffGrad(fun,G_1);


% EM iterations
figure; plot(xMat(:),x_truedec(:),'k.'); hold on; plot(xMat(:),x_1(:),'b.')

Cinit = C_1;
xinit = x_1;
rinit = U_1*x_1;


EMIters = 25;

xRecord = zeros(Nx,T,EMIters);
rRecord = zeros(Nr,T,EMIters);
CostVec = zeros(EMIters,1);

options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','tolX',1e-3,'MaxFunEvals',500,'GradObj','on','TolFun',1e-3,'MaxIter',250);

for iterem = 1:EMIters
    
    if iterem > 20
        options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','tolX',1e-4,'MaxFunEvals',500,'GradObj','on','TolFun',1e-4,'MaxIter',250);
    elseif iterem > 35
        options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','tolX',1e-5,'MaxFunEvals',500,'GradObj','on','TolFun',1e-5,'MaxIter',250);
    end
    
    disp(iterem);

    fun     = @(G_1)NegLogLikelihoodCost(rMat, hMat, P_AS_1, P_BS_1, lam, Qpr, Qobs, J_1, G_1, U_1);
    
    G_1     = fminunc(fun,G_1,options);

    XMat    = reshape(P_BS_1, Nx,K*T);
    XSum    = reshape(sum(P_BS_1, 2),Nx,T);
    xcov    = XMat*XMat';
    rxcov   = rMat*XSum';
    % This estimate of U is obtained by setting the gradient of the
    % Loglikelihood cost w.r.t U to zero. 

    U_1    = rxcov/xcov; %+ 1e-2*randn(Nr,Nx);

    [x_1, P_AS_1, P_BS_1] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, U_1, J_1, G_1,useprior);
    [C_1, dG_1]     = NegLogLikelihoodCost(rMat, hMat, P_AS_1, P_BS_1, lam, Qpr, Qobs, J_1, G_1, U_1);
    
    disp(C_1);
    

    xRecord(:,:,iterem) = x_1;

    rRecord(:,:,iterem) = U_1*x_1;
    
    CostVec(iterem) = C_1;
end



