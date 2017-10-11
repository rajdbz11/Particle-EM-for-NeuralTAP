% Script to learn the TAP parameters (U, J and G) from the neural activity 
% Here, we learn the parameters from 2 different initial conditions and
% compare the obtained solutions


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
K = 90; % No. of particles

[x_truedec, P_AS_truedec, P_BS_truedec] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, U, J, G,useprior);
r_truedec = U*x_truedec;

% Compute the log likelihood cost
theta = [G; JMatToVec(J)];
[C_truedec, dtheta_truedec] = NegLogLikelihoodCost(rMat, hMat, P_AS_truedec, P_BS_truedec, lam, Qpr, Qobs, U, theta);


% Run the PF with true U and zero J and G
U_1 = randn(Nr,Nx); 
G_1 = randn(27,1);
J_1 = sparsePDMatrix(Nx,sp)/2; % random initialization for J as well

[x_1, P_AS_1, P_BS_1] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, U_1, J_1, G_1,useprior);
r_1 = U_1*x_1;

theta_1 = [G_1; JMatToVec(J_1)];
[C_1, dtheta_1]     = NegLogLikelihoodCost(rMat, hMat, P_AS_1, P_BS_1, lam, Qpr, Qobs, U_1, theta_1);


% Second initialization
U_2 = randn(Nr,Nx); 
G_2 = randn(27,1);
J_2 = sparsePDMatrix(Nx,sp)/2; % random initialization for J as well

[x_2, P_AS_2, P_BS_2] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, U_2, J_2, G_2,useprior);
r_2 = U_2*x_2;

theta_2 = [G_2; JMatToVec(J_2)];
[C_2, dtheta_2]     = NegLogLikelihoodCost(rMat, hMat, P_AS_2, P_BS_2, lam, Qpr, Qobs, U_2, theta_2);


figure; plot(xMat(:),x_truedec(:),'k.'); hold on; plot(xMat(:),x_1(:),'b.'); plot(xMat(:),x_2(:),'r.')
figure; plot(rMat(:),r_truedec(:),'k.'); hold on; plot(rMat(:),r_1(:),'b.'); plot(rMat(:),r_2(:),'r.')

Cinit1 = C_1;
xinit1 = x_1;
rinit1 = r_1;

Cinit2 = C_2;
xinit2 = x_2;
rinit2 = r_2;


% EM iterations

EMIters = 20;

xRecord1 = zeros(Nx,T,EMIters);
rRecord1 = zeros(Nr,T,EMIters);
CostVec1 = zeros(EMIters,1);

xRecord2 = zeros(Nx,T,EMIters);
rRecord2 = zeros(Nr,T,EMIters);
CostVec2 = zeros(EMIters,1);

options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','tolX',1e-4,'MaxFunEvals',100,'GradObj','on','TolFun',1e-4,'MaxIter',50);

for iterem = 1:EMIters

    disp(iterem);
    
    % for initialization 1
    
    theta_1 = [G_1; JMatToVec(J_1)];

    fun     = @(theta_1)NegLogLikelihoodCost(rMat, hMat, P_AS_1, P_BS_1, lam, Qpr, Qobs, U_1, theta_1);
    
    theta_1 = fminunc(fun,theta_1,options);

    XMat    = reshape(P_BS_1, Nx,K*T);
    XSum    = reshape(sum(P_BS_1, 2),Nx,T);
    xcov    = XMat*XMat';
    rxcov   = rMat*XSum';
    
    U_1     = rxcov/xcov;
    G_1     = theta_1(1:27);
    J_1     = JVecToMat(theta_1(28:end));

    [x_1, P_AS_1, P_BS_1] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, U_1, J_1, G_1,useprior);
    r_1 = U_1*x_1;
    
    [C_1, dtheta_1]     = NegLogLikelihoodCost(rMat, hMat, P_AS_1, P_BS_1, lam, Qpr, Qobs, U_1, theta_1);
    
    disp(C_1);
    

    xRecord1(:,:,iterem) = x_1;

    rRecord1(:,:,iterem) = r_1;
    
    CostVec1(iterem)     = C_1;
    
    % for initialization 2
    
    theta_2 = [G_2; JMatToVec(J_2)];

    fun     = @(theta_2)NegLogLikelihoodCost(rMat, hMat, P_AS_2, P_BS_2, lam, Qpr, Qobs, U_2, theta_2);
    
    theta_2 = fminunc(fun,theta_2,options);

    XMat    = reshape(P_BS_2, Nx,K*T);
    XSum    = reshape(sum(P_BS_2, 2),Nx,T);
    xcov    = XMat*XMat';
    rxcov   = rMat*XSum';
    
    U_2     = rxcov/xcov;
    G_2     = theta_2(1:27);
    J_2     = JVecToMat(theta_2(28:end));

    [x_2, P_AS_2, P_BS_2] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, U_2, J_2, G_2,useprior);
    r_2 = U_2*x_2;
    
    [C_2, dtheta_2]     = NegLogLikelihoodCost(rMat, hMat, P_AS_2, P_BS_2, lam, Qpr, Qobs, U_2, theta_2);
    
    disp(C_2);
    

    xRecord2(:,:,iterem) = x_2;

    rRecord2(:,:,iterem) = r_2;
    
    CostVec2(iterem)     = C_2;
end


a1vec = -0.2:0.05:1.2; 
a2vec = -0.2:0.05:1.2; 

L1 = length(a1vec);
L2 = length(a2vec);
TVec = zeros(L1,L2); T2Vec = zeros(L1,L2);

for k1 = 1:L1
    a1 = a1vec(k1);
    disp(a1);
    for k2 = 1:L2
        a2 = a1vec(k2);
        Unew = (1-a1)*(1-a2)*U + a1*U_1 + a2*U_2;
        Gnew = (1-a1)*(1-a2)*G + a1*G_1 + a2*G_2;
        Jnew = (1-a1)*(1-a2)*J + a1*J_1 + a2*J_2;
        theta = [Gnew; JMatToVec(Jnew)];
        [x_n, P_AS_n, P_BS_n] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, Unew, Jnew, Gnew,useprior);
        TVec(k1,k2) = NegLogLikelihoodCost(rMat, hMat, P_AS_n, P_BS_n, lam, Qpr, Qobs, Unew, theta);
        T2Vec(k1,k2) = norm(rMat(:) - mv(Unew*x_n));
    end
end


b1vec = -0.2:0.05:1.2;  

L1 = length(b1vec);

M1Vec = zeros(L1,1);
M2Vec = zeros(L1,1); 

for k1 = 1:L1
    b1      = b1vec(k1);
    disp(b1);
    Unew    = (1-b1)*U + b1*U_1;
    Gnew    = (1-b1)*G + b1*G_1;
    Jnew    = (1-b1)*J + b1*J_1;
    theta   = [Gnew; JMatToVec(Jnew)];
    
    [~, P_AS_n, P_BS_n] = particlefilter(rMat(:,1:250), hMat(:,1:250), K, lam, Qpr, Qobs, Unew, Jnew, Gnew,useprior);
    M1Vec(k1) = NegLogLikelihoodCost(rMat(:,1:250), hMat(:,1:250), P_AS_n, P_BS_n, lam, Qpr, Qobs, Unew, theta);
    
    [~, P_AS_n, P_BS_n] = particlefilter(rMat(:,251:500), hMat(:,251:500), K, lam, Qpr, Qobs, Unew, Jnew, Gnew,useprior);
    M2Vec(k1) = NegLogLikelihoodCost(rMat(:,251:500), hMat(:,251:500), P_AS_n, P_BS_n, lam, Qpr, Qobs, Unew, theta);

end

