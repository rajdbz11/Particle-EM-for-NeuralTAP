Nx  = 3;    % No. of variables
Nr  = 10;    % No. of neurons
T   = 500;  % No. of time steps
Nh  = 25;   % No. of time steps for which h is the same
lam = 0.2;  % low pass filtering constant for the TAP dynamics

% True values of the representation (U), graphical model parameters (J) and
% global hyperparameters (G)

sp  = 0.4;  % fraction of zero entries in the coupling matrix 
J   = sparsePDMatrix(Nx,0.4); % Generate coupling matrix
G   = [0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,4,-4,0,-8,8,0,0,0]'; % These are the global hyperparameters for the true TAP model dynamics 
U   = randn(Nr,Nx); % Matrix for embedding the TAP dynamics into neural activity

% Noise covariances
Qpr     = 1e-3*eye(Nx); % process noise
Qobs    = 4e-3*eye(Nr); % observation  noise

hMat = generateH(Nx,T,Nh,0.4);


% Initial values for the TAP dynamics
x0      = rand(Nx,1); % This is drawn from the prior distribution on x0
 
% Generate the latent dynamics and observations for one experimental trial!
[xMat, rMat] = runTAP(x0, hMat, lam, Qpr, Qobs, U, J, G);

% Run the particle filter with values for all the parameters of interest (U, J, G)
K = 100; % No. of particles
[x_truedec, PS_truedec] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, U, J, G);

[x_1, PS_1] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, randn(Nr,Nx), J*0, G*0);

figure; plot(xMat(:),x_truedec(:),'b.'); hold on
plot(xMat(:),x_1(:),'g.')
