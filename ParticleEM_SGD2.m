Nx  = 3;    % No. of variables
Nr  = 5;    % No. of neurons
T   = 500;  % No. of time steps
Nh  = 20;   % No. of time steps for which h is the same
lam = 0.25;  % low pass filtering constant for the TAP dynamics

% True values of the representation (U), graphical model parameters (J) and global hyperparameters (G)

sp  = 0.5;  % fraction of zero entries in the coupling matrix 
gj  = 0.25; % scaling for the coupling matrix
J   = gj*sparsePDMatrix(Nx,sp); % Generate coupling matrix
G   = [0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,4,-4,0,-8,8,0,2,2]'; % G for the true TAP model dynamics 
U   = randn(Nr,Nx); % Matrix for embedding the TAP dynamics into neural activity

% Noise covariances
Qpr     = 1e-4*eye(Nx); % process noise
Qobs    = 4e-4*eye(Nr); % observation  noise

% Generate inputs
gh      = 0.5; % gain for h
hMat    = generateH(Nx,T,Nh,gh);

% Initial values for the TAP dynamics
x0      = rand(Nx,1); % This is drawn from the prior distribution on x0
 
% ----------- Generate the latent dynamics and observations ---------------
[xMat, rMat, sinps] = runTAP(x0, hMat, lam, Qpr, Qobs, U, J, G); 


% ----------- Run the particle filter with true values of (U, J, G) -------

K = 100; % No. of particles

[x_truedec, P_AS_truedec, P_BS_truedec] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, U, J, G);
r_truedec = U*x_truedec;

% Compute the negative log likelihood cost using these particles
theta           = [G; JMatToVec(J); U(:)];
[C_truedec, ~]  = NegLL(rMat, hMat, P_AS_truedec, P_BS_truedec, lam, Qpr, Qobs, theta);


% ----------- Now we try to learn the parameters from data using PF-EM ----

% Choose initial values for parameters
H   = sigmoid(hMat);
U_1 = rMat*pinv(H);
G_1 = randn(27,1); G_1(1:10) = 0; G_1(19) = 0;
% J_1 = sparsePDMatrix(Nx,sp)/2;
J_1 = J;

% Run the PF with the initial values of the parameters
tic;
[x_1, P_AS_1, P_BS_1] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, U_1, J_1, G_1);
r_1 = U_1*x_1;
toc;

% Compute negative log likelihood cost
tic;
theta_1     = [G_1; JMatToVec(J_1); U_1(:)];
[C_1, ~]    = NegLL(rMat, hMat, P_AS_1, P_BS_1, lam, Qpr, Qobs, theta_1);
toc;

% Plot the true vs decoded latents and neural responses
figure; plot(xMat(:),x_truedec(:),'k.'); hold on; plot(xMat(:),x_1(:),'b.'); 
figure; plot(rMat(:),r_truedec(:),'k.'); hold on; plot(rMat(:),r_1(:),'b.'); 

Cinit = C_1;
xinit = x_1;
rinit = r_1;


% ---------------------------- EM iterations ------------------------------

EMIters = 2000;

options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','tolX',1e-3,'MaxFunEvals',50,'GradObj','on','TolFun',1e-3,'MaxIter',10);


BS = 20; % batch size

% Initialize the batch
si      = randi(T-BS+1);
idx     = si:si+BS-1;
rB      = rMat(:,idx); % pick the observations for the mini batch
hB      = hMat(:,idx);
P_AS_B  = P_AS_1(:,:,idx);
P_BS_B  = P_BS_1(:,:,idx);
NJ      = Nx*(Nx+1)/2;

for iterem = 1:EMIters
    
    disp(iterem);
    
    theta_1 = [G_1; JMatToVec(J_1); U_1(:)];
    
    fun     = @(theta_1)NegLL(rB, hB, P_AS_B, P_BS_B, lam, Qpr, Qobs, theta_1);
    
    theta_1 = fminunc(fun,theta_1,options);
    
    G_1     = theta_1(1:27);
    J_1     = JVecToMat(theta_1(28:27+NJ));
    U_1     = reshape(theta_1(28+NJ:end),Nr,Nx);
    
    % Pick a new batch and run the particle filter with the updated parameters
    
    si  = randi(T-BS+1);
    idx = si:si+BS-1;
    rB  = rMat(:,idx); % pick the observations for the mini batch
    hB  = hMat(:,idx);

    [x_B, P_AS_B, P_BS_B] = particlefilter(rB, hB, K, lam, Qpr, Qobs, U_1, J_1, G_1);
    
    
end

% Run the PF on the full data set using the estimated parameters

[x_1, P_AS_1, P_BS_1] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, U_1, J_1, G_1);
[C_1, ~]     = NegLL(rMat, hMat, P_AS_1, P_BS_1, lam, Qpr, Qobs, theta_1);
r_1 = U_1*x_1;


% -------------------  Run the cross validation ---------------------------

hMatCV  = generateH(Nx,T,Nh,0.2);
x0CV    = rand(Nx,1); % This is drawn from the prior distribution on x0
 
% Generate the latent dynamics and observations for one experimental trial!
[xMatCV, rMatCV]    = runTAP(x0CV, hMatCV, lam, Qpr, Qobs, U, J, G);

% Run PF with true parameters
[xCV_truedec, ~, ~] = particlefilter(rMatCV, hMatCV, K, lam, Qpr, Qobs, U, J, G);
rCV_truedec         = U*xCV_truedec;

% Run PF with parameters learnt using EM
[xCV_1, ~, ~]       = particlefilter(rMatCV, hMatCV, K, lam, Qpr, Qobs, U_1, J_1, G_1);
rCV_1               = U_1*xCV_1;

% Find a linear transformation that maps xMatCV to xCV_1
A           = xCV_1*pinv(xMatCV);
xMatCV_a    = A*xMatCV;

figure; plot(xCV_1(:),xMatCV_a(:),'b.')

% Check if U_1 maps to U
Uhat = U_1*A;
figure; plot(U(:),Uhat(:),'b*')


