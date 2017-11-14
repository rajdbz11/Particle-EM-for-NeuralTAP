Nx  = 3;    % No. of variables
Nr  = 20; % No. of neurons in the RNN
T   = 2000;  % No. of time steps
Nh  = 10;   % No. of time steps for which h is the same
lam = 0.5;  % low pass filtering constant for the TAP dynamics

% True values of the representation (U), graphical model parameters (J) and
% global hyperparameters (G)

sp  = 0.2;  % fraction of zero entries in the coupling matrix 
J   = sparsePDMatrix(Nx,sp)/4; % Generate coupling matrix
G   = [0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,4,-4,0,-8,8,0,0,0]'; % These are the global hyperparameters for the true TAP model dynamics 
U   = randn(Nr,Nx); % Matrix for embedding the TAP dynamics into neural activity


% Noise covariances
Qpr     = 1e-4*eye(Nx); % process noise
Qobs    = 4e-4*eye(Nr); % observation  noise

hMatFull = generateH(Nx,T,Nh,0.75);


% Initial values for the TAP dynamics
x0      = rand(Nx,1); % This is drawn from the prior distribution on x0
 
% Generate the latent dynamics and observations for one experimental trial!
xMatFull = runTAP(x0, hMatFull, lam, Qpr, Qobs, U, J, G);




% Using RNN implementation
Wx      = 2/sqrt(Nr)*randn(Nr,Nx);
Wh      = 2/sqrt(Nr)*randn(Nr,Nx);
bVec    = 2/sqrt(Nr)*randn(Nr,1);

% Use linear regression to learn the weights V; think about adding some
% regularization to the weights later
inp = sigmoid(Wx*xMatFull(:,1:T-1) + Wh*hMatFull(:,1:T-1) + repmat(bVec,1,T-1));
out = xMatFull(:,2:T);
V   = out*pinv(inp);

rRNNMat = zeros(Nr,T);

%rold = pinv(V)*xMatFull(:,1); % Initialize r(t) using initial value of x(t=1)
rold  = rand(Nr,1);
rRNNMat(:,1) = rold;

for tt = 2:T
    rnew = sigmoid(Wx*V*rold + Wh*hMatFull(:,tt) + bVec) + 4e-4*randn(Nr,1); 
    rold = rnew;
    rRNNMat(:,tt) = rnew;
end

xhatFull = V*rRNNMat;
figure; plot(mv(xMatFull(:,20:end)),mv(xhatFull(:,20:end)),'b.'); hold on; plot([0,1],[0,1],'k')


% Now use only a subset of the data to run the PF
T = 500;
of = 20; % offset
xMat = xMatFull(:,1+of:of+T);
xapprox = xhatFull(:,1+of:of+T); 

hMat = hMatFull(:,1+of:of+T);
rMat = rRNNMat(:,1+of:of+T);

rhat = pinv(V)*xapprox;
x1   = V*rMat;
x2   = V*rhat;

% figure; plot(rMat(:),rhat(:),'k.')
% figure; plot(x1(:),x2(:),'k.') 


% Run the particle filter with true values for all the parameters of interest (U, J, G)
useprior = 0;
K = 100; % No. of particles

% Actually we don't know the true U here. We want to learn it using EM
H = sigmoid(hMat);
U = rMat*pinv(H);

[x_truedec, P_AS_truedec, P_BS_truedec] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, U, J, G,useprior);
r_truedec = U*x_truedec;

% Compute the log likelihood cost
theta           = [G; JMatToVec(J); U(:)];
[C_truedec, ~]  = NegLL(rMat, hMat, P_AS_truedec, P_BS_truedec, lam, Qpr, Qobs, theta);


% Run the PF with some initial values for U, G and J
H   = sigmoid(hMat);
U_1 = rMat*pinv(H);

G_1 = G;
J_1 = J;

tic;
[x_1, P_AS_1, P_BS_1] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, U_1, J_1, G_1,useprior);
r_1 = U_1*x_1;
toc;

tic;
theta_1     = [G_1; JMatToVec(J_1); U_1(:)];
[C_1, ~]    = NegLL(rMat, hMat, P_AS_1, P_BS_1, lam, Qpr, Qobs, theta_1);
toc;

figure; plot(xMat(:),x_truedec(:),'k.'); hold on; plot(xMat(:),x_1(:),'b.'); 
figure; plot(rMat(:),r_truedec(:),'k.'); hold on; plot(rMat(:),r_1(:),'b.'); 

Cinit = C_1;
xinit = x_1;
rinit = r_1;


% EM iterations

EMIters = 500;


options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','tolX',1e-3,'MaxFunEvals',50,'GradObj','on','TolFun',1e-3,'MaxIter',40);


BS = 20; % batch size

% Initialize the batch
% pick the batch
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

    [x_B, P_AS_B, P_BS_B] = particlefilter(rB, hB, K, lam, Qpr, Qobs, U_1, J_1, G_1,useprior);
    
    
end

theta_1 = [G_1; JMatToVec(J_1); U_1(:)];

[x_1, P_AS_1, P_BS_1] = particlefilter(rMat, hMat, K, lam, Qpr, Qobs, U_1, J_1, G_1,useprior);
[C_1, ~]     = NegLL(rMat, hMat, P_AS_1, P_BS_1, lam, Qpr, Qobs, theta_1);
r_1 = U_1*x_1;




% % % Run the particle filter with true values for all the parameters of interest (V, J, G)
% 
% 
% % Different one for the PF because of the special formulation
% Qobs    = 1e-4*eye(Nx); % observation  noise 
% 
% K = 100;
% 
% 
% [x_truedec, P_AS_truedec, P_BS_truedec] = particlefilterRNN(rMat, hMat, K, lam, Qpr, Qobs, V, J, G);
% theta    = [G; JMatToVec(J); V(:)];
% [C_truedec, ~]  = NegLL_RNN(rMat, hMat, P_AS_truedec, P_BS_truedec, lam, Qpr, Qobs, theta);
% 
% 
% 
% % Run the PF with some initial values for V, G and J
% % H   = sigmoid(hMat);
% % V_1 = H*pinv(rMat);
% V_1 = V;
% G_1 = randn(27,1); G_1(1:10) = 0; G_1(19) = 0;
% J_1 = sparsePDMatrix(Nx,sp)/2;
% 
% tic;
% [x_1, P_AS_1, P_BS_1] = particlefilterRNN(rMat, hMat, K, lam, Qpr, Qobs, V_1, J_1, G_1);
% % r_1 = V_1*x_1;
% toc;
% 
% tic;
% theta_1     = [G_1; JMatToVec(J_1); V_1(:)];
% [C_1, ~]    = NegLL_RNN(rMat, hMat, P_AS_1, P_BS_1, lam, Qpr, Qobs, theta_1);
% toc;
% 
% figure; plot(xMat(:),x_truedec(:),'k.'); hold on; plot(xMat(:),x_1(:),'b.'); 
% % figure; plot(rMat(:),r_truedec(:),'k.'); hold on; plot(rMat(:),r_1(:),'b.'); 
%  
% Cinit = C_1;
% xinit = x_1;
% 
% 
% 
% % ------------------------------ Do EM ------------------------------
% 
% EMIters = 500;
% 
% 
% 
% options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','tolX',1e-3,'MaxFunEvals',50,'GradObj','on','TolFun',1e-3,'MaxIter',40);
% 
% 
% BS = 20; % batch size
% 
% % Initialize the batch
% % pick the batch
% si      = randi(T-BS+1);
% idx     = si:si+BS-1;
% rB      = rMat(:,idx); % pick the observations for the mini batch
% hB      = hMat(:,idx);
% P_AS_B  = P_AS_1(:,:,idx);
% P_BS_B  = P_BS_1(:,:,idx);
% NJ      = Nx*(Nx+1)/2;
% 
% for iterem = 1:EMIters
%     
%     disp(iterem);
%     
%     theta_1 = [G_1; JMatToVec(J_1); V_1(:)];
%     
%     fun     = @(theta_1)NegLL_RNN(rB, hB, P_AS_B, P_BS_B, lam, Qpr, Qobs, theta_1);
%     
%     theta_1 = fminunc(fun,theta_1,options);
%     
%     G_1     = theta_1(1:27);
%     J_1     = JVecToMat(theta_1(28:27+NJ));
%     V_1     = reshape(theta_1(28+NJ:end),Nx,Nr);
%     
%     % Pick a new batch and run the particle filter with the updated parameters
%     
%     si  = randi(T-BS+1);
%     idx = si:si+BS-1;
%     rB  = rMat(:,idx); % pick the observations for the mini batch
%     hB  = hMat(:,idx);
% 
%     [x_B, P_AS_B, P_BS_B] = particlefilterRNN(rB, hB, K, lam, Qpr, Qobs, V_1, J_1, G_1);
%     
%     
% end
% 
% theta_1 = [G_1; JMatToVec(J_1); V_1(:)];
% 
% [x_1, P_AS_1, P_BS_1] = particlefilterRNN(rMat, hMat, K, lam, Qpr, Qobs, V_1, J_1, G_1);
% [C_1, ~]     = NegLL_RNN(rMat, hMat, P_AS_1, P_BS_1, lam, Qpr, Qobs, theta_1);
% 
% 
% xtilde = [V_1*rMat; ones(1,T)];
% Atilde = xMat*pinv(xtilde);
% x_1mappedtoxMat = Atilde*xtilde;
