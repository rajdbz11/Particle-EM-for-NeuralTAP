function [xMat, rMat, sigmoidinputs] = runTAP(x0, hMat, lam, Qpr, Qobs, U, J, G)

% Function that generates the TAP dynamics

% Inputs: 
% x0    : latent variables at time t = 0
% hMat  : of size Nx x T, specifies inputs h(t) for t = 1,..,T
% lam   : low pass fitlering constant for TAP dynamics
% Qpr   : covariance of process noise
% Qobs  : covariance of measurement noise
% U     : embedding matrix for neural activity
% J     : coupling matrix of the underlying distribution
% G     : global hyperparameters

% Outputs: 
% xMat  : latent variables 
% rMat  : neural activity. r = Ux + noise
% sigmoidinputs: argument of the sigmoid for each time step


T    = size(hMat,2);    % no. of time steps
Nx   = length(x0);      % no. of latent variables
Nr   = size(U,1);       % no. of neurons

J_p  = powersofJ(J,2);  % element-wise powers of the J matrix
xold = x0;              % initial value of x

xMat            = zeros(Nx,T);     
sigmoidinputs   = zeros(Nx,T); 

for tt = 1:T  
    ht          = hMat(:,tt); 
    [out, argf] = TAPF(xold,ht,J_p,G);
    xnew        = (1-lam)*xold + lam*out;               % Low pass filter 
    xMat(:,tt)  = xnew + mvnrnd(zeros(1,Nx),Qpr,1)';    % Add process noise 
    xold        = xnew;
    sigmoidinputs(:,tt) = argf;
end

rMat = U*xMat + mvnrnd(zeros(1,Nr),Qobs,T)'; % Adding independent observation noise to each time step
