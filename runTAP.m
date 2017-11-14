function [xMat, rMat,sigmoidinputs] = runTAP(x0, hMat, lam, Qpr, Qobs, U, J, G)
% Function that generates the TAP dynamics

% Inputs: 
% x0    :latent variables at time t = 0
% hMat  :of size Nx x T, specifies h(t) for t = 1,..,T
% lam   :low pass fitlering constant for TAP dynamics
% Qpr   :covariance of process noise
% Qobs  :covariance of observation noise
% U     :embedding matrix, r = Ux + noise
% J     :coupling matrix
% G     :global hyperparameters

% Outputs: xMat: latent variables, rMat: neural activity

% Constants and initializations
T   = size(hMat,2);
J_p = powersofJ(J,2);
Nx  = length(x0);

xMat = zeros(Nx,T);
xold = x0;

sigmoidinputs = [];

for tt = 1:T
    
    ht          = hMat(:,tt); 
    [out, argf] = TAPF(xold,ht,J_p,G);
    % xnew        = (1-lam)*xold + lam*TAPF(xold,ht,J_p,G); % Low pass filtering done here
    xnew        = (1-lam)*xold + lam*out; % Low pass filtering done here
    
    xMat(:,tt)  = xnew + mvnrnd(zeros(1,Nx),Qpr,1)'; % Adding process noise at each time step
    xold        = xnew;
    sigmoidinputs = [sigmoidinputs; argf];
end

Nr = size(U,1);

rMat = U*xMat + mvnrnd(zeros(1,Nr),Qobs,T)'; % Adding independent observation noise to each time step
