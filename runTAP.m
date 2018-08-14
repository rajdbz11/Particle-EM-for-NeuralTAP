function [xMat, rMat] = runTAP(x0, hMat, lam, Qpr, Qobs, U, V, J, G, nltype)

% Function that generates the TAP dynamics

% Inputs: 
% x0    : latent variables at time t = 0
% hMat  : of size Nx x T, specifies inputs h(t) for t = 1,..,T
% lam   : low pass fitlering constant for TAP dynamics
% Qpr   : covariance of process noise
% Qobs  : covariance of measurement noise
% U     : embedding matrix from latent space to neural activity
% V     : emedding matrix from input space to latent variable space
% J     : coupling matrix of the underlying distribution
% G     : global hyperparameters

% Outputs: 
% xMat  : latent variables 
% rMat  : neural activity. r = Ux + noise


T    = size(hMat,2);    % no. of time steps
Nx   = length(x0);      % no. of latent variables
Nr   = size(U,1);       % no. of neurons

xold = x0;              % initial value of x

xMat = zeros(Nx,T);     

J2   = J.^2;

TAPFn = @(x,ht)(nonlinearity( V*ht + G(1)*J*x + G(2)*J2*x + G(3)*J2*(x.^2) + G(4)*x.*(J2*x) + G(5)*x.*(J2*(x.^2)), nltype));


for tt = 1:T  
    ht          = hMat(:,tt); 
    out         = TAPFn(xold,ht);
    xnew        = (1-lam)*xold + lam*out + mvnrnd(zeros(1,Nx),Qpr,1)'; % Low pass filter and add process noise
    % xMat(:,tt)  = xnew + mvnrnd(zeros(1,Nx),Qpr,1)';    % Add process noise 
    xMat(:,tt)  = xnew;      % Add process noise 
    xold        = xnew;
end

rMat = U*xMat + mvnrnd(zeros(1,Nr),Qobs,T)'; % Adding independent observation noise to each time step