function [ LL, xhat, ParticlesAll, WVec, ESSVec] = particlefilter(rMat, hMat, K, P, M, RG, theta, nltype)

% Particle filter function specific to the TAP dynamics
% Type of particle filter: standard SIR filter

% Inputs: 
% rMat  : observations r(t)
% hMat  : inputs h(t)
% K     : No. of particles
% lam   : low pass filtering constant for the TAP dynamics
% P     : covariance of process noise
% M     : covariance of observation noise
% theta : parameter vector which contains G J and U
% U     : embedding matrix, r = Ux + noise
% V     : input embedding matrix
% J     : coupling matrix
% G     : global hyperparameters
% nltype: nonlinearity used in the TAP dynamics

% Ouputs:
% LL    : data log likelihood
% xhat  : decoded latent variables xhat(t)
% ParticlesAll: set of particles for all time steps
% WVec  : weights of the particles
% ESSVec: Effective sample size at each time


[Nr,T]   = size(rMat);
Nx      = size(P,2);
Nh      = size(hMat,1);

if RG % this parameter tells us if we are using a restricted set of Gs or the full set 
    lG = 5;
else
    lG = 18;
end

% Extract the required parameters
lam     = theta(1);
theta   = theta(2:end);
G       = mv(theta(1:lG));
NJ      = Nx*(Nx+1)/2;
JVec    = theta(lG+1:lG+NJ);
J       = JVecToMat(JVec);
U       = reshape(theta(lG+1+NJ:lG+NJ+Nr*Nx),Nr,Nx);
V       = reshape(theta(lG+NJ+Nr*Nx+1:end),Nx,Nh);

J2      = J.^2;

ParticlesAll = zeros(Nx,K,T+1);
% ParticlesOld = rand(Nx,K); % initialize particles
ParticlesOld = pinv(U)*rMat(:,1) + mvnrnd(zeros(1,Nx),P,K)';

ParticlesAll(:,:,1) = ParticlesOld;

WVec    = ones(K,1)/K;
ESSVec  = zeros(T,1);

Pinv    = inv(P);
Q_postinv  = Pinv + U'*(M\U);
Q_post = inv(Q_postinv);

Q_post = (Q_post + Q_post')/2; %just to ensure it is perfectly symmetric (numerical errors creepy)

TAPFn = @(x,ht)(nonlinearity( V*ht + G(1)*J*x + G(2)*J2*x + G(3)*J2*(x.^2) + G(4)*x.*(J2*x) + G(5)*x.*(J2*(x.^2) ), nltype));

LL = 0; %log likelihood log(p(r))

for tt = 1:T
    
    ht              = hMat(:,tt);
    rt              = rMat(:,tt);
    Minvr           = (M\rt);
    rMinvr          = rt'*Minvr;
    UMinvr          = U'*Minvr;
    
    % sampling x(t) from the proposal distribution p(x(t)|x(t-1), r(t))
    % p(x(t)|x(t-1),r(t)) = 1/Z*p(x(t)|x(t-1))*p(r(t)|x(t))
    
    outmat      = TAPFn(ParticlesOld, ht);
    f_tap       = (1-lam)*ParticlesOld + lam*outmat;
    Pinvf_tap   = P\f_tap; 
    v           = Pinvf_tap + UMinvr;
    mu_post     = Q_postinv\v; % mean of the proposal distribution

    % draw new particles from this proposal distribution
    ParticlesNew  = mvnrnd(mu_post',Q_post)';

    % assigning weights to the particles proportional to p(r(t)|x(t-1))
    w_ii = exp(-0.5*( rMinvr + sum(f_tap.*Pinvf_tap - v.*mu_post)' )) + 1e-128; %adding a small constant to avoid nan problem
    WVec = WVec.*w_ii; 
    
    LL = LL + log(sum(WVec));
    
    ParticlesAll(:,:,tt+1) = ParticlesNew;
    
    WVec = WVec/sum(WVec); % Normalize the weights
    if isnan(sum(WVec))
        keyboard;
    end
    
    % Resample the particles based on their weights
    
    ESS = 1/sum(WVec.^2);
    ESSVec(tt) = ESS;
    
    if ESS < K/2 && tt ~= T
        idx = resampleSystematic(WVec);
        ParticlesAll(:,:,1:tt+1) = ParticlesAll(:,idx,1:tt+1);
        WVec = ones(K,1)/K;
    end

    ParticlesOld = ParticlesAll(:,:,tt+1);
 
            
end

xhat = bsxfun(@times,ParticlesAll,repmat(reshape(WVec,1,K),Nx,1));
xhat = reshape(sum(xhat,2),Nx,T+1);