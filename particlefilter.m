function [xhat, ParticlesAll, WVec, ESSVec] = particlefilter(rMat, hMat, K, lam, P, M, U, J, G)

% Particle filter function specific to the TAP dynamics
% Type of particle filter: standard SIR filter

% Inputs: 
% rMat  : observations r(t)
% hMat  : inputs h(t)
% K     : No. of particles
% lam   : low pass filtering constant for the TAP dynamics
% P     :covariance of process noise
% M     :covariance of observation noise
% U     :embedding matrix, r = Ux + noise
% J     :coupling matrix
% G     :global hyperparameters

% Ouputs:
% xhat  : decoded latent variables xhat(t)
% ParticlesAll: set of particles for all time steps


[~,T]   = size(rMat);
Nx      = size(U,2);
J_p     = powersofJ(J,2);

ParticlesAll    = zeros(Nx,K,T+1);
ParticlesOld    = rand(Nx,K); % initialize particles
ParticlesAll(:,:,1) = ParticlesOld;

WVec    = ones(K,1)/K;
ESSVec  = zeros(T,1);

Pinv    = inv(P);
Q_postinv  = Pinv + U'*(M\U);
Q_post = inv(Q_postinv);

for tt = 1:T
 
    ht              = hMat(:,tt);
    rt              = rMat(:,tt);
    ParticlesNew    = zeros(Nx,K);
    Minvr           = (M\rt);
    rMinvr          = rt'*Minvr;
    UMinvr          = U'*Minvr;
    
    for k = 1:K
        % sampling x(t) from the proposal distribution p(x(t)|x(t-1), r(t))
        % p(x(t)|x(t-1),r(t)) = 1/Z*p(x(t)|x(t-1))*p(r(t)|x(t))
        
        out = TAPF(ParticlesOld(:,k),ht,J_p,G);
        
        f_tap    = (1-lam)*ParticlesOld(:,k) + lam*out;
        Pinvf_tap = P\f_tap; 
        
        v = Pinvf_tap + UMinvr;
        mu_post = Q_postinv\v; % mean of the proposal distribution
        
        % draw a sample from this proposal distribution
        ParticlesNew(:,k)  = mvnrnd(mu_post',Q_post,1)';

        % assigning weights to the particles proportional to p(r(t)|x(t-1))
        w_ii = exp(-0.5*(rMinvr + f_tap'*Pinvf_tap - v'*mu_post));
        WVec(k)    = WVec(k)*w_ii; 
        
    end
    
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

xhat = mean(ParticlesAll,2);
xhat = reshape(xhat,Nx,T+1);