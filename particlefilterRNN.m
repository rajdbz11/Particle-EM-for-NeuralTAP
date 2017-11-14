function [xhat, ParticlesAll_AS, ParticlesAll_BS] = particlefilterRNN(rMat, hMat, K, lam, Qpr, Qobs, V,J,G)

% Particle filter function specific to the TAP dynamics
% Implementing the standard SIR filter here
% Inputs: 
% rMat  : observations r(t)
% hMat  : inputs h(t)
% K     : No. of particles
% lam   : low pass filtering constant for the TAP dynamics
% Qpr   :covariance of process noise
% Qobs  :covariance of observation noise
% U     :embedding matrix, r = Ux + noise
% J     :coupling matrix
% G     :global hyperparameters

% Ouputs:
% xhat  : decoded latent variables xhat(t)
% ParticlesAll_AS: set of particles (after resampling) for each time step

[~,T]   = size(rMat);
Nx      = size(V,1);

ParticlesAll_BS = zeros(Nx,K,T); % particles before sampling
ParticlesAll_AS = zeros(Nx,K,T); % particles after sampling

ParticlesOld    = rand(Nx,K); % initialize particles
% Drawing from a uniform distribution here, because we know that x E (0,1)


J_p = powersofJ(J,2);

for tt = 1:T
    
    ht              = hMat(:,tt);
    ParticlesNew    = zeros(Nx,K);
    WVec            = zeros(K,1);
    rt              = rMat(:,tt);
    
    for ii = 1:K
        % sampling x(t) from the proposal distribution p(x(t)|x(t-1))
        pNew                = (1-lam)*ParticlesOld(:,ii) + lam*TAPF(ParticlesOld(:,ii),ht,J_p,G);
        % ParticlesNew(:,ii)  = mvnrnd(pNew',Qpr,1)';
        % Changing the proposal distribution to the posterior
        
        %Q_post  = inv(inv(Qpr) + U'*(Qobs\U));
        %mu_post = Q_post*(Qpr\pNew + U'*(Qobs\rt));
        Q_post = inv(inv(Qpr) + inv(Qobs));
        mu_post = Q_post*(Qpr\pNew + Qobs\(V*rt));
        
        ParticlesNew(:,ii)  = mvnrnd(mu_post',Q_post,1)'; 
        
        % assigning weights to the particles = p(r(t)|x(t))
        mu                  = ParticlesNew(:,ii);
        %WVec(ii)            = mvnpdf(rMat(:,tt)',mu',Qobs) + 1e-64;
        WVec(ii) = exp(-0.5*(mu - V*rt)'*(Qobs\(mu - V*rt))) + 1e-64;
    end
    
    WVec = WVec/sum(WVec); % Normalizing the weights
    
    ParticlesAll_BS(:,:,tt) = ParticlesNew;
    
    % Now to resample the particles by drawing samples from a multinomial
    % distribution with parameters WVec
    % NVec(ii) = no. of children on particle ii 
    NVec        = mnrnd(K,WVec)'; 
    
    ParticlesRS = [];
    
    for ii = 1:K
        if NVec(ii) ~= 0
            repP        = repmat(ParticlesNew(:,ii),1,NVec(ii));
            ParticlesRS = [ParticlesRS, repP];
        end
    end
    

    ParticlesAll_AS(:,:,tt) = ParticlesRS;
    ParticlesOld            = ParticlesRS;
        
    
end

xhat = mean(ParticlesAll_AS,2);
xhat = reshape(xhat,Nx,T);