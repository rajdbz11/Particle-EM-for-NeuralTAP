function [C, dG] = CostFn(G_1)

load Data/dataset.mat

% Function for computing the Log Likelihood cost for the probabilistic
% model for the TAP dynamics
% Inputs:
% rMat  : observations r(t)
% hMat  : inputs h(t)
% P_AS_1  : Particles(t) after re-sampling step 
% P_BS_1  : Particles(t) before re-sampling
% lam   : low pass filtering constant for the TAP dynamics
% Qpr   :covariance of process noise
% Qobs  :covariance of observation noise
% U_1     :embedding matrix, r = Ux + noise
% J_1     :coupling matrix
% G_1     :global hyperparameters 

% Output: 
% Cost C and gradient w.r.t G_1 

[~,K,T] = size(P_BS_1);


C1      = 0;
C2      = 0;
% C1Mat   = zeros(T,K);
% C2Mat   = zeros(T,K);

J_p         = powersofJ(J_1,2);

dG = 0;

for t = 2:T
    
    r_t     = rMat(:,t);
    ht      = hMat(:,t);
    
    for k = 1:K
        
        x_old  = P_AS_1(:,k,t-1);
        x_curr = P_BS_1(:,k,t);

        % out    = TAPF(x_old,ht,J_p,G_1);
        [out, argf, Im1Mat, ~] = TAPF(x_old,ht,J_p,G_1);
        xpred  = (1-lam)*x_old + lam*out; %Prediction based on the old particles
        
        
        C1 = C1 + 0.5*(x_curr - xpred)'*(Qpr\(x_curr - xpred));
        C2 = C2 + 0.5*(r_t - U_1*x_curr)'*(Qobs\(r_t - U_1*x_curr));
        % C1Mat(t,k) = 0.5*(x_curr - xpred)'*(Qpr\(x_curr - xpred));
        % C2Mat(t,k) = 0.5*(r_t - U_1*x_curr)'*(Qobs\(r_t - U_1*x_curr));
        
        temp = repmat(argf,1,length(G_1));
        % dG = dG + lam*(x_curr - xpred)'*inv(Qpr)*(sigmoid(temp).*(1 - sigmoid(temp)).*Im1Mat);
        dG = dG + lam*(x_curr - xpred)'*(Qpr\(sigmoid(temp).*(1 - sigmoid(temp)).*Im1Mat));
    end
end

C = (C1 + C2)/K; 


dG = -dG'/K;
