function [C, dtheta] = NegLogLikelihoodCost(rMat, hMat, P_AS, P_BS, lam, Qpr, Qobs, U, theta)

% Function for computing the Log Likelihood cost for the probabilistic
% model for the TAP dynamics
% Inputs:
% rMat  : observations r(t)
% hMat  : inputs h(t)
% P_AS  : Particles(t) after re-sampling step 
% P_BS  : Particles(t) before re-sampling
% lam   : low pass filtering constant for the TAP dynamics
% Qpr   :covariance of process noise
% Qobs  :covariance of observation noise
% U     :embedding matrix, r = Ux + noise
% J     :coupling matrix
% G     :global hyperparameters 

% Output: 
% Cost C and gradient w.r.t G 

G       = theta(1:27);
JVec    = theta(28:end);
J       = JVecToMat(JVec);

[Nx,K,T] = size(P_BS);


C1      = 0;
C2      = 0;
% C1Mat   = zeros(T,K);
% C2Mat   = zeros(T,K);

J_p         = powersofJ(J,2);

dG = G*0;
dJ = J*0;

for t = 2:T
    
    r_t     = rMat(:,t);
    ht      = hMat(:,t);
    
    for k = 1:K
        
        x_old  = P_AS(:,k,t-1);
        x_curr = P_BS(:,k,t);

        % out    = TAPF(x_old,ht,J_p,G);
        [out, argf, Im1Mat, ~, EtaMat] = TAPF(x_old,ht,J_p,G);
        xpred  = (1-lam)*x_old + lam*out; %Prediction based on the old particles
        
        
        C1 = C1 + 0.5*(x_curr - xpred)'*(Qpr\(x_curr - xpred));
        C2 = C2 + 0.5*(r_t - U*x_curr)'*(Qobs\(r_t - U*x_curr));
        
        if nargout > 1
        
            sigder      = sigmoid(argf).*(1 - sigmoid(argf));
            repsigder   =  repmat(sigder,1,length(G));

            temp        = lam*(x_curr - xpred)'*(Qpr\(repsigder.*Im1Mat));
            dG          = dG + temp';

            dJtemp = J*0;

            for ii = 1:Nx
                for jj = 1:ii
                    df = zeros(Nx,1);
                    if ii == jj
                        df(ii) = sigder(ii)*EtaMat(ii,ii);
                    else
                        df(ii) = sigder(ii)*EtaMat(ii,jj);
                        df(jj) = sigder(jj)*EtaMat(jj,ii);
                    end
                    dJtemp(ii,jj) = lam*(x_curr - xpred)'*(Qpr\df);
                end
            end



            dJ = dJ + dJtemp;
        
        end

    end
end

C = (C1 + C2)/K; 


dG = -dG/K;

dJ = -dJ/K;


dtheta = [dG; JMatToVec(dJ)];