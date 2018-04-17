function [C, dtheta] = NegLL(rMat, hMat, P_S, WVec, lam, P, M, theta)

% Function for computing the Log Likelihood cost for the probabilistic
% model for the TAP dynamics
% Inputs:
% rMat  : observations r(t)
% hMat  : inputs h(t)
% P_S   : Particles trajectories
% WVec  : Weights of the particles trajectories
% lam   : low pass filtering constant for the TAP dynamics
% P     :covariance of process noise
% M     :covariance of measurement noise
% theta : parameter vector with the following subcomponents
% G     :global hyperparameters 
% J     :coupling matrix
% U     :embedding matrix, r = Ux + noise


% Output: 
% Cost C and gradient w.r.t G 

[Nr, T] = size(rMat);  % No. of neurons and time steps
Nx      = size(P_S,1); % No. of latent variables
K       = size(P_S,2); % No. of particles

% Extract the required parameters
G       = mv(theta(1:18));
NJ      = Nx*(Nx+1)/2;
JVec    = theta(19:18+NJ);
J       = JVecToMat(JVec);
U       = reshape(theta(19+NJ:end),Nr,Nx);

J_p     = powersofJ(J,2);

% two components of the cost
C1      = 0;
C2      = 0;

% Initialize the gradients
dG = G*0;
dJ = J*0;
dU = U*0;

for t = 1:T % think about including the first time step also
    
    r_t     = rMat(:,t);
    ht      = hMat(:,t);
    
    for k = 1:K
        
        x_old  = P_S(:,k,t);
        x_curr = P_S(:,k,t+1);

        [out, argf, Im1Mat, ~, EtaMat] = TAPF(x_old, ht, J_p, G);
        xpred  = (1-lam)*x_old + lam*out; %Prediction based on the old particles
        
        
        C1 = C1 + 0.5*WVec(k)*(x_curr - xpred)'*(P\(x_curr - xpred));
        C2 = C2 + 0.5*WVec(k)*(r_t - U*x_curr)'*(M\(r_t - U*x_curr));
        
        if nargout > 1
        
            sigder      = sigmoid(argf).*(1 - sigmoid(argf));
            repsigder   = repmat(sigder,1,length(G));

            temp        = lam*(x_curr - xpred)'*(P\(repsigder.*Im1Mat));
            dG          = dG + WVec(k)*temp';

            dJtemp = J*0;

            for ii = 1:Nx
                for jj = 1:ii
                    df = zeros(Nx,1);
                    if ii == jj
                        % df(ii) = sigder(ii)*EtaMat(ii,ii);
                        df(ii) = 0;
                    else
                        df(ii) = sigder(ii)*EtaMat(ii,jj);
                        df(jj) = sigder(jj)*EtaMat(jj,ii);
                    end
                    dJtemp(ii,jj) = lam*(x_curr - xpred)'*(P\df);
                end
            end



            dJ = dJ + WVec(k)*dJtemp;
            
            dU = dU + WVec(k)*(M\((r_t - U*x_curr)*x_curr'));
        
        end

    end
end

C  = C1 + C2; 

dG = -dG;
dJ = -dJ; 
dU = -dU; 


dtheta = [dG; JMatToVec(dJ); dU(:)];