function [C, dtheta] = NegLL(rMat, hMat, P_S, WVec, lam, P, M, RG, nltype, theta)

% Function for computing the Log Likelihood cost for the probabilistic
% model for the TAP dynamics
% Inputs:
% rMat  : observations r(t)
% hMat  : inputs h(t)
% P_S   : Particles trajectories
% WVec  : Weights of the particles trajectories
% lam   : low pass filtering constant for the TAP dynamics
% P     : covariance of process noise
% M     : covariance of measurement noise
% RG    : indicates whether G is of reduced size or not
% theta : parameter vector with the following subcomponents
% G     :global hyperparameters 
% J     :coupling matrix
% U     :embedding matrix, r = Ux + noise


% Output: 
% Cost C and gradient w.r.t G 

[Nr, T] = size(rMat);  % No. of neurons and time steps
Nx      = size(P_S,1); % No. of latent variables
K       = size(P_S,2); % No. of particles


if RG % this parameter tells us if we are using a restricted set of Gs or the full set 
    lG = 5;
else
    lG = 18;
end

% Extract the required parameters
G       = mv(theta(1:lG));
NJ      = Nx*(Nx+1)/2;
JVec    = theta(lG+1:lG+NJ);
J       = JVecToMat(JVec);
U       = reshape(theta(lG+1+NJ:end),Nr,Nx);

J2      = J.^2;

Argfn = @(x,ht)( ht + G(1)*J*x + G(2)*J2*x + G(3)*J2*(x.^2) + G(4)*x.*(J2*x) + G(5)*x.*(J2*(x.^2)) );

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
    x_old   = P_S(:,:,t);
    x_curr  = P_S(:,:,t+1);
    farg    = Argfn(x_old,ht);
    [fx, dfx] = nonlinearity(farg,nltype);
    x_pred  = (1-lam)*x_old + lam*fx;
    dx      = x_curr - x_pred;
    dr      = r_t - U*x_curr;
    
    % update the cost
    C1      = C1 + 0.5*sum(dx.*(P\dx))*WVec;
    C2      = C2 + 0.5*sum(dr.*(M\dr))*WVec;
    
    if nargout > 1
        
        % gradient for U
        dU      = dU - (M\dr)*(WVec.*x_curr'); 

        % gradient for G
        Im1     = lam*(P\dx).*(WVec').*dfx;
        x_old2  = x_old.^2;

        dG(1)   = dG(1) - sum(sum(Im1.*(J*x_old))); 
        dG(2)   = dG(2) - sum(sum(Im1.*(J2*x_old))); 
        dG(3)   = dG(3) - sum(sum(Im1.*(J2*x_old2))); 
        dG(4)   = dG(4) - sum(sum(Im1.*(x_old.*(J2*x_old)))); 
        dG(5)   = dG(5) - sum(sum(Im1.*(x_old.*(J2*x_old2)))); 

        % gradient for J 
        for ii = 1:Nx
            for jj = 1:ii
                dA = zeros(Nx,K);
                if ii == jj
                    dA(ii,:) = G(1)*x_old(ii,:) + ...
                         2*J(ii,ii)*( G(2)*x_old(ii,:) + (G(3) + G(4))*x_old2(ii,:) +  G(5)*(x_old(ii,:).^3));
                    % dA(ii,:) = 0;
                else
                    dA(ii,:) = G(1)*x_old(jj,:) + ...
                        2*J(ii,jj)*( G(2)*x_old(jj,:) + G(3)*x_old2(jj,:) + ...
                        G(4)*(x_old(ii,:).*x_old(jj,:)) +  G(5)*(x_old(ii,:).*x_old2(jj,:)) );

                    dA(jj,:) = G(1)*x_old(ii,:) + ...
                        2*J(ii,jj)*( G(2)*x_old(ii,:) + G(3)*x_old2(ii,:) + ...
                        G(4)*(x_old(jj,:).*x_old(ii,:)) +  G(5)*(x_old(jj,:).*x_old2(ii,:)) );
                end
                dJ(ii,jj) = dJ(ii,jj) - sum(sum(Im1.*dA));
            end
        end
    
    end
    

end


C = C1 + C2;

% Add L1 norm of J and L2 norm of G
a1 = 0;
a2 = 0;
C = C + a1*sum(G.^2) + a2*sum(abs(JMatToVec(J)));

% Add gradient of L1 norm of J
dJ = dJ + a2*sign(J);
dJ = JMatToVec(dJ);

% Add gradient of L2 norm of G
dG = dG + a1*2*G;

dtheta  = [dG; dJ; dU(:)];