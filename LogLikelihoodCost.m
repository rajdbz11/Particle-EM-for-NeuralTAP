function [C, dG] = LogLikelihoodCost(PS_d,rMat,Q,R,hMat,Nh,lam,J_d,Ghat,U_d)
% Function for computing the Log Likelihood cost


[Nx,Ns,T] = size(PS_d);

% First construct the nonlinear function given particles at each time step
Fof_ParticlesAll    = zeros(Nx,Ns,T);
SigInp_ParticlesAll = zeros(Nx,Ns,T); % Input to sigmoid function for all particles at all times
 
 
% C       = 0;
C1      = 0;
C2      = 0;
dG      = zeros(size(Ghat));
Qinv    = inv(Q);
Rinv    = inv(R);
J_p     = powersofJ(J_d,2);

for tt = 2:T
    
    r_t     = rMat(:,tt);
    ht      = hMat(:,ceil((tt)/Nh));
    
    for ss = 1:Ns
        
        x_stm1  = PS_d(:,ss,tt-1);
        x_st    = PS_d(:,ss,tt);

        [out, argf, GamMat]               = nonlinearF(x_stm1,ht,J_p,Ghat);
        f_stm1                          = (1-lam)*x_stm1 + lam*out;
        Fof_ParticlesAll(:,ss,tt)       = f_stm1; % One time step lag 
        SigInp_ParticlesAll(:,ss,tt)    = argf; % One time step lag
        
        C1 = C1 + 0.5*(x_st - f_stm1)'*Qinv*(x_st - f_stm1);
        C2 = C2 + 0.5*(r_t - U_d*x_st)'*Rinv*(r_t - U_d*x_st);
        
        
        df = lam*repmat(sigmoidder(argf),1,length(Ghat)).*GamMat;
        dG = dG + (x_st - f_stm1)'*Qinv*df;
    end
end

% C = -C/Ns;
% dG = dG/Ns;


C = (C1 + C2)/Ns; % This is actually negative log likelihood .. want to minimize this
dG = -dG/Ns; % gradient of negative LL .. for the fminunc function
