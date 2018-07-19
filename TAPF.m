function [out, argf, Im1Mat, Im2Mat, EtaMat] = TAPF(xt,ht,J_p,G)

% Function that implements the TAP (Thouless Anderson Palmer) approximation
% The coefficients G that correspond to the TAP approximation are: 
% G   = [0,2,0,0,0,0,0,0,0,0,4,-4,0,-8,8,0,0,0]';
% G   = [2,4,-4,-8,8]; % reduced parameter case
% This function however can take as input arbitrary coefficients G

% Inputs: 
% xt    : latent variable at time t
% ht    : input at time t
% J_G   : elementwise powers of J; i.e., 1, Jij,Jij^2, ..
% G     : global hyperparameters 

% Outputs:
% out: f(x(t),h(t)|J,G) = sigmoid(argf(x(t),h(t)|J,G))
% Im1Mat: intermediate computation 1 that corresponds to the sum_j(Jij^a *xi^b *xj^c)
% Im2Mat: intermediate computation 2 that corresponds to the G_abc*sum_j(Jij^a *xi^b *xj^c)


Nx = size(xt,1);

LUT = ...
    [0     0     0
     0     0     1
     0     0     2
     0     1     0
     0     1     1
     0     1     2
     0     2     0
     0     2     1
     0     2     2
     1     0     0
     1     0     1
     1     0     2
     1     1     0
     1     1     1
     1     1     2
     1     2     0
     1     2     1
     1     2     2
     2     0     0
     2     0     1
     2     0     2
     2     1     0
     2     1     1
     2     1     2
     2     2     0
     2     2     1
     2     2     2];
 
% Here we are assuming the indices of G_abc take values: 
% a in {1,2} and b,c in {0,1,2}

xt_p = ones(Nx,3); 

for ii = 1:2
    xt_p(:,ii+1) = xt.^ii;
end

    
Im1Mat = zeros(Nx,length(G));
Im2Mat = zeros(Nx,length(G));
EtaMat = zeros(Nx,Nx,length(G)); %for derivative wrt J

if length(G) == 5
    ReducedG = 1;
else
    ReducedG = 0;
end

if (ReducedG)
    idx = [2,11,12,14,15];
    for kk = 1:length(idx)
        ii = idx(kk);
        a   = LUT(ii+9,1)+1;
        b   = LUT(ii+9,2)+1;
        c   = LUT(ii+9,3)+1;
        Ja  = J_p(:,:,a);
        xb  = xt_p(:,b);
        xc  = xt_p(:,c);
        Im1Mat(:,kk)   = xb.*(Ja*xc);
        Im2Mat(:,kk)   = G(kk)*Im1Mat(:,kk);
        EtaMat(:,:,kk) = G(kk)*(a-1)*J_p(:,:,a-1).*(xb*xc');  
    end    
else
    for ii = 1:length(G)
        a   = LUT(ii+9,1)+1;
        b   = LUT(ii+9,2)+1;
        c   = LUT(ii+9,3)+1;
        Ja  = J_p(:,:,a);
        xb  = xt_p(:,b);
        xc  = xt_p(:,c);
        Im1Mat(:,ii)   = xb.*(Ja*xc);
        Im2Mat(:,ii)   = G(ii)*Im1Mat(:,ii);
        EtaMat(:,:,ii) = G(ii)*(a-1)*J_p(:,:,a-1).*(xb*xc');  
    end
end

argf    = sum(Im2Mat,2) + ht;
out     = sigmoid(argf);
EtaMat  = sum(EtaMat,3);