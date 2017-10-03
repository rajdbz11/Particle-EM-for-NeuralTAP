function [out, argf, Im1Mat, Im2Mat] = TAPF(xt,ht,J_p,G)
% Function that corresponds to the TAP approximation
% Inputs: 
% xt    : latent variable at time t
% ht    : input at time t
% J_G   : elementwise powers of the coupling matrix; i.e., Jij,Jij^2, ..
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
 
% Here we are assuming the indices a,b,c in G_abc are in {0,1,2}
xt_p = ones(Nx,3); 

for ii = 1:2
    xt_p(:,ii+1) = xt.^ii;
end

    
Im1Mat = zeros(Nx,length(G));
Im2Mat = zeros(Nx,length(G));
        
for ii = 1:length(G)
    a   = LUT(ii,1)+1;
    b   = LUT(ii,2)+1;
    c   = LUT(ii,3)+1;
    Ja  = J_p(:,:,a);
    xb  = xt_p(:,b);
    xc  = xt_p(:,c);
    Im1Mat(:,ii)    = xb.*(Ja*xc);
    Im2Mat(:,ii)    = G(ii)*Im1Mat(:,ii);
end

argf    = sum(Im2Mat,2) + ht;
out     = sigmoid(argf);
