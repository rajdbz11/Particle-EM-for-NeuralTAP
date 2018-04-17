function J = Create_J(Nx, sp, Jtype, SelfCoupling)
% Generate a sparse, symmetric coupling matrix with desired
% kind of itneractions

% Inputs: 
% Nx    : No. of x's
% sp    : degree of sparsity of J
% Jtype : type of J matrix. ferr, antiferr, nonferr
% SelfCoupling: determines if J matrix has self coupling or not

% Output
% J     : coupling matrix of the Ising model

% Create the mask for zeros
H   = rand(Nx,Nx);
H   = tril(H,-1);
H(H < sp)     = 0;
H(H >= sp)    = 1;
if SelfCoupling
    H   = H + H' + eye(Nx);
else
    H   = H + H';
end

% Create full coupling matrix with required kind of interaction
if strcmp(Jtype,'ferr')
    J = tril(rand(Nx,Nx),-1);
    J = J + J' + diag(rand(Nx,1));
    J = J/Nx;
elseif strcmp(Jtype,'antiferr')
    J = tril(rand(Nx,Nx),-1);
    J = J + J' + diag(rand(Nx,1));
    J = -J;
    J = J/sqrt(Nx);
else %nonferr
    J = tril(0.5*randn(Nx,Nx),-1);
    J = J + J' + diag(0.5*randn(Nx,1));
    J = J/sqrt(Nx);
end

% Apply mask
if sp ~= 0
    J   = J.*H;
end