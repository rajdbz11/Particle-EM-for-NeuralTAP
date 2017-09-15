function J = sparsePDMatrix(Nx,sparseness)
% function to generate a sparse, symmetric, positive definite matrix


% Create the mask for zeros
sparseness = sparseness/2; % because we zero out only entries in lower half. and by symmetry, those entries would be zero in upper half too.
H   = rand(Nx,Nx);
H   = tril(H,-1);
H(H<sparseness)     = 0;
H(H>=sparseness)    = 1;
H   = H + H' + eye(Nx);

% Symmetric matrix: entries in lower (or upper) triangle are normally distributed
J   = tril(randn(Nx,Nx),-1);
J   = J + J' + diag(randn(Nx,1));

% Apply mask
J   = J.*H;

% Make J positive definite
mineigval   = min(eig(J));
eps         = 0.1;
J           = J - diag(mineigval*ones(Nx,1) - eps);