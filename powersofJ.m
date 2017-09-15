function J_p = powersofJ(J,P)
% Function that creates elementwise powers of the matrix J
% powers from 0 to P: i.e., Jij^0 to Jij^P

[Nx,Ny]  = size(J);
J_p = ones(Nx,Ny,P+1);

for ii = 1:P
    J_p(:,:,ii+1) = J.^ii;
end
