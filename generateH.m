function hMat = generateH(Nx,T,Nh,scaling)
% Function to generate h(t), the input to the TAP dynamics
% Modeling h(t) such that it stays constant for every Nh time steps.

% First generate only T/Nh independent values of h
hInd    = scaling*randn(Nx,ceil(T/Nh));
hMat = zeros(Nx,T);

% Then repeat each independent h for Nh time steps
for t = 1:T
    hMat(:,t) = hInd(:,ceil(t/Nh));
end
    