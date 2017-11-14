function margpVec = ComputeTrueProbsIsing(J,h)

Nx = length(h);

stable = 1-2*truthtable(Nx);

pVec = zeros(2^Nx,1);

for ii = 1:2^Nx
    sVec = stable(ii,:)';
    pVec(ii) = exp(sVec'*J*sVec + h'*sVec);
end

pVec = pVec/sum(pVec);

margpVec = zeros(Nx,1);

for ii = 1:Nx
    idx = find(stable(:,ii) == 1);
    margpVec(ii) = sum(pVec(idx));
   
end

