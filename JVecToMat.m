function JMat = JVecToMat(JVec)

L   = length(JVec);
Nv  = roots([0.5 0.5 -L]);
Nv  = Nv(find(Nv > 0));
Nv = round(Nv);

JMat = zeros(Nv,Nv);

for k = 1:Nv
    JMat(k:end,k) = JVec(1:Nv-k+1);
    JMat(k,k:end) = JVec(1:Nv-k+1)';
    JVec(1:Nv-k+1) = [];
end