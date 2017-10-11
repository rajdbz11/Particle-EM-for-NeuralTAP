function JVec = JMatToVec(JMat)

N = size(JMat,1);

JVec = []; 

for kk = 1:N
    JVec = [JVec; JMat(kk:end,kk)]; 
end