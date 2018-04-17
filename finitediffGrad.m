function grad = finitediffGrad(fun,X)

% X may be a vector 
grad = zeros(length(X),1);
y0 = fun(X);

h = 1e-4;

for k = 1:length(X)
    X1 = X;
    X1(k) = X1(k) + h;
    grad(k) = (fun(X1) - y0)/h;
end