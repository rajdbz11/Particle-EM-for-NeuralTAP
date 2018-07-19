function H = finitediffHess(fun,X)

N = length(X);
H = zeros(N,N);

f0 = fun(X);

dx = 1e-2;

for ii = 1:N
    for jj = 1:ii
        if jj == ii
            Xp = X; % positive difference
            Xp(jj) = Xp(jj) + dx;
            Xn = X; % negative difference
            Xn(jj) = Xn(jj) - dx;
            H(jj,jj) = (fun(Xp) + fun(Xn) - 2*f0)/(dx^2);
        else
            Xpp = X; 
            Xpp(ii) = Xpp(ii) + dx;
            Xpp(jj) = Xpp(jj) + dx;
            Xpn = X; 
            Xpn(ii) = Xpn(ii) + dx;
            Xpn(jj) = Xpn(jj) - dx;
            Xnp = X; 
            Xnp(ii) = Xnp(ii) - dx;
            Xnp(jj) = Xnp(jj) + dx;
            Xnn = X; 
            Xnn(ii) = Xnn(ii) - dx;
            Xnn(jj) = Xnn(jj) - dx;
            
            out = (fun(Xpp) - fun(Xpn) - fun(Xnp) + fun(Xnn))/(4*dx^2);
            
            H(ii,jj) = out;
            H(jj,ii) = out;  
        end
    end
end