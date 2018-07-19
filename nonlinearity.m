function [y, dy] = nonlinearity(x,nltype)
if (nargin<2)
    nltype='sigmoid';
end
switch lower(nltype)
    case 'sigmoid'
        y   = 1./(1 + exp(-x)); % sigmoid
        dy  = y.*(1-y);
    case 'expsqrt'
        y   = sqrt(log(1 + exp(x)));
        dy  = exp(x)./(y.*(1+exp(x)));
    case 'funky'
        y   = .5*(1 + tanh(4*x)./(1 + x.^2));
        dy  = 2*(sech(4*x).^2)./(1 + x.^2) - (1+tanh(4*x)).*x/(1+x.^2).^2;
    case 'dgauss'
        y   = .5 + x.*exp(-x.^2);
        dy  = exp(-x.^2).*(1-2*x.^2);
    case 'xcauchy'
        y   = .5 + x./(1+x.^2);
        dy  = (1-x.^2)./(1+x.^2).^2;
    case 'xcauchytanh'
        y   = .5 + x./(1+x.^2) + .05*tanh(x);
        dy  = 1./(1 + x.^2) + x./((1 + x.^2).^2) + 0.05*(sech(x).^2);
    case 'xabs'
        y   = .5*(1 + x./(1+abs(x)));
        dy  = 0.5./(1 + abs(x)) - x.*sign(x)./(1 + abs(x)).^2;
    otherwise
        disp('Error: unknown nonlinearity')
        y = 0*x;
end
