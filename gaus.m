%Gaussian function with x, mu, FWHM
function G = gaus(x,mu,FWHM)
sigma = FWHM / sqrt(8 * log(2));
G = exp(-(x-mu).^2/2/sigma^2)/(sqrt(2*pi)*sigma);
% G = G/max(G);
end