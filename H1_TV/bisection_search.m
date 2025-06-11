% Copyright (C) 202 Tobias Wolf <tobias.wolf@aau.at>

% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.

% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.

% You should have received a copy of the GNU General Public License
% along with this program. If not, see <https://www.gnu.org/licenses/>.

function [lambda,u,v] = bisection_search(ratio,lambda_0,lambda_1,f,kernel,tau,sigma,maxits,delta,tol_search)
%performs bisection search on the ratio of the regularization parameters for
%Tikhonov regularization with H^1-TV penalty functions until the residual
%is close to the noise level

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Input: ratio alpha/beta of regularization parameters, upper and lower
%bounds lambda_0, lambda_1 for alpha, data f and blurring kernel,
%parameters tau, sigma and maxits for PDHG, noise level delta and tolerance
%tol_search to determine stopping of bisection
%Output: regularization parameter lambda = alpha, corresponding pair of
%Tikhonov minimizers u and v


kernel_four = psf2otf(kernel,size(f));
l2 = @(u) sqrt(sum(abs(u).^2,'all'));
err = @(u,v) delta -l2(f-real(ifft2(kernel_four.*fft2(u+v))));

%initialize value for alpha
lambda = (lambda_0+lambda_1)/2;

error = delta;
while true
    alpha = lambda;
    beta = alpha/ratio;
    %compute minimizers of Tikhonov regularization
    [u,v] = Variational_H1_TV_deblurring(f,kernel_four,alpha,beta,0,0,tau,sigma,maxits);
    error = err(u,v);

    %check if bisection can be terminated
    if abs(error)/delta <= tol_search
        break;
    end

    %update lambda
    if error >0
        lambda_0 = lambda;
        lambda = (lambda+lambda_1)/2;

    else
        lambda_1 = lambda;
        lambda = (lambda+lambda_0)/2;
    end
end
