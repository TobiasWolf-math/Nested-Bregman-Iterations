% Copyright (C) 2025 Tobias Wolf <tobias.wolf@aau.at>

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

function [u,v,stopping,subdiff] = Bregman_H1_TV(f,kernel,alpha,beta,p,q,tau,sigma,maxits,bregman_its,tol)
%performs Bregman iterations with penalty term J(u,v) = alpha/2|| \nabla u ||_2^2 + beta TV(v) -
% -<p,u> - <q,v> and stops via discrepancy principle

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Input: data f and blurring kernel, regularization parameters alpha and
%beta, subdifferentials p and q, parameters tau and sigma for PDHG, maximum
%number of iterations of PDHG maxits, maximum number of iterations for
%Bregman iteration bregman_its, tolerance tol for stopping criterion
%Output: regularized solutions u and v, stopping index, subdifferential at
%final iterate

l2 = @(x) sqrt(sum(abs(x).^2,'all'));
kernel_four = psf2otf(kernel,size(f));
kernel_abs = abs(kernel_four).^2;
f_adj = conj(kernel_four).*fft2(f);

f_breg = f;
subdiff = p;
for l = 1:bregman_its

    %compute Tikhonov minimzer
    [u,v] = Variational_H1_TV_deblurring(f_breg,kernel_four,alpha,beta,p,q,tau,sigma,maxits);

    %update f
    f_breg = f_breg+f-real(ifft2(kernel_four.*fft2(u+v)));

    %compute subdifferential
    subdiff = subdiff + real(ifft2(f_adj-kernel_abs.*fft2(u+v)));
    %chck stopping criterion
    if l2(f-real(ifft2(kernel_four.*fft2(u+v)))) < tol*l2(f)
        break;
    end
end
stopping = l;
end