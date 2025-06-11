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

function [u,v] = Morozov_H1_TV_deblurring(f,kernel_four,alpha,c,p,q,tau,sigma,maxits)
%uses Primal-Dual Hybrid Gradient Algorithm  (PDHG, see Antonin Chambolle, Thomas Pock. A first-order primal-dual algorithm for convex problems with
%applications to imaging. 2010. ffhal-00490826f ) to compute minimizers of
%the constraint problem
% min alpha/2|| \nabla u ||_2^2 +  TV(v) -<p,u> - <q,v>
%s.t. || k*(u+v) -f||_2 <= c

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Input: observation f and Fourier transform of kernel kernel_four,
%regularization parameter alpha, upper bound c for constraint, subdifferentials p and q,
%arameters tau, sigma for PDHG and maximum number of iterations maxits
%Output: approximate minimizers u and v

[n,m] = size(f);
l2 = @(x) sqrt(sum(abs(x).^2,'all'));


%backwards difference w.r.t. x
dxm = @(u) [u(:,1:end-1) zeros(n,1)] - [zeros(n,1) u(:,1:end-1)];

%forward difference w.r.t. x
dxp = @(u) [u(:,2:end) u(:,end)] - u;

%backwards difference w.r.t. y
dym = @(u) [u(1:end-1,:);zeros(1,m)] - [zeros(1,m);u(1:end-1,:)];

%forwards difference w.r.t. y
dyp = @(u) [u(2:end,:); u(end,:)] - u;



%primal variables
u = zeros(n,m);
v = zeros(n,m);

%dual variables
y1 = zeros(n,m,2);
y2 = zeros(n,m,2);
y3 = zeros(n,m);

%relaxation variables
u_bar = zeros(n,m);
v_bar = zeros(n,m);

%functions for updates
prox_ball = @(z1,z2) z1./max(1,sqrt(z1.^2+z2.^2));

for l = 1:maxits
    %remember old iterates
    u_old = u;
    v_old = v;


    %dual update
    y1(:,:,1) = (y1(:,:,1) + sigma*dxp(u_bar))./(1+sigma/alpha);
    y1(:,:,2) = (y1(:,:,2) + sigma*dyp(u_bar))./(1+sigma/alpha);

    y2(:,:,1) = prox_ball(y2(:,:,1) +sigma*dxp(v_bar),y2(:,:,2) + sigma*dyp(v_bar));
    y2(:,:,2) = prox_ball(y2(:,:,2) + sigma*dyp(v_bar),y2(:,:,1) +sigma*dxp(v_bar));

    arg = y3+sigma*real(ifft2(fft2(u_bar+v_bar).*kernel_four));
    arg = arg-sigma*f;
    if l2(arg)-sigma*c >0
        y3 = (l2(arg)-sigma*c)*arg/l2(arg);
    else
        y3 = zeros(n,m);
    end


    %primal update
    u = u-tau*(real(ifft2(fft2(y3).*conj(kernel_four)))-dxm(y1(:,:,1)) - dym(y1(:,:,2)))+tau*p;
    v = v-tau*(real(ifft2(fft2(y3).*conj(kernel_four)))-dxm(y2(:,:,1)) - dym(y2(:,:,2)))+tau*q;

    %relaxation
    u_bar = 2*u-u_old;
    v_bar = 2*v-v_old;


end


end