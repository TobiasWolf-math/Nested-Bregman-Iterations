% Copyright (C) 2024 Tobias Wolf <tobias.wolf@aau.at>

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

function [u1,u2,w1,w2] = TGV_Morozov(f,c,alpha1,alpha2,beta1,beta2,cxx,cyy,cxy,p_u,q_u,p_w,q_w,epsilon,tau,sigma,maxits)
%uses Primal-Dual Hybrid Gradient Algorithm  (PDHG, see Antonin Chambolle, Thomas Pock. A first-order primal-dual algorithm for convex problems with
%applications to imaging. 2010. ffhal-00490826f ) to compute minimizers of
%the constraint problem
% min alpha1 || nabla u1 - w1 ||_1 + beta 1 || E w1 ||_1 + alpha2 || nabla u2 - w2 ||_1 + beta 2 || E w2 + Cu ||_1 -<p,u> - <q,v>
%s.t. || (u+v) -f||_2 <= c
% In this algorithm the 1-norms are approximated by the Huber function and
% E denotes the symmetrized derivative. For more details see  Y. Gao and K. Bredies. Infimal convolution of oscillation total generalized variation for the recovery of images with structured texture. SIAM Journal on Imaging Sciences, 11(3):2021-2063, 2018. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Input: observation f, upper bound c for constraint, parameters alpha1,
%alpha2, beta1, beta2, cxx, cyy, cxy for TGV and oscillation TGV,
%subdifferentialsp_u, q_u, p_w, q_w, parameter epsilon for huber functional, parametes tau, sigma for PDHG and maximal number of iterations maxits
%Output: approximate minimizers u1, u2, w1, w2

[N,M] = size(f);

l2 = @(u) sqrt(sum(abs(u).^2,'all'));



% backwards difference w.r.t. x
dxm = @(u) [u(:,1:end-1) zeros(N,1)] - [zeros(N,1) u(:,1:end-1)];

% forward difference w.r.t. x
dxp = @(u) [u(:,2:end) u(:,end)] - u;

% backwards difference w.r.t. y
dym = @(u) [u(1:end-1,:);zeros(1,M)] - [zeros(1,M);u(1:end-1,:)];

% forwards difference w.r.t. y
dyp = @(u) [u(2:end,:); u(end,:)] - u;

%initialize primal variables
u1 = zeros(N,M);
u2 = zeros(N,M);
w1 = zeros(N,M,2);
w2 = zeros(N,M,2);

%initialize dual variables
x1 = zeros(N,M,2);
x2 = zeros(N,M,2);
y1 = zeros(N,M,3);
y2 = zeros(N,M,3);
z = zeros(N,M);

%initialize relaxation variables
u1_bar = zeros(N,M);
u2_bar = zeros(N,M);
w1_bar = zeros(N,M,2);
w2_bar = zeros(N,M,2);



%Chambolle-Pock algorithm
for l = 1:maxits

    u1_old =u1;
    u2_old = u2;
    w1_old = w1;
    w2_old =w2;

    %update dual variables
    x1(:,:,1) = x1(:,:,1) + sigma*(dxp(u1_bar(:,:))-w1_bar(:,:,1));
    x1(:,:,2) = x1(:,:,2) + sigma*(dyp(u1_bar(:,:))-w1_bar(:,:,2));

    factor = max(1+sigma*epsilon,sqrt(x1(:,:,1).^2+x1(:,:,2).^2)/alpha1);

    x1(:,:,1) = x1(:,:,1)./factor;
    x1(:,:,2) = x1(:,:,2)./factor;

    x2(:,:,1) = x2(:,:,1) + sigma*(dxp(u2_bar(:,:))-w2_bar(:,:,1));
    x2(:,:,2) = x2(:,:,2) + sigma*(dyp(u2_bar(:,:))-w2_bar(:,:,2));

    factor = max(1+sigma*epsilon,sqrt(x2(:,:,1).^2+x2(:,:,2).^2)/alpha2);

    x2(:,:,1) = x2(:,:,1)./factor;
    x2(:,:,2) = x2(:,:,2)./factor;

    y1(:,:,1) = y1(:,:,1) + sigma*(dxm(w1_bar(:,:,1)));
    y1(:,:,2) = y1(:,:,2) + sigma*(dym(w1_bar(:,:,2)));
    y1(:,:,3) = y1(:,:,3) + sigma*((dym(w1_bar(:,:,1))+dxm(w1_bar(:,:,2)))/2);

    factor = max(1+sigma*epsilon,sqrt(y1(:,:,1).^2+y1(:,:,2).^2+2*y1(:,:,3).^2)/beta1);


    y1(:,:,1) = y1(:,:,1)./factor;
    y1(:,:,2) = y1(:,:,2)./factor;
    y1(:,:,3) = y1(:,:,3)./factor;

    y2(:,:,1) = y2(:,:,1) + sigma*(dxm(w2_bar(:,:,1)) +cxx*u2_bar);
    y2(:,:,2) = y2(:,:,2) + sigma*(dym(w2_bar(:,:,2))+cyy*u2_bar);
    y2(:,:,3) = y2(:,:,3) + sigma*((dym(w2_bar(:,:,1))+dxm(w2_bar(:,:,2)))/2+cxy*u2_bar);

    factor = max(1+sigma*epsilon,sqrt(y2(:,:,1).^2+y2(:,:,2).^2+2*y2(:,:,3).^2)/beta2);

    y2(:,:,1) = y2(:,:,1)./factor;
    y2(:,:,2) = y2(:,:,2)./factor;
    y2(:,:,3) = y2(:,:,3)./factor;



    arg = z+sigma*(u1_bar+u2_bar);
    arg = arg-sigma*f;
    if l2(arg)-sigma*c >0
        z = (l2(arg)-sigma*c)*arg/l2(arg);
    else
        z = zeros(N,M);
    end



    %update primal variables
    u1 = u1-tau*(z-(dxm(x1(:,:,1))+dym(x1(:,:,2))))+tau*p_u;

    u2 = u2-tau*(z-(dxm(x2(:,:,1))+dym(x2(:,:,2)))+cxx*y2(:,:,1)+cyy*y2(:,:,2)+2*cxy*y2(:,:,3))+tau*q_u;

    w1(:,:,1) = w1(:,:,1) +tau*(x1(:,:,1) + dxp(y1(:,:,1))+dyp(y1(:,:,3)))+tau*p_w(:,:,1);
    w1(:,:,2) = w1(:,:,2) +tau*(x1(:,:,2) + dxp(y1(:,:,3))+dyp(y1(:,:,2)))+tau*p_w(:,:,2);

    w2(:,:,1) = w2(:,:,1) +tau*(x2(:,:,1) + dxp(y2(:,:,1))+dyp(y2(:,:,3)))+tau*q_w(:,:,1);
    w2(:,:,2) = w2(:,:,2) +tau*(x2(:,:,2) + dxp(y2(:,:,3))+dyp(y2(:,:,2)))+tau*q_w(:,:,2);


    %relaxation
    u1_bar = 2*u1-u1_old;

    u2_bar = 2*u2-u2_old;

    w1_bar = 2*w1-w1_old;

    w2_bar = 2*w2-w2_old;



end
end