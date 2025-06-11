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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%conducts the numerical experiments for Nested Bregman iterations with TGV
%and oscillation TGV as penalty terms and saves the created data

close all
clear
clc
rng('default')

l2 = @(u) sqrt(sum(abs(u).^2,'all'));
l1 = @(u) sum(abs(u),'all');


%create ground truth
n = 75;
mat = repmat(1:n,n,1);
omega1=0.25;
omega2=0.5;
osci = cos(omega1*mat+omega2*mat') +2*sin(omega1*mat+omega2*mat');
osci = 0.4*osci;
shape = repmat(abs(linspace(-0.9,0.9,n))+0.1,n,1);

comp1 = [zeros(n),zeros(n),zeros(n); zeros(n), shape, zeros(n);  zeros(n), zeros(n), zeros(n)];
comp2 = [osci,osci,osci;osci,zeros(n),osci;osci,osci,osci];

imag = comp1+comp2;

%add noise
noise = normrnd(0,5e-2,size(imag));
f = imag+noise;
delta = l2(noise);
noiselevel = delta/l2(f);
image(256*f); colormap(gray);


[N,M] = size(f);


%derivatives

% backwards difference w.r.t. x
dxm = @(u) [u(:,1:end-1) zeros(N,1)] - [zeros(N,1) u(:,1:end-1)];

% forward difference w.r.t. x
dxp = @(u) [u(:,2:end) u(:,end)] - u;

% backwards difference w.r.t. y
dym = @(u) [u(1:end-1,:);zeros(1,M)] - [zeros(1,M);u(1:end-1,:)];

% forwards difference w.r.t. y
dyp = @(u) [u(2:end,:); u(end,:)] - u;







%matrix entries for oscillation TGV
cxx = 2-2*cos(omega1);
cyy=2-2*cos(omega2);
cxy = 1+cos(omega1-omega2)-cos(omega1)-cos(omega2);

%regularization parameters
alpha1 =5;
beta1  = alpha1;
alpha2 = 1;
beta2  = alpha2;

%epsilon for huberization
epsilon = 1e-8;


p_u=zeros(N,M);
q_u=zeros(N,M);
p_w = zeros(N,M,2);
q_w = zeros(N,M,2);

c = delta;


%parameters for PDHG
L2 = 20;
tau = 1/sqrt(L2);
sigma = 1/tau/L2;
maxits = 500000;


outer_its = 10;

u_nested = zeros(N,M,outer_its);
v_nested = zeros(N,M,outer_its);
psnr_nested_u = zeros(1,outer_its);
psnr_nested_v = zeros(1,outer_its);
psnr_nested_x = zeros(1,outer_its);

cross_corr = zeros(1,outer_its);
ind_stopping_corr = outer_its;
first_min = true;


%Nested Bregman iuterations with Morozov regularization in inner loop

for l = 1:outer_its
    %Morozov regularization
    [u1,u2,w1,w2] = TGV_Morozov(f,c,alpha1,alpha2,beta1,beta2,cxx,cyy,cxy,p_u,q_u,p_w,q_w,epsilon,tau,sigma,maxits);

    %compute u subgradient
    grad_u1_x = dxp(u1);
    grad_u1_y = dyp(u1);
    [der_huber_u1_x,der_huber_u1_y] = derivative_Huber_2D(grad_u1_x-w1(:,:,1),grad_u1_y-w1(:,:,2),epsilon);
    subdiff_u = -alpha1*(dxm(der_huber_u1_x)+dym(der_huber_u1_y));



    %compute w subgradient
    der_sym_w1_xx = dxm(w1(:,:,1));
    der_sym_w1_yy = dym(w1(:,:,2));
    der_sym_w1_xy = 0.5*(dym(w1(:,:,1))+dxm(w1(:,:,2)));


    [der_huber_w1_xx,der_huber_w1_yy,der_huber_w1_xy] = derivative_Huber_3D(der_sym_w1_xx,der_sym_w1_yy,der_sym_w1_xy,epsilon);
    subdiff_w(:,:,1) = -alpha1*der_huber_u1_x-beta1*(dxp(der_huber_w1_xx)+dyp(der_huber_w1_xy));
    subdiff_w(:,:,2) = -alpha1*der_huber_u1_y - beta1*(dxp(der_huber_w1_xy)+dyp(der_huber_w1_yy));



    %update subdifferentials
    p_u = subdiff_u;
    p_w = subdiff_w;




    u_nested(:,:,l) = u1;
    v_nested(:,:,l) = u2;

    psnr_nested_u(l) = psnr(u1,comp1);
    psnr_nested_v(l) = psnr(u2,comp2);
    psnr_nested_x(l) = psnr(u1+u2,imag);
    cross_corr(l) = l2(convn(u1,u2(end:-1:1,end:-1:1)))/(l2(u1)*l2(u2));

    if l>2 && cross_corr(l) > cross_corr(l-1) && cross_corr(l-2) > cross_corr(l-1) && first_min
        ind_stopping_corr = l-1;
        first_min = false;
    end

end
psnr_nested_sum = psnr_nested_u+psnr_nested_v;
ind_max_psnr = find(psnr_nested_sum == max(psnr_nested_sum));


%%
save('data.mat')
