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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%conducts the numerical experiments for Tikhonov regularization, Bregman iterations and Nested Bregman iterations with
%H^1-TV penalties and saves the created data

rng('default')
close all
clear
format long
clc

l2 = @(u) sqrt(sum(abs(u).^2,'all'));
l1 = @(u) sum(abs(u),'all');

%create ground truth
n = 100;
grid = linspace(-1,1,n);
u_true = abs(grid'*grid);
u_true = fftshift(u_true);
v_true = blkdiag(zeros(33,33),0.5*ones(33,33),zeros(34,34));
x_true = u_true + v_true;

%blurring
[n,m] = size(u_true);
kernel = fspecial('gaussian',[m,n],1);
kernel_four = psf2otf(kernel,size(x_true));
f = real(ifft2(kernel_four.*fft2(x_true)));

%add noise
noise  = normrnd(0,5e-2,size(f));
f = f+noise;
delta = l2(noise);




%derivatives
[n,m] = size(f);

%backwards difference w.r.t. x
dxm = @(u) [u(:,1:end-1) zeros(n,1)] - [zeros(n,1) u(:,1:end-1)];

%forward difference w.r.t. x
dxp = @(u) [u(:,2:end) u(:,end)] - u;

%backwards difference w.r.t. y
dym = @(u) [u(1:end-1,:);zeros(1,m)] - [zeros(1,m);u(1:end-1,:)];

%forwards difference w.r.t. y
dyp = @(u) [u(2:end,:); u(end,:)] - u;

tv = @(u) l1(dxp(u)+1i*dyp(u));
H1 = @(u) 0.5*l2(dxp(u))^2 + l2(dyp(u))^2;



%parameters for PDHG
sigma = 0.9999;
L = 10;
tau = 1/L^2;

maxits = 3e4;





%subgradient for outer Bregman iteration
p = 0;
q = 0;

%%
%Tikhonov regularization and bisection search for optimal parameters

%parameters for bisection search
lambda_0 = 0;
lambda_1 = 120;
tol_search = 1e-3;

%test ratios
ratios = [1e3,47,10,2];

u_ratios = zeros(n,m,length(ratios));
v_ratios = u_ratios;
alphas = zeros(1,length(ratios));
betas = alphas;
psnr_ratios_x = alphas;
psnr_ratios_u = alphas;
psnr_ratios_v = alphas;

parfor l = 1:length(ratios)
    ratio = ratios(l);
    [lambda,u,v]  = bisection_search(ratio,lambda_0,lambda_1,f,kernel,tau,sigma,maxits,delta,tol_search);
    psnr_ratios_x(l) = psnr(u+v,x_true);
    psnr_ratios_u(l) = psnr(u,u_true);
    psnr_ratios_v(l) = psnr(v,v_true);
    alphas(l) = lambda;
    betas(l) = lambda/ratio;
    u_ratios(:,:,l) = u;
    v_ratios(:,:,l) = v;
    err_rel = (delta-l2(f-real(ifft2(kernel_four.*fft2(u+v)))))/delta;
end

%%
%Bregman iterations
bregman_its = 5000;
tol = (1+1e-3)*delta/l2(f);

psnr_bregman_ratios_x = zeros(1,length(ratios));
psnr_bregman_ratios_u = zeros(1,length(ratios));
psnr_bregman_ratios_v = zeros(1,length(ratios));

u_Bregman_ratios = zeros(m,n,length(ratios));
v_Bregman_ratios = zeros(m,n,length(ratios));

stopping_ratios = zeros(1,length(ratios));

parfor l = 1:length(ratios)
    alpha = 4*alphas(l);
    beta = 4*betas(l);
    ratio = ratios(l);
    [u,v,stopping,subdiff] = Bregman_H1_TV(f,kernel,alpha,beta,p,q,tau,sigma,maxits,bregman_its,tol);
    psnr_bregman_ratios_x(l) = psnr(u+v,x_true);
    psnr_bregman_ratios_u(l) = psnr(u,u_true);
    psnr_bregman_ratios_v(l) = psnr(v,v_true);
    u_Bregman_ratios(:,:,l) = u;
    v_Bregman_ratios(:,:,l) = v;
    stopping_ratios(l) = stopping;
end

%%
%nested Bregman iterations with Morozov regularization in inner loop


%initialize subgradient for outer Bregman iteration
p = 0;
q = 0;
outer_its = 50;

u_nested_M = zeros(n,m,outer_its);
v_nested_M = u_nested_M;
psnr_nested_x_M = zeros(1,outer_its);
psnr_nested_u_M = zeros(1,outer_its);
psnr_nested_v_M = zeros(1,outer_its);

alpha =1/0.001;

cross_corr_M = zeros(1,outer_its);
ind_stopping_M = outer_its;
first_min_M = true;


for l = 1:outer_its
    %Morozov regularization
    [u_M,v_M] = Morozov_H1_TV_deblurring(f,kernel_four,alpha,delta,p,q,tau,sigma,maxits);

    %compute subgradient
    subdiff = -(alpha)*(dxm(dxp(u_M))+dym(dyp(u_M)));
    p = subdiff;

    psnr_nested_x_M(l) = psnr(u_M+v_M,x_true);
    psnr_nested_u_M(l) = psnr(u_M,u_true);
    psnr_nested_v_M(l) = psnr(v_M,v_true);
    u_nested_M(:,:,l) = u_M;
    v_nested_M(:,:,l)= v_M;

    %compute cross-correlation and check stopping criterion
    cross_corr_M(l) = l2(convn(u_nested_M(:,:,l),v_nested_M(end:-1:1,end:-1:1,l)))/(l2(u_nested_M(:,:,l))*l2(v_nested_M(:,:,l)));
    if l>2 && cross_corr_M(l-2) > cross_corr_M(l-1) && cross_corr_M(l) > cross_corr_M(l-1) && first_min_M
        ind_stopping_M = l-1;
        first_min_M = false;
    end
end
psnr_sum_M = psnr_nested_u_M+psnr_nested_v_M;
ind_psnr_M = find(psnr_sum_M == max(psnr_sum_M));

%%
%nested Bregman iterations with Bregman iterations in inner loop
close all

bregman_its = 5000;
tol = (1+1e-3)*delta/l2(f);

p = 0;
q = 0;
outer_its = 50;

u_nested_B = zeros(n,m,outer_its);
v_nested_B = u_nested_B;

stopping_nested = zeros(1,outer_its);

psnr_nested_x_B = zeros(1,outer_its);
psnr_nested_u_B = zeros(1,outer_its);
psnr_nested_v_B = zeros(1,outer_its);

cross_corr_B = zeros(1,outer_its);
ind_stopping_B = outer_its;
first_min_B = true;

alpha = 4*alphas(1);
beta = 4*betas(1);

for l = 1:outer_its
    %Bregman iteration
    [u_B,v_B,stopping,subdiff] = Bregman_H1_TV(f,kernel,alpha,beta,p,q,tau,sigma,maxits,bregman_its,tol);

    %compute subgradient
    p = subdiff;
    psnr_nested_x_B(l) = psnr(u_B+v_B,x_true);
    u_nested_B(:,:,l)  = u_B;
    v_nested_B(:,:,l) = v_B;
    stopping_nested(l) = stopping;

    psnr_nested_x_B(l) = psnr(u_B+v_B,x_true);
    psnr_nested_u_B(l) = psnr(u_B,u_true);
    psnr_nested_v_B(l) = psnr(v_B,v_true);

    %compute cross-correlation and check stopping criterion
    cross_corr_B(l) = l2(convn(u_nested_B(:,:,l),v_nested_B(end:-1:1,end:-1:1,l)))/(l2(u_nested_B(:,:,l))*l2(v_nested_B(:,:,l)));
    if l>2 && cross_corr_B(l-2) > cross_corr_B(l-1) && cross_corr_B(l) > cross_corr_B(l-1) && first_min_B
        ind_stopping_B = l-1;
        first_min_B = false;
    end
end

psnr_sum_B = psnr_nested_u_B+psnr_nested_v_B;
ind_psnr_B = find(psnr_sum_B == max(psnr_sum_B));
%%
%save data
save('data_plotting.mat')