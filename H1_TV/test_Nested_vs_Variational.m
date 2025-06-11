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

%conducts the numerical experiments comparing Morozov regularization and Nested Bregman iterations with
%H^1-TV penalties and saves data and the corresponding plots

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

delta = 0;

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

%%

maxits = 3e4;

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

ind_psnr_x_M = find(psnr_nested_x_M == max(psnr_nested_x_M));
ind_psnr_u_M = find(psnr_nested_u_M == max(psnr_nested_u_M));
ind_psnr_v_M = find(psnr_nested_v_M == max(psnr_nested_v_M));
psnr_sum_M = psnr_nested_u_M+psnr_nested_v_M;
ind_psnr_sum_M = find(psnr_sum_M == max(psnr_sum_M));

%% Variational
p = 0;
q = 0;
alphas = logspace(-2,4,1000);
num_alphas = length(alphas);
u_nested_V = zeros(n,m,num_alphas);
v_nested_V = u_nested_V;
psnr_nested_x_V = zeros(1,num_alphas);
psnr_nested_u_V = zeros(1,num_alphas);
psnr_nested_v_V = zeros(1,num_alphas);





cross_corr_V = zeros(1,num_alphas);
ind_stopping_V = num_alphas;
first_min_V = true;

for l = 1:num_alphas

    alpha = alphas(l);
    [u_V,v_V] = Morozov_H1_TV_deblurring(f,kernel_four,alpha,delta,p,q,tau,sigma,maxits);


    psnr_nested_x_V(l) = psnr(u_V+v_V,x_true);
    psnr_nested_u_V(l) = psnr(u_V,u_true);
    psnr_nested_v_V(l) = psnr(v_V,v_true);
    u_nested_V(:,:,l) = u_V;
    v_nested_V(:,:,l)= v_V;




    %compute cross-correlation and check stopping criterion
    cross_corr_V(l) = l2(convn(u_nested_V(:,:,l),v_nested_V(end:-1:1,end:-1:1,l)))/(l2(u_nested_V(:,:,l))*l2(v_nested_V(:,:,l)));
    if l>2 && cross_corr_V(l-2) > cross_corr_V(l-1) && cross_corr_V(l) > cross_corr_V(l-1) && first_min_V
        ind_stopping_V = l-1;
        first_min_V = false;
    end
    
end


psnr_sum_V = psnr_nested_u_V+psnr_nested_v_V;
%ind_psnr_V = find(psnr_sum_V == max(psnr_sum_V));

save('Nested_vs_Variational1_extensive.mat')
%%
%create plots


dir = strcat(pwd,'/Nested_vs_Variational_extensive');
mkdir(dir);

figure('visible','off');
semilogx(alphas, psnr_nested_u_V, color =  'black'); hold on; semilogx(alphas, psnr_nested_u_M(ind_stopping_M)*ones(1,num_alphas), '--', color = 'black'); semilogx(alphas, psnr_nested_u_M(ind_psnr_u_M)*ones(1,num_alphas), ':', color = 'black');
xlabel('$\alpha$','Interpreter','latex'); ylabel('PSNR (dB)'); legend('Morozov regularization', 'Nested Bregman at stopping index', 'maximal PSNR Nested Bregman','Location','southwest')
hold on
ylim( [min(psnr_nested_u_V) - 0.1*range(psnr_nested_u_V),  max(psnr_nested_u_V)+0.1*range(psnr_nested_u_V)]);
title('PSNR  of u-component')
exportgraphics(gcf,strcat(dir,'/PSNR_u.pdf'),'ContentType','vector');close;

figure('visible','off');
semilogx(alphas, psnr_nested_v_V, color =  'black'); hold on; semilogx(alphas, psnr_nested_v_M(ind_stopping_M)*ones(1,num_alphas), '--', color = 'black'); semilogx(alphas, psnr_nested_v_M(ind_psnr_v_M)*ones(1,num_alphas), ':', color = 'black');
xlabel('$\alpha$','Interpreter','latex'); ylabel('PSNR (dB)'); legend('Morozov regularization', 'Nested Bregman at stopping index', 'maximal PSNR Nested Bregman','Location','southwest')
hold on
ylim( [min(psnr_nested_v_V) - 0.1*range(psnr_nested_v_V),  max(psnr_nested_v_V)+0.1*range(psnr_nested_v_V)]);
title('v')
title('PSNR  of v-component')
exportgraphics(gcf,strcat(dir,'/PSNR_v.pdf'),'ContentType','vector');close;


figure('visible','off');
semilogx(alphas, psnr_nested_x_V, color =  'black'); hold on; semilogx(alphas, psnr_nested_x_M(ind_stopping_M)*ones(1,num_alphas), '--', color = 'black'); semilogx(alphas, psnr_nested_x_M(ind_psnr_x_M)*ones(1,num_alphas), ':', color = 'black');
xlabel('$\alpha$','Interpreter','latex'); ylabel('PSNR (dB)'); legend('Morozov regularization', 'Nested Bregman at stopping index', 'maximal PSNR Nested Bregman','Location','southwest')
hold on
ylim( [min(psnr_nested_x_V) - 0.1*range(psnr_nested_x_V),  max(psnr_nested_x_V)+0.1*range(psnr_nested_x_V)]);
title('PSNR of the entire reconstruction')
exportgraphics(gcf,strcat(dir,'/PSNR_x.pdf'),'ContentType','vector');close;

figure('visible','off');
semilogx(alphas, psnr_sum_V, color =  'black'); hold on; semilogx(alphas, psnr_sum_M(ind_stopping_M)*ones(1,num_alphas), '--', color = 'black'); semilogx(alphas, psnr_sum_M(ind_psnr_sum_M)*ones(1,num_alphas), ':', color = 'black');
xlabel('$\alpha$','Interpreter','latex'); ylabel('PSNR (dB)'); legend('Morozov regularization', 'Nested Bregman at stopping index', 'maximal PSNR Nested Bregman','Location','southwest')
hold on
ylim( [min(psnr_sum_V) - 0.1*range(psnr_sum_V),  max(psnr_sum_V)+0.1*range(psnr_sum_V)]);
title('Sum of individual PSNR values for u- and v-component')
exportgraphics(gcf,strcat(dir,'/PSNR_sum.pdf'),'ContentType','vector');close;

figure('visible','off'); 
yyaxis left;
plot(psnr_nested_u_V,'Color','black')
xlabel('$\alpha$','Interpreter','latex')
set(gca,'ycolor','black')
ylabel('PSNR (dB)');
yyaxis right
plot(cross_corr_V,'--','Color','black')
set(gca,'ycolor','black')
ylabel('$\mathcal{C}(u,v)$','interpreter','latex')
legend('PSNR','Correlation')
title('u')
exportgraphics(gcf,strcat(dir,'/corr_u.pdf'),'ContentType','vector');close;

figure('visible','off'); 
yyaxis left;
plot(psnr_nested_v_V,'Color','black')
xlabel('$\alpha$','Interpreter','latex')
set(gca,'ycolor','black')
ylabel('PSNR (dB)');
yyaxis right
plot(cross_corr_V,'--','Color','black')
set(gca,'ycolor','black')
ylabel('$\mathcal{C}(u,v)$','interpreter','latex')
legend('PSNR','Correlation')
title('v')
exportgraphics(gcf,strcat(dir,'/corr_v.pdf'),'ContentType','vector');close;

figure('visible','off'); 
yyaxis left;
plot(psnr_nested_x_V,'Color','black')
xlabel('$\alpha$','Interpreter','latex')
set(gca,'ycolor','black')
ylabel('PSNR (dB)');
yyaxis right
plot(cross_corr_V,'--','Color','black')
set(gca,'ycolor','black')
ylabel('$\mathcal{C}(u,v)$','interpreter','latex')
legend('PSNR','Correlation')
title('x')
exportgraphics(gcf,strcat(dir,'/corr_x.pdf'),'ContentType','vector');close;

figure('visible','off'); 
yyaxis left;
plot(psnr_sum_V,'Color','black')
xlabel('$\alpha$','Interpreter','latex')
set(gca,'ycolor','black')
ylabel('PSNR (dB)');
yyaxis right
plot(cross_corr_V,'--','Color','black')
set(gca,'ycolor','black')
ylabel('$\mathcal{C}(u,v)$','interpreter','latex')
legend('PSNR','Correlation')
title('sum')
exportgraphics(gcf,strcat(dir,'/corr_sum.pdf'),'ContentType','vector');close;



