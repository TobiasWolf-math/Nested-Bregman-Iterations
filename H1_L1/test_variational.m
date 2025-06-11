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

% conducts the numerical experiments for comparing Nested Bregman iterations and Morozov regularization with
% L^1-H^2 penalty term and creates the corresponding plots

clear;
close all
clc;
format long
rng(23);


l2 = @(u) sqrt(sum(abs(u).^2,'all'));
l1 = @(u) sum(abs(u),'all');

n = 300;
indices = 1:n;
comp_1 = zeros(n,1);

%create true signal
comp_1(mod(indices,50)== 0) = 0.5;
comp_1(mod(indices,80)== 45) = -0.4;
comp_2 = sin(linspace(-2*pi,0*pi,n)');
x_true = comp_1 + comp_2;

%create observation
A = eye(n);
f = A*x_true;
noise = normrnd(0,0.05,[n,1]);
f = f+noise;
delta = l2(noise);






%parameters
alpha = 1000;
c = delta;

%derivatives
Df = eye(n) - diag(ones(n-1,1),1);
Df(end,:) = 0;


%Db = -eye(n) + diag(ones(n-1,1),-1);
Db = Df';

%%
%nested Bregman iterations with Morozov regularization in inner loop
outer_loops_morozov = 100;
p=zeros(size(f));
q=zeros(size(f));
u_morozov = zeros(n,outer_loops_morozov);
v_morozov = zeros(n,outer_loops_morozov);

val_g_morozov = zeros(1,outer_loops_morozov);
val_h_morozov = zeros(1,outer_loops_morozov);

psnr_u_morozov = zeros(1,outer_loops_morozov);
psnr_v_morozov = zeros(1,outer_loops_morozov);
psnr_x_morozov = zeros(1,outer_loops_morozov);

cross_corr_morozov = zeros(1,outer_loops_morozov);
ind_min_corr_morozov = outer_loops_morozov;
first_min = true;

for l = 1:outer_loops_morozov
    %Solve Morzov type subproblem with CVX
    cvx_begin quiet
    variables u(n) v(n)
    minimize (alpha/2*sum_square(Df*u)+ norm(v,1)-sum(p.*u)-sum(q.*v))
    subject to
    norm(f-A*(u+v),2) <= c;
    cvx_end

    %compute subdifferential
    subdiff = alpha*Db*Df*u;
    p = subdiff;


    u_morozov(:,l) = u;
    v_morozov(:,l) =v;
    psnr_u_morozov(l) = psnr(u,comp_2);
    psnr_v_morozov(l) = psnr(v,comp_1);
    psnr_x_morozov(l) = psnr(u+v,x_true);
    %evaluate penalty functions
    val_g_morozov(l) = alpha/2*sum(abs(Df*u).^2,'all');
    val_h_morozov(l) = l1(v);

    %compute cross correlation and check stopping criterion
    cross_corr_morozov(l) = l2(convn(u,v(end:-1:1)))/(l2(u)*l2(v));
    if l>2 && cross_corr_morozov(l-2)>= cross_corr_morozov(l-1) && cross_corr_morozov(l) >cross_corr_morozov(l-1) && first_min
        ind_min_corr_morozov = l-1;
        first_min = false;
    end



end
ind_psnr_best_u_morozov = find(psnr_u_morozov == max(psnr_u_morozov));
ind_psnr_best_v_morozov = find(psnr_v_morozov == max(psnr_v_morozov));
ind_psnr_best_x_morozov = find(psnr_x_morozov == max(psnr_x_morozov));

psnr_sum_morozov = psnr_u_morozov+psnr_v_morozov;
ind_psnr_best_sum_morozov = find(psnr_sum_morozov == max(psnr_sum_morozov));


%%

alphas = logspace(-1,4,1000);
num_alphas = length(alphas);
u_variational = zeros(n,num_alphas);
v_variational = zeros(n,num_alphas);

psnr_u_variational = zeros(1,num_alphas);
psnr_v_variational = zeros(1,num_alphas);
psnr_x_variational = zeros(1,num_alphas);

val_g_variational = zeros(1,num_alphas);
val_h_variational = zeros(1,num_alphas);

cross_corr_variational = zeros(1,num_alphas);
ind_min_corr_variational = num_alphas;
first_min = true;

for l = 1:num_alphas
    alpha = alphas(l);
    cvx_begin quiet
    variables u(n) v(n)
    minimize (alpha/2*sum_square(Df*u)+ norm(v,1))
    subject to
    norm(f-A*(u+v),2) <= c;
    cvx_end

    u_variational(:,l) = u;
    v_variational(:,l) = v;

    val_g_variational(l) = alpha/2*sum(abs(Df*u).^2,'all');
    val_h_variational(l) = l1(v);

    psnr_u_variational(l) = psnr(u,comp_2);
    psnr_v_variational(l) = psnr(v,comp_1);
    psnr_x_variational(l) = psnr(v+u,x_true);




        cross_corr_variational(l) = l2(convn(u,v(end:-1:1)))/(l2(u)*l2(v));
    if l>2 && cross_corr_variational(l-2)>= cross_corr_variational(l-1) && cross_corr_variational(l) >cross_corr_variational(l-1) && first_min
        ind_min_corr_variational = l-1;
        first_min = false;
    end
end

val_g = alpha/2*sum(abs(Df*comp_2).^2,'all');
val_h = l1(comp_1);
psnr_sum_variational = psnr_u_variational+psnr_v_variational;




%%
%create and save plots

dir = strcat(pwd,'/Nested_vs_Variational_extensive');
mkdir(dir);

figure('visible','off');
semilogx(alphas, psnr_x_variational, color =  'black'); hold on; semilogx(alphas, psnr_x_morozov(ind_min_corr_morozov)*ones(1,num_alphas), '--', color = 'black'); semilogx(alphas, psnr_x_morozov(ind_psnr_best_x_morozov)*ones(1,num_alphas), ':', color = 'black');
xlabel('$\alpha$','Interpreter','latex'); ylabel('PSNR (dB)'); legend('Morozov regularization', 'Nested Bregman at stopping index', 'maximal PSNR Nested Bregman','Location','southwest')
hold on
ylim([min(psnr_x_variational)-0.1*range(psnr_x_variational), max(psnr_x_variational)+0.1*range(psnr_x_variational)]);
title('PSNR of the entire reconstruction')
exportgraphics(gcf,strcat(dir,'/psnr_x.pdf'),'ContentType','vector');close;

figure('visible','off');
semilogx(alphas, psnr_u_variational, 'black'); hold on; semilogx(alphas, psnr_u_morozov(ind_min_corr_morozov)*ones(1,num_alphas), '--', color = 'black'); semilogx(alphas, psnr_u_morozov(ind_psnr_best_u_morozov)*ones(1,num_alphas), ':', color = 'black');
xlabel('$\alpha$','Interpreter','latex'); ylabel('PSNR (dB)'); legend('Morozov regularization', 'Nested Bregman at stopping index', 'maximal PSNR Nested Bregman','Location','southwest')
hold on
ylim([min(psnr_u_variational)-0.1*range(psnr_u_variational), max(psnr_u_variational)+0.1*range(psnr_u_variational)]);
title('PSNR of u-component')
exportgraphics(gcf,strcat(dir,'/psnr_u.pdf'),'ContentType','vector');close;

figure('visible','off');
semilogx(alphas, psnr_v_variational, 'black'); hold on; semilogx(alphas, psnr_v_morozov(ind_min_corr_morozov)*ones(1,num_alphas), '--', color = 'black'); semilogx(alphas, psnr_v_morozov(ind_psnr_best_v_morozov)*ones(1,num_alphas), ':', color = 'black');
xlabel('$\alpha$','Interpreter','latex'); ylabel('PSNR (dB)'); legend('Morozov regularization', 'Nested Bregman at stopping index', 'maximal PSNR Nested Bregman','Location','southwest')
hold on
ylim([min(psnr_v_variational)-0.1*range(psnr_v_variational), max(psnr_v_variational)+0.1*range(psnr_v_variational)]);
title('PSNR of v-component')
exportgraphics(gcf,strcat(dir,'/psnr_v.pdf'),'ContentType','vector');close;

figure('visible','off');
semilogx(alphas, psnr_sum_variational, 'black'); hold on; semilogx(alphas, psnr_sum_morozov(ind_min_corr_morozov)*ones(1,num_alphas), '--', color = 'black'); semilogx(alphas, psnr_sum_morozov(ind_psnr_best_sum_morozov)*ones(1,num_alphas), ':', color = 'black');
xlabel('$\alpha$','Interpreter','latex'); ylabel('PSNR (dB)'); legend('Morozov regularization', 'Nested Bregman at stopping index', 'maximal PSNR Nested Bregman','Location','southwest')
hold on
ylim([min(psnr_sum_variational)-0.1*range(psnr_sum_variational), max(psnr_sum_variational)+0.1*range(psnr_sum_variational)]);
title('Sum of individual PSNR values for u- and v-component')
exportgraphics(gcf,strcat(dir,'/psnr_sum.pdf'),'ContentType','vector');close;

figure('visible','off'); 
yyaxis left;
plot(psnr_u_variational,'Color','black')
xlabel('$\alpha$','Interpreter','latex')
set(gca,'ycolor','black')
ylabel('PSNR (dB)');
yyaxis right
plot(cross_corr_variational,'--','Color','black')
set(gca,'ycolor','black')
ylabel('$\mathcal{C}(u,v)$','interpreter','latex')
legend('PSNR','Correlation')
title('u')
exportgraphics(gcf,strcat(dir,'/corr_u.pdf'),'ContentType','vector');close;

figure('visible','off'); 
yyaxis left;
plot(psnr_v_variational,'Color','black')
xlabel('$\alpha$','Interpreter','latex')
set(gca,'ycolor','black')
ylabel('PSNR (dB)');
yyaxis right
plot(cross_corr_variational,'--','Color','black')
set(gca,'ycolor','black')
ylabel('$\mathcal{C}(u,v)$','interpreter','latex')
legend('PSNR','Correlation')
title('v')
exportgraphics(gcf,strcat(dir,'/corr_v.pdf'),'ContentType','vector');close;

figure('visible','off'); 
yyaxis left;
plot(psnr_x_variational,'Color','black')
xlabel('$\alpha$','Interpreter','latex')
set(gca,'ycolor','black')
ylabel('PSNR (dB)');
yyaxis right
plot(cross_corr_variational,'--','Color','black')
set(gca,'ycolor','black')
ylabel('$\mathcal{C}(u,v)$','interpreter','latex')
legend('PSNR','Correlation')
title('x')
exportgraphics(gcf,strcat(dir,'/corr_x.pdf'),'ContentType','vector');close;

figure('visible','off'); 
yyaxis left;
plot(psnr_sum_variational,'Color','black')
xlabel('$\alpha$','Interpreter','latex')
set(gca,'ycolor','black')
ylabel('PSNR (dB)');
yyaxis right
plot(cross_corr_variational,'--','Color','black')
set(gca,'ycolor','black')
ylabel('$\mathcal{C}(u,v)$','interpreter','latex')
legend('PSNR','Correlation')
title('sum')
exportgraphics(gcf,strcat(dir,'/corr_sum.pdf'),'ContentType','vector');close;






