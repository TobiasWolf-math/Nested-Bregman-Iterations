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

% conducts the numerical experiments for Nested Bregman iterations with
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
    %evaluate penalty functions
    val_g_morozov(l) = alpha/2*sum(abs(Df*u),'all');
    val_h_morozov(l) = l1(v);

    %compute cross correlation and check stopping criterion
    cross_corr_morozov(l) = l2(convn(u,v(end:-1:1)))/(l2(u)*l2(v));
    if l>2 && cross_corr_morozov(l-2)>= cross_corr_morozov(l-1) && cross_corr_morozov(l) >cross_corr_morozov(l-1) && first_min
        ind_min_corr_morozov = l-1;
        first_min = false;
    end



end
psnr_sum_morozov = psnr_u_morozov+psnr_v_morozov;
ind_psnr_best_morozov = find(psnr_sum_morozov == max(psnr_sum_morozov));

%%
%nested Bregman iterations with Bregman iterations in inner loop
tau = 1.01;
tol = tau*delta;
beta = 1;
Bregman_its = 100;

p=zeros(size(f));
q=zeros(size(f));
outer_loops_nested = 100;

u_nested = zeros(n,outer_loops_nested);
v_nested = zeros(n,outer_loops_nested);

val_g_nested = zeros(1,outer_loops_nested);
val_h_nested = zeros(1,outer_loops_nested);

psnr_u_nested = zeros(1,outer_loops_nested);
psnr_v_nested = zeros(1,outer_loops_nested);

cross_corr_nested = zeros(1,outer_loops_nested);
ind_min_corr_nested = outer_loops_nested;
first_min = true;

subdiff = 0;

for l = 1:outer_loops_nested

    %Bregman iterations for regularized solution using CVX
    f_breg = f;

    for ll = 1:Bregman_its
        cvx_begin quiet
        variables u(n) v(n)
        minimize (0.5*sum_square(f_breg-A*(u+v))+alpha/2*sum_square(Df*u)+ beta*norm(v,1)-sum(p.*u)-sum(q.*v))
        cvx_end
        f_breg = f_breg + (f-A*(u+v));
        subdiff = subdiff+A'*(f-A*(u+v));


        if l2(f-A*(u+v)) <= tol

            break;
        end
    end

    %compute subdifferential
    p = subdiff;
    u_nested(:,l) = u;
    v_nested(:,l) = v;


    psnr_u_nested(l) = psnr(u,comp_2);
    psnr_v_nested(l) = psnr(v,comp_1);

    %evaluate penalty functions
    val_g_nested(l) = alpha/2*sum(abs(Df*u),'all');
    val_h_nested(l) = beta*l1(v);

    %compute cross correlation and check stopping criterion
    cross_corr_nested(l) = l2(convn(u,v(end:-1:1)))/(l2(u)*l2(v));
    if l>2 && cross_corr_nested(l-2)>= cross_corr_nested(l-1) && cross_corr_nested(l) >cross_corr_nested(l-1) && first_min
        ind_min_corr_nested = l-1;
        first_min = false;
    end
end
psnr_sum_nested = psnr_u_nested+psnr_v_nested;
ind_psnr_best_nested = find(psnr_sum_nested == max(psnr_sum_nested));

%%
%plotting
close all

mkdir(strcat(pwd,'/Images_nested/Images_Morozov'));
mkdir(strcat(pwd,'/Images_nested/Images_Nested'));


%values of penalty vs theoretical bound
val_g = alpha/2*sum(abs(Df*x_true),'all');
val_h = l1(x_true);

figure('visible','off')
semilogy(1:outer_loops_morozov,val_h_morozov,'--',Color='black');
hold on
semilogy(1:outer_loops_morozov,val_h_nested,'.',Color='black');
semilogy(1:outer_loops_morozov,ones(outer_loops_morozov,1)*val_g./(1:outer_loops_morozov)',Color='black');
hold off
legend('Algortihm 4.2','Algorithm 4.3','theoretical bound')
xlabel('iterations')
ylabel('value of $\Vert v \Vert_1$','Interpreter','latex')
ylim([1e-1,1e4]);
xlim([1,100])
exportgraphics(gcf,'Images_Nested/Comparison_Morozov_Nested.pdf','ContentType','vector');
close;

%psnr and stopping
figure('visible','off');
yyaxis left;
plot(psnr_sum_morozov,'o-','MarkerEdgeColor','black','MarkerIndices',ind_psnr_best_morozov,'Color','black')
set(gca,'ycolor','black')
ylabel('PSNR (dB)');
yyaxis right
plot(cross_corr_morozov,'*--','MarkerFaceColor','black','MarkerEdgeColor','black','MarkerIndices',ind_min_corr_morozov,'Color','black')
set(gca,'ycolor','black')
ylabel('$\mathcal{C}(u,v)$','interpreter','latex');
xlabel('iterations');
legend('PSNR','Correlation');
exportgraphics(gcf,'Images_Nested/PSNR_Morozov.pdf','ContentType','vector');close;


figure('visible','off');
yyaxis left;
plot(psnr_sum_nested,'o-','MarkerEdgeColor','black','MarkerIndices',ind_psnr_best_nested,'Color','black')
set(gca,'ycolor','black')
ylabel('PSNR (dB)');
yyaxis right
plot(cross_corr_nested,'*--','MarkerFaceColor','black','MarkerEdgeColor','black','MarkerIndices',ind_min_corr_nested,'Color','black')
set(gca,'ycolor','black')
ylabel('$\mathcal{C}(u,v)$','interpreter','latex')
xlabel('iterations');
legend('PSNR','Correlation');
exportgraphics(gcf,'Images_Nested/PSNR_Nested.pdf','ContentType','vector');close;


%images of components
for l = [1,ind_psnr_best_morozov,ind_min_corr_morozov,outer_loops_morozov]
    figure('visible','off'); plot(u_morozov(:,l),'color','black'); title(strcat('$u_{',num2str(l),'}$'),'Interpreter','latex',FontSize=20);ylim([-1.5,1.5]); exportgraphics(gcf,strcat('Images_nested/Images_Morozov/u_',num2str(l),'.pdf'),'ContentType','vector'); close;
    figure('visible','off'); plot(v_morozov(:,l),'color','black'); title(strcat('$v_{',num2str(l),'}$'),'Interpreter','latex',FontSize=20);ylim([-1.5,1.5]); exportgraphics(gcf,strcat('Images_nested/Images_Morozov/v_',num2str(l),'.pdf'),'ContentType','vector'); close;
    figure('visible','off'); plot(u_morozov(:,l)+v_morozov(:,l),'color','black'); title(strcat('$u_{',num2str(l),'}+v_{',num2str(l),'}$'),'Interpreter','latex',FontSize=20);ylim([-1.5,1.5]); exportgraphics(gcf,strcat('Images_nested/Images_Morozov/u_',num2str(l),'+v_',num2str(l),'.pdf'),'ContentType','vector');ylim([-1.5,1.5]); close;

end

for l = [1,ind_psnr_best_nested,ind_min_corr_nested,outer_loops_nested]
    figure('visible','off'); plot(u_nested(:,l),'color','black'); title(strcat('$u_{',num2str(l),'}$'),'Interpreter','latex',FontSize=20);ylim([-1.5,1.5]); exportgraphics(gcf,strcat('Images_nested/Images_Nested/u_',num2str(l),'.pdf'),'ContentType','vector'); close;
    figure('visible','off'); plot(v_nested(:,l),'color','black'); title(strcat('$v_{',num2str(l),'}$'),'Interpreter','latex',FontSize=20);ylim([-1.5,1.5]); exportgraphics(gcf,strcat('Images_nested/Images_Nested/v_',num2str(l),'.pdf'),'ContentType','vector'); close;
    figure('visible','off'); plot(u_nested(:,l)+v_nested(:,l),'color','black'); title(strcat('$u_{',num2str(l),'}+v_{',num2str(l),'}$'),'Interpreter','latex',FontSize=20); ylim([-1.5,1.5]);exportgraphics(gcf,strcat('Images_nested/Images_Nested/u_',num2str(l),'+v_',num2str(l),'.pdf'),'ContentType','vector');ylim([-1.5,1.5]); close;
end

figure('visible','off'); plot(comp_2,'color','black'); title('$u^\dagger$',Interpreter='latex',FontSize=20);ylim([-1.5,1.5]); exportgraphics(gcf,strcat('Images_nested/u_real.pdf'),'ContentType','vector'); close;
figure('visible','off'); plot(comp_1,'color','black'); title('$v^\dagger$',Interpreter='latex',FontSize=20);ylim([-1.5,1.5]); exportgraphics(gcf,strcat('Images_nested/v_real.pdf'),'ContentType','vector'); ylim([-1.5,1.5]);close;
figure('visible','off'); plot(f,'color','black'); title('$f^\delta$',Interpreter='latex',FontSize=20);ylim([-1.5,1.5]); exportgraphics(gcf,strcat('Images_nested/f.pdf'),'ContentType','vector');ylim([-1.5,1.5]); close;


