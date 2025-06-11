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

%creates the plots corresponding to the numerical experiments for Tikhonov regularization, Bregman iterations and Nested Bregman iterations with
%H^1-TV penalties


close all;
clear;
clc;
load('data_plotting.mat');


%create directory
dir = strcat(pwd,'/Images_Regularization');
mkdir(dir);

%plot true images and observation

figure('visible','off');
image(255*u_true); colormap(gray); axis off; pbaspect([m,n,1]); title('$u^\dagger$','Interpreter','latex',FontSize=20);exportgraphics(gcf,strcat(dir,'/u.pdf'),'ContentType','vector');close
figure('visible','off');
image(255*v_true); colormap(gray); axis off; pbaspect([m,n,1]); title('$v^\dagger$','Interpreter','latex',FontSize=20);exportgraphics(gcf,strcat(dir,'/v.pdf'),'ContentType','vector');close;
figure('visible','off');
image(255*(u_true+v_true)); colormap(gray); axis off; pbaspect([m,n,1]); title('$u^\dagger+v^\dagger$','Interpreter','latex',FontSize=20);exportgraphics(gcf,strcat(dir,'/u+v.pdf'),'ContentType','vector');close;
figure('visible','off');
image(255*f); colormap(gray); axis off; pbaspect([m,n,1]); title('${f^\delta}$','Interpreter','latex','FontWeight','bold',FontSize=20);exportgraphics(gcf,strcat(dir,'/f.pdf'),'ContentType','vector');close;



%plots for Tikhonov and Bregman iterations


for l = 1:length(ratios)
    dir_T = char(strcat(dir,'/Tikhonov/rat_',num2str(ratios(l))));
    dir_B = char(strcat(dir,'/Bregman/rat_',num2str(ratios(l))));
    mkdir(dir_T);
    mkdir(dir_B);
    

    alpha = alphas(l);
    beta = betas(l);

    u = u_ratios(:,:,l);
    v = v_ratios(:,:,l);


    figure('visible','off'); image(255*u); colormap(gray); axis off; pbaspect([m,n,1]); title({strcat('$u\, (\alpha = ', num2str(alpha),')$'),strcat('$PSNR = ', num2str(psnr_ratios_u(l)),'$')},'Interpreter','latex',FontSize=20); exportgraphics(gcf,strcat(dir_T,'/u.pdf'),'ContentType','vector'); close
    figure('visible','off'); image(255*v); colormap(gray); axis off; pbaspect([m,n,1]); title({strcat('$v\, (\beta = ', num2str(beta), ')$'),strcat('$PSNR = ', num2str(psnr_ratios_v(l)),'$')},'Interpreter','latex',FontSize=20); exportgraphics(gcf,strcat(dir_T,'/v.pdf'),'ContentType','vector'); close
    figure('visible','off');image(255*(u+v)); colormap(gray); axis off;pbaspect([m,n,1]); pbaspect([m,n,1]); title(strcat('$u+v, \; PSNR = ', num2str(psnr_ratios_x(l)),'$'),Interpreter="latex",FontSize=20);exportgraphics(gcf,strcat(dir_T,'/u+v.pdf'),'ContentType','vector'); close
    figure('visible','off');image(255*real(ifft2(kernel_four.*fft2(u+v)))); colormap(gray); axis off; pbaspect([m,n,1]); tit ='$A(u+v)$';title(tit,'Interpreter','Latex',FontSize=20);exportgraphics(gcf,strcat(dir_T,'/Au+v.pdf'),'ContentType','vector'); close


    alpha = 4*alpha;
    beta = 4*beta;

    u = u_Bregman_ratios(:,:,l);
    v = v_Bregman_ratios(:,:,l);

    stopping = stopping_ratios(l);

    figure('visible','off'); image(255*u); colormap(gray); axis off; pbaspect([m,n,1]); title({strcat('$u\, (\alpha = ', num2str(alpha),')$'),strcat('$PSNR = ', num2str(psnr_bregman_ratios_u(l)),'$')},'Interpreter','latex',FontSize=20); exportgraphics(gcf,strcat(dir_B,'/u.pdf'),'ContentType','vector'); close
    figure('visible','off'); image(255*v); colormap(gray); axis off; pbaspect([m,n,1]); title({strcat('$v \, (\beta = ', num2str(beta),')$'),strcat('$PSNR = ', num2str(psnr_bregman_ratios_v(l)),'$')},'Interpreter','latex',FontSize=20); exportgraphics(gcf,strcat(dir_B,'/v.pdf'),'ContentType','vector'); close
    figure('visible','off');image(255*(u+v)); colormap(gray); axis off;pbaspect([m,n,1]); pbaspect([m,n,1]); title(strcat('$u+v,\; PSNR = ', num2str(psnr_bregman_ratios_x(l)),'$'),Interpreter="latex",FontSize=20);exportgraphics(gcf,strcat(dir_B,'/u+v.pdf'),'ContentType','vector'); close
    figure('visible','off');image(255*real(ifft2(kernel_four.*fft2(u+v)))); colormap(gray); axis off; pbaspect([m,n,1]); tit = strcat('$A(u+v) \; ','\mathbf{\quad (k^\delta =   }',num2str(stopping_ratios(l)),')$');title(tit,'Interpreter','Latex',FontSize=20);exportgraphics(gcf,strcat(dir_B,'/Au+v.pdf'),'ContentType','vector'); close


end



%plots for Nested Bregman iterations

dir = strcat(pwd,'/Images_Nested');
dir_M = strcat(dir,'/Nested_Morozov');
dir_B = strcat(dir,'/Nested_Bregman');

mkdir(dir);
mkdir(dir_M);
mkdir(dir_B);


%psnr and stopping
figure('visible','off');
yyaxis left;
plot(psnr_sum_M,'o-','MarkerEdgeColor','black','MarkerIndices',ind_psnr_M,'Color','black')
xlabel('iterations')
set(gca,'ycolor','black')
ylabel('PSNR (dB)');
yyaxis right
plot(cross_corr_M,'*--','MarkerFaceColor','black','MarkerEdgeColor','black','MarkerIndices',ind_stopping_M,'Color','black')
set(gca,'ycolor','black')
ylabel('$\mathcal{C}(u,v)$','interpreter','latex')
legend('PSNR','Correlation')
exportgraphics(gcf,strcat(dir_M,'/PSNR.pdf'),'ContentType','vector');close;



figure('visible','off');
yyaxis left;
plot(psnr_sum_B,'o-','MarkerEdgeColor','black','MarkerIndices',ind_psnr_B,'Color','black')
xlabel('iterations')
set(gca,'ycolor','black')
ylabel('PSNR (dB)');
yyaxis right
plot(cross_corr_B,'*--','MarkerFaceColor','black','MarkerEdgeColor','black','MarkerIndices',ind_stopping_B,'Color','black')
set(gca,'ycolor','black')
ylabel('$\mathcal{C}(u,v)$','interpreter','latex')
legend('PSNR','Correlation')
exportgraphics(gcf,strcat(dir_B,'/PSNR.pdf'),'ContentType','vector');close;


%images of components
alpha =1/0.001;
beta = 1;
for l = [1,ind_psnr_M,ind_stopping_M,outer_its]
    figure('visible','off'); image(255*u_nested_M(:,:,l)); colormap(gray); axis off; pbaspect([m,n,1]); title(strcat('$u_{',num2str(l),'}', ',\; PSNR = ',num2str(psnr_nested_u_M(l)),'$'),'Interpreter','latex',FontSize=20); exportgraphics(gcf,strcat(dir_M,'/u_',num2str(l),'.pdf'),'ContentType','vector'); close
    figure('visible','off'); image(255*v_nested_M(:,:,l)); colormap(gray); axis off;pbaspect([m,n,1]); title(strcat('$v_{',num2str(l),'} ', ',\; PSNR = ', num2str(psnr_nested_v_M(l)),'$'),'Interpreter','latex',FontSize=20); exportgraphics(gcf,strcat(dir_M,'/v_',num2str(l),'.pdf'),'ContentType','vector'); close
    figure('visible','off');image(255*(u_nested_M(:,:,l)+v_nested_M(:,:,l))); colormap(gray); axis off;pbaspect([m,n,1]); pbaspect([m,n,1]); title(strcat('$u_{',num2str(l),'}+v_{',num2str(l),'}', ',\; PSNR = ', num2str(psnr(u_nested_M(:,:,l) + v_nested_M(:,:,l),x_true)),'$'),Interpreter="latex",FontSize=20);exportgraphics(gcf,strcat(dir_M,'/u_',num2str(l),'+v_',num2str(l),'.pdf'),'ContentType','vector'); close
    figure('visible','off');image(255*real(ifft2(kernel_four.*fft2(u_nested_M(:,:,l)+v_nested_M(:,:,l))))); colormap(gray); axis off; pbaspect([m,n,1]); tit = strcat('$A(u_{',num2str(l),'}+v_{',num2str(l),'})$');title(tit,'Interpreter','Latex',FontSize=20);exportgraphics(gcf,strcat(dir_M,'/A(u_',num2str(l),'+v_',num2str(l),').pdf'),'ContentType','vector'); close
end

alpha = 4*alphas(1);
beta = 4*betas(1);
for l = [1,ind_psnr_B,ind_stopping_B,outer_its]
    stopping = stopping_nested(l);
    figure('visible','off'); image(255*u_nested_B(:,:,l)); colormap(gray); axis off; pbaspect([m,n,1]); title(strcat('$u_{',num2str(l),'},', '\; PSNR = ',num2str(psnr_nested_u_B(l)),'$'),'Interpreter','latex',FontSize=20); exportgraphics(gcf,strcat(dir_B,'/u_',num2str(l),'.pdf'),'ContentType','vector'); close
    figure('visible','off'); image(255*v_nested_B(:,:,l)); colormap(gray); axis off;pbaspect([m,n,1]); title(strcat('$v_{',num2str(l),'},', '\; PSNR = ',num2str(psnr_nested_v_B(l)),'$'),'Interpreter','latex',FontSize=20); exportgraphics(gcf,strcat(dir_B,'/v_',num2str(l),'.pdf'),'ContentType','vector'); close
    figure('visible','off'); image(255*(u_nested_B(:,:,l)+v_nested_B(:,:,l))); colormap(gray); axis off;pbaspect([m,n,1]); pbaspect([m,n,1]); title(strcat('$u_{',num2str(l),'}+v_{',num2str(l),'}$, PSNR = ', num2str(psnr(u_nested_B(:,:,l)+v_nested_B(:,:,l),x_true))),Interpreter="latex",FontSize=20);exportgraphics(gcf,strcat(dir_B,'/u_',num2str(l),'+v_',num2str(l),'.pdf'),'ContentType','vector'); close
    figure('visible','off'); image(255*real(ifft2(kernel_four.*fft2(u_nested_B(:,:,l)+v_nested_B(:,:,l))))); colormap(gray); axis off; pbaspect([m,n,1]); tit = strcat('$A(u_{',num2str(l),'}+v_{',num2str(l),'})  ','\, \mathbf{\quad (k^\delta =   ',num2str(stopping_nested(l)),')}$');title(tit,'Interpreter','Latex',FontSize=20);exportgraphics(gcf,strcat(dir_B,'/A(u_',num2str(l),'+v_',num2str(l),').pdf'),'ContentType','vector'); close
end

