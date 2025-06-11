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

%creates the plots corresponding to the numerical experiments for Tikhonov regularization, Bregman iterations and Nested Bregman iterations with
%TGV and oscillation TGV as penalties

close all
clear
clc
load('data.mat')

%create directory
dir = strcat(pwd,'/Images_Nested');
mkdir(dir);


%plot true images and observation
figure('visible','off');
yyaxis left;
plot(psnr_nested_sum,'o-','MarkerEdgeColor','black','MarkerIndices',ind_max_psnr,'Color','black')
set(gca,'ycolor','black')
ylabel('PSNR (dB)');
yyaxis right
plot(cross_corr,'*--','MarkerFaceColor','black','MarkerEdgeColor','black','MarkerIndices',ind_stopping_corr,'Color','black')
set(gca,'ycolor','black')
ylabel('$\mathcal{C}(u,v)$','interpreter','latex')
xlabel('iterations')
legend('PSNR','Correlation');
exportgraphics(gcf,strcat(dir,'/PSNR.pdf'),'ContentType','vector');close;

figure('visible', 'off'); image(255*comp1); colormap(gray); axis off; pbaspect([n,n,1]); title('$u^\dagger$',Interpreter='latex',FontSize=20); exportgraphics(gcf,strcat(dir,'/u_true.pdf'),'ContentType','vector');close;
figure('visible', 'off');image(256*comp2); colormap(gray); axis off; pbaspect([n,n,1]); title('$v^\dagger$',Interpreter='latex',FontSize=20); exportgraphics(gcf,strcat(dir,'/v_true.pdf'),'ContentType','vector');close;
figure('visible', 'off');image(255*f); colormap(gray); axis off; pbaspect([n,n,1]); title('$f^\delta$',Interpreter='latex',FontSize=20); exportgraphics(gcf,strcat(dir,'/x_true.pdf'),'ContentType','vector');close;


%plots for Nested Bregman iterations
for l = [1,ind_stopping_corr,ind_max_psnr,outer_its]
   figure('visible', 'off'); image(255*u_nested(:,:,l));colormap(gray);axis off; pbaspect([n,n,1]);title(strcat('$u_{',num2str(l),'}$',', PSNR = ',num2str(psnr_nested_u(l))),Interpreter="latex",FontSize=20); exportgraphics(gcf,strcat(dir,'/u_',num2str(l),'.pdf'),'ContentType','vector'); close;
   figure('visible', 'off'); image(255*v_nested(:,:,l));colormap(gray);axis off; pbaspect([n,n,1]);title(strcat('$v_{',num2str(l),'}$',', PSNR  = ', num2str(psnr_nested_v(l))),Interpreter="latex",FontSize=20); exportgraphics(gcf,strcat(dir,'/v_',num2str(l),'.pdf'),'ContentType','vector'); close;
   figure('visible', 'off'); image(255*(u_nested(:,:,l)+v_nested(:,:,l)));colormap(gray);axis off; pbaspect([n,n,1]);title(strcat('$u_{',num2str(l),'}+v_{',num2str(l),'}$', ', PSNR = ', num2str(psnr_nested_x(l))),Interpreter="latex",FontSize=20); exportgraphics(gcf,strcat(dir,'/x_',num2str(l),'.pdf'),'ContentType','vector'); close;
end

