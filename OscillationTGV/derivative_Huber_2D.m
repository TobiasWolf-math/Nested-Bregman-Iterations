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
function [der_x,der_y] = derivative_Huber_2D(u_x,u_y,epsilon)
%computes the derivative of the 2D-Huber function in directions x and y 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input: x-component u_x, y-component u_y and parameter epsilon for Huber
%function
%Output components der_x and der_y of gradient

norm_u = sqrt(u_x.^2+u_y.^2);
quad = find(norm_u <= epsilon);
l1 = find(norm_u > epsilon);
der_x = zeros(size(u_x));
der_y = zeros(size(u_y));
der_x(quad) = u_x(quad)/epsilon;
der_y(quad) = u_y(quad)/epsilon;
der_x(l1) = u_x(l1)./norm_u(l1);
der_y(l1) = u_y(l1)./norm_u(l1);
end