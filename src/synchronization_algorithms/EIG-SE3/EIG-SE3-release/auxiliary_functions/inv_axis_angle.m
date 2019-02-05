
function R = inv_axis_angle(theta,u)
% Computes the orthonormal matrix corresponding to the rotation through the
% angle theta about the axis u
%
% Author: Federica Arrigoni, 2013

if (length(u) ~= 3)
    error('L''asse di rotazione deve essere un vettore di 3 elementi!!');
end

if theta ==0
    u=[1;1;1];
end

u=u/norm(u); % unit vector
u=u(:); % column vector 

% Rodriguez Formula
R=cos(theta)*eye(3)+sin(theta)*star(u)+(1-cos(theta))*u*u';

end


