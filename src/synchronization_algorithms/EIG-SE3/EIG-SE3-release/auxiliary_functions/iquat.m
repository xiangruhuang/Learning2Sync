
function   q = iquat(R)
% Computes the quaternion form of a rotation matrix

% Format compatible with Lourakis' bundle adjustment: (scalar, vector)
% BUT not with the quaternion toolbox which uses (vector, scalar).

% Reference: B. K. P. Horn, "Closed-form solution of absolute orientation 
% using unit quaternions", Journal of the Optical Society of America A,
% 1987


% if abs(det(R) - 1.0)> 0.00001
%     error('R must be a rotation')
% end

q(1)=0.5*sqrt(trace(R)+1);
q(2)=0.5*sqrt(1+R(1,1)-R(2,2)-R(3,3));
q(3)=0.5*sqrt(1-R(1,1)+R(2,2)-R(3,3));
q(4)=0.5*sqrt(1-R(1,1)-R(2,2)+R(3,3));

[~,i]=max(q);

switch i
    case 1
        q(2) = (R(3,2)-R(2,3))/(4*q(1));
        q(3) = (R(1,3)-R(3,1))/(4*q(1));
        q(4) = (R(2,1)-R(1,2))/(4*q(1));
    case 2
        q(1) = (R(3,2)-R(2,3))/(4*q(2));
        q(3) = (R(2,1)+R(1,2))/(4*q(2));
        q(4) = (R(1,3)+R(3,1))/(4*q(2));
    case 3
        q(1) = (R(1,3)-R(3,1))/(4*q(3));
        q(2) = (R(1,2)+R(2,1))/(4*q(3));
        q(4) = (R(2,3)+R(3,2))/(4*q(3));
    case 4
        q(1) = (R(2,1)-R(1,2))/(4*q(4));
        q(2) = (R(1,3)+R(3,1))/(4*q(4));
        q(3) = (R(2,3)+R(3,2))/(4*q(4));
end




end