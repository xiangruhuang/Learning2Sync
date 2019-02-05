
function [theta,u,v] = axis_angle(R)
%
% Computes the rotation axis u and angle theta corresponding to the 
% rotation matrix R
%
% Reference: "Metrics for 3D rotations: comparison and analysis" Huyn - 
% Appendix A
%
% Author: Federica Arrigoni, 2013


theta = acos((trace(R)-1)/2); % due to numerical errors, theta may be complex
theta=real(theta); 


% R-R' is the skew-symmetric matrix corresponding to the cross product with
% u
u = [R(3,2)-R(2,3), R(1,3)-R(3,1), R(2,1)-R(1,2)]'; 

% particular cases:
if (nnz(R-R')==0) % R-R'=0
    if (nnz(R-eye(3))==0) % R = I
        theta=0;
        u=[1;1;1]; % u can be any vector (the angle is zero!)
    else %R +I=2*u*u' has rank 1
        theta=pi;
        A=R+eye(3);
        u=A(:,1); % u can be computed by normalizing any column of R+I
    end       
end

u=u/norm(u);

v=u*theta;

end
