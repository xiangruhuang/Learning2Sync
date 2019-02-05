
function d=phi_6(R1,R2)
%
% d=phi_6(R1,R2)
% Computes the distance d between the rotation matrices R1,R2 using the
% function phi_6 (angular distance) -> rotation angle of R1*R2' in the
% angle-axis representation
%
% Reference: "Metrics for 3D rotations: comparison and analysis"
% Author: Federica Arrigoni, 2013


R=R1*R2';
d = acos((trace(R)-1)/2);

% NOTE: due to numerical errors, it may happen that abs(q1'*q2) is greater
% than 1, such that acos returns a complex number. In such a case we
% consider the real part only
d=real(d);

end
