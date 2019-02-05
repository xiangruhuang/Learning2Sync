function  R = quat(q)
%IQUAT Compute the rotation matrix given a unit quaternion 

% format compatible with Lourakis' bundle adjustment: (scalar, vector)
% BUT not with the quaternion toolbox which uses (vector, scalar).

%fprintf('IQUAT del CVLAB\n');

% normalize the quaternion first
q = q./norm(q);

R = [ q(1)^2+q(2)^2-q(3)^2-q(4)^2,  2*(q(2)*q(3)-q(1)*q(4)), 2*(q(2)*q(4)+q(1)*q(3))   
    2*(q(2)*q(3)+q(1)*q(4)), q(1)^2-q(2)^2+q(3)^2-q(4)^2,  2*(q(3)*q(4)-q(1)*q(2))   
    2*(q(2)*q(4)-q(1)*q(3)),  2*(q(3)*q(4)+q(1)*q(2)),  q(1)^2-q(2)^2-q(3)^2+q(4)^2 ];


