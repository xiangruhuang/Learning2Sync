function R = eul(a,t)
%EUL Returns a rotation matrix from the Euler angles
%    R = eul(a,'RPY') return the matrix specified by the RPY (Roll,Pithch,Yaw)
%    system, namely R = R_z(a(3)) * R_y(a(2)) * R_x(a(1)). Note that
%    a=[Yaw,Pitch,Roll] 
%
%    R = eul(a,'ZYZ') return the matrix specified by the ZYZ system.
%
%    R = eul(a) is the same as eul(a,'RPY')
%
%    See also: IEUL


% Authors: A. Fusiello, M. Mancini.


%Controllo del formato dei parametri di input
na=length(a);
if na~=3
   error('Gli angoli devo essere 3!!')
end

if (nargin < 2)
    t='RPY';
end
    

phi   = a(3);
theta = a(2);
psi   = a(1);

switch t
    case 'RPY'
        R = [ cos(phi)*cos(theta) cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi) cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi)
              sin(phi)*cos(theta) sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi) sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi)
              -sin(theta)                    cos(theta)*sin(psi)                              cos(theta)*cos(psi)];
    case 'ZYZ'
        R = [ cos(phi)*cos(theta)*cos(psi)-sin(phi)*sin(psi) -cos(phi)*cos(theta)*sin(psi)-sin(phi)*cos(psi) cos(phi)*sin(theta)
              sin(phi)*cos(theta)*cos(psi)+cos(phi)*sin(psi) -sin(phi)*cos(theta)*sin(psi)+cos(phi)*cos(psi) sin(phi)*sin(theta)
                -sin(theta)*cos(psi)                                       sin(theta)*sin(psi)                  cos(theta)];

    otherwise
        error('Tipo di angoli non supportato!!');
end     
