function [ Rm ] = dqq_rotation_quaternion_initialization( R )
%DQQ_R_Q_INITIALIZATION Summary of this function goes here
%   Detailed explanation goes here
%   This is a function which provides an initialization for givin mean rotation  
%	matrix R.
%	Please refer to
%	'Rotation Averaging with Application to Camera-Rig Calibration'
%	for details.




QR=zeros(size(R,3),4);
for i=1:size(R,3)
    QR(i,:) = iquat(R(:,:,i));
end
SR=R(:,:,1);
SQ=iquat(SR);


% QR=dcm2quat(R);
% 
% SR=R(:,:,1);
% SQ=dcm2quat(SR);


for i=1:size(R,3)
    if norm(QR(i,:)+SQ)<norm(QR(i,:)-SQ)
        QR(i,:)=-QR(i,:);
    end
end


barQR=sum(QR,1)/size(R,3);

barQR=barQR/norm(barQR);

%Rm=quat2dcm(barQR);

Rm=quat(barQR);



end






