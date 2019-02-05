

function [err_R,err_C,nrmse,C1] = error_R_T(R1,R2,T1,T2)

% R1 = estimated absolute rotations
% R2 = ground truth absolute rotations
% T1 = estimated absolute translations
% T2 = ground truth absolute translations
% Computes the optimal rotation, translation and scale necessary to map
% R1,T1 to R2,T2 and evaluates the errors
%
% Author: Federica Arrigoni, 2015


ncams=size(R1,3);


% compute the optimal rotation that maps R1 to R2 by applying L1 single
% rotation averaging (Weiszfeld algorithm)

iter_max=20;
R_estimates=zeros(3,3,ncams);
for i=1:ncams
    R_estimates(:,:,i)=R1(:,:,i)'*R2(:,:,i);
end
Rmean = L1_single_averaging(R_estimates,iter_max);


% transform the rotations, evaluate the error and update camera centres
err_R=zeros(1,ncams);
C1=zeros(3,ncams);
C2=zeros(3,ncams);
for i=1:ncams
    R1(:,:,i)=R1(:,:,i)*Rmean;
    err_R(i)=phi_6(R1(:,:,i),R2(:,:,i))*180/pi;
    
    %err_R(i) = norm(R1(:,:,i)-R2(:,:,i),'fro');
    
    C1(:,i)=-R1(:,:,i)'*T1(:,i);
    C2(:,i)=-R2(:,:,i)'*T2(:,i);
end

% compute the optimal translation and scale that maps C1 to C2 in the least
% squares sense
A=zeros(3*ncams,4);
b=zeros(3*ncams,1);
for i=1:ncams
    A(3*i-2:3*i,:) = [C1(:,i) eye(3)];
    b(3*i-2:3*i) = C2(:,i);
end
x=A\b;
scale_opt=x(1);
T_opt=x(2:4);


% transform the centres and evaluate the error
err_C=zeros(1,ncams);
for i=1:ncams
    C1(:,i) = scale_opt*C1(:,i)+T_opt;
    err_C(i) = norm(C1(:,i)-C2(:,i));
end


% compute nrmse
C_mean=mean(C2,2);
nrmse=sqrt(sum(err_C.^2)/norm(C1-repmat(C_mean,1,ncams),'fro')^2);




end





