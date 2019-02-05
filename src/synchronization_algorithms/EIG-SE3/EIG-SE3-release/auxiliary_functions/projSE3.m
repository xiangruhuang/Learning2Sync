
function [M,R,t]=projSE3(U,n)

% Perform projection onto SE(3)

M=zeros(4,4,n);
R=zeros(3,3,n);
t=zeros(3,n);

for i = 1:n
    
    M(:,:,i) = U(4*i-3:4*i,:);
    [u,~,v]=svd(M(1:3,1:3,i));
    M(1:3,1:3,i) = u*diag([1,1,det(u*v')])*v';
    
    M(:,4,i) = M(:,4,i)./M(4,4,i);
    M(4,1:3,i) = 0;
    
    R(:,:,i) = M(1:3,1:3,i);
    t(:,i) = M(1:3,4,i);
    
end


end