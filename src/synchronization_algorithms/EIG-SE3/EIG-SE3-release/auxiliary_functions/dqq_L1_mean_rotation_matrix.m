function [ Rmean ] = dqq_L1_mean_rotation_matrix( R,iter_max)
%DQQ_L1_MEAN_ROTATION_MATRIX Summary of this function goes here
%   This function calculate the mean rotation matrix of the given 3*3*n R matrix
%	under L1 norm by Weiszfeld algorithm.
%	Please refer to the paper: 
%	'L1 rotation averaging using the Weiszfeld algorithm', Richard Hartley, etc, CVPR 2011
%	for details.

S(:,:,1) = dqq_rotation_quaternion_initialization( R );
nofR=size(R);

iter=1;


while isreal(S(:,:,iter)) && iter<=iter_max;
    
    iter=iter+1;
    sum_vmatrix_normed(:,:,iter)=zeros(3,3);
    
    
    conta=0;
    for j=1:nofR(3)
        
        
%        vmatrix(:,:,j)=logm(R(:,:,j)*(S(:,:,iter-1))^(-1));
        
        [theta,u] = axis_angle(R(:,:,j)*S(:,:,iter-1)');
        vmatrix(:,:,j)=star(theta*u);

        
        vmatrix_normed(:,:,j)=vmatrix(:,:,j)/norm(vmatrix(:,:,j));
        sum_vmatrix_normed(:,:,iter)=sum_vmatrix_normed(:,:,iter)+vmatrix_normed(:,:,j);
        inv_norm_vmatrix(j)=1/norm(vmatrix(:,:,j));
        
        if norm(vmatrix(:,:,j))<1e-10
            conta=conta+1;
        end
        
    end
    

    if conta==nofR(3)
        break
    end
    
    delta(:,:,iter)=sum_vmatrix_normed(:,:,iter)/sum(inv_norm_vmatrix);
    
    
    
%     S(:,:,iter)=expm(delta(:,:,iter))*S(:,:,iter-1);
    
    u=inv_star(delta(:,:,iter));
    theta=norm(u);
    u=u/theta;
    S(:,:,iter) = inv_axis_angle(theta,u)*S(:,:,iter-1);
    

    
%     if abs(1-det(S(:,:,iter)*S(:,:,iter)'))<10^(-10)
%         disp('break')
%         break;
%     end
    
    
    
end

Rmean=S(:,:,iter-1);

end

