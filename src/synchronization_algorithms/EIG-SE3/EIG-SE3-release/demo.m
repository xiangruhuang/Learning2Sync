
clear,clc,close all
addpath('./auxiliary_functions')

nviews = 100; % number of cameras
prob_out=0.3; % fraction of outliers
sigma_a=5; % noise (degrees)
sigma_t=0.05; % noise(translations)
fraction_missing = 0.5; % fraction of missing data


%% EIG parameters

use_mex=0;
use_sparse=1;
nmax_eig_irls=50;
thresh_eig_irls=1e-5;


%% generate ground truth

M_gt = zeros(4,4,nviews); % motions
R_gt = zeros(3,3,nviews); % rotations
T_gt = zeros(3,nviews); % translations
C_gt = zeros(3,nviews); % centres

for i = 1:nviews
    
    Ri=eul(randn(1,3)); % random Euler angles
    ci = randn(3,1); % random translation
    ti = -Ri*ci; 
    
    M_gt(:,:,i) = [Ri, ti; 0 0 0 1];
    R_gt(:,:,i) = Ri;
    T_gt(:,i) = ti;
    C_gt(:,i) = ci;
    
end

Y = reshape(permute(M_gt,[1,3,2]),[],4); % absolute motions
Yt = blkinvt(Y); % inverses of absolute motions
X_gt = Y * Yt; % relative motions

%% add noise on relative motions

X = X_gt;
for i=1:nviews
    for j=i+1:nviews
        
        % random axis
        r = rand(3,1)-0.5;
        if norm(r)~=0
            r=r/norm(r);
        end
        
        % small angle
        angle = randn()+sigma_a;
        angle=angle*pi/180;
        noise = [inv_axis_angle(angle,r), randn(3,1)*sigma_t; 0 0 0 1];
        
        X(4*i-3:4*i,4*j-3:4*j) = X(4*i-3:4*i,4*j-3:4*j) * noise;
        X(4*j-3:4*j,4*i-3:4*i) = inv( X(4*i-3:4*i,4*j-3:4*j) );
        
    end
end

%% generate the epipolar graph

if fraction_missing==0
    A=ones(nviews);
else
    n_conn=2;
    while n_conn~=1 % generate a new graph if the current one is not connected
        A=rand(nviews)>=fraction_missing;
        A=tril(A,-1);
        A=A+A'+eye(nviews); % symemtric Adjacency matrix
        [n_conn] = graphconncomp(sparse(A),'Directed', false);
    end
    
end

%% put missing blocks in correspondence of missing pairs

X = X.*kron(A,ones(4));

%% Generate the outlier graph

[I,J]=find(triu(A));
n_pairs=length(I);

n_conn=2;
while n_conn~=1
    
    W=rand(n_pairs,1)<prob_out;
    A_outliers=sparse(I,J,W,nviews,nviews);
    A_outliers=A_outliers+A_outliers'; % graph representing outlier rotations
    
    % graph representing inlier rotations: it must be connected
    A_inliers=not(A_outliers)&A; 
    [n_conn] = graphconncomp(sparse(A_inliers),'Directed', false);
end


%% Add outliers among relative motions

for i=1:nviews
    for j=i+1:nviews
        
        if A_outliers(i,j)==1
            
            [u,~,v]=svd(rand(3));
            Rij=u*diag([1 1 det(u*v')])*v';
            tij=rand(3,1);
            Mij=[Rij tij;0 0 0 1];
            
            X(4*i-3:4*i,4*j-3:4*j)=Mij;
            X(4*j-3:4*j,4*i-3:4*i)=inv(Mij);
            
        end
    end
end


%% Cmpute absolute motions: EIG-SE(3)+IRLS

tic
%[M_EIG,R_EIG,T_EIG] = SE3_eig(X,A,'top',use_sparse); % This is not robust to outliers
[M_EIG,R_EIG,T_EIG,iter] = SE3_eig_IRLS(X,A,nmax_eig_irls,thresh_eig_irls,'top',use_mex,use_sparse);
toc

[err_R_EIG,err_T_EIG]=error_R_T(R_EIG,R_gt,T_EIG,T_gt);
disp('Angular error (degrees) - translation error')

[mean(err_R_EIG),mean(err_T_EIG)]





