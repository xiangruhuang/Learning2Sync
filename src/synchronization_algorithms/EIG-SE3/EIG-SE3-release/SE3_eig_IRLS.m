

function [M,R,T,k,W] = SE3_eig_IRLS(X,A,nmax,thresh,method,use_mex,use_sparse)
%
% Estimate global rigid motion by averaging in SE(3)
% Inspired  the "eigenvalue" approach in SO(3) by Amit Singer (2011)
% Note: each relative/absolute motion is represented as a 4x4 matrix
%
% INPUT 
% X = (4n x 4n) matrix with pairwise motions
% A = adjacency matrix of the view-graph
% nmax = maximum number of iterations of IRLS
% thresh = thrsehold on relative error to check convergence of IRLS
% method = 'top' -> compute top eigenvectors
% method = 'null' -> compute null-space
% use_mex = true -> use mex function to compute weights (which speeds up
% the for loop)
% use_mex = false -> do not use mex function to compute weights
% use_sparse = true -> use sparse Matlab solvers (eigs/svds)
% use_sparse = false -> do not use sparse Matlab solvers (eig/svd)
%
% OUTPUT
% M = 4x4xn matrix with absolute motions
% R = 3x3xn matrix with absolute rotations
% T = 3xn matrix with absolute translations
% k = number of effective IRLS iterations
% W = final IRLS weights
%
% Author: Federica Arrigoni, 2015
% Reference: 
% F. Arrigoni, B. Rossi, A. Fusiello, Spectral Synchronization of Multiple
% Views in SE(3), SIAM Journal on Imaging Sciences, 2016.
%

n = size(X,1)/4; % number of local reference frames (cameras/sensors/scans)

A=sparse(A);
X=sparse(X);

% Initialize parameters
W=A; % current weights
W_old = A; % old weights
k=1; % number of iterations
deltaW=2*thresh; % increment on weights

while k<=nmax && deltaW>thresh
    
    % Perform eigenvalue decomposition and projection onto SE(3)
    [M,R,T] = SE3_eig(X,W,method,use_sparse);
    
    % Update weights
    W=update_weights(X,M,A,'cauchy',use_mex);
    
    % compute the increment on weights
    deltaW = norm(W_old -W,'fro')/(norm(W,'fro')*n);
    W_old = W;
    
    k=k+1;
     
end

end




function [W]=update_weights(X,M,A,weight_fun,use_mex)

ncams=size(A,1);
[I,J]=find(triu(A,1));

if use_mex
    [res]=residuals_EIG_IRLS_mex(I,J,full(X),M);
else
    [res]=residuals_EIG_IRLS(I,J,X,M);
end

theta=2; % SIIMS
% theta=2.385; % default

th =  theta/0.6745*mad(res(:),1);

if strcmp(weight_fun,'huber')
    weights=1.*(res<=th)+th./res.*(res>th);
elseif strcmp(weight_fun,'cauchy')
    weights=1./(1+(res/th).^2);
elseif strcmp(weight_fun,'bisquare')
    weights = (1-(res/th).^2).^2;
    weights(res>=th)=0;
else
    error('')
end

W=sparse([I;J;(1:ncams)'],[J;I;(1:ncams)'],[weights;weights;max(weights)*ones(ncams,1)],ncams,ncams);

end



