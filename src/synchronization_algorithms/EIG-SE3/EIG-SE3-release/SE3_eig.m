
function [M,R,t] = SE3_eig(X,A,method,use_sparse)
% Estimate global rigid motion by averagning in SE(3)
% Inspired  the "eigenvalue" approach in SO(3) by Arie-Nachimson et al.
%
% input: 
% X = (4n x 4n) matrix with relative transformations
% A = n x n Adjacency matrix of the graph
% method = 'top' (spectral solution) or 'null' (null-space solution)
% use_sparse = 1 (use sparse solvers - RECOMMENDED) 
% use_sparse = 0 (do not use sparse solvers) 
%
% ouptut:
% M = (4 x 4 x n) matrix with absolute transformations
% R = (3 x 3 x n) matrix with absolute rotations
% t = (3 x n) matrix with absolute translations
%
% Author: F. Arrigoni, A. Fusiello, 2015
% Reference: F. Arrigoni, B. Rossi, A. Fusiello. "Spectral synchronization
% of multiple views in SE(3)" SIAM Journal on Imaging Sciences, 2016

n = size(X,1)/4;

A=sparse(A);
X=sparse(X);

X = X.*kron(A,ones(4));


if strcmp(method,'top') % Top eigenvectors
    
    %D_inv = kron(diag(1./sum(A,2)),eye(4)); % e' piu' lento
    D_inv = kron(diag(sum(A,2).^(-1)),eye(4));
    
    if use_sparse
        
        [U,~]=eigs(D_inv*X,4);        

    else
        
        [U,D] = eig(full(D_inv*X));
        [~,I] = sort(abs(diag(D)),'descend');
        U = U(:, I);
        U = U(:,1:4);
        
    end
    
    
    % compute a linear combination of the columns of U in order to have 
    % [0 0 0 1] in rows multiple of 4
    B=U(4:4:end,:);
    [~,~,alpha]=svd(B);
    alpha=alpha(:,end-2:end);
    beta=B\ones(n,1);
    U=[U*alpha U*beta];
    
    % force U to be real
    U=real(U);
    
    
elseif strcmp(method,'null') % null-space
    
    D=kron(diag(sum(A,2)),eye(4));

    if use_sparse
        
        [~,~,U]=svds(X-D,4,0); 
        
    else
        
        [~,~,U]=svd(full(X-D));
        U=U(:,end-3:end);
        
    end
    
    % compute a linear combination of the columns of U in order to have 
    % [0 0 0 1] in rows multiple of 4
    B=U(4:4:end,:);
    [~,~,alpha]=svd(B);
    alpha=alpha(:,end-2:end);
    beta=B\ones(n,1);
    U=[U*alpha U*beta];
    
else
    error('unknown method: top|null')
end


% permute first 3 columns so that det >0
for I = perms(1:3)'
    J = [I', 4]';
    U = U(:, J);
    if  det(U(1:3,1:3)) > 0
        break
    end
end

% projection onto SE(3)
[M,R,t]=projSE3(U,n);


end
