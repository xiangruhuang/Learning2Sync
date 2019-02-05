% Implementation of the H-matrix based transformation synchronisation
% method for synchronising the set of relative transformations such that
% they are transitively consistent. The method generalises the work
% described in [1] since it can deal with missing transformations, i.e. not
% all K^2 pairwise transformations need to be available. For details, refer
% [2].

%
% INPUT:
%  Tcell: K-by-K cell containing invertible D-by-D transformation matrices,
%  where item Tcell{i,j} denotes the transformation from i to j.
%  Post-multiplication is used, i.e. the 1-by-D vector x from coordinate
%  system i is transformed to coordinate system j using x*Tcell{i,j}. If
%  the transformation Tcell{i,j} is missing, it must be equals to
%  zeros(D,D).
%
%  transformationType: 'linear', 'affine', 'similarity',
%  'similarityNoReflection', 'euclidean', 'rigid', 'translationOnly',
%  'orthogonal'
% 
% OUTPUT:
%  GsyncedCell: K-by-K cell containing the synchronised D-by-D
%  transformation matrices.
%
%  GiCell: K-by-1 cell, where the transformation matrix at GiCell{i}
%  denotes the transformation  from coordinate system i to the reference
%  system (denoted by \star, refer [1] for details). The transformation
%  belongs to the class of the transformationType specified as input, i.e.
%  it is projected onto the specified set. 
%
%  GiInvNonprojectedCell: K-by-1
%  cell, where the transformation matrix at GiInvNonprojectedCell{i}
%  denotes the transformation  from the reference system (denoted by \star,
%  refer [1] for details) to coordinate system i. The transformation does
%  not necessarily belong to the class of the transformationType specified
%  as input, i.e. it is NOT projected onto the specified set.
%
% EXAMPLE USAGE: see files demo2.m and experimentsNoisyTransformations.m
%
% Author: Florian Bernard, f.bernardpi [at] gmail [dot] com
%
% [1] F. Bernard, J. Thunberg, P. Gemmar, F. Hertel, A. Husch, J.
% Goncalves: A Solution for Multi-Alignment by Transformation
% Synchronisation.  IEEE Conference on Computer Vision and Pattern
% Recognition (CVPR). 2015
%
% [2] J. Thunberg, F. Bernard, J. Goncalves: On Transitive Consistency for
% Linear Invertible Transformations between Euclidean Coordinate Systems.

function [GsyncedCell, GiCell, GiInvNonprojectedCell] = ...
    synchroniseTransformationsH(TCell, transformationType)

    if ( strcmpi(transformationType, 'linear') || ...
			strcmpi(transformationType, 'orthogonal') )
        isLinear = true;
    else
        isLinear = false;
    end
    
    K = size(TCell,1);
    
    if ( isLinear )
        D = size(TCell{1,1},1);
        Dhom = D;
    else
        Dhom = size(TCell{1,1},1);
        D = Dhom-1;
    end
    
    W = cell2mat(TCell);
    
    % create adjacency matrix
    A = cell2mat(cellfun(@(x) any(x(:)), TCell, 'UniformOutput', false));
    
	% construct matrices Wbar, Z and Z2
	Wbar = W';
    % graph laplacian
%     L = diag(A*ones(K,1)) - A;
    
    Z = kron(diag(A*ones(K,1)), eye(Dhom)) - W;
	
	blkDiag = kron(eye(K),ones(Dhom,Dhom));
	Z2 = (Wbar*Wbar').*blkDiag - Wbar;
	
	% construct H matrix
	H = Z + Z2;
	
	% find D-dimensional null space of H matrix
	[~,~,v] = svd(H);
	
	% extract last D columns
	V = v(:,(end-(Dhom-1)):end);
	V1 = V(1:Dhom,1:Dhom);
	Vnormalised = V/V1;

    % reconstruct unnoisy similarity transformation matrices
    GiCell = cell(K,1);
    GiInvNonprojectedCell = cell(K,1);

    for k=1:K
        GiInv = Vnormalised((k-1)*Dhom+1:k*Dhom,1:Dhom);
		Gi = inv(GiInv);
		
        switch transformationType
            case 'affine'
                A = Gi(1:D,1:D);
            case 'linear'
                A = Gi(1:D,1:D);
            case 'similarity'
                [Utmp,SigmaTmp,Vtmp] = svd(Gi(1:D,1:D));
                Q = Utmp*Vtmp';
                
                s = (prod(diag(SigmaTmp)))^(1/D); % <=> abs(det(A))^(1/D)
                A = s*Q; 
            case {'orthogonal', 'euclidean'}
                [Utmp,~,Vtmp] = svd(Gi(1:D,1:D));
                Q = Utmp*Vtmp';
                A = Q;
            case 'rigid'
                [Utmp,~,Vtmp] = svd(Gi(1:D,1:D));
                Q = Utmp*diag([ones(1,D-1) det(Vtmp*Utmp)])*Vtmp';
                A = Q;
            case 'translationOnly'
                A = eye(D);
        end

        T = zeros(Dhom,Dhom);
        T(1:D,1:D) = A;
        if ( ~isLinear )
            t = Gi(end,1:D);
            
            T(end,end) = 1;
            T(end,1:D) = t;
		end

		GiInvNonprojectedCell{k} = GiInv.*(sqrt(K));
        GiCell{k} = T;
	end
	
	% reconstruct (transitively consistent) pairwise transformations
	GsyncedCell = cell(K,K);
	for i=1:K
		for j=1:K
			Ti = GiCell{i};
			Tj = GiCell{j};
			
			Tij = inv(Ti)*Tj;
			GsyncedCell{i,j} = Tij;
		end
	end
end
