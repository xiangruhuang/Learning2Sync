% Implementation of the transformation synchronisation method for
% synchronising the set of pairwise relative transformations such that they
% are transitively consistent. For more information, see:
%
% F. Bernard, J. Thunberg, P. Gemmar, F. Hertel, A. Husch, J. Goncalves: A
% Solution for Multi-Alignment by Transformation Synchronisation.  IEEE
% Conference on Computer Vision and Pattern Recognition (CVPR). 2015
%
% INPUT:
% 
% Tcell: K-by-K cell containing invertible D-by-D transformation matrices,
% where item Tcell{i,j} denotes the transformation from i to j.
% Post-multiplication is used, i.e. the 1-by-D vector x from coordinate
% system i is transformed to coordinate system j using x*Tcell{i,j}
%
% transformationType: 'linear', 'affine', 'similarity',
% 'similarityNoReflection', 'euclidean', 'rigid', 'translationOnly',
% 'orthogonal'
% 
% stableTrans: if true (default), use a numerically stable computation when
% dealing with large translations, otherwise, use the naive
% implementation
%
% OUTPUT:
%  TsyncedCell: K-by-K cell containing the synchronised D-by-D
%  transformation matrices.
%
% EXAMPLE USAGE:
% see files demo.m and experimentsNoisyTransformations.m
%
% Author: Florian Bernard, f.bernardpi [at] gmail [dot] com
%

function TsyncedCell = ...
	synchroniseTransformations(Tcell, transformationType, stableTransFlag)

	if ( ~exist('stableTransFlag', 'var') )
		if ( strcmp(transformationType, 'linear') )
			stableTransFlag = false;
		else
			stableTransFlag = true;
		end
	end
	
	if ( strcmpi(transformationType, 'linear') || ...
			strcmpi(transformationType, 'orthogonal') ) % do not consider homogeneous transformation matrices
		isLinear = true;
		D = size(Tcell{1,1},1); % data dimension
		Dhom = D; % homogeneous dimension equals data dimension
	else % homogeneous transformation matrices
		isLinear = false;
		Dhom = size(Tcell{1,1},1); % homogeneous dimension
		D = Dhom-1; % data dimension 
	end
	
	K = size(Tcell,1); % number of objects with pairwise transformations
	
	if ( stableTransFlag && ~isLinear ) 
		% make sure the constant homogeneous part is exactly [0 ... 0 1]
		% in floating point representation22
		% (necessary for numerically stable approach)
		for i=1:K
			for j=1:K
				Tcell{i,j}(:,end) = zeros(Dhom,1);
				Tcell{i,j}(end,end) = 1;
			end
		end
	end
	

	% construct matrices
	W = cell2mat(Tcell);
	Z = W - K*eye(K*Dhom);

	% find least-squares approximation of the nullspace of Z
	if ( ~isLinear && stableTransFlag ) % numerically stable implementation using block-decomposition of Z
		[forwardPerm,~,r,s] = dmperm(Z);
		P = eye(Dhom*K);
		P = P(:,forwardPerm);
		PTZP = P'*Z*P;

		Z11 = PTZP(r(1):r(2)-1,s(1):s(2)-1);
		Z22 = PTZP(r(2):r(3)-1,s(2):s(3)-1);
		Z12 = PTZP(r(1):r(2)-1,s(2):s(3)-1);

		% find nullspace of Z22
		[~,~,V2] = svd(Z22);
		V2 = V2(:,(end-D+1):end);

		% remove ones(K,1) from nullspace by adding the row ones(1,K) to
		% A = [Z11] and zeros(1,D) to b, and solve Ax = b
		V1 = [Z11;ones(1,K)]\[-Z12*V2; zeros(1,D)];

		U1 = P*[V1;V2];
	else
		if ( ~isLinear ) 
			% remove homoegeneous parts [0 ... 0 1]' from nullspace of Z by
			% adding it as row to Z
			zeroOneRowVec = repmat([zeros(1,D) 1], 1, K);
			Z = [Z; zeroOneRowVec];
		end
		[~,~,U1tmp] = svd(Z);

		% extract last D columns to obtain U1
		U1 = U1tmp(:,(end-D+1):end);
	end

	if ( ~isLinear ) % add constant homogeneous part to U1
		U1 = [U1 repmat([zeros(D,1); 1], K, 1)];
	end
	% reconstruct unnoisy similarity transformation matrices
	TsyncedCell = cell(K,K);

	% normalise, such that, w.l.o.g., the first transformation is identity
	idx = 1; 
	unnoisyT1star = U1((idx-1)*Dhom+1:idx*Dhom,1:Dhom);
	U1normalised = U1/unnoisyT1star;

	% project transformations onto the desired transformation type
	for k=1:K
		Tij = U1normalised((k-1)*Dhom+1:k*Dhom,1:Dhom);

		switch transformationType
			case 'affine'
				A = Tij(1:D,1:D);
			case 'linear'
				A = Tij(1:D,1:D);
			case 'similarity'
				[Utmp,SigmaTmp,Vtmp] = svd(Tij(1:D,1:D));
				Q = Utmp*Vtmp';

				s = (prod(diag(abs(SigmaTmp))))^(1/D); % <=> abs(det(A))^(1/D)
				% s = trace(SigmaTmp)/D; 
				A = s*Q;
			case 'similarityNoReflection'
				[Utmp,SigmaTmp,Vtmp] = svd(Tij(1:D,1:D));
				Q = Utmp*diag([ones(1,D-1) det(Vtmp'*Utmp)])*Vtmp';

				s = (prod(diag(abs(SigmaTmp))))^(1/D); % <=> abs(det(A))^(1/D)
				% s = trace(SigmaTmp)/D; 
				A = s*Q;
			case {'orthogonal', 'euclidean'}
				[Utmp,~,Vtmp] = svd(Tij(1:D,1:D));
				Q = Utmp*Vtmp';
				A = Q;
			case 'rigid'
				[Utmp,~,Vtmp] = svd(Tij(1:D,1:D));
				Q = Utmp*diag([ones(1,D-1) det(Vtmp*Utmp)])*Vtmp';
				A = Q;
			case 'translationOnly'
				A = eye(D);
		end

		T = zeros(Dhom,Dhom);
		T(1:D,1:D) = A;
		if ( ~isLinear )
			t = Tij(end,1:D);

			T(end,end) = 1;
			T(end,1:D) = t;
		end

		TsyncedCell{k,idx} = T;
		TsyncedCell{idx,k} = inv(T);
	end
	
	% recover all K^2 pairwise transformations that are transitively
	% consistent (synchronised)
	for i=1:K
		for j=1:K
			Tiref = TsyncedCell{i,idx};
			Trefj = TsyncedCell{idx,j};

			TsyncedCell{i,j} = Tiref*Trefj;
		end
	end
end
