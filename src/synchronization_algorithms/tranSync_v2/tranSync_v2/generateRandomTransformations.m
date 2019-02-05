% Generate "random" transformations according to the scheme described in 
%
% F. Bernard, J. Thunberg, P. Gemmar, F. Hertel, A. Husch, J. Goncalves: A
% Solution for Multi-Alignment by Transformation Synchronisation.  IEEE
% Conference on Computer Vision and Pattern Recognition (CVPR). 2015
%
% INPUT: 
% d: dimensionality
%
% k: number of transformations
%
% transformationType: 'linear', 'affine', 'similarity',
% 'similarityNoReflection', 'euclidean', 'rigid', 'translationOnly',
% 'orthogonal'
%
% n: number of points in reference shape

% OUTPUT:
% transformations: generated transformations
%
% refShape: reference shape that is generated for creating random
% transformations
%
% Author: Florian Bernard, f.bernardpi [at] gmail [dot] com
%


function [transformations, refShape] = ...
    generateRandomTransformations(d, k, transformationType, n)

	refShape = [];
	while ( rank(refShape) < d )
		% create random reference shape
		if ( exist('n', 'var') )
			refShape = rand(n,d);
		else
			refShape = rand(d,d);
		end
	end
    %% generate linear part of transformation first
    Tlinear = cell(1,k);
    
    for i=1:k
        T = nan;
        while ( any(isnan(T(:))) || any(isinf(T(:))) ) % avoid that per chance we generate a singular matrix
            % pick d+1 random points
            randPoints = rand(d,d);
            
            % find affine transformation that best aligns the random points to the
            % first d points in refShape
            T = randPoints \ refShape(1:d,:);
            
            Tlinear{i} = T;
        end
    end

    transformations = cell(1,k);
    if (strcmpi(transformationType, 'linear') || ...
			strcmpi(transformationType, 'orthogonal')) % only d-by-d matrix
        for i=1:k
            % SVD of Tlinear
            [U,~,V] = svd(Tlinear{i});
            
            Q = U*V';
			switch transformationType
				case 'linear'
					scalingFactor = 1+(rand(1)-0.5); % random number in ]0.5 ; 1.5[
					
					randNoise = eye(d) + randn(d,d)*0.1;
					A = scalingFactor*Q*randNoise;
				case 'orthogonal'
					A = Q;
			end
            transformations{i} = A;
        end
    else % homogeneous (d+1)-by-(d+1) matrix
        for i=1:k
            % SVD of Tlinear
            [U,~,V] = svd(Tlinear{i});
            
            % generate linear part/rotation/scalingFactor
            switch transformationType
                case 'affine' % (d+1)-by-(d+1) homogeneous matrix, with linear part and translation
                    Q = U*V';
                    scalingFactor = 1+(rand(1)-0.5); % random number in [0.5 ; 1.5]
                    
                    randNoise = eye(d) + randn(d,d)*0.1;
                    A = scalingFactor*Q*randNoise;
                case 'similarity' % (d+1)-by-(d+1) homogeneous matrix, isotropic scaling, rotation and translation
                    Q = U*V';
                    scalingFactor = 1+(rand(1)-0.5); % random number in [0.5 ; 1.5]
                    A = scalingFactor*Q;
                case 'similarityNoReflection' % (d+1)-by-(d+1) homogeneous matrix, isotropic scaling, rotation and translation
                    Q = U*diag([ones(1,d-1) det(V'*U)])*V';
                    scalingFactor = 1+(rand(1)-0.5); % random number in [0.5 ; 1.5]
                    A = scalingFactor*Q;
                case 'euclidean' % (d+1)-by-(d+1) homogeneous matrix, rotation and translation
                    Q = U*V';
                    A = Q;
                case 'rigid' % (d+1)-by-(d+1) homogeneous matrix, rotation, no reflection and translation
                    Q = U*diag([ones(1,d-1) det(V'*U)])*V';
                    A = Q;
                case 'translationOnly'
                    A = eye(d);
            end
            T = zeros(d+1,d+1);
            T(end,end) = 1;
            T(1:d,1:d) = A;
            
            % random translation
			translation = 5*(rand(1,d)-0.5); % random translation in [-2.5 ; 2.5]^(1 x d)
			
            T(end,1:d) = translation;
            
            transformations{i} = T;
        end
    end
end   