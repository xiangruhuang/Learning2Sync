% Add Gaussian noise to transformations as described in
%
% F. Bernard, J. Thunberg, P. Gemmar, F. Hertel, A. Husch, J. Goncalves: A
% Solution for Multi-Alignment by Transformation Synchronisation.  IEEE
% Conference on Computer Vision and Pattern Recognition (CVPR). 2015
%
% INPUT: 
% transformations: cell containing transformations
%
% sigma: standard deviation of Gaussian noise
%
% transformationType: 'linear', 'affine', 'similarity',
% 'similarityNoReflection', 'euclidean', 'rigid', 'translationOnly'
%
% OUTPUT:
% noisyTransformations: cell containing noisy variants of the input
% transformations
%
% Author: Florian Bernard, f.bernardpi [at] gmail [dot] com
%


function noisyTransformations = ...
    addGaussianNoiseToTransformations(transformations, sigma, transformationType)
% add noise to transformations
    if ( ~iscell(transformations) )
        noisyTransformations = {transformations};
    else
        noisyTransformations = transformations;
    end

    dMatrix = size(noisyTransformations{1},1);
    
    for i=1:numel(noisyTransformations)
        if ( numel(noisyTransformations) > 1 && ...
                size(transformations,1) == size(transformations,2) )
            % if we have a square cell of transformations, do not add noise
            % to diagonal elements, as these should remain identity
            [r,c] = ind2sub(size(transformations), i);
            if ( r == c )
                continue;
            end
        end
        
        if ( strcmpi(transformationType, 'linear') ) % d-by-d matrix
            noisyTransformations{i} = ...
                noisyTransformations{i} + randn(dMatrix,dMatrix)*sigma;
        else
            T = noisyTransformations{i};
			T(1:dMatrix,1:dMatrix-1) = T(1:dMatrix,1:dMatrix-1) + ...
                randn(dMatrix,dMatrix-1)*sigma;
            noisyTransformations{i} = T;
		end
    end
    if ( ~iscell(transformations) )
        noisyTransformations = noisyTransformations{1};
    end
end
