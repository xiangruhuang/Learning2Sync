% Function to compute the error between two sets of transformations
%
% INPUT: 
% pairwiseProcTrans/gtPairwiseProcTrans: cells containing
% invertible D-by-D transformation matrices
%
% OUTPUT:
% errorVal: Error between both sets of transformations
%
% Author: Florian Bernard, f.bernardpi [at] gmail [dot] com
%

function errorVal = errorTransformations(pairwiseProcTrans, gtPairwiseProcTrans)
    if ( ~iscell(pairwiseProcTrans) )
        pairwiseProcTrans = {pairwiseProcTrans};
        gtPairwiseProcTrans = {gtPairwiseProcTrans};
    end

    errorVal = 0;
    for i=1:numel(gtPairwiseProcTrans)
        errorVal = errorVal + ...
            norm(pairwiseProcTrans{i} - gtPairwiseProcTrans{i}, 'fro').^2;
    end
    errorVal = errorVal/numel(gtPairwiseProcTrans);
end