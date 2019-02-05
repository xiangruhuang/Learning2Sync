% Function to compute transitive error of a set of pairwise transformations.
%
% INPUT:
% Tcell: K-by-K cell containing invertible D-by-D transformation matrices,
% where item Tcell{i,j} denotes the transformation from i to j.
% Post-multiplication is used, i.e. the 1-by-D vector x from coordinate
% system i is transformed to coordinate system j using x*Tcell{i,j}
%
% OUTPUT:
% transitiveError: Scalar value for transitive error
%
% Author: Florian Bernard, f.bernardpi [at] gmail [dot] com
%

function transitiveError = transitiveErrorOfPairwiseTransformations(Tcell)
    k = size(Tcell,1);

    transitiveError = 0; 
    for star=1:k
        for i=1:k
            for j=1:k
                Tistar = Tcell{i,star};
                Tstarj = Tcell{star,j};
                
                TijDirect = Tcell{i,j};
               
                transitiveError = transitiveError + ...
                    norm(Tistar*Tstarj-TijDirect, 'fro');
            end
        end
    end
    transitiveError = transitiveError/(k^3); % average
end