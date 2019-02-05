
function  x = inv_star(X)
%
% returns the vector x corresponding to the skew-symmetric matrix X
% X=[   0    -x(3)  x(2)
%      x(3)    0   -x(1)
%     -x(2)  x(1)   0   ];
%
% Author: Federica Arrigoni

if (nnz(X+X')~=0)
    warning('La matrice deve essere antisimmetrica ');
    X=(X-X')/2;
end

x=[X(3,2) X(1,3) X(2,1)]';

end
