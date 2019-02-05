function  S = star(x)
%STAR  Returns the  matrix S s.t.  S*y=0 iff x and y are collinear
%
%   In other words, S is s.t. ker(S) = x
%   If lenght(x)=3 returns the skew-symmetric matrix S s.t. cross(x,y) = S*y
%
%   Named after the Hodge star operator, to which it is somehow related. 

% Author: A. Fusiello

if size(x,2) ~=1 x = x'; end

if (size(x,2) ~=1)
    error('Argument must be a vector');
end

n = length(x);
S =  [];

for i=1:n
    S = [S;
        zeros(n-i, i-1), -x(i+1:end), x(i) * eye(n-i)];
end

% if n==3 S is the skew-symmetric matrix related to x up to: exchanging 1st
% and 3rd rows and  changing the sign of the 2nd row
if n == 3
    t = S(3,:); S(3,:)=S(1,:); S(1,:)=t;
    S(2,:)=-S(2,:);
end

% Old implementation worked only for 3d vectors

%STAR  Returns the skew-symmetric matrix X s.t. cross(x,y) = X*y
%
%   Returns the skew-symmetric matrix X s.t. cross(x,y) = X*y
%   X is s.t. ker(X) = x
% if (length(x) ~= 3)
%     error('Vector must be  3-dimensional');
% end
%
%
%
% X=[   0    -x(3)  x(2)
%      x(3)    0   -x(1)
%     -x(2)  x(1)   0   ];


