

function [res]=residuals_EIG_IRLS(I,J,X,M) %#codegen

% Compute residuals between absolute and relative motions (considering only
% rotation components) with Frobenus norm

nedges=length(I);
res=zeros(nedges,1);

for k=1:nedges
    i=I(k); j=J(k);
    res(k)=norm(M(1:3,1:3,i)*M(1:3,1:3,j)'-X(4*i-3:4*i-1,4*j-3:4*j-1),'fro');
end

end

% codegen residuals_EIG_IRLS -args {coder.typeof(0, [Inf, 1]),coder.typeof(0, [Inf, 1]),coder.typeof(0, [Inf, Inf]),coder.typeof(0,[4,4,Inf])}