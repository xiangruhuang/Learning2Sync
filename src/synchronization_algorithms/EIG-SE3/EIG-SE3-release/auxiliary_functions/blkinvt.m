function B = blkinvt(A)

% Compute inverse of each block in A and concatenate them
n = size(A,2);

% n must divide the number of rows of A
if mod(size(A,1),n) ~=0 
    error('')
end

B=[];

for i = 1:size(A,1)/n
   
   B= [B, inv(A(n*(i-1)+1:n*i,:))];

end
   