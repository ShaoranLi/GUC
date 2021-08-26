function [ y, sqrtm_y ] = corr_matrix( M, L,rho, rolloff )
%input x is temp
x=eye(M);
y=diag(diag(x));
for i=1:size(x,1)
    for j=1:L
        if(i+j>0 && i+j <=size(x,2))
            y(i,i+j)=sqrt(x(i,i)*x(i+j,i+j))*rho^((1-rolloff)+rolloff*j);
            y(i+j,i)=y(i,i+j);
        end
    end
end
sqrtm_y=sqrtm(y);
for i=1:size(y,1)
    sqrtm_y(i,:)=sqrtm_y(i,:)/norm(sqrtm_y(i,:));
end


