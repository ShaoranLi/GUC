function [ diagL ] = get_L_diag( x,L )
sizex=size(x,1);
diagL=zeros(L+1,sizex);
for i=1:L+1
    if(i>1)
        diagL(i,:)=[diag(x,i-1)' zeros(1,i-1)]*2;
    else
        diagL(i,:)=[diag(x,i-1)' zeros(1,i-1)];
    end
end

