function  [para_obj,power,time] = GUC_on_GPU(Paras)
%running GUC on CPU to verify correctness
epsilon=Paras.epsilon;
num_sub=Paras.num_sub;
num_rb=Paras.num_rb;
num_user=Paras.num_user;
num_PU=Paras.num_PU;
Imax=Paras.Imax;
mu=Paras.mu;
h=Paras.h;
sig=Paras.sig;
max_power=Paras.max_power;
w=Paras.w;
num_promising=Paras.num_promising;
L=Paras.corrL;
interf=zeros((L+2)*num_PU,num_rb);
temp_cri=zeros(1,num_rb);
for i=1:num_PU
    interf((i-1)*(L+2)+1:i*(L+2)-1,:)=get_L_diag(abs(reshape(sig(i,:,:),num_rb,num_rb))*(1-epsilon)/epsilon,L);
    interf(i*(L+2),:)=mu(i,:);
    r_interf=sum(abs(reshape(sig(i,:,:),num_rb,num_rb))*(1-epsilon)/epsilon,1);
    temp_cri=temp_cri+sqrt(r_interf)+mu(i,:);
end
%Write data to M2C.txt
M2C_data=[h';w';interf;mean(temp_cri,1)];
for i=1:num_PU*(L+2)+3
    M2C_data(i,:)=reshape(reshape(M2C_data(i,:),num_sub,num_user)',1,num_rb);
end
%Run on GPU
M2C = fopen(['M2C.txt'],'w');
format=[repmat('%f ',1,num_rb-1) '%f\n'];
fprintf(M2C,format,M2C_data');
fclose(M2C);
system(['Picocell' num2str(num_sub) '_' num2str(num_user) '_' num2str(L) '.exe']);
%system(['Picocell.exe']);
%Read data from C2M.txt
C2M = fopen('C2M.txt','r');
p_cpp=fscanf(C2M,'%f',[1 num_sub*2+3]);
fclose(C2M);
obj_cpp= M2C_data(2,1+p_cpp(3:num_sub+2))* log2(1+M2C_data(1,1+p_cpp(3:num_sub+2)).*p_cpp(num_sub+3:2*num_sub+2))';
para_obj=p_cpp(2)*log(2);
time=p_cpp(1);
for i_PU=1:num_PU
    power=zeros(num_rb,1);
    power(1+p_cpp(3:num_sub+2),1)=p_cpp(num_sub+3:2*num_sub+2);
    power=reshape(reshape(power,num_user,num_sub)',num_rb,1);
end

% 
% for i=1:num_PU
%     I(i)=sqrt(M2C_data(2*i+1,1+p_cpp(3:num_sub+2)).*p_cpp(num_sub+3:2*num_sub+2)*p_cpp(num_sub+3:2*num_sub+2)')+...
%         M2C_data(2*i+2,1+p_cpp(3:num_sub+2))*p_cpp(num_sub+3:2*num_sub+2)';
% end

