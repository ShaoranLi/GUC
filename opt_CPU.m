function [obj_opt, time,mytime_Gurobi,my_power] = opt_CPU(Paras)
epsilon=Paras.epsilon;
num_sub=Paras.num_sub;
num_rb=Paras.num_rb;
num_user=Paras.num_user;
num_PU=Paras.num_PU;
MIPGap_req=Paras.MIPGap_req;
Imax=Paras.Imax;
mu=Paras.mu;
V=Paras.V;
h=Paras.h;
max_power_extend=Paras.max_power_extend;
max_power=Paras.max_power;
w=Paras.w;
%% Optimization Modeling by SOCP in cvx
%scheduling decision
A1=zeros(num_sub,num_rb);
for i=1:num_sub
    A1(i,i:num_sub:num_rb)=1;
end
%allocate power
A2=zeros(num_user,num_rb);
for i=1:num_user
    A2(i,(i-1)*num_sub+1:i*num_sub)=1;
end
%power and decision
%SOCP
%linearization
beta=[0:0.005:0.2 0.25:0.05:1 ]'; % segment of the whole transmission power
% beta=0:0.02:1;
cvx_clear
cvx_begin quiet
cvx_solver Gurobi_2
cvx_precision low
%cvx_solver_settings( 'dumpfile', 'Test' )
cvx_solver_settings( 'MIPGap', MIPGap_req )
cvx_solver_settings('TIMELIMIT', 100)
variable x(num_rb) binary
variable p(num_rb) nonnegative
variable c(num_rb)
maximize(w'*c )
subject to
A1*x<=1
A2*p<=max_power'
p <= x.*max_power_extend
for i=1:num_PU
    sqrt((1-epsilon)/epsilon)*norm(reshape(V(i,:,:),num_rb,num_rb)*p) + mu(i,:)*p <= Imax
end
for i=1:num_rb
    tempp=beta*max_power_extend(i);
    slope=h(i)./(1+h(i)*tempp);
    fix_b=log(1+h(i)*tempp);
    c(i) <= slope.*(p(i)-tempp)+fix_b;
end
%c <= log(1+h.*p/1e3);
tic
myruntime=cvx_end;
time=toc;
mytime_Gurobi=myruntime;
obj_opt=cvx_optval;
%w'*log(1+h.*p/1e3)-obj_opt
my_power=p;

