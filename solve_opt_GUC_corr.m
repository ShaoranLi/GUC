opt_GUC_corr(32,20,5,6,[1 1],'topo_PU_inside.mat',45);

%real cpu time \appro act -0.3
       save(['GPU_' num2str(num_sub) '_' num2str(num_user) '_' num2str(outputfile) 'final.mat'],...
            't1','t2','t4','t5','t6','t8','K1','K2','K4','K5','K6','K8','vio1','vio3','vio4','vio6');
%             save(['GPU_' num2str(num_sub) '_' num2str(num_user) '_' num2str(outputfile) '60limit.mat'],...
%               't1','t2','t5','t6','K1','K2','K5','K6','vio1','vio4');

function [] = opt_GUC_corr(num_sub,num_user,num_PU,corrL,opt_solve_flag,filename,outputfile)
N=100;
%generate problme setting
num_rb=num_sub*num_user;
max_sample=10000;
num_promising=2;
lossto_PU_extend=zeros(num_rb,num_PU);
ep_pol=[0.01 0.05 0.1 0.2 0.3 0.4 0.5];
I_pool=3;% e-7 mW
% pd = makedist('Nakagami',2,1);
% pd = makedist('rayl',1/sqrt(2));
load(filename);
dis_SU=dis_SU_full(:,1:num_user);
dis_PU=dis_PU_full(:,1:num_user);
weight=weight_full(1:num_user);
weight=weight/sum(weight); %normalized
w=reshape(repmat(weight,num_sub,1),num_rb,1);
inter_micro_SU=10^((46-128.1-37.6*log10(0.4))/10)*1e-3;
lossin_SU=10.^(3.8+3*log10(dis_SU))*(inter_micro_SU+1e-10); %noise 1e-7 mW
lossto_PU=10.^(3.8+3*log10(dis_PU));
for PU_i=1:num_PU
    lossto_PU_extend(:,PU_i)=reshape(repmat(lossto_PU(PU_i,:),num_sub,1),num_rb,1);
end
Imax=I_pool(1,:)/1; % Imax=3 -> 3e-7 mW
max_power=ones(1,num_user); % 0.1 W, times by 10
max_power_extend=reshape(repmat(max_power,num_sub,1),num_rb,1);
Paras.num_sub=num_sub;
Paras.num_rb=num_rb;
Paras.num_user=num_user;
Paras.num_PU=num_PU;
Paras.MIPGap_req=0.01;
Paras.Imax=Imax;
Paras.max_power_extend=max_power_extend;
Paras.max_power=max_power;
Paras.w=w;
Paras.num_promising=num_promising;

for sta_k=1:N
    [h_ind_rand, h_corr_rand, mu_ind, mu_corr, sig_ind, sig_corr, g_ind_rand, g_corr_rand]= gen_corr_ray( corrL, sqrt(0.7),num_sub,num_user,max_sample);
    
    %% Independent Channel
    %Transmission channel gains in the SU, divided by 10, power times by 10
    lossin_SU_extend=reshape(repmat(lossin_SU,num_sub,1),num_rb,1);
    h=h_ind_rand'./lossin_SU_extend/10;
    %h=1./lossin_SU_extend/10;
    %Intereference channel gains in the SU, times by 10^9
    %Statistics
    mu=zeros(num_PU,num_rb);
    sig=zeros(num_PU,num_rb,num_rb);
    V=zeros(num_PU,num_rb,num_rb);
    for PU_i=1:num_PU
        mu(PU_i,:)=mu_ind./lossto_PU_extend(:,PU_i)'*1e9;
        sig(PU_i,:,:)=sig_ind.*(1./lossto_PU_extend(:,PU_i)*1e9).^2;
        V(PU_i,:,:)=sqrtm(reshape(sig(PU_i,:,:),num_rb,num_rb));
    end
    disp(['sta_k=' num2str(sta_k) ' Independent']);
    for ep_i=1:length(ep_pol)
        epsilon=ep_pol(ep_i);
        
        Paras.epsilon=epsilon;
        Paras.mu=mu;
        Paras.sig=sig;
        Paras.V=V;
        Paras.h=h;
        Paras.corrL=0;
        if(opt_solve_flag(1)==1)
            % Results from CPU
            [obj_opt, time_CPU,mytime_Gurobi,my_power] = opt_CPU(Paras);
            t1(ep_i,sta_k)=time_CPU;
            t2(ep_i,sta_k)=mytime_Gurobi;
            K1(ep_i,sta_k)=obj_opt;
            K2(ep_i,sta_k)=w'*log(1+h.*my_power);
            for PU_i=1:num_PU
                %vio1(PU_i,ep_i,sta_k)=sum(reshape(g(PU_i,:,:),num_rb,max_sample)'*power>Imax)/max_sample;
                vio1(PU_i,ep_i,sta_k)=sum(g_ind_rand./lossto_PU_extend(:,PU_i)'*1e9*my_power > Imax)/max_sample;
            end
        else
            t1(ep_i,sta_k)=0;
            t2(ep_i,sta_k)=0;
            K1(ep_i,sta_k)=1;
            K2(ep_i,sta_k)=1;
            vio1(PU_i,ep_i,sta_k)=0;
        end
        
        [para_obj,power_GPU,time_GUC_GPU] = GUC_on_GPU(Paras);
        t4(ep_i,sta_k)=time_GUC_GPU;
        K4(ep_i,sta_k)=para_obj;
        for PU_i=1:num_PU
            vio3(PU_i,ep_i,sta_k)=sum(g_ind_rand./lossto_PU_extend(:,PU_i)'*1e9*power_GPU > Imax)/max_sample;
        end
    end
    
    %% Correlated Channel
    %Transmission channel gains in the SU, divided by 10, power times by 10
    lossin_SU_extend=reshape(repmat(lossin_SU,num_sub,1),num_rb,1);
    h=h_corr_rand'./lossin_SU_extend/10;
    %Intereference channel gains in the SU, times by 10^9
    %Statistics
    mu=zeros(num_PU,num_rb);
    sig=zeros(num_PU,num_rb,num_rb);
    V=zeros(num_PU,num_rb,num_rb);
    for PU_i=1:num_PU
        mu(PU_i,:)=mu_corr./lossto_PU_extend(:,PU_i)'*1e9;
        sig(PU_i,:,:)=sig_corr.*(1./lossto_PU_extend(:,PU_i)*1e9).^2;
        V(PU_i,:,:)=sqrtm(reshape(sig(PU_i,:,:),num_rb,num_rb));
    end
    disp(['sta_k=' num2str(sta_k) ' Correlated']);
    for ep_i=1:length(ep_pol)
        epsilon=ep_pol(ep_i);
        %display(['epsilon=' num2str(epsilon) ', sta_k=' num2str(sta_k)]);
        
        Paras.epsilon=epsilon;
        Paras.mu=mu;
        Paras.sig=sig;
        Paras.V=V;
        Paras.h=h;
        Paras.corrL=corrL;
        if(opt_solve_flag(2)==1)
            % Results from CPU
            [obj_opt, time_CPU,mytime_Gurobi,my_power] = opt_CPU(Paras);
            t5(ep_i,sta_k)=time_CPU;
            t6(ep_i,sta_k)=mytime_Gurobi;
            K5(ep_i,sta_k)=obj_opt;
            K6(ep_i,sta_k)=w'*log(1+h.*my_power);
            for PU_i=1:num_PU
                %vio1(PU_i,ep_i,sta_k)=sum(reshape(g(PU_i,:,:),num_rb,max_sample)'*power>Imax)/max_sample;
                vio4(PU_i,ep_i,sta_k)=sum(g_corr_rand./lossto_PU_extend(:,PU_i)'*1e9*my_power > Imax)/max_sample;
            end
        else
            t5(ep_i,sta_k)=0;
            t6(ep_i,sta_k)=0;
            K5(ep_i,sta_k)=1;
            K6(ep_i,sta_k)=1;
            vio4(PU_i,ep_i,sta_k)=0;
        end
        
        [para_obj,p_t,time_GUC_GPU] = GUC_on_GPU(Paras);
        t8(ep_i,sta_k)=time_GUC_GPU;
        K8(ep_i,sta_k)=para_obj;
        for PU_i=1:num_PU
            vio6(PU_i,ep_i,sta_k)=sum(g_corr_rand./lossto_PU_extend(:,PU_i)'*1e9*p_t > Imax)/max_sample;
        end
    end
    %}
    
    
    if(mod(sta_k,10)==0)
        save(['GPU_' num2str(num_sub) '_' num2str(num_user) '_' num2str(outputfile) 'final.mat'],...
            't1','t2','t4','t5','t6','t8','K1','K2','K4','K5','K6','K8','vio1','vio3','vio4','vio6');
        % save(['GPU_' num2str(num_sub) '_' num2str(num_user) '_' num2str(outputfile) '60limit.mat'],...
        %          't1','t2','t5','t6','K1','K2','K5','K6','vio1','vio4');
    end
end
end