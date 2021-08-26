function [ h_ind_rand, h_corr_rand, mu_ind, mu_corr, sig_ind, sig_corr, g_ind_rand, g_corr_rand] = gen_corr_ray( L, rho,num_sub,num_user,max_sample)
num_rb=num_sub*num_user;
h_channel=1/sqrt(2)*(mvnrnd(zeros(num_rb,1),eye(num_rb),1)+1j*mvnrnd(zeros(num_rb,1),eye(num_rb),1)); %original Gaussian chl
g_channel=1/sqrt(2)*(mvnrnd(zeros(num_rb,1),eye(num_rb),max_sample)+1j*mvnrnd(zeros(num_rb,1),eye(num_rb),max_sample)); %original Gaussian chl
% Independent
h_ind_rand=abs(h_channel).^2; %rayleigh
g_ind_rand=abs(g_channel).^2;
mu_ind=mean(g_ind_rand);
sig_ind=cov(g_ind_rand).*(eye(num_rb));
% Correlated
target_corr_piece=corr_matrix( num_sub, L,rho, 1);
target_corr=zeros(num_rb,num_rb);
for i=1:num_user
    target_corr((i-1)*num_sub+1:i*num_sub,(i-1)*num_sub+1:i*num_sub)=target_corr_piece;
end
corr_mask=target_corr>0;
h_corr_rand=abs(h_channel*sqrtm(target_corr)).^2;
g_channel=g_channel*sqrtm(target_corr);
g_corr_rand=abs(g_channel).^2;
mu_corr=mean(g_corr_rand);
sig_corr=cov(g_corr_rand).*corr_mask;
end

