% chapter 9.2.1 using MCMC Gibbs sampler
%
% NPQ $2019.11.02$

clear
x1 = [0,0,0,0,0,0,1,1,1,1,1,1]';     % 0/1: running/aerobic
x2 = [23,22,22,25,27,20,31,23,27,28,22,24]'; % age
y = [-0.87,-10.74,-3.27,-1.97,7.50,-7.25,17.05,4.96,10.40,11.05,0.26,2.51]';
X = [ones(size(x1)), x1, x2, x1.*x2];
[n,p] = size(X);

%% OLS
beta_ols = inv(X'*X)*(X'*y)             % coefficent
SSR = sum((y-X*beta_ols).^2)            % SSR of OLS
sigma2_ols = SSR/(n-p)                  % unbiased estimate of the std of the innovation term
beta_ols_std = sqrt(diag(inv(X'*X)*sigma2_ols))     % std of coefficients

%% MCMC for sigma2 and beta
% --- prior
nu0 = 1; sigma20 = 225;  % for sigma2 (innovation)
beta0 = beta_ols;        % mean for beta
Sigma20 = diag([150, 30, 6, 5].^2);    % covariance for beta's distribution

% --- start values
sigma2i = sigma2_ols;

% --- MCMC
nmcmc = 5000;
sigma2_mcmc = nan(nmcmc,1);
beta_mcmc = nan(nmcmc,p);

for ii=1:nmcmc
    % sample beta
    V_beta = inv(inv(Sigma20) + X'*X/sigma2i);      % pxp
    E_beta = V_beta*(inv(Sigma20)*beta0 + X'*y/sigma20);    % px1
    betai = mvnrnd(E_beta,V_beta);              % 1xp
    
    % sample sigam2
    sigma2i = 1./random('Gamma',(nu0+n)/2, 2/(nu0*sigma20+sum((y-X*betai(:)).^2)));
    
    % collect
    sigma2_mcmc(ii) = sigma2i;
    beta_mcmc(ii,:) = betai;
end

beta2_list = linspace(-85,130,1000);
pd_beta2 = fitdist(beta_mcmc(:,2),'Kernel','Kernel','normal');    % use normal kenerl
pdf_beta2_kernel = pdf(pd_beta2,beta2_list);

beta4_list = linspace(-5,5,1000);
pd_beta4 = fitdist(beta_mcmc(:,4),'Kernel','Kernel','normal');    % use normal kenerl
pdf_beta4_kernel = pdf(pd_beta4,beta4_list);

figure
subplot(1,3,1);
hold on; box on;
plot(beta2_list,pdf_beta2_kernel)
plot([0,0],ylim)
xlabel('\beta_2');

subplot(1,3,2);
hold on; box on;
plot(beta4_list,pdf_beta4_kernel)
plot([0,0],ylim)
xlabel('\beta_4');

subplot(1,3,3);
scatter(beta_mcmc(:,2),beta_mcmc(:,4),'.');
box on;
set(gca,'xlim',[-60,60],'ylim',[-2,2]);
xlabel('\beta_2');
ylabel('\beta_4');

