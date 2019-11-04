% chapter 8.4: identical variance for groups
%
% * Need multiESS from https://github.com/lacerbi/multiESS
% * autocorr requires Econometrics Toolbox
%
% NPQ $2019.11.02$

clear
s0 = readtable(fullfile(pwd,'data','nels.csv'),'ReadRowNames',true);
stats = grpstats(s0,'school',{'mean','var'});


%% plot data
figure
plot(s0.school,s0.mathscore,'b.');
xlabel('school');
ylabel('math score');
title('Figure 8.4');

figure
subplot(1,2,1);
histogram(stats.mean_mathscore);
xlabel('sample mean');
ylabel('frequency');
subplot(1,2,2);
scatter(stats.GroupCount,stats.mean_mathscore);
box on;
xlabel('sample size');
ylabel('sample mean');
sgtitle('Figure 8.5');

%% MCMC
% --- prior
nu0 = 1; sigma20 = 100; 
eta0 = 1; tau20 = 100;
mu0 = 50; gamma20 = 25;

% --- start values
thetai = stats.mean_mathscore;
sigma2i = mean(stats.var_mathscore);
mui = mean(thetai);
tau2i = var(thetai);

% --- mcmc
m = height(stats);      % # of groups
n = height(s0);         % # of all samples
nm = stats.GroupCount;  % # of samples in each group
nmcmc = 5000;
sigma2_mcmc = nan(nmcmc,1);
tau2_mcmc = nan(nmcmc,1);
mu_mcmc = nan(nmcmc,1);
theta_mcmc = nan(nmcmc,m);
for ii=1:nmcmc
    % sample new theta for each group
    thetai = random('Normal',(nm.*stats.mean_mathscore/sigma2i + mui/tau2i)./(nm/sigma2i+1/tau2i),sqrt(1./(nm/sigma2i+1/tau2i)));
    
    % sample sigma2
    all_ss = 0; % square sum of all
    for jj=1:m
        all_ss = all_ss + sum((s0.mathscore(s0.school == jj)-thetai(jj)).^2);
    end
    sigma2i = 1./random('Gamma',(nu0+n)/2, 2/(nu0*sigma20+ all_ss));
    
    % sample mu
    mui = random('Normal',(m*mean(thetai)/tau2i + mu0/gamma20)/(m/tau2i+1/gamma20), sqrt(1/(m/tau2i+1/gamma20)));
    
    % sample tau2
    tau2i = 1./random('Gamma',(eta0+m)/2, 2/(eta0*tau20+sum((thetai-mui).^2)));
    
    % collect
    theta_mcmc(ii,:) = thetai(:)';
    sigma2_mcmc(ii) = sigma2i;
    mu_mcmc(ii) = mui;
    tau2_mcmc(ii) = tau2i;  
end

%% MCMC diagnostics
ESS = [multiESS(mu_mcmc),multiESS(sigma2_mcmc),multiESS(tau2_mcmc)]; % effective sample size
fprintf('Effective sample sizes for [mu, sigma2, and tau2] are [%d, %d, %d]\n',round(ESS));
fprintf('   Corresponding approximated MC is [%f, %f, %f] if direclty using MC result\n',std([mu_mcmc,sigma2_mcmc,tau2_mcmc])./sqrt(ESS));
% do not use std(mu_mcmc) because many of the samples are not independent. 

[ESS_theta,Sigma_theta] = multiESS(theta_mcmc);    % book assume no correlation between thetas 
theta_std1 = sqrt(diag(Sigma_theta)/ESS_theta);     % use cov from multiESS
theta_std2 = sqrt(diag(cov(theta_mcmc))/ESS_theta);     % use raw theta_mcmc
fprintf('Effective sample size for theta is %d\n',round(ESS_theta));
fprintf('   Corresponding range of MC is\n');
fprintf('      [%f,%f] using multiESS result\n',min(theta_std1),max(theta_std1));
fprintf('      [%f,%f] directly using MC result\n',min(theta_std2),max(theta_std2));

figure
subplot(1,3,1);
boxplot(reshape(mu_mcmc,500,[]),'Labels',cellstr(num2str((500:500:5000)')));
xlabel('iteration');
ylabel('\mu');
subplot(1,3,2);
boxplot(reshape(sigma2_mcmc,500,[]),'Labels',cellstr(num2str((500:500:5000)')));
xlabel('iteration');
ylabel('\sigma^2');
subplot(1,3,3);
boxplot(reshape(tau2_mcmc,500,[]),'Labels',cellstr(num2str((500:500:5000)')));
xlabel('iteration');
ylabel('\tau^2');
sgtitle('Figure 8.6');

% -- autorcorrelation
figure
subplot(1,3,1);
autocorr(mu_mcmc);
title('\mu');
subplot(1,3,2)
autocorr(sigma2_mcmc);
title('\sigma^2');
subplot(1,3,3)
autocorr(tau2_mcmc);
title('\tau^2');

%% --- marginal posterior 
mu_list = linspace(40,60,1000);
pd_mu = fitdist(mu_mcmc,'Kernel','Kernel','normal');    % use normal kenerl
pdf_mu_kernel = pdf(pd_mu,mu_list);

sigma2_list = linspace(70,100,1000);
pd_sigma2 = fitdist(sigma2_mcmc,'Kernel','Kernel','normal');    % use normal kenerl
pdf_sigma2_kernel = pdf(pd_sigma2,sigma2_list);

tau2_list = linspace(10,50,1000);
pd_tau2 = fitdist(tau2_mcmc,'Kernel','Kernel','normal');    % use normal kenerl
pdf_tau2_kernel = pdf(pd_tau2,tau2_list);

figure
subplot(1,3,1);
hold on; box on;
plot(mu_list,pdf_mu_kernel)
plot(repmat(mean(mu_mcmc),1,2),ylim);
xlabel('\mu');
ylabel('p(\mu|y_1,...,y_m)');

subplot(1,3,2);
hold on; box on;
plot(sigma2_list,pdf_sigma2_kernel)
plot(repmat(mean(sigma2_mcmc),1,2),ylim);
xlabel('\sigma^2');
ylabel('p(\sigma^2|y_1,...,y_m)');

subplot(1,3,3);
hold on; box on;
plot(tau2_list,pdf_tau2_kernel)
plot(repmat(mean(tau2_mcmc),1,2),ylim);
xlabel('\tau^2');
ylabel('p(\tau^2|y_1,...,y_m)');
sgtitle('Figure 8.7');

%% shrinkage
figure
subplot(1,2,1);
hold on;
scatter(stats.mean_mathscore,mean(theta_mcmc,1))
plot(ylim,ylim)
box on;
xlabel('$\bar{y}$','Interpreter','latex');
ylabel('$\hat{\mu}$','Interpreter','latex');

subplot(1,2,2);
hold on;
scatter(nm,stats.mean_mathscore -  mean(theta_mcmc,1)')
plot(xlim,[0 0])
box on;
xlabel('sample size');
ylabel('$\bar{y}-\hat{\mu}$','Interpreter','latex');

%% rank schools
theta_list = linspace(20,70,1000);
pd_theta46 = fitdist(theta_mcmc(:,46),'Kernel','Kernel','normal');    % use normal kenerl
pdf_theta46_kernel = pdf(pd_theta46,theta_list);
pd_theta82 = fitdist(theta_mcmc(:,82),'Kernel','Kernel','normal');    % use normal kenerl
pdf_theta82_kernel = pdf(pd_theta82,theta_list);
figure
hold on; box on;
plot(theta_list,pdf_theta46_kernel,'b')
plot(theta_list,pdf_theta82_kernel,'r')
plot(repmat(mean(mu_mcmc),1,2),ylim,'k--')
scatter(s0.mathscore(s0.school == 46),-0.02*ones(nm(46),1),[],'b')
scatter(stats.mean_mathscore(46),-0.02,200,'b','x')
plot(xlim,[-0.02,-0.02],'b-');
scatter(s0.mathscore(s0.school == 82),-0.04*ones(nm(82),1),[],'r')
scatter(stats.mean_mathscore(82),-0.04,200,'r','x')
plot(xlim,[-0.04,-0.04],'r-');
legend({'school 46','school 82','E[\mu|y_1,...,y_m]'},'Location','best');
xlabel('math score')
title('Figure 8.9');