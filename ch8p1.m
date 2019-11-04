% chapter 8.1
%
% NPQ $2019.11.02$

clear
y1 = readtable('school1.csv','ReadRowNames',true);
y2 = readtable('school2.csv','ReadRowNames',true);
y1 = y1{:,:};
y2 = y2{:,:};
n1 = length(y1);
n2 = length(y2);

%% 
figure
subplot(1,2,1)
boxplot([y1;y2],[ones(size(y1));2*ones(size(y2))],...
    'Labels',{'school 1','school 2'},'whisker',1000); % large whisker in order not to label any outliers
ylabel('score');


% --- t-test
[h,p,ci,stats]=ttest2(y1,y2);
subplot(1,2,2);
hold on; box on;
t_list = linspace(-4,4,1000);
plot(t_list,tpdf(t_list,n1+n2-2));
plot(repmat(stats.tstat,1,2),ylim)
xlabel('t');
ylabel('pdf');

sgtitle('Figure 8.1');


%% MCMC
% --- prior
mu0 = 50;  gamma20 = 625;
delta0 = 0; tau20 = 625;
nu0 = 1; sigma20 = 100;

nmcmc = 5000;
mu_mcmc = nan(nmcmc,1);
delta_mcmc = nan(nmcmc,1);
sigma2_mcmc = nan(nmcmc,1);
mui = (mean(y1)+mean(y2))/2;
deltai = (mean(y1)-mean(y2))/2;
for ii=1:nmcmc
    % sample sigma
    nun = nu0+n1+n2;
    sigma2i = 1/random('Gamma',nun/2,2/(nun*sigma20+sum((y1-(mui+deltai)).^2)+sum((y2-(mui-deltai)).^2)));
    % sample mu
    gamma2n = 1/(1/gamma20+(n1+n2)/sigma2i);
    mun = gamma2n*(mu0/gamma20+sum(y1-deltai)/sigma2i+sum(y2+deltai)/sigma2i);
    mui = random('Normal',mun,sqrt(gamma2n));
    % sample delta
    tau2n = 1/(1/tau20+(n1+n2)/sigma2i);
    deltan = tau2n*(delta0/tau20+sum(y1-mui)/sigma2i-sum(y2-mui)/sigma2i);
    deltai = random('Normal',deltan,sqrt(tau2n));
    % collect
    mu_mcmc(ii) = mui;
    delta_mcmc(ii) = deltai;
    sigma2_mcmc(ii) = sigma2i;
end

%% --- is school 1 better than school 2? 
% --- difference quantile
quantile(2*delta_mcmc,[0.025,0.975])

% --- probability of theta1>theta2
mean(delta_mcmc(end-999:end)>0)

% --- probablity of Y1>Y2
y1_mcmc = mu_mcmc + delta_mcmc + random('Normal',0,sqrt(sigma2_mcmc));
y2_mcmc = mu_mcmc - delta_mcmc + random('Normal',0,sqrt(sigma2_mcmc));
mean(y1_mcmc(end-999:end)>y2_mcmc(end-999:end))

%% plot
mu_list = linspace(30,70,1000);
pd_mu = fitdist(mu_mcmc(end-1000+1:end),'Kernel','Kernel','normal');    % use normal kenerl
pdf_mu_kernel = pdf(pd_mu,mu_list);

delta_list = linspace(-20,20,1000);
pd_delta = fitdist(delta_mcmc(end-1000+1:end),'Kernel','Kernel','normal');    % use normal kenerl
pdf_delta_kernel = pdf(pd_delta,delta_list);

figure
subplot(1,2,1);
hold on; box on;
plot(mu_list,pdf_mu_kernel);
plot(mu_list,normpdf(mu_list,mu0,sqrt(gamma20)));
xlabel('\mu');
ylabel('pdf');
legend({'posterior','prior'},'Location','best');
subplot(1,2,2);
hold on; box on;
plot(delta_list,pdf_delta_kernel);
plot(delta_list,normpdf(delta_list,delta0,sqrt(tau20)));
xlabel('\delta');
ylabel('pdf');
legend({'posterior','prior'},'Location','best');
sgtitle('Figure 8.2');