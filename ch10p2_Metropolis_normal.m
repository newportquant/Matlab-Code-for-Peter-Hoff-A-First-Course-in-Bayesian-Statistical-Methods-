% chapter 10.2
%
% NPQ $2019.11.02$

clear

y = [9.37, 10.18, 9.16, 11.60, 10.33];
n = length(y);

% --- prior
mu0 = 5; tau20 = 10;    % for theta
sigma20 = 1;

% --- proposal distribution
delta2 = 2;

% --- start values
thetai = 0;

% --- MCMC
nmcmc = 10000;
theta_mcmc = nan(nmcmc,1);

for ii=1:nmcmc
    % propose a theta near thetai from proposal distribution
    theta_tmp =  random('Normal',thetai,sqrt(delta2));
    
    % calculate log ratio
    logr = sum(log(normpdf(y,theta_tmp,sigma20))) - sum(log(normpdf(y,thetai,sigma20))) ...
        + log(normpdf(theta_tmp,mu0,sqrt(tau20))) - log(normpdf(thetai,mu0,sqrt(tau20)));
    
    % take it or not
    if log(rand)<logr
        thetai = theta_tmp;
    end
    
    % collect
    theta_mcmc(ii,1) = thetai;
end

% --- known posterior
mun = (mu0/tau20+n*mean(y)/sigma20)/(1/tau20+n/sigma20);
tau2n = 1/(1/tau20+n/sigma20);
th_list = linspace(0,20,1000);
pdf_th_post = normpdf(th_list,mun,sqrt(tau2n));

% --- plot
figure
subplot(1,2,1);
plot(theta_mcmc);
xlabel('iteration');
ylabel('\theta');
subplot(1,2,2)
histogram(theta_mcmc,50,'Normalization','pdf');
hold on; box on;
plot(th_list,pdf_th_post,'LineWidth',2);
set(gca,'xlim',[8.5,11.5]);
xlabel('\theta');
ylabel('pdf');
sgtitle('Figure 10.3');

%% check tunning parameter delta2
delta2_a = 1/32;
delta2_c = 64;
thetai_a = 0;
thetai_c = 0;
theta_mcmc_a = nan(nmcmc,1);
theta_mcmc_c = nan(nmcmc,1);
for ii=1:nmcmc
    % propose a theta near thetai from proposal distribution
    theta_tmp_a =  random('Normal',thetai_a,sqrt(delta2_a));
    theta_tmp_c =  random('Normal',thetai_c,sqrt(delta2_c));

    % calculate log ratio
    logr_a = sum(log(normpdf(y,theta_tmp_a,sigma20))) - sum(log(normpdf(y,thetai_a,sigma20))) ...
        + log(normpdf(theta_tmp_a,mu0,sqrt(tau20))) - log(normpdf(thetai_a,mu0,sqrt(tau20)));
    logr_c = sum(log(normpdf(y,theta_tmp_c,sigma20))) - sum(log(normpdf(y,thetai_c,sigma20))) ...
        + log(normpdf(theta_tmp_c,mu0,sqrt(tau20))) - log(normpdf(thetai_c,mu0,sqrt(tau20)));    
    
    % take it or not
    if log(rand)<logr_a
        thetai_a = theta_tmp_a;
    end
    if log(rand)<logr_c
        thetai_c = theta_tmp_c;
    end
    
    % collect
    theta_mcmc_a(ii,1) = thetai_a;
    theta_mcmc_c(ii,1) = thetai_c;
end

ha=[];
figure
ha(1) = subplot(1,3,1);
hold on; box on;
plot(theta_mcmc_a);
plot(xlim,[mun,mun]);
xlabel('iteration');
ylabel('\theta');
title('\delta^2=1/32');

ha(2) = subplot(1,3,2);
hold on; box on;
plot(theta_mcmc);
plot(xlim,[mun,mun]);
xlabel('iteration');
ylabel('\theta');
title('\delta^2=2');

ha(3) = subplot(1,3,3);
hold on; box on;
plot(theta_mcmc_c);
plot(xlim,[mun,mun]);
xlabel('iteration');
ylabel('\theta');
title('\delta^2=64');

linkaxes(ha,'xy');
set(gca,'xlim',[-1,500],'ylim',[0,15]);
sgtitle('Figure 10.4');