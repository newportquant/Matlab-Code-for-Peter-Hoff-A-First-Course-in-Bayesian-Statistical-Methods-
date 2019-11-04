% chapter 8.5: different variance for groups
%
% Need gendist from Matlab Central to sample random number from discrete
% distribution (https://www.mathworks.com/matlabcentral/fileexchange/34101-random-numbers-from-a-discrete-distribution)
%
% NPQ $2019.11.02$


clear
s0 = readtable(fullfile(pwd,'data','nels.csv'),'ReadRowNames',true);
stats = grpstats(s0,'school',{'mean','var'});


%% MCMC
% --- prior
eta0 = 1; tau20 = 100;      % for gamma2
mu0 = 50; gamma20 = 25;     % for mu
a0 = 1; b0 = 1/100;         % for sigma20 (sigam2 for each group is sampled using sigma20, nu0, theta)
alpha0 = 1;                  % prior for nu0 is an exponential (page 144)

% --- start values
thetai = stats.mean_mathscore;
mui = mean(thetai);
tau2i = var(thetai);
sigma2i = stats.var_mathscore;  % sigma2 are different for each group
nu0i = 10;   % start value fo nu0
sigma20i = 1/mean(1./sigma2i);  % start value of sigma20 from where sigma2i is sampled

% --- mcmc
m = height(stats);      % # of groups
n = height(s0);         % # of all samples
nm = stats.GroupCount;  % # of samples in each group
nmcmc = 5000;
sigma2_mcmc = nan(nmcmc,m);     % sigma2 for each group is sampled using sigma20, nu0, and theta
sigma20_mcmc = nan(nmcmc,1);
nu0_mcmc = nan(nmcmc,1);
tau2_mcmc = nan(nmcmc,1);
mu_mcmc = nan(nmcmc,1);
theta_mcmc = nan(nmcmc,m);
nu0_list = 1:5000;
tic
for ii=1:nmcmc
    % sample new theta for each group (equation before 8.4 on page 143)
    thetai = random('Normal',(nm.*stats.mean_mathscore./sigma2i + mui/tau2i)./(nm./sigma2i+1/tau2i),sqrt(1./(nm./sigma2i+1/tau2i)));
    
    % sample sigma2 each group (equation below 8.4 on page 143)
    for jj=1:m
        sigma2i(jj) = 1./random('Gamma',(nu0i+nm(jj))/2, 2/(nu0i*sigma20i+ sum((s0.mathscore(s0.school==jj)-thetai(jj)).^2) ));
    end
    
    % sample sigma20
    sigma20i = random('Gamma',(a0+m*nu0i)/2, 1/(b0+nu0i*sum(1./sigma2i)/2));

    % sample nu0 (calcualte in log scale to avoid Inf)
    log_pdf_nu0 = m*(nu0_list/2.*log(nu0_list*sigma20i/2)-gammaln(nu0_list/2)) + ...
        (nu0_list/2+1)*sum(log(1./sigma2i)) - nu0_list*(alpha0+sigma20i*sum(1./sigma2i)/2);
    log_pdf_nu0 = log_pdf_nu0 - max(log_pdf_nu0);   % normalize pdf in log scale
    pdf_nu0 = exp(log_pdf_nu0);
    pdf_nu0 = pdf_nu0/trapz(nu0_list,pdf_nu0);  % normalize 
    nu0i = gendist(pdf_nu0,1,1);    % need gendist.m from Matlab central

      
    % sample mu
    mui = random('Normal',(m*mean(thetai)/tau2i + mu0/gamma20)/(m/tau2i+1/gamma20), sqrt(1/(m/tau2i+1/gamma20)));
    
    % sample tau2
    tau2i = 1./random('Gamma',(eta0+m)/2, 2/(eta0*tau20+sum((thetai-mui).^2)));
    
    % collect
    theta_mcmc(ii,:) = thetai(:)';
    sigma2_mcmc(ii,:) = sigma2i(:)';
    mu_mcmc(ii) = mui;
    tau2_mcmc(ii) = tau2i;
    nu0_mcmc(ii) = nu0i;
    sigma20_mcmc(ii) = sigma20i;    
end
toc

%% MCMC analysis
figure
subplot(2,2,1);
histogram(mu_mcmc,50,'Normalization','pdf');
xlabel('\mu');
ylabel('$p(\mu|y_1,\dots,y_m)$','Interpreter','latex');
subplot(2,2,2);
histogram(tau2_mcmc,50,'Normalization','pdf');
xlabel('\tau^2');
ylabel('$p(\tau^2|y_1,\dots,y_m)$','Interpreter','latex');
subplot(2,2,3);
histogram(nu0_mcmc,100,'Normalization','probability');
xlabel('\nu0');
ylabel('$p(\nu_0|y_1,\dots,y_m)$','Interpreter','latex');
subplot(2,2,4);
histogram(sigma20_mcmc,50,'Normalization','pdf');
xlabel('\sigma_0^2');
ylabel('$p(\sigma_0^2|y_1,\dots,y_m)$','Interpreter','latex');
sgtitle('Figure 8.11');

%% shrinkage
figure
subplot(1,2,1);
hold on;
scatter(stats.var_mathscore,mean(sigma2_mcmc,1))
plot(ylim,ylim)
box on;
xlabel('$\mathrm{var}(y)$','Interpreter','latex');
ylabel('$\hat{\sigma^2}$','Interpreter','latex');

subplot(1,2,2);
hold on;
scatter(nm,stats.var_mathscore -  mean(sigma2_mcmc,1)')
plot(xlim,[0 0])
box on;
xlabel('sample size');
ylabel('$\mathrm{var}(y)-\hat{\sigma^2}$','Interpreter','latex');

