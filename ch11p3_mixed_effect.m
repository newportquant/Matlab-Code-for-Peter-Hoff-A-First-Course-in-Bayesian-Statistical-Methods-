% chapter 11.3
%
% autocorr requires Econometrics Toolbox
% 
% NPQ $2019.11.02$

clear
s0 = readtable(fullfile(pwd,'data','nelsSES.csv'),'ReadRowNames',true); % SES already centered
stats = grpstats(s0,'sch_id');

n = height(s0); % total number of samples

ids = stats.sch_id;
m = height(stats);  % number of groups (schools)

p = 2;  % dimension of X variable (with intercept)
nm = stats.GroupCount;  % length of samples in each group

% --- contruct data cell
X = cell(m,1);
y = cell(m,1);
for ii=1:m
    si = s0(s0.sch_id==ids(ii),:);
    X{ii} = [ones(height(si),1), si.stu_ses];
    y{ii} = si.stu_mathscore;    
end

%% OLS
beta_ols = nan(m,p);
Sigma_ols = cell(m,1);    % covariance of beta coefficients
sigma2_ols = nan(m,1);   % sample variance (innovations)  
for jj=1:m
    Xj = X{jj};
    yj = y{jj};
    betai = inv(Xj'*Xj)*(Xj'*yj);
    beta_ols(jj,:) = betai;
    SSR = sum((yj-Xj*betai).^2);            % SSR of OLS
    sigma2_ols(jj) = SSR/(nm(jj)-p);                  % unbiased estimate of the std of the innovation term
    Sigma_ols{jj} = sigma2_ols(jj)*inv(Xj'*Xj);         % covraince of beta
end

%% generalized linear regression on mixed effect using MCMC
% --- prior
mu0 = mean(beta_ols); Lambda0 = mean(cat(3,Sigma_ols{:}),3);     % for theta (to generate beta): use ols from samples
eta0 = p+2; Sigma0 = cov(beta_ols);         % for Sigma0 (to generate beta): using cov of beta for groups
nu0 = 1; sigma20 = mean(sigma2_ols);    % for sigma2 (to generate y samples): assume identical for every group during MCMC

% --- start values
Sigmai = Sigma0;    % use prior values
sigma2i = sigma20;  % use prior values
thetai = mu0;       % use prior values

% MCMC
nmcmc = 10000;
beta_mcmc = cell(nmcmc,1);
theta_mcmc = nan(nmcmc,p);
Sigma_mcmc = cell(nmcmc,1);
sigma2_mcmc = nan(nmcmc,1);
tic
for ii=1:nmcmc
    % Gibbs: sample beta
    betai = nan(m,p);    
    for jj=1:m
        Xj = X{jj};
        yj = y{jj};       
        V_beta = inv(inv(Sigmai)+Xj'*Xj/sigma2i);
        E_beta = V_beta*(inv(Sigmai)*thetai'+Xj'*yj/sigma2i);
        betai(jj,:) = mvnrnd(E_beta,V_beta);
    end
    
    % Gibbs: sample theta
    Lambdam = inv(inv(Lambda0)+m*inv(Sigmai));
    mum = Lambdam*(inv(Lambda0)*mu0'+m*inv(Sigmai)*mean(betai)');
    thetai = mvnrnd(mum,Lambdam);
    
    % Gibbs: sample Sigma
    Sigmath = (betai-thetai)'*(betai-thetai);
    Sigmai = iwishrnd(Sigmath+Sigma0,eta0+m);      % nu0+m is the degree of freedom
    
    % Gibbs: sigma2
    SSRi = 0;
    for jj=1:m
        Xj = X{jj};
        yj = y{jj};       
        SSRi = SSRi + sum((yj-Xj*betai(jj,:)').^2);
    end
    sigma2i = 1./random('Gamma',(nu0+n)/2,2/(nu0*sigma20+SSRi));

    % collect
    beta_mcmc{ii} = betai;
    theta_mcmc(ii,:) = thetai;
    Sigma_mcmc{ii} = Sigmai;
    sigma2_mcmc(ii) = sigma2i;
end
toc

%% autocorrelation of theta 
figure
subplot(1,2,1);
autocorr(theta_mcmc(:,1))
title('\theta_1')
subplot(1,2,2);
autocorr(theta_mcmc(:,2))
title('\theta_2')

%% anaysis of MCMC
beta_mcmc_mat = cat(3,beta_mcmc{:});

% KDE of slope
slope_list = linspace(-6,8,1000);

pd_slope_theta = fitdist(theta_mcmc(:,2),'Kernel','Kernel','normal');    % use normal kenerl
pdf_slope_theta = pdf(pd_slope_theta,slope_list);

slope_beta = squeeze(beta_mcmc_mat(:,2,:));
slope_beta = slope_beta(:);
fprintf('Probability of beta2<0 is %f\n',mean(slope_beta<0))  % probably of beta2<0

pd_slope_beta = fitdist(slope_beta(:),'Kernel','Kernel','normal');    % use normal kenerl
pdf_slope_beta = pdf(pd_slope_beta,slope_list);

figure
subplot(1,2,1);
hold on; box on;
plot(slope_list,pdf_slope_theta);
plot(slope_list,pdf_slope_beta);
plot(repmat(quantile(theta_mcmc(:,2),0.025),1,2),ylim,'k--');
plot(repmat(quantile(theta_mcmc(:,2),0.975),1,2),ylim,'k--');
% histogram(theta_mcmc(:,2),'Normalization','pdf')
% histogram(beta_mcmc{end}(:,2),50,'Normalization','pdf')
xlabel('slope parameter');
ylabel('posterior pdf');
legend({'\theta_2','\beta_2','CI=95%'},'Location','best')

% --- prediction
xlist = linspace(min(s0.stu_ses),max(s0.stu_ses),200)';
ypred = [ones(size(xlist,1),1),xlist]*mean(beta_mcmc_mat,3)';
ypred_theta = [ones(size(xlist,1),1),xlist]*mean(theta_mcmc)';

subplot(1,2,2);
hold on; box on;
plot(xlist,ypred,'g')
plot(xlist,ypred_theta,'linewidth',4);
xlabel('SES');
ylabel('math score');
sgtitle('Figure 11.3');