% chapter 11.4
%
% multiESS requires https://github.com/lacerbi/multiESS
% 
% NPQ $2019.11.02$

clear
s0 = readtable(fullfile(pwd,'data','tumorLocation.csv'),'ReadRowNames',true);
Y = s0{:,:};    % response data
m = size(Y,1);  % number of groups (each mouse is a group)
x = (1:width(s0))/width(s0);   % location 

y_grpmean = mean(Y);    % mean within groups for # of tumors at each location

%% polyfit to determine the degree of model for the parameter of Poisson distribution
figure
subplot(1,2,1);
plot(x,Y,'g')
hold on; box on;
plot(x,y_grpmean,'LineWidth',3);
xlabel('location');
ylabel('# of tumors');

subplot(1,2,2);
plot(x,log(y_grpmean),'LineWidth',3);
hold on; box on;
plot(x,polyval(polyfit(x,log(y_grpmean),2),x))
plot(x,polyval(polyfit(x,log(y_grpmean),3),x))
plot(x,polyval(polyfit(x,log(y_grpmean),4),x))
xlabel('location');
ylabel('log(average # of tumors)');
legend({'data','quadratic (p=3)','cubic (p=4)','quartic (p=5)'},'Location','best');

%% MCMC
p = 5; % choose quartic models based on polyfit
n = length(x);  % # of samples in each group
X = x'.^(0:p-1);    % nx5

% --- OLS to 5th on each mouse (group)
beta_ols = nan(m,p);
SSR_ols = nan(m,1);
sigma2_ols = nan(m,1);
Sigma_ols = cell(m,1);
for jj=1:m
    Xj = X;
    yj = log(Y(jj,:)+ 1/n)';
    beta_ols(jj,:) =  inv(Xj'*Xj)*(Xj'*yj);   % on log values 
    SSR_ols(jj) = sum((yj-Xj*beta_ols(jj,:)').^2);
    sigma2_ols(jj) = SSR_ols(jj)/(n-p);
    Sigma_ols{jj} = inv(X'*X)*sigma2_ols(jj);
end

% --- prior
mu0 = mean(beta_ols); Lambda0 = cov(beta_ols);      % for theta; may also use mean(cat(3,Sigma_ols{:}),3) as prior for Lambda0
eta0 = p+2; Sigma0 = cov(beta_ols);                 % for Sigma; may also use mean(cat(3,Sigma_ols{:}),3) as prior for Sigma0
% sigma20 is not needed, because sigma2 is the square residual of the log
% of an Poisson variable so it does not follow multivariate normal
% distribution. There is no close-loop full conditional posterior
% distribution for sigma2. Hence, Metropolis instead of Gibbs will be used
% to update beta, and no need to update sigma2

% --- start values
betai = beta_ols;
Sigmai = Sigma0;

% --- MCMC
nmcmc = 50000;
theta_mcmc = nan(nmcmc,p);
Sigma_mcmc = cell(nmcmc,1);
beta_mcmc = cell(nmcmc,1);
count_accept = 0;
tic
for ii=1:nmcmc
    % Gibbs: sample theta
    Lambdam = inv(inv(Lambda0)+m*inv(Sigmai));
    mum = Lambdam*(inv(Lambda0)*mu0'+m*inv(Sigmai)*mean(betai)');
    thetai = mvnrnd(mum,Lambdam);
    
    % Gibbs: sample Sigma
    Sigmath = (betai-thetai)'*(betai-thetai);
    Sigmai = iwishrnd(Sigmath+Sigma0,eta0+m);      % nu0+m is the degree of freedom
    
    % Gibbs: sigma2 (no need to update if we sample beta with Metropolis)
    % we will sample beta with Metropolis intead of Gibbs)      

    % Metroplis: beta (update one group each time)
    % --- use vector operation (much faster than for loop)
    betap = mvnrnd(betai,Sigmai/2);   % acceptance rate for new proposals is highre for Sigmai/2 than Sigmai
    logr = sum(log(poisspdf(Y',exp(X*betap')))) - sum(log(poisspdf(Y',exp(X*betai')))) ...
        + log(mvnpdf(betap,thetai,Sigmai)') - log(mvnpdf(betai(jj,:),thetai,Sigmai)');
    idx_update = log(rand(size(logr))) < logr;
    betai(idx_update,:) = betap(idx_update,:);
    count_accept = count_accept+nnz(idx_update);
    
    % --- use for loop
%     for jj=1:m
%         % propose a new value around betai, not around thetai (because of
%         % Metropolis not Gibbs)
%         betap = mvnrnd(betai(jj,:),Sigmai/2);   % acceptance rate is highre for Sigmai/2 than Sigmai
%         % log ratio
%         Xj = X;
%         Yj = Y(jj,:)';        % log(y) does NOT follow mutlivariate distribution, must directly use poisson distribution
%         logr = sum(log(poisspdf(Yj,exp(Xj*betap')))) - sum(log(poisspdf(Yj,exp(Xj*betai(jj,:)')))) ...
%             + log(mvnpdf(betap,thetai,Sigmai)) - log(mvnpdf(betai(jj,:),thetai,Sigmai));
%         % accept it or not
%         if log(rand)<logr
%             betai(jj,:) = betap;
%             count_accept = count_accept+1;
%         end
%     end
        
    % collect
    theta_mcmc(ii,:) = thetai;
    Sigma_mcmc{ii} = Sigmai;
    beta_mcmc{ii} = betai;
end
toc

%% analysis of MCMC
fprintf('Acceptatnce rate for updating beta: %f\n',count_accept/(nmcmc*m));
fprintf('Effective sample size of Sigma (condiering the covariance) is %.0f\n',multiESS(reshape(cat(3,Sigma_mcmc{:}),25,[])'));
fprintf('Effective sample size of theta (condering covariance) is %.0f\n',multiESS(theta_mcmc));
fprintf('Effective sample size of theta (w/o condering covariance) are [%.0f,%.0f,%.0f,%.0f,%.0f]\n',...
    multiESS(theta_mcmc(:,1)),...
    multiESS(theta_mcmc(:,2)),...
    multiESS(theta_mcmc(:,3)),...
    multiESS(theta_mcmc(:,4)),...
    multiESS(theta_mcmc(:,5)));

figure
subplot(1,3,1);
plot(x,quantile(exp(theta_mcmc*X'),[0.025,0.5,0.975]));
xlabel('location');
ylabel('# of tumors');
set(gca,'ylim',[0,18])
legend({'2.5%','50%','97.5%'},'Location','best');
title('exp(\thetax)')

subplot(1,3,2);
plot(x,quantile(exp(cat(1,beta_mcmc{:})*X'),[0.025,0.5,0.975]));
xlabel('location');
ylabel('# of tumors');
set(gca,'ylim',[0,18])
legend({'2.5%','50%','97.5%'},'Location','best');
title('exp(\beta^Tx)')

subplot(1,3,3);
plot(x,quantile(poissrnd(exp(cat(1,beta_mcmc{:})*X')),[0.025,0.5,0.975]));
xlabel('location');
ylabel('# of tumors');
set(gca,'ylim',[0,18])
legend({'2.5%','50%','97.5%'},'Location','best');
title('\{Y|x\}')

sgtitle('Figure 11.5')



