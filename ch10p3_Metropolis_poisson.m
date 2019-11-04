% chapter 10.3
%
% autocorr requires Econometrics Toolbox
% multiESS requires https://github.com/lacerbi/multiESS
%
% NPQ $2019.11.02$


clear

s0 = readtable(fullfile(pwd,'data','sparrows.csv'),'ReadRowNames',true);

x = s0.age;
y = s0.fledged;

X = [ones(height(s0),1),x,x.^2];
[n,p] = size(X);


%% MCMC
% --- prior
mu0 = [0,0,0]; Sigma0 = 100*eye(p);   % prior of beta

% --- proposal distribution
Sigmap = var(log(y+0.5))*inv(X'*X);     % pxp covairance (page 180 choice of proposal covariance)

% --- start values
betai = [0 0 0]; 

% --- MCMC with Metropolis
nmcmc = 10000;
beta_mcmc = nan(nmcmc,p);
count_accept = 0;
for ii=1:nmcmc
    % propose a new betai
    beta_tmp = mvnrnd(betai,Sigmap);
    
    % calculate log ratio
    logr = sum(log(poisspdf(y,exp(X*beta_tmp')))) - sum(log(poisspdf(y,exp(X*betai')))) ...
        + log(mvnpdf(beta_tmp,mu0,Sigma0)) - log(mvnpdf(betai,mu0,Sigma0));

    % take it or not
    if log(rand)<logr
        betai = beta_tmp;
        count_accept = count_accept + 1;
    end
    
    % collect
    beta_mcmc(ii,:) = betai;
end

%% analysis
fprintf('Acceptance rate is %f\n',count_accept/nmcmc);
[ESS_beta,Cov_beta] = multiESS(beta_mcmc);
fprintf('Effective sample size for beta is %d condering the covariance \n',round(ESS_beta));
ESS_beta1 = multiESS(beta_mcmc(:,1));
ESS_beta2 = multiESS(beta_mcmc(:,2));
ESS_beta3 = multiESS(beta_mcmc(:,3));
fprintf('Effective sample size for beta is [%d, %d, %d] w/o condering the covariance between beta''s \n',round(ESS_beta1),round(ESS_beta2),round(ESS_beta3));


figure
subplot(1,3,1);
plot(beta_mcmc(:,3));
xlabel('iteration');
ylabel('\beta_3');

subplot(1,3,2);
% - use xcorr does not provide CI
% [acf,lag] = xcorr(normalize(beta_mcmc(:,3),'center'),40,'coeff');
% stem(lag(lag>=0),acf(lag>=0))
% - use aucorr for automatic plot
autocorr(beta_mcmc(:,3),40);

subplot(1,3,3);
autocorr(beta_mcmc(1:10:end,3),30);
xlabel('Lag/10');

sgtitle('Figure 10.5');

%% get # of offsprings quantiles
y_mcmc = nan(n,nmcmc);
for ii=1:nmcmc
    y_mcmc(:,ii) = exp(X*beta_mcmc(ii,:)');
end

x_unique = unique(x);
y_unique = cell(length(x_unique),1);
for ii=1:length(x_unique)
    ytmp = y_mcmc(x==x_unique(ii),:);
    y_unique{ii} = ytmp(:);    
end
y_quantiles = cell2mat(cellfun(@(x)quantile(x,[0.025,0.5,0.975]),y_unique,'UniformOutput',false));

figure
subplot(1,3,1);
histogram(beta_mcmc(:,2),'Normalization','pdf')
xlabel('\beta_2');
ylabel('pdf');
title('marginal posterior');

subplot(1,3,2);
histogram(beta_mcmc(:,3),'Normalization','pdf')
xlabel('\beta_3');
ylabel('pdf');
title('marginal posterior');

subplot(1,3,3);
hold on; box on;
plot(x_unique,y_quantiles)
legend({'2.5%','50%','97.5%'},'Location','best');
xlabel('age');
ylabel('# of offsprings');
title('posterior quantiles');

sgtitle('Figure 10.6');