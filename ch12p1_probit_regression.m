% chapter 12.1
%
% OrdinalRankings requires https://www.mathworks.com/matlabcentral/fileexchange/19496-rankings?focused=3856857&tab=function
%
% NPQ $2019.11.02$

clear
s0 = readtable(fullfile(pwd,'data','socmob.csv'),'ReadRowNames',1);

% --- remove missing values
idx = any(ismissing([s0.DEGREE,s0.AGE,s0.CHILDREN,s0.PDEGREE]),2);
s0(idx,:) = [];

deg = s0.DEGREE+1;  % offset by one so 1 for no degree
age = s0.AGE;
child = s0.CHILDREN;
pdeg = s0.PDEGREE>2;

X = [child, pdeg];


%% fitlm
mdl = fitlm(X,deg,'interactions') % fitlm automatically ignore rows containing NaN

figure
subplot(1,2,1);
histogram(deg,'Normalization','pdf')
xlabel('DEG')
ylabel('pdf');
subplot(1,2,2);
histogram(child,'Normalization','pdf')
xlabel('CHILD')
ylabel('pdf');
sgtitle('Figure 12.1');

%% MCMC
X = [child, pdeg, child.*pdeg];     % expand X; no intercept term needed
y = deg;
[n,p] = size(X);
[ycat,~,ic] = unique(y);    % # of ordinal categories; ic is the rank 
k = length(ycat);
ycat = (1:k)';
y = ycat(ic);      % replace y with ordinal integers from 1 to k

% --- prior
mu = zeros(1,k-1); sigma = 100*ones(1,k-1);   %  for gi

% --- start values
% rankings = OrdinalRankings(y);      % ascending tied ranking
rankings = OrdinalRankings2(y);     % random tied ranking
zi = norminv(rankings/(n+1));

% --- mcmc
nmcmc = 10000;
beta_mcmc = nan(nmcmc,3);
z_mcmc = nan(n,nmcmc);
g_mcmc = nan(nmcmc,k-1);

tic
for ii=1:nmcmc
    % Gibbs: update g
    gi = nan(1,k-1);
    ai = nan(1,k-1);
    bi = nan(1,k-1);
    for kk=1:k-1
        ai(kk) = max(zi(y==kk));
        bi(kk) = min(zi(y==kk+1));
    end
    u = unifrnd( normcdf((ai-mu)./sigma), normcdf((bi-mu)./sigma) );
    gi = mu + norminv(u).*sigma;
    
    
    % Gibbs: update beta
    V_betai = n/(n+1)*inv(X'*X);    % variance of beta
    E_betai =V_betai *(X'*zi);     % expectation of beta
    betai = mvnrnd(E_betai,V_betai);
    
    % Gibbs: update z
    E_zi = X*betai';
    ai = nan(n,1);
    bi = nan(n,1);    
    for jj=1:n
        if y(jj) == 1
            ai(jj) = -Inf;
        else
            ai(jj) = gi(y(jj)-1);
        end
        if y(jj) == k
            bi(jj) = Inf;
        else
            bi(jj) = gi(y(jj));
        end
    end
    u = unifrnd( normcdf(ai-E_zi), normcdf(bi-E_zi) );
    zi = E_zi + norminv(u);        
    
    % collect
    g_mcmc(ii,:) = gi;
    beta_mcmc(ii,:) = betai;
    z_mcmc(:,ii) = zi;
end
toc

%% MCMC analysis
z_plot = z_mcmc(:,1:25:end);

figure
subplot(1,2,1);
hold on; box on;
scatter(child(pdeg==0)-0.1,z_mcmc(pdeg==0,end),'b');
scatter(child(pdeg==1)+0.1,z_mcmc(pdeg==1,end),'r');
plot(child(pdeg==0),predict(fitlm(child(pdeg==0),z_mcmc(pdeg==0,end)),child(pdeg==0)),'b','linewidth',2);
plot(child(pdeg==1),predict(fitlm(child(pdeg==1),z_mcmc(pdeg==1,end)),child(pdeg==1)),'r','linewidth',2);
xlabel('# of children');
ylabel('z (last scan');
legend({'PDEG=0','PDEG=1'},'Location','best');

xlist = linspace(-0.5,0.5,100)';
beta_prior_Sigma = n*inv(X'*X);
subplot(1,2,2);
hold on; box on;
histogram(beta_mcmc(:,3),'Normalization','pdf')
plot(xlist,normpdf(xlist,0,sqrt(beta_prior_Sigma(3,3))))
xlabel('\beta_3');
ylabel('pdf');
legend({'posterior','prior'},'Location','best');
sgtitle('Figure 12.2');

fprintf('95%% quantile of beta3 is (%f,%f)\n',quantile(beta_mcmc(:,3),[0.025,0.975]))

%% 12.1.2 MCMC without g (slower than the above MCMC with g's help)
zi = norminv(rankings/(n+1));

beta_mcmc2 = nan(nmcmc,3);
z_mcmc2 = nan(n,nmcmc);

tic
for ii=1:nmcmc    
    % Gibbs: update beta
    V_betai = n/(n+1)*inv(X'*X);    % variance of beta
    E_betai =V_betai *(X'*zi);     % expectation of beta
    betai = mvnrnd(E_betai,V_betai);
    
    % Gibbs: update z
    E_zi = X*betai';
    ai = nan(n,1);
    bi = nan(n,1);    
    for jj=1:n
        ai(jj) = max([-Inf; zi(y<y(jj))]);
        bi(jj)= min([Inf;zi(y(jj)<y)]);
    end
    u = unifrnd( normcdf(ai-E_zi), normcdf(bi-E_zi) );
    zi = E_zi + norminv(u);        
    
    % collect
    beta_mcmc2(ii,:) = betai;
    z_mcmc2(:,ii) = zi;
end
toc

%% MCMC analysis
figure
subplot(1,3,1);
hold on; box on;
histogram(beta_mcmc(:,1),'Normalization','pdf');
histogram(beta_mcmc2(:,1),'Normalization','pdf');
xlabel('\beta_1');
ylabel('pdf')
legend({'oridnal probit\newlineregression','rank likelihood'},'Location','best');

subplot(1,3,2);
hold on; box on;
histogram(beta_mcmc(:,2),'Normalization','pdf');
histogram(beta_mcmc2(:,2),'Normalization','pdf');
xlabel('\beta_2');
ylabel('pdf')

subplot(1,3,3);
hold on; box on;
histogram(beta_mcmc(:,3),'Normalization','pdf');
histogram(beta_mcmc2(:,3),'Normalization','pdf');
xlabel('\beta_3');
ylabel('pdf')
sgtitle('Figure 12.3');

