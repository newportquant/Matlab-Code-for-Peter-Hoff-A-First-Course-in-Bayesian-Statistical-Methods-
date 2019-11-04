% chapter 10.5
%
% autocorr requires Econometrics Toolbox
% multiESS requires https://github.com/lacerbi/multiESS
%
% NPQ $2019.11.02$


clear
s0 = readtable(fullfile(pwd,'data','icecore.csv'),'ReadRowNames',true);
t = s0.year;
co2 = s0.co2;
tmp = s0.tmp;

% standardized values
co2s = normalize(s0.co2,'zscore');
tmps = normalize(s0.tmp,'zscore');

% prepare using raw data
y = tmp;        
X = [ones(size(y)),co2];
[n,p] = size(X);
    

%% plot
figure
subplot(1,3,[1,2]);
plot(t,tmps,t,co2s);
legend('temp','co2');
xlabel('year');
ylabel('standardized measurement');
subplot(1,3,3);
scatter(co2,tmp);
box on;
xlabel('CO2 (ppmv)');
ylabel('temperature difference (degC)');

%% OLS
mdl = fitlm(co2,tmp)    % on non-standardized data
figure
subplot(1,2,1);
histogram(mdl.Residuals.Raw);
xlabel('residual');
ylabel('frequency');
subplot(1,2,2);
autocorr(mdl.Residuals.Raw,23);


%% MCMC with Gibbs+MH (on standardized data)
% --- prior 
beta0 = zeros(1,p); Sigma0 = eye(p)*1000;    % for beta
nu0 = 1; sigma20 = 1;   % for sigma2 for each data point

% --- proposal distribution for rho (uniform)
delta = 0.1;    % delta controls the convergence speed of MCMC

% --- start values using OLS result
betai = mdl.Coefficients.Estimate';     % use OLS value 
sigma2i = mdl.MSE;  % or mdl.SSE/(n-p)
acf = autocorr(mdl.Residuals.Raw,1);
rhoi = acf(2);  % acf of lag 1 

% --- MCMC
nmcmc = 25000;
beta_mcmc = nan(nmcmc,p);
sigma2_mcmc = nan(nmcmc,1);     % coefficient of serially correlated y variance
rho_mcmc = nan(nmcmc,1);
count_accept = 0;   % counter for acceptance
tic
for ii=1:nmcmc
    % --- Gibbs: sample betai
    Crhoi = toeplitz(rhoi.^(0:n-1));     % nxn
    V_beta = inv(X'*inv(Crhoi)*X/sigma2i + inv(Sigma0));
    E_beta = V_beta*(X'*inv(Crhoi)*y/sigma2i + inv(Sigma0)*beta0');
    betai = mvnrnd(E_beta,V_beta);  % 1xp
    
    % --- Gibbs: sample sigma2
    SSRrho = (y-X*betai')'*inv(Crhoi)*(y-X*betai');
    sigma2i = 1./random('Gamma',(nu0+n)/2,2/(nu0*sigma20+SSRrho));
    
    % --- MH: sample rho
    % proposae a new rho (reflecting random walk)
    rhoi_tmp = abs(unifrnd(rhoi-delta,rhoi+delta));
    rhoi_tmp = min(rhoi_tmp,2-rhoi_tmp); 
    % get acceptance ratio (log scale)
    Crhoi_tmp = toeplitz(rhoi_tmp.^(0:n-1));     % nxn    
    logr = log(mvnpdf(y,(X*betai'),sigma2i*Crhoi_tmp)) - log(mvnpdf(y,(X*betai'),sigma2i*Crhoi));
    % accept it or not
    if log(rand)<logr
        rhoi = rhoi_tmp;
        count_accept = count_accept + 1;
    end
    
    % collect
    beta_mcmc(ii,:) = betai;
    sigma2_mcmc(ii) = sigma2i;
    rho_mcmc(ii) = rhoi;
end
toc

%% analysis
fprintf('Acceptance rate is %f\n',count_accept/nmcmc);
fprintf('Effective sample size of the 1000 MC sequence of rho is %0.0f\n',multiESS(rho_mcmc(1:1000)));

figure
subplot(1,2,1)
plot(rho_mcmc(1:1000));
xlabel('scan');
ylabel('\rho');
subplot(1,2,2);
autocorr(rho_mcmc,30)
sgtitle('Figure 10.9');

figure
subplot(1,2,1)
plot(rho_mcmc(1:25:end));
xlabel('scan/25');
ylabel('\rho');
subplot(1,2,2);
autocorr(rho_mcmc(1:25:end),30)
xlabel('Lag/25');
sgtitle('Figure 10.10');

% --- posterior prediction (GLS estimate)
co2_list = linspace(min(co2),max(co2),100)';
Xlist = [ones(length(co2_list),1),co2_list];
y_ols = predict(mdl,co2_list);
y_gls = Xlist*mean(beta_mcmc)';
figure
subplot(1,2,1)
histogram(beta_mcmc(:,2),'Normalization','pdf');
xlabel('\beta_2');
ylabel('posterior');
subplot(1,2,2)
hold on; box on;
scatter(co2,tmp);
h1 = plot(co2_list,y_gls,'LineWidth',2);
h2 = plot(co2_list,y_ols,'LineWidth',2);
xlabel('CO_2');
ylabel('temperature');
legend([h1,h2],{'GLS','OLS'},'Location','best');
sgtitle('Figure 10/11');
