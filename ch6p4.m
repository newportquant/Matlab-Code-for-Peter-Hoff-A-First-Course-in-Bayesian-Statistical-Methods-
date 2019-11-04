% chapter 6.4
%
% NPQ $2019.11.02$

clear

s0 = readtable(fullfile(pwd,'data','midge.csv'),'ReadRowNames',true);
s0=convertvars(s0,{'Species'},'categorical');

s = s0{s0.Species == 'Af','Wing_Length'};

%% parameters
% --- prior parameters
% - for sigma
sigma0 = 0.1; nu0 = 1;
% - for theta (alternatively can use k0 which is defined such that tau0 = sigma/sqrt(k0))
mu0 = 1.9; tau0 = 0.95;

% --- sampling
n = length(s);  % length of samplings
y_mean = mean(s);
y_var = var(s);


%% MCMC for joint posterior distribution
ns = 1000;
th_mc = nan(ns,1);
sigma_mc = nan(ns,1);
sigmai = 1/sqrt(y_var);     % start value of sigmai
for ii=1:ns
    % --- sample theta from its full conditional posterior distribution
    % conditional on y and known sigma (formula on page 75 or page 89)
    mun = (mu0/tau0^2 + n*y_mean/sigmai^2)/(1/tau0^2 + n/sigmai^2);
    taun = sqrt(1/(1/tau0^2+n/sigmai^2));    
    thi = random('Normal',mun,taun);
    
    % ---- sample sigma from its full posterior distribution conditional on y
    % and known theta (formula on page 93, NOT page 75)
    nun = nu0+n;
    sn2 = mean((s-thi).^2);
    sigman = sqrt(1/nun*(nu0*sigma0^2+n*sn2));
    sigmai = 1/sqrt(random('Gamma',nun/2,1/(nun*sigman^2/2)));
    
    % --- collect MC result
    sigma_mc(ii) = sigmai;
    th_mc(ii) = thi;
end

%% plot
figure
subplot(1,3,1);
plot(th_mc(1:5),1./(sigma_mc(1:5)).^2,'o-');
xlabel('\theta');
ylabel('$\tilde{\sigma^2}$','Interpreter','latex')
subplot(1,3,2);
plot(th_mc(1:15),1./(sigma_mc(1:15)).^2,'o-');
xlabel('\theta');
ylabel('$\tilde{\sigma^2}$','Interpreter','latex')
subplot(1,3,3);
plot(th_mc(1:100),1./(sigma_mc(1:100)).^2,'o-');
xlabel('\theta');
ylabel('$\tilde{\sigma^2}$','Interpreter','latex')
sgtitle('Figure 6.2');

%% KDE
th_list0 = linspace(1.6,2.0,1000);
pd_th = fitdist(th_mc,'Kernel','Kernel','normal');    % use normal kenerl
pdf_th_kernel = pdf(pd_th,th_list0);

sigma2inv_list0 = linspace(0,200,1000);
pd_sigma2inv = fitdist(1./(sigma_mc).^2,'Kernel','Kernel','normal');    % use normal kenerl
pdf_sigma2inv_kernel = pdf(pd_sigma2inv,sigma2inv_list0);

figure
subplot(1,3,1);
scatter(th_mc,1./(sigma_mc).^2,'Marker','.');
box on;
set(gca,'xlim',[1.6,2.0]);
xlabel('\theta');
ylabel('$\tilde{\sigma^2}$','Interpreter','latex')
title('Joint');

subplot(1,3,2);
plot(th_list0,pdf_th_kernel);
xlabel('\theta');
ylabel('$p(\theta|y_1,\dots,y_n)$','Interpreter','latex')
title('Marginal');

subplot(1,3,3);
plot(sigma2inv_list0,pdf_sigma2inv_kernel);
xlabel('$\tilde{\sigma^2}$','Interpreter','latex')
ylabel('$p(\tilde{\sigma^2}|y_1,\dots,y_n)$','Interpreter','latex')
title('Marginal');
sgtitle('Figure 6.3');

%% quantiles
fprintf('Quantiles for th are [%f,%f,%f]\n',quantile(th_mc,[0.025,0.5,0.975]))
fprintf('Quantiles for 1/sigma^2 are [%f,%f,%f]\n',quantile(1./(sigma_mc).^2,[0.025,0.5,0.975]))
fprintf('Quantiles for sigma are [%f,%f,%f]\n',quantile(sigma_mc,[0.025,0.5,0.975]))

