% ch5.3
% Unknown theta and sigma
%
% NPQ $2019.11.02$

clear
s0 = readtable(fullfile(pwd,'data','midge.csv'),'ReadRowNames',true);
s0=convertvars(s0,{'Species'},'categorical');

s = s0{s0.Species == 'Af','Wing_Length'};

%% joint posterior distribution of (theta,sigma2)
% --- prior parameters
% - for sigma
sigma0 = 0.1;   
nu0 = 1;
% - for theta
mu0 = 1.9; 
k0 = 1;

% --- sampling
n = length(s);  % length of samplings
y_mean = mean(s);
y_var = var(s);

% --- posterior paremters 
% - for theta
kn = k0+n;
mun = (k0*mu0+n*y_mean)/kn;
% - for sigma
nun = nu0+n;
sigman=sqrt(1/nun*(nu0*sigma0^2+(n-1)*y_var+k0*n/kn*(y_mean-mu0)^2));

% --- numerically calculate joint distribution
sigma2_list = linspace(0.,0.05,1000);
pdf_sigma2 = pdf('Gamma',1./sigma2_list,nun/2,1/(nun*sigman^2/2));

sigma2inv_list = linspace(0,200,1000);
pdf_sigma2inv = pdf('Gamma',sigma2inv_list,nun/2,1/(nun*sigman^2/2));

th_list = linspace(1.6,2.0,1100);
pdf_th_for_sigma2 = normpdf(th_list,mun,sqrt(sigma2_list(:)/kn));
pdf_2D_for_sigma2 = pdf_th_for_sigma2.*pdf_sigma2(:);
pdf_th_for_sigma2inv = normpdf(th_list,mun,sqrt(1./sigma2inv_list(:)/kn));
pdf_2D_for_sigma2inv = pdf_th_for_sigma2inv.*pdf_sigma2inv(:);

% --- plot 
figure
subplot(1,2,1);
imagesc(th_list,sigma2inv_list,pdf_2D_for_sigma2inv);
set(gca,'ydir','normal');
axis square
xlabel('\theta');
ylabel('$\tilde{\sigma^2}$','Interpreter','latex');

subplot(1,2,2);
imagesc(th_list,sigma2_list,pdf_2D_for_sigma2);
set(gca,'ydir','normal');
axis square
xlabel('\theta');
ylabel('$\sigma^2$','Interpreter','latex');
sgtitle('Figure 5.4');

%% MC for marginal samples to calculate quantiles 
% --- MC
ns = 10000;
sigma2_post_mc = 1./random('Gamma',nun/2,1/(nun*sigman^2/2),[ns,1]);       % random numbers
th_post_mc = random('Normal',mun,sqrt(sigma2_post_mc/kn),[ns,1]);

% --- kernal density estimate of marginal posterior for sigma2
sigma2_list0 = linspace(0,0.08,1000);
pd_sigma2 = fitdist(sigma2_post_mc,'Kernel','Kernel','normal');    % use normal kenerl
pdf_sigma2_kernel = pdf(pd_sigma2,sigma2_list0);

% --- kernal density estimate of marginal posterior for theta
th_list0 = linspace(1.6,2.0,1000);
pd_th = fitdist(th_post_mc,'Kernel','Kernel','normal');    % use normal kenerl
pdf_th_kernel = pdf(pd_th,th_list0);

% --- plot
figure
subplot(2,2,[1,2]);
scatter(th_post_mc,sigma2_post_mc,'.');
box on;
set(gca,'xlim',[1.6,2.0]);
set(gca,'ylim',[0,0.07]);
xlabel('\theta');
ylabel('\sigma^2');

subplot(2,2,3);
plot(sigma2_list0,pdf_sigma2_kernel);
xlabel('\sigma^2');
ylabel('$p(\sigma^2|y_1,\dots,y_n)$','Interpreter','latex');
title('KDE for \sigma^2');

subplot(2,2,4);
hold on; box on;
plot(th_list0,pdf_th_kernel);
plot(repmat(icdf(pd_th,0.025),1,2),ylim,'k--')
plot(repmat(icdf(pd_th,0.975),1,2),ylim,'k--')
plot(repmat(tinv(0.025,nu0+n)*sigman/sqrt(kn)+mun,1,2),ylim,'m--');
plot(repmat(tinv(0.975,nu0+n)*sigman/sqrt(kn)+mun,1,2),ylim,'m--');
title('KDE for \theta');
xlabel('\theta^2');
ylabel('$p(\theta|y_1,\dots,y_n)$','Interpreter','latex');
sgtitle('Figure 5.5');


