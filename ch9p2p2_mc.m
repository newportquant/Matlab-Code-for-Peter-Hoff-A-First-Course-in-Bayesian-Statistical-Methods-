% chapter 9.2.2 MC 
%
% NPQ $2019.11.02$

clear
x1 = [0,0,0,0,0,0,1,1,1,1,1,1]';     % 0/1: running/aerobic
x2 = [23,22,22,25,27,20,31,23,27,28,22,24]'; % age
y = [-0.87,-10.74,-3.27,-1.97,7.50,-7.25,17.05,4.96,10.40,11.05,0.26,2.51]';
X = [ones(size(x1)), x1, x2, x1.*x2];
[n,p] = size(X);

%% OLS
beta_ols = inv(X'*X)*(X'*y)             % coefficent
SSR = sum((y-X*beta_ols).^2)            % SSR of OLS
sigma2_ols = SSR/(n-p)                  % unbiased estimate of the std of the innovation term
beta_ols_std = sqrt(diag(inv(X'*X)*sigma2_ols))     % std of coefficients

%% MC
% --- prior
nu0 = 1; sigma20 = sigma2_ols;  % for sigma2 (innovation)
g = n; % g-prior for beta

% --- from samples
SSRg = y'*(eye(n)-g/(g+1)*X*inv(X'*X)*X')*y;    % SSR shrinked by g factor

% --- MC (page 158-159)
nmc = 5000;
% sample sigma2
sigma2_mc = 1./random('Gamma',(nu0+n)/2,2/(nu0*sigma20+SSRg),nmc,1);
% sample beta
beta_mc = nan(nmc,p);
E_beta = g/(g+1)*beta_ols;
for ii=1:nmc
    V_beta =  g/(g+1)*sigma2_mc(ii)*inv(X'*X);
    beta_mc(ii,:) = mvnrnd(E_beta,V_beta);     % normal distribtion
end

%% statistics
fprintf('posterior means of beta are [%f,%f,%f,%f]\n',mean(beta_mc))
fprintf('posterior std of beta are [%f,%f,%f,%f]\n',std(beta_mc))

%% --- plot
% --- posterior distribution by KDE
beta2_list = linspace(-70,70,1000);
pd_beta2 = fitdist(beta_mc(:,2),'Kernel','Kernel','normal');    % use normal kenerl
pdf_beta2_kernel = pdf(pd_beta2,beta2_list);

beta4_list = linspace(-4,4,1000);
pd_beta4 = fitdist(beta_mc(:,4),'Kernel','Kernel','normal');    % use normal kenerl
pdf_beta4_kernel = pdf(pd_beta4,beta4_list);

% --- beta's prior distribution
Sigma20 = g*sigma20*inv(X'*X);  % (prior equation at the top of page 157)
% - assume t localation-scaled distribution
pd02 = makedist('tLocationScale','mu',0,'sigma',sqrt(Sigma20(2,2)),'nu',nu0);
pdf_beta02 = pdf(pd02,beta2_list);
pd04 = makedist('tLocationScale','mu',0,'sigma',sqrt(Sigma20(4,4)),'nu',nu0);
pdf_beta04 = pdf(pd04,beta4_list);
% % - assume normal distribution
% pdf_beta02 = normpdf(beta2_list,0,sqrt(Sigma20(2,2)));
% pdf_beta04 = normpdf(beta4_list,0,sqrt(Sigma20(4,4)));

figure
subplot(1,3,1);
hold on; box on;
plot(beta2_list,pdf_beta2_kernel)
plot(beta2_list,pdf_beta02)
plot([0,0],ylim)
xlabel('\beta_2');
legend({'post','prior'},'Location','best')
title('Marginal posterior');

subplot(1,3,2);
hold on; box on;
plot(beta4_list,pdf_beta4_kernel)
plot(beta4_list,pdf_beta04)
plot([0,0],ylim)
xlabel('\beta_4');
legend({'post','prior'},'Location','best')
title('Marginal posterior');

subplot(1,3,3);
scatter(beta_mc(:,2),beta_mc(:,4),'.');
box on;
set(gca,'xlim',[-70,70],'ylim',[-3,3]);
xlabel('\beta_2');
ylabel('\beta_4');
title('Joint posterior');
sgtitle('Figure 9.3');


age = 20:31;
figure
boxplot(beta_mc(end-999:end,2) + beta_mc(end-999:end,4)*age)
%set(gca,'ylim',[-10,15]);
set(gca,'Xtick',1:length(age),'XTickLabel',cellstr(num2str(age')));
xlabel('age');
ylabel('\beta_2+\beta_4age')
title('Figure 9.4');
