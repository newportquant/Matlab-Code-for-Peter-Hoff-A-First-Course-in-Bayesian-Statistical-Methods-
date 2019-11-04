% chapter 7.3
%
% NPQ $2019.11.02$

clear
s0 = readtable(fullfile(pwd,'data','reading.csv'),'ReadRowNames',true);
y = s0{:,:};

% --- prior for Theta = score [before, after].
Mu0 = [50, 50];
Lambda0 = [
    625, 312.5; 
    312.5, 625];

% --- prior for Sigma (2x2 covariance matrix)
nu0 = 4;        % The larger the nu0, the more certain about the covarience near the prior value (narrower PDF in Wishart distribution)
S0 = Lambda0;    % S0 is related to expected covariance through nu0 (page 110)

% --- samples
n = size(y,1);
Mu_sam = mean(y);   % sample mean
Sigma_sam = cov(y);     % sample covariance

%% MCMC
nmcmc = 5000;
Theta_mcmc = nan(nmcmc,2);
Sigma_mcmc = cell(nmcmc,1);     % cell to save 2x2 covariance matrix
y_mcmc = nan(nmcmc,2);  % also sampling y 
Sigmai = Sigma_sam;     % initialize with sample's covariance
for ii=1:nmcmc
    % sample theta from its conditional posterior distribution (on y and Sigmai); page 108 
    Lambdan = inv(inv(Lambda0) + n*inv(Sigmai));
    Mun = (Mu0/Lambda0+n*Mu_sam/Sigmai) * Lambdan; 
    Thetai = mvnrnd(Mun,Lambdan);
    
    % sample Sigma from its conditional posterior (on y and Thetai); page 111
    Sth = (y-Thetai)'*(y-Thetai);
    Sigmai = iwishrnd(Sth+S0,nu0+n);      % nu0+n is the degree of freedom
    
    % collect
    Theta_mcmc(ii,:) = Thetai;
    Sigma_mcmc{ii} = Sigmai; 
    y_mcmc(ii,:) = mvnrnd(Thetai,Sigmai); % sample y (not to be iterated used for the MCMC loop)
end

fprintf('Quantiles for th2-th1 are [%f,%f,%f]\n',quantile(Theta_mcmc(:,2)-Theta_mcmc(:,1),[0.025,0.5,0.975]))

%% 
figure
subplot(1,2,1);
hold on; 
scatter(Theta_mcmc(:,1),Theta_mcmc(:,2),'Marker','.');
plot([0,100],[0,100]);
box on;
xlabel('\theta_1');
ylabel('\theta_2');
set(gca,'xlim',[35,60],'ylim',[40,65]);
title(['p(\theta_2>\theta_1|y_1,...,y_n)=',num2str(mean(Theta_mcmc(:,2)>Theta_mcmc(:,1))*100),'%'])

subplot(1,2,2);
hold on;
scatter(y_mcmc(:,1),y_mcmc(:,2),'Marker','.');
plot([0,100],[0,100]);
box on;
xlabel('y_1');
ylabel('y_2');
set(gca,'xlim',[0,100],'ylim',[0,100]);
title(['p(Y_2>Y_1|y_1,...,y_n)=',num2str(mean(y_mcmc(:,2)>y_mcmc(:,1))*100),'%'])
sgtitle('Figure 7.2');