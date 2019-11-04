% chapter 7.5
%
% NPQ $2019.11.02$

clear

%% load and matrix plot full data set
sfull0 = readtable(fullfile(pwd,'data','pima_full.csv'),'ReadRowNames',true);
sfull = sfull0{:,:};

% --- matrix plot for full
figure
[S,AX,BigAx,H,HAx] = plotmatrix(sfull);
for ii=1:size(sfull,2)
    ylabel(AX(ii,1),sfull0.Properties.VariableNames{ii})
    xlabel(AX(end,ii),sfull0.Properties.VariableNames{ii})
end
title(BigAx,'Figure 7.3');

%% data set with missing values
% --- load missing data set
s0 = readtable(fullfile(pwd,'data','pima_miss.csv'),'ReadRowNames',true);
y = s0{:,:};

O = isfinite(y);    % missing matrix
[n,p] = size(y);

% --- prior of theta
Mu0 = [120, 64, 26, 26];
Lambda0 = diag((Mu0/2).^2);     % choosen arbitrarily such that Mu0 is very likely positive (i.e. Mu0-2*std>0)

% --- prior of Sigma (4x4);
nu0 = p+2;
S0 = Lambda0;

%% MCMC
nmcmc = 5000;
Sigmai = nancov(y);         % Use sample's covariance for Sigma's start value
yfulli = fillmissing(y,'constant',nanmean(y));                   % fill missing values with mean values
Theta_mcmc = nan(nmcmc,p);
Sigma_mcmc = cell(nmcmc,1);     % cell to save 2x2 covariance matrix
yfull_mcmc = cell(nmcmc,1);     % also sampling y
for ii=1:nmcmc
    % sample theta from its conditional posterior distribution (on y with missing values filled, and Sigmai); page 108
    Mu_sam = mean(yfulli);  % mean of sampel with missing values filled
    Lambdan = inv(inv(Lambda0) + n*inv(Sigmai));
    Mun = (Mu0/Lambda0+n*Mu_sam/Sigmai) * Lambdan;
    Thetai = mvnrnd(Mun,Lambdan);
    
    % sample Sigma from its conditional posterior (on y with missing values filled, and Thetai); page 111
    Sth = (yfulli-Thetai)'*(yfulli-Thetai);
    Sigmai = iwishrnd(Sth+S0,nu0+n);      % nu0+n is the degree of freedom
    
    % sample missing values from its conditional posterior on (ymiss, Thetai, and Sigmai); page 118
    yfulli = y;
    for jj=1:n
        if all(O(jj,:))     % skip if no missing
            continue;
        end
        a = find(O(jj,:));
        b = find(~O(jj,:));
        Theta_ba = Thetai(b) + reshape(Sigmai(b,a)/Sigmai(a,a) * (y(jj,a) - Thetai(a))', size(Thetai(b)));
        Sigma_ba = Sigmai(b,b) - Sigmai(b,a)/Sigmai(a,a) * Sigmai(a,b);
        %         % or equivalently
        %         Theta_ba = Thetai(b) + (y(jj,a) - Thetai(a)) * (Sigmai(b,a)/Sigmai(a,a))';
        %          Sigma_ba = Sigmai(b,b) - Sigmai(a,b)' * (Sigmai(b,a)/Sigmai(a,a))';
        yfulli(jj,b) = mvnrnd(Theta_ba,Sigma_ba);
    end
    
    % collect
    Theta_mcmc(ii,:) = Thetai;
    Sigma_mcmc{ii} = Sigmai;
    yfull_mcmc{ii} = yfulli;
end

%% statistics and plot
Theta_mean = mean(Theta_mcmc(end-1000+1:end,:))      % mean Theta values
Rho_mcmc = cellfun(@corrcov,Sigma_mcmc,'UniformOutput',false);     % Correlation matrix
Rho_mcmc = cat(3,Rho_mcmc{:});      % convert to 4 x 4 x nmcmc
Rho_mean = mean(Rho_mcmc(:,:,end-1000+1:end),3)     % mean of Correlation 

Rho_ci = [quantile(Rho_mcmc(:,:,end-1000+1:end),[0.025,0.975],'dim',3)];

hfig = figure;
for ii=1:p
    subplot(p,2,ii*2-1)
    for jj=1:p
        if jj==ii
            continue;
        end
        hold on; box on
        plot(jj,Rho_mean(ii,jj),'b.');
        plot([jj,jj],squeeze(Rho_ci(ii,jj,:)),'r-');
    end
    set(gca,'xlim',[0.5,p+0.5],'ylim',[0,0.8]);
    set(gca,'XTick',1:p);
    set(gca,'XTickLabel',[]);
    ylabel(s0.Properties.VariableNames{ii});
end
subplot(p,2,2*p-1);
set(gca,'XTickLabel',s0.Properties.VariableNames);

%% regression coefficients
% Right panel of figure 7.4 incorrectly use correlation matrix to calculate
% beta. Beta must be calcuated using covariance matrix, not correlation
% matrix.  
cbeta = nan(p,p,nmcmc);     % coefficent beta
for ii=1:nmcmc
    cbeta_tmp = nan(p,p);
    for jj=1:p
        b = jj;
        a = setxor(1:p,b);
        cbeta_tmp(b,a) = Sigma_mcmc{ii}(b,a)/Sigma_mcmc{ii}(a,a);   % cbeta_tmp(b,a) is wrong; must transpose of cbeta
    end
    cbeta(:,:,ii) = cbeta_tmp;
end
cbeta_ci = (quantile(cbeta(:,:,end-1000+1:end),[0.025,0.975],'dim',3));
cbeta_mean = mean(cbeta(:,:,end-1000+1:end),3);     % mean of beta

% --- plot
figure(hfig);
for ii=1:p
    subplot(p,2,ii*2)
    for jj=1:p
        if jj==ii
            continue;
        end
        hold on; box on
        plot(jj,cbeta_mean(ii,jj),'b.');
        plot([jj,jj],squeeze(cbeta_ci(ii,jj,:)),'r-');
    end
    set(gca,'xlim',[0.5,p+0.5]);
    set(gca,'ylim',[-0.2,1.6]);
    
    set(gca,'XTick',1:p);
    set(gca,'XTickLabel',[]);
    ylabel(s0.Properties.VariableNames{ii});
end
subplot(p,2,2*p);
set(gca,'XTickLabel',s0.Properties.VariableNames);
sgtitle('Figure 7.4');


%% prediction vs true values
yfull_mcmc_cat = cat(3,yfull_mcmc{:});
figure
for ii=1:p
    subplot(2,2,ii)
    hold on;
    scatter(sfull(~O(:,ii),ii), mean(yfull_mcmc_cat(~O(:,ii),ii,:),3))%,'Marker','.');
    line(ylim,ylim);
    box on;
    xlabel('True value');
    ylabel({'Predicted missing value','(posterior expectations)'});
    title(s0.Properties.VariableNames{ii})
end
sgtitle('Figure 7.5');