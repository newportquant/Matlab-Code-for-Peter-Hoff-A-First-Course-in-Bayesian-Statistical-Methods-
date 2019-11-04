% Ch3.1 
%
% NPQ $2019.11.02$

clear
s0 = readtable(fullfile(pwd,'data','gss.csv'),'ReadRowNames',true);
s1 = s0(s0.FEMALE==1 & s0.AGE>=65 & s0.YEAR==1998 & s0.HAPUNHAP<=4,:).HAPUNHAP;
s = nan(size(s1)); 
s(s1>2) = 0;
s(s1<=2) = 1;

nh = nnz(s==1);   % # of happy
nu = nnz(s==0);  % # of unhappy

%% probabilities
th = linspace(0,1,1000); % 
lld = th.^nh.*(1-th).^nu;  % likelihood
prior = ones(size(th)); % assume prior distribution of th is uniform (i.e. no knowledge about it)
% posterior = lld.*prior/(gamma(nh+1)*gamma(nu+1)/gamma(nh+1+nu+1));
posterior = lld.*prior/beta(nh+1,nu+1);

%% figure 3.1
figure
subplot(2,1,1);
plot(th,lld*1e17);
legend('likelihood * 10^{17}','Location','best')
xlabel('\theta')
subplot(2,1,2);
plot(th,posterior,th,prior);
legend({'posterior','prior'},'Location','best')
xlabel('\theta')
sgtitle('Figure 3.1');

%% figure 3.2 and 3.3 binomial distribution
n = 10;
y = 0:n;
th = 0.2;
figure
subplot(1,2,1);
stem(y,binopdf(y,n,th),'Marker','none','LineWidth',2)
xlabel('y (sum of n trail results)');
ylabel('p(Y=y|\theta=0.2,n=10)')
set(gca,'XMinorTick','on');
th = 0.8;
subplot(1,2,2);
stem(y,binopdf(y,n,th),'Marker','none','LineWidth',2)
xlabel('y (sum of n trail results)');
ylabel('p(Y=y|\theta=0.8,n=10)')
set(gca,'XMinorTick','on');
sgtitle('Figure 3.2');

n = 100;
y = 0:n;
th = 0.2;
figure
subplot(1,2,1);
stem(y,binopdf(y,n,th),'Marker','none','LineWidth',2)
xlabel('y (sum of n trail results)');
ylabel('p(Y=y|\theta=0.2,n=100)')
set(gca,'XMinorTick','on');
th = 0.8;
subplot(1,2,2);
stem(y,binopdf(y,n,th),'Marker','none','LineWidth',2)
xlabel('y (sum of n trail results)');
ylabel('p(Y=y|\theta=0.8,n=100)')
set(gca,'XMinorTick','on');
sgtitle('Figure 3.3');

%% figure 3.4
th = linspace(0,1,100);

figure
subplot(2,2,1);
a = 1; b = 1;   % prior
n = 5; y = 1;   % sampling function
hold on; box on;
plot(th,betapdf(th,a,b));
plot(th,betapdf(th,a+y,b+n-y));
legend({'prior','posterior'},'Location','best');
xlabel('\theta');
ylabel('P(\theta|y)');

subplot(2,2,2);
a = 3; b = 2;   % prior
n = 5; y = 1;   % sampling function
hold on; box on;
plot(th,betapdf(th,a,b));
plot(th,betapdf(th,a+y,b+n-y));
ylabel('P(\theta|y)');
xlabel('\theta');

subplot(2,2,3);
a = 1; b = 1;   % prior
n = 100; y = 20;   % sampling function
hold on; box on;
plot(th,betapdf(th,a,b));
plot(th,betapdf(th,a+y,b+n-y));
ylabel('P(\theta|y)');
xlabel('\theta');

subplot(2,2,4);
a = 3; b = 2;   % prior
n = 100; y = 20;   % sampling function
hold on; box on;
plot(th,betapdf(th,a,b));
plot(th,betapdf(th,a+y,b+n-y));
ylabel('P(\theta|y)');
xlabel('\theta');

sgtitle('Figure 3.4');


%% figure 3.5
th = linspace(0,1,1000);
a = 1; b=1;
n=10; y=2;
pd = makedist('Beta','a',a+y,'b',b+n-y);
figure
hold on; box on;
plot(th,pdf(pd,th));
ylims = get(gca,'YLim');
plot(icdf(pd,0.025)*ones(1,2),ylims,'k--');
plot(icdf(pd,0.975)*ones(1,2),ylims,'k--');
xlabel('\theta');
ylabel('p(\theta|y)');
sgtitle('Figure 3.5');

%% figure 3.6
% numerically find HDP
[~,idx_max] = max(pdf(pd,th));
% --- get th pairs of identical pdf and calculate pValue
pValue = nan(idx_max,1);
th_left = nan(idx_max,1);
th_right = nan(idx_max,1);
for ii=1:idx_max
    th_left(ii) = th(ii);
    [~,idx_right] = min(abs(pdf(pd,th(idx_max:end)) - pdf(pd,th_left(ii))));
    th_right(ii) = th(idx_right+idx_max-1);
    pValue(ii) = abs(diff(cdf(pd,[th_left(ii),th_right(ii)])));
end

figure
hold on; box on;
plot(th,pdf(pd,th));
ylims = get(gca,'YLim');

% 50% HPD
[~,idx] = min(abs(pValue-0.5));
th_lr = [th_left(idx),th_right(idx)];
p = pdf(pd,th_lr);
h1 = plot(th_lr,p);
plot(repmat(th_lr,2,1),[ylims(1)*ones(1,2);p],'k:');

% 75% HPD
[~,idx] = min(abs(pValue-0.75));
th_lr = [th_left(idx),th_right(idx)];
p = pdf(pd,th_lr);
h2 = plot(th_lr,p);
plot(repmat(th_lr,2,1),[ylims(1)*ones(1,2);p],'k:');

% 95% HPD
[~,idx] = min(abs(pValue-0.95));
th_lr = [th_left(idx),th_right(idx)];
p = pdf(pd,th_lr);
h3 = plot(th_lr,p);
plot(repmat(th_lr,2,1),[ylims(1)*ones(1,2);p],'k:');

% 95% quantile-based
th_lr = icdf(pd,[0.025,0.975]);
p = pdf(pd,th_lr);
h4 = plot(th_lr,p);
plot(repmat(th_lr,2,1),[ylims(1)*ones(1,2);p],'k:');

legend([h1,h2,h3,h4],{'50% HPD','75% HPD','95% HPD','95% quantile-based'});
xlabel('\theta');
ylabel('p(\theta|y)');
sgtitle('Figure 3.6');