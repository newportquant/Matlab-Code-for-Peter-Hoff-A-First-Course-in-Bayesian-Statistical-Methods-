% Ch4.1
%
% NPQ $2019.11.02$

clear

%% figure 4.1
a = 2; b=1;         % prior
n=44; c = 66;     % experiment sampling
A = a+c;
B = 1/(1/b+n);

th_list0 = linspace(0.5,2.5,1000);

% --- random numbers
ns = 100;
th = random('Gamma',A,B,[ns,1]);       % random numbers
[counts,edges] = histcounts(th,'Normalization','pdf');
th_list = (edges(2:end)+edges(1:end-1))/2;

% --- Nonparametric representation of the probability density (kernel density estimation)
pd = fitdist(th,'Kernel','Kernel','normal');    % use normal kenerl
y_kernel = pdf(pd,th_list0);

% --- plot
linewidth = 2;
figure
hold on; box on;
stem(th_list,counts,'Marker','none','LineWidth',linewidth);
plot(th_list0,y_kernel,'LineWidth',linewidth);
plot(th_list0,gampdf(th_list0,A,B),'LineWidth',linewidth);
legend({'MC','Kernel density','true density'});
xlabel('\theta');
ylabel('PDF');
title(sprintf('Figure 4.1 (S=%d)',ns));

%% figure 4.2
ns = 1000;
th = random('Gamma',A,B,[ns,1]);       % random numbers
cmean = cumsum(th)./(1:ns)';    
ccdf = nan(size(th));
cquantile = nan(size(th));
for ii=1:ns
%     % --- cumulative cdf: method 1 (method 1 is more accurate when # of MC samples is small, but slower)
%     [f,x] = ecdf(th(1:ii));     % empirical cdf
%     [~,idx] = min(abs(x-1.75));
%     ccdf(ii) = f(idx);
    % --- cumulative cdf: method 2
    ccdf(ii) = nnz(th(1:ii)<=1.75)/ii;
    cquantile(ii) = quantile(th(1:ii),0.975);
end

figure
subplot(1,3,1);
hold on; box on;
plot(1:ns,cmean)
plot([1,ns],repmat(gamstat(A,B),1,2))
xlabel('# of MC samples');
ylabel('cumulative mean');

subplot(1,3,2);
hold on; box on;
plot(1:ns,ccdf)
plot([1,ns],repmat(cdf('Gamma',1.75,A,B),1,2))
xlabel('# of MC samples');
ylabel('cumulative cdf at \theta=1.75');

subplot(1,3,3);
hold on; box on;
plot(1:ns,cquantile)
plot([1,ns],repmat(icdf('Gamma',0.975,A,B),1,2))
xlabel('# of MC samples');
ylabel('cumulative 97.5% quantile');
