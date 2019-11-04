% chapter 6.5
%
% autocorr needs Econometrics Toolbox
%
% NPQ $2019.11.02$

clear

% --- prior 
pdf_delta0 = [0.45,0.1,0.45];
mu0 = [-3,0,3];
sigma0 = sqrt([1/3,1/3,1/3]);

% --- mc
ns = 10000;
delta_mc = random_delta(pdf_delta0,ns,1);
th_mc = reshape(random('Normal',mu0(delta_mc),sigma0(delta_mc)),size(delta_mc));

[counts,edges] = histcounts(th_mc,'Normalization','pdf');
th_list = (edges(2:end)+edges(1:end-1))/2;

% --- true marginal pdf of th
th_list0 = linspace(-6,6,1000)';
pdf_th = [normpdf(th_list0,mu0(1),sigma0(1)), normpdf(th_list0,mu0(2),sigma0(2)), normpdf(th_list0,mu0(3),sigma0(3))]*pdf_delta0';

% --- plot
linewidth = 2;
figure
hold on; box on;
stem(th_list,counts,'Marker','none','LineWidth',linewidth);
%plot(th_list0,y_kernel,'LineWidth',linewidth);
plot(th_list0,pdf_th,'linewidth',linewidth);
legend({'MC','True'});
xlabel('\theta');
ylabel('p(\theta)');
title('Figure 6.4');

%% MCMC
nmcmc = 10000;
th_mcmc = nan(nmcmc,1);
delta_mcmc = nan(nmcmc,1);
thi = 0;    % start th value
for ii=1:nmcmc
    % sample delta from its conditional posterior
    pdf_deltai = pdf_delta0.*normpdf(thi,mu0,sigma0);
    pdf_deltai = pdf_deltai/sum(pdf_deltai);
    deltai = random_delta(pdf_deltai,1,1);
    % sample delta from its conditional posterior
    thi = random('Normal',mu0(deltai),sigma0(deltai));
    % collect
    th_mcmc(ii) = thi;
    delta_mcmc(ii) = deltai;
end

%% analyisis of MCMC
fprintf('Effective sample size for theta is %d\n',round(multiESS(th_mcmc)));

% plot 1000 samples
[counts,edges] = histcounts(th_mcmc(1:1000),'Normalization','pdf');
th_list = (edges(2:end)+edges(1:end-1))/2;
figure
subplot(1,2,1);
hold on; box on;
stem(th_list,counts,'Marker','none','LineWidth',linewidth);
plot(th_list0,pdf_th,'linewidth',linewidth);
legend({'MCMC','True'});
xlabel('\theta');
ylabel('p(\theta)');

subplot(1,2,2);
plot(th_mcmc(1:1000));
xlabel('iteration');
ylabel('\theta');
sgtitle('Figure 6.5');

% --- plot 10000 samples
[counts,edges] = histcounts(th_mcmc(1:10000),'Normalization','pdf');
th_list = (edges(2:end)+edges(1:end-1))/2;
figure
subplot(1,2,1);
hold on; box on;
stem(th_list,counts,'Marker','none','LineWidth',linewidth);
plot(th_list0,pdf_th,'linewidth',linewidth);
legend({'MCMC','True'});
xlabel('\theta');
ylabel('p(\theta)');

subplot(1,2,2);
plot(th_mcmc(1:10000));
xlabel('iteration');
ylabel('\theta');
sgtitle('Figure 6.6');

%% auto correlation
figure
autocorr(th_mcmc,100);


%% --- function to randomly generate delta given its pdf (marginal or
% condiitonal)
function y = random_delta(pdf_delta,m,n)
r = rand(n,m);
y = nan(size(r));
y(r>=0 & r<pdf_delta(1)) = 1;
y(r>=pdf_delta(1) & r<1-pdf_delta(3)) = 2;
y(r>=1-pdf_delta(3)) = 3;
end


