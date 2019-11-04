% ch5.2
% Given known sigma
%
% NPQ $2019.11.02$

clear
s0 = readtable(fullfile(pwd,'data','midge.csv'),'ReadRowNames',true);
s0=convertvars(s0,{'Species'},'categorical');

s = s0{s0.Species == 'Af','Wing_Length'};

%%
% --- prior of theta
mu0 = 1.9; 
tau0 = 0.95;

% --- sampling
n = length(s);  % length of samplings
y_mean = mean(s);
y_var = var(s);

% --- theta conditional on sigma2
sigma = sqrt(y_var);

% --- post of theta
mun = (mu0/tau0^2 + n*y_mean/sigma^2)/(1/tau0^2 + n/sigma^2);
taun = sqrt(1/(1/tau0^2+n/sigma^2));

% --- plot
th = linspace(0,4,1000);
figure
hold on; box on;
plot(th,normpdf(th,mun,taun));
plot(th,normpdf(th,mu0,tau0));
xlabel('\theta');
ylabel('$p(\theta|y_1,...y_n,\sigma^2=0.017)$','Interpreter','latex');
legend('conditional posterior','prior');
title('Figure 5.3');