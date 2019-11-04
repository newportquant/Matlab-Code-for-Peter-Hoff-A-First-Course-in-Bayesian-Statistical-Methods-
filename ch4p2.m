% CH4.2
%
% NPQ $2019.11.02$

clear

%% figure 4.3
g_list0 = linspace(-5,5,1000);
ns = 10000;

a = 1; b=1;     % prior
n = 860; c = 441;   % sampling 

% --- MC for prior
th_prior_mc = random('Beta',a,b,[ns,1]);       % random numbers
g_prior_mc = log(th_prior_mc./(1-th_prior_mc));
[counts_prior,edges] = histcounts(g_prior_mc,'Normalization','pdf');
g_prior_list = (edges(2:end)+edges(1:end-1))/2;
% --- Nonparametric representation of the probability density (kernel density estimation)
pd_prior = fitdist(g_prior_mc,'Kernel','Kernel','normal');    % use normal kenerl
y_prior_kernel = pdf(pd_prior,g_list0);


% ---  MC for posterior
th_post_mc = random('Beta',a+c,b+n-c,[ns,1]);       % random numbers
g_post_mc = log(th_post_mc./(1-th_post_mc));
[counts_post,edges] = histcounts(g_post_mc,'Normalization','pdf');
g_post_list = (edges(2:end)+edges(1:end-1))/2;
% --- Nonparametric representation of the probability density (kernel density estimation)
pd_post = fitdist(g_post_mc,'Kernel','Kernel','normal');    % use normal kenerl
y_post_kernel = pdf(pd_post,g_list0);

% --- plot
linewidth = 2;
figure
subplot(1,2,1);
hold on; box on;
stem(g_prior_list,counts_prior,'Marker','none','LineWidth',linewidth);
plot(g_list0,y_prior_kernel,'LineWidth',linewidth);
legend({'MC','Kernel density'});
xlabel('log-odds (\gamma)');
ylabel('p(\gamma)');
set(gca,'xlim',[-4,4]);
title('prior');

subplot(1,2,2);
hold on; box on;
plot(g_list0,y_post_kernel,'LineWidth',linewidth);
plot(g_list0,y_prior_kernel,'LineWidth',linewidth);
xlabel('log-odds (\gamma)');
ylabel('p(\gamma|y_1,...y_n)');
set(gca,'xlim',[-4,4]);
legend({'posterior','prior'});
title('posterior');

sgtitle('Figure 4.3');

%% figure 4.4
s0 = readtable(fullfile(pwd,'data','gss.csv'),'ReadRowNames',true);
s = s0(s0.FEMALE==1 & s0.AGE==40 & s0.YEAR>=1990 & isfinite(s0.DEGREE),:).CHILDS;
s1 = s0(s0.FEMALE==1 & s0.AGE==40 & s0.YEAR>=1990 & s0.DEGREE<3,:).CHILDS;      % less than bachelor degree
s2 = s0(s0.FEMALE==1 & s0.AGE==40 & s0.YEAR>=1990 & s0.DEGREE>=3,:).CHILDS;     % with bachelor degree or higher

a = 2; b=1;
A1 = a+sum(s1); B1 = 1/(1/b+length(s1));
A2 = a+sum(s2); B2 = 1/(1/b+length(s2));

ns = 10000;
th1_post_mc = random('Gamma',A1,B1,[ns,1]);       % random numbers
th2_post_mc = random('Gamma',A2,B2,[ns,1]);       % random numbers

g_post_mc = th1_post_mc./th2_post_mc;
pd_post = fitdist(g_post_mc,'Kernel','Kernel','normal');    % use normal kenerl
g_list0 = linspace(min(g_post_mc),max(g_post_mc),1000);
y_post_kernel = pdf(pd_post,g_list0);

figure
plot(g_list0,y_post_kernel);
xlabel('\gamma=\theta_1/\theta_2');
ylabel('p(\gamma|y_1,y_2)');
title('Figure 4.4');
