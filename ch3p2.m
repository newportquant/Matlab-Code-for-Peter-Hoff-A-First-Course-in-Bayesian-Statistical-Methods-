% Ch3.2 
%
% NPQ $2019.11.02$

clear

s0 = readtable(fullfile(pwd,'data','gss.csv'),'ReadRowNames',true);
s = s0(s0.FEMALE==1 & s0.AGE==40 & s0.YEAR>=1990 & isfinite(s0.DEGREE),:).CHILDS;
s1 = s0(s0.FEMALE==1 & s0.AGE==40 & s0.YEAR>=1990 & s0.DEGREE<3,:).CHILDS;      % less than bachelor degree
s2 = s0(s0.FEMALE==1 & s0.AGE==40 & s0.YEAR>=1990 & s0.DEGREE>=3,:).CHILDS;     % with bachelor degree or higher


%% figure 3.7
nc = min(s):max(s);     % # of children
counts = histcounts(s,min(s)-0.5:1:max(s)+0.5,'Normalization','probability');

pd = makedist('Poisson','lambda',mean(s));
figure
subplot(1,2,1);
hold on; box on;
stem(nc-0.05,pdf(pd,nc),'Marker','none','LineWidth',2)
stem(nc+0.05,counts,'Marker','none','LineWidth',2)
xlabel('# of children');
ylabel('p(Y_i=y_i)')
set(gca,'XMinorTick','on');
legend({'Possion model','empirical'},'Location','best');

subplot(1,2,2);
pd = makedist('Poisson','lambda',mean(s)*10);
stem(0:50,pdf(pd,0:50),'Marker','none','LineWidth',2)
xlabel('# of children');
ylabel('p(\SigmaY_i=y|\theta=1.83)')
set(gca,'XMinorTick','on');

sgtitle('Figure 3.7')

%% figure 3.8
th = linspace(0,10,1000);
figure
subplot(2,3,1);
a = 1; b = 1; % gamma distribution
plot(th,pdf(makedist('Gamma','a',a,'b',b),th));
title(sprintf('a=%d,b=1/%d',a,1/b));
xlabel('\theta');
ylabel('p(\theta)');

subplot(2,3,2);
a = 2; b = 1/2; % gamma distribution
plot(th,pdf(makedist('Gamma','a',a,'b',b),th));
title(sprintf('a=%d,b=1/%d',a,1/b));

subplot(2,3,3);
a = 4; b = 1/4; % gamma distribution
plot(th,pdf(makedist('Gamma','a',a,'b',b),th));
title(sprintf('a=%d,b=1/%d',a,1/b));

subplot(2,3,4);
a = 2; b = 1; % gamma distribution
plot(th,pdf(makedist('Gamma','a',a,'b',b),th));
title(sprintf('a=%d,b=1/%d',a,1/b));

subplot(2,3,5);
a = 8; b = 1/4; % gamma distribution
plot(th,pdf(makedist('Gamma','a',a,'b',b),th));
title(sprintf('a=%d,b=1/%d',a,1/b));

subplot(2,3,6);
a = 32; b = 1/16; % gamma distribution
plot(th,pdf(makedist('Gamma','a',a,'b',b),th));
title(sprintf('a=%d,b=1/%d',a,1/b));
sgtitle('Figure 3.8');

%% figure 3.9
nc1 = min(s1):max(s1);     % # of children
counts1 = histcounts(s1,min(s1)-0.5:1:max(s1)+0.5,'Normalization','count');
nc2 = min(s2):max(s2);     % # of children
counts2 = histcounts(s2,min(s2)-0.5:1:max(s2)+0.5,'Normalization','count');

figure
subplot(1,2,1);
stem(nc1,counts1,'Marker','none','LineWidth',2)
xlabel('# of children');
ylabel('n_1(y)')
set(gca,'XMinorTick','on');
title('Less than bachelor''s');

subplot(1,2,2);
stem(nc2,counts2,'Marker','none','LineWidth',2)
xlabel('# of children');
ylabel('n_2(y)')
set(gca,'XMinorTick','on');
title('Bachelor''s or higher');

%% figure 3.10
th = linspace(0,5,1000);
figure
a = 2; b = 1;   % prior (b defined as in matlab)
subplot(1,2,1);
hold on; box on;
plot(th,gampdf(th,a+sum(s1),1/(1/b+length(s1))));   % inverse b in matlab
plot(th,gampdf(th,a+sum(s2),1/(1/b+length(s2))));
plot(th,pdf(makedist('Gamma','a',a,'b',b),th));
xlabel('\theta');
ylabel('probability');
legend({'posterior (less than bachelor''s)','posterior (bachelor''s or higher)','prior'},'Location','best');

y = 0:12;
subplot(1,2,2);     % predictive is negative binomial 
hold on; box on;
stem(y-0.1,pdf(makedist('NegativeBinomial','R',a+sum(s1),'p',(1/b+length(s1))/(1/b+length(s1)+1)),y),'Marker','none','LineWidth',2)
stem(y+0.1,pdf(makedist('NegativeBinomial','R',a+sum(s2),'p',(1/b+length(s2))/(1/b+length(s2)+1)),y),'Marker','none','LineWidth',2)
xlabel('y_{n+1}');
ylabel('p(y_{n+1}|y_1,y_2,...,y_n)');
legend({'less than bachelor''s','bachelor''s or higher'},'Location','best');
set(gca,'xlim',[y(1)-0.5,y(end)+0.5]);

sgtitle('Figure 3.10');