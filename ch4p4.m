% chapter 4.4
%
% NPQ $2019.11.02$

clear

s0 = readtable(fullfile(pwd,'data','gss.csv'),'ReadRowNames',true);
s = s0(s0.FEMALE==1 & s0.AGE==40 & s0.YEAR>=1990 & isfinite(s0.DEGREE),:).CHILDS;
s1 = s0(s0.FEMALE==1 & s0.AGE==40 & s0.YEAR>=1990 & s0.DEGREE<3,:).CHILDS;      % less than bachelor degree
s2 = s0(s0.FEMALE==1 & s0.AGE==40 & s0.YEAR>=1990 & s0.DEGREE>=3,:).CHILDS;     % with bachelor degree or higher

a = 2; b=1;
A1 = a+sum(s1); B1 = 1/(1/b+length(s1));
A2 = a+sum(s2); B2 = 1/(1/b+length(s2));

nc1 = min(s1):max(s1);     % # of children
counts1 = histcounts(s1,min(s1)-0.5:1:max(s1)+0.5,'Normalization','probability');

figure
subplot(1,2,1);
hold on; box on;
stem(nc1-0.05,counts1,'Marker','none','LineWidth',2)
stem(nc1+0.1,pdf(makedist('NegativeBinomial','R',a+sum(s1),'p',(1/b+length(s1))/(1/b+length(s1)+1)),nc1),'Marker','none','LineWidth',2)
xlabel('# of children');
ylabel('p(Y_i=y_i)');
legend({'empiriical','predictive'});

%% MC
nmc = 10000;
t = nan(nmc,1);
for ii=1:nmc
    th1_post_mc = random('Gamma',A1,B1,1);       % random numbers
    y1_post_mc = random('Poisson',th1_post_mc,1,length(s1));
    t(ii) = nnz(y1_post_mc == 2)/nnz(y1_post_mc == 1);
end
[counts,edges] = histcounts(t,20,'Normalization','pdf');
t_list = (edges(2:end)+edges(1:end-1))/2;

subplot(1,2,2);
hold on; box on;
stem(t_list,counts,'Marker','none','LineWidth',2)
plot(repmat(nnz(s1==2)/nnz(s1==1),1,2),[0,mean(ylim)]);
xlabel('$t(\tilde{Y})$','Interpreter','latex')
ylabel('p(t)');
legend({'MC','observation'},'Location','best');
sgtitle('Figure 4.6');