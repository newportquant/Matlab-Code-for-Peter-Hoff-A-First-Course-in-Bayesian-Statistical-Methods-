% chapter 11.1
%
% NPQ $2019.11.02$

clear
s0 = readtable(fullfile(pwd,'data','nelsSES.csv'),'ReadRowNames',true); % SES already centered
stats = grpstats(s0,'sch_id');

ids = stats.sch_id;
nschool = height(stats);

p = 2;  % dimension of X variable (with intercept)

% --- contruct data cell
X = cell(nschool,1);
y = cell(nschool,1);
for ii=1:nschool
    si = s0(s0.sch_id==ids(ii),:);
    X{ii} = [ones(height(si),1), si.stu_ses];
    y{ii} = si.stu_mathscore;    
end

%% OLS
beta_ols = nan(nschool,p);
for ii=1:nschool
    Xi = X{ii};
    yi = y{ii};
    beta_ols(ii,:) = inv(Xi'*Xi)*(Xi'*yi);
end

xlist = linspace(min(s0.stu_ses),max(s0.stu_ses),200)';
ypred = [ones(size(xlist,1),1),xlist]*beta_ols';
figure
subplot(1,3,1);
hold on; box on;
plot(xlist,ypred,'g');
plot(xlist,mean(ypred,2),'linewidth',4);
xlabel('SES');
ylabel('math score');

subplot(1,3,2);
hold on;  box on;
scatter(stats.GroupCount,beta_ols(:,1));
plot(xlim,repmat(mean(beta_ols(:,1)),1,2),'linewidth',2);
xlabel('sample size');
ylabel('intercept')

subplot(1,3,3);
hold on;  box on;
scatter(stats.GroupCount,beta_ols(:,2));
plot(xlim,repmat(mean(beta_ols(:,2)),1,2),'linewidth',2);
xlabel('sample size');
ylabel('slope')

sgtitle('Figure 11.1');
