% chapter 9.3
%
% NPQ $2019.11.02$

clear

s0 = readtable(fullfile(pwd,'data','diabetes.csv'),'ReadRowNames',true);
s0 = normalize(s0,'zscore');    % normalize data

n = height(s0);
ntrain = 342;
ntest = n-ntrain;

c = cvpartition(n,'HoldOut',ntest/n);

Xtrain = s0{c.training,2:end};
ytrain = s0.y(c.training);
Xtest = s0{c.test,2:end};
ytest = s0.y(c.test);


%% linear fit
mdl0 = fitlm(Xtrain,ytrain)
ypred0 = predict(mdl0,Xtest);
fprintf('\nMSE of linear fit on test data is %f\n',mean((ypred0-ytest).^2))

%% stepwise 
mdl = stepwiselm(Xtrain,ytrain,'linear','Upper','linear','Criterion','sse','PRemove',0.1)
ypred = predict(mdl,Xtest);
fprintf('\nMSE of stepwise fit on test data is %f\n',mean((ypred-ytest).^2))

%% plot
figure
subplot(1,3,1);
scatter(ytest,ypred0);
hold on; box on;
plot(xlim,xlim);
set(gca,'ylim',[-1.5,2.5]);
xlabel('y_{test}');
ylabel('y_{pred}');

subplot(1,3,2);
linewidth=2;
stem(mdl0.Coefficients.Estimate,'Marker','none','LineWidth',linewidth);
xlabel('regressor index');
ylabel('\beta_{ols}');

subplot(1,3,3)
scatter(ytest,ypred);
hold on; box on;
plot(xlim,xlim);
set(gca,'ylim',[-1.5,2.5]);
xlabel('y_{test}');
ylabel('y_{pred}');
sgtitle('Figure 9.5');


%% randomly permute ytrain
ytrain_rnd = ytrain(randperm(ntrain));
mdl0 = fitlm(Xtrain,ytrain_rnd);
mdl = stepwiselm(Xtrain,ytrain_rnd,'linear','Upper','linear','Criterion','sse','PRemove',0.1);

figure
subplot(1,2,1);
stem(mdl0.Coefficients.tStat,'Marker','none','LineWidth',linewidth);
set(gca,'ylim',[-4,4])
xlabel('regressor index');
ylabel('t-stat');
subplot(1,2,2);
stem([1;1+str2double(extractAfter(mdl.CoefficientNames(2:end)','x'))], mdl.Coefficients.tStat,'Marker','none','LineWidth',linewidth);
set(gca,'ylim',[-4,4])
xlabel('regressor index');
ylabel('t-stat');
sgtitle('Figure 9.6');