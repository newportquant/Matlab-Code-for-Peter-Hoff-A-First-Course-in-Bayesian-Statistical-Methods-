% chapter 9.1
%
% NPQ $2019.11.02$

clear

x2 = [0,0,0,0,0,0,1,1,1,1,1,1]';     % 0/1: running/aerobic
x3 = [23,22,22,25,27,20,31,23,27,28,22,24]'; % age
x1 = ones(size(x3));    % for intercept term
x4 = x2.*x3;
y = [-0.87,-10.74,-3.27,-1.97,7.50,-7.25,17.05,4.96,10.40,11.05,0.26,2.51]';
X = [x1, x2, x3, x4];

[n,p] = size(X);

figure
hold on;
scatter(x3(x2==1),y(x2==1));
scatter(x3(x2==0),y(x2==0));
box on;
xlabel('age');
ylabel('change in maximal uptake');
legend({'aerobic','running'},'Location','best');

% --- OLS
beta_ols = inv(X'*X)*(X'*y)             % coefficent
SSR = sum((y-X*beta_ols).^2)            % SSR of OLS
sigma2_ols = SSR/(n-p)                  % unbiased estimate of the std of the innovation term
beta_ols_std = sqrt(diag(inv(X'*X)*sigma2_ols))     % std of coefficients