% Chapter 9.3.2
%
% NPQ $2019.11.02$

clear
s0 = readtable(fullfile(pwd,'data','diabetes.csv'),'ReadRowNames',true);
s0 = normalize(s0,'zscore');    % normalize data

n = height(s0);
p = width(s0)-1;
ntrain = 342;
ntest = n-ntrain;

c = cvpartition(n,'HoldOut',ntest/n);

Xtrain = s0{c.training,2:end};
ytrain = s0.y(c.training);
Xtest = s0{c.test,2:end};
ytest = s0.y(c.test);

%% MCMC for model selection
% --- prior
% z_j:0/1 {j=1,...,p} is uniform distribution (50% chance for each z_j to take 0 or 1)
g = ntrain;
nu0 = 1;    % for sigma2

% --- start values of zi (with intercept, will be p+1 dimesion. The
% intercept is always included)
% zi = logical(randi([0 1],1,p)); 
zi = true(1,p);
lpyi = fcn_lpy([ones(ntrain,1),Xtrain(:,zi)],ytrain,g,nu0);

nmcmc = 2000;
z_mcmc = nan(nmcmc,p);
tic
for ii=1:nmcmc
    % --- sample z
    orderz = randperm(p); % randomize the order
    for jj=1:p
        jz = orderz(jj);
        z_tmp = zi; 
        z_tmp(jz) = ~z_tmp(jz);     % flip sign 
        lpy_tmp = fcn_lpy([ones(ntrain,1),Xtrain(:,z_tmp)],ytrain,g,nu0);
        prob_flip = exp(lpy_tmp - lpyi)/(1+exp(lpy_tmp - lpyi));  % probability to flip
        flipi = binornd(1,prob_flip);
        if flipi == 1   % if need to flip
            zi = z_tmp;
            lpyi = lpy_tmp;
        end
    end
    z_mcmc(ii,:) = zi;
end
z_mcmc = logical(z_mcmc);   % convert to logical

%% prediction
beta_mcmc = zeros(p+1,nmcmc);
ypred_mcmc = nan(ntest,nmcmc);
for ii=1:nmcmc
    Xi = [ones(ntrain,1),Xtrain(:,z_mcmc(ii,:))];
    betai = inv(Xi'*Xi)*(Xi'*ytrain);  % OLS coefficent or (X'*X)\(X'*y)
    ypred_mcmc(:,ii) = [ones(ntest,1),Xtest(:,z_mcmc(ii,:))] * betai;
    beta_mcmc([1,1+find(z_mcmc(ii,:))],ii) = betai;
end

%ypred = mean(ypred_mcmc,2);

% --- equivalently, ypred can be calculated using beta means.
ypred = [ones(ntest,1),Xtest]*mean(beta_mcmc,2); 

fprintf('MSE on test data is %f\n',mean((ypred - ytest).^2))

%% plot
figure
subplot(1,2,1);
stem(mean(z_mcmc),'Marker','none','LineWidth',2);
xlabel('regressor index j');
ylabel('P(z_j=1|y,X)');
subplot(1,2,2);
hold on;
scatter(ytest,ypred);
box on;
plot(xlim,xlim);
set(gca,'ylim',[-1.5,2.5]);
xlabel('y_{test}');
ylabel('y_{pred}');
sgtitle('Figure 9.7');

function lpy = fcn_lpy(X,y,g,nu0)    % calculate P(y|X,z)
    [n,pz] = size(X);

    % --- OLS
    beta = inv(X'*X)*(X'*y);  % OLS coefficent or (X'*X)\(X'*y)
    sigma2 = sum((y-X*beta).^2)/(n-pz);      % unbiased estimate of the std of the innovation term

    SSRg = y'*(eye(n)-g/(g+1)*X*inv(X'*X)*X')*y;    % SSR shrinked by g factor
    lpy = -n/2*log(pi) + gammaln((nu0+n)/2) - gammaln(nu0/2) - pz/2*log(1+g) + nu0/2*log(nu0*sigma2) - (nu0+n)/2*log(nu0*sigma2+SSRg);
end