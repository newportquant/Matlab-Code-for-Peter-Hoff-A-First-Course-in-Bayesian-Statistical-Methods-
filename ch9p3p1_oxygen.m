% chapter 9.3.1 model selection by bayesian analysis
%
% NPQ $2019.11.02$

clear
x1 = [0,0,0,0,0,0,1,1,1,1,1,1]';     % 0/1: running/aerobic
x2 = [23,22,22,25,27,20,31,23,27,28,22,24]'; % age
y = [-0.87,-10.74,-3.27,-1.97,7.50,-7.25,17.05,4.96,10.40,11.05,0.26,2.51]';
X = [ones(size(x1)), x1, x2, x1.*x2];
[n,p] = size(X);

%%
zmat = logical([
    1 0 0 0
    1 1 0 0
    1 0 1 0
    1 1 1 0
    1 1 1 1]);

nmdl = size(zmat,1);    % number of models

% --- overall prior
g = n;
nu0 = 1; 

% --- probability of y conditional on X and z
logp_y = nan(nmdl,1);
for ii=1:nmdl
    z = zmat(ii,:);
    Xi = X(:,z);
    pz = nnz(z);
    
    % --- OLS
    betai = inv(Xi'*Xi)*(Xi'*y);             % coefficent
    SSRi = sum((y-Xi*betai).^2);            % SSR of OLS
    sigma2i = SSRi/(n-pz);                     % unbiased estimate of the std of the innovation term
    
    SSRgi = y'*(eye(n)-g/(g+1)*Xi*inv(Xi'*Xi)*Xi')*y;    % SSR shrinked by g factor
    logp_y(ii) = log(pi^(-n/2)*gamma((nu0+n)/2)/gamma(nu0/2)*(1+g)^(-pz/2)*(nu0*sigma2i)^(nu0/2)/(nu0*sigma2i+SSRgi)^((nu0+n)/2));
end
p_y = exp(logp_y);

% --- posterior of z conditional on X and y
p_z_post = p_y/sum(p_y)    % assume p_z_prior is uniform.


