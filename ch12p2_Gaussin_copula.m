% chapter 12.2
% Gaussian copula (Equaiton 12.7)
%
% OrdinalRankings requires https://www.mathworks.com/matlabcentral/fileexchange/19496-rankings?focused=3856857&tab=function
%
% NPQ $2019.11.02$

clear
s0 = readtable(fullfile(pwd,'data','socmob.csv'),'ReadRowNames',1);

% degree and pdegree are ordinal categorical variable ([1,2,3,4,5] and NaN for missing)
deg = s0.DEGREE+1;  % offset by one so 1 for no degree
pdeg = s0.PDEGREE + 1;

% pincome is ordinal categorical
pinc = s0.PINCOME;

% income is numerical
inc = s0.INCOME;

% # of children is numerical
child = s0.CHILDREN;
pchild = s0.PCHILDREN;

% age is numerical
age = s0.AGE;

Y = [inc, deg, child, pinc, pdeg, pchild, age];

%% MCMC for correlation of Y (with missing values considered)
[n,p] = size(Y);

% --- prior 
nu0 = p+2; Sigma0 = inv((p+2)*eye(7));  % Sigma of Y convariance

% --- start values for Sigma 
Sigmai = cov(Y,'omitrows');     % or cov(Y,'partialrows')

% --- start values for z (Eq. 12.7 so that the covariance is Sigma, not
% Rho)
zi = nan(n,p);
% - deg
idx_finite = isfinite(deg);
rankings = OrdinalRankings2(deg(idx_finite));     % random tied ranking
zi(idx_finite,2) = norminv(rankings/(nnz(idx_finite)+1));
% - pdeg
idx_finite = isfinite(pdeg);
rankings = OrdinalRankings2(pdeg(idx_finite));     % random tied ranking
zi(idx_finite,5) = norminv(rankings/(nnz(idx_finite)+1));
% - pinc
idx_finite = isfinite(pinc);
rankings = OrdinalRankings2(pinc(idx_finite));     % random tied ranking
zi(idx_finite,4) = norminv(rankings/(nnz(idx_finite)+1));
% - normalize others
zi(:,[1,3,6,7]) = normalize(Y(:,[1,3,6,7]),'zscore');
% - fillmissing with mean
zi = fillmissing(zi,'constant',nanmean(zi));

% --- mcmc
nmcmc = 1000; %25000
Sigma_mcmc = cell(nmcmc,1);
z_mcmc = cell(nmcmc,1);
tic
for ii=1:nmcmc
    % Gibbs: sample z
    V_z = nan(1,p);
    E_z = nan(n,p);
    a = nan(n,p);
    b = nan(n,p);
    for jj=1:p
        % get Vz and Ez
        Sigmai_cjcj = Sigmai(setdiff(1:p,jj),setdiff(1:p,jj));     % Sigmai(-j,-j)
        Sigmai_jcj = Sigmai(jj,setdiff(1:p,jj));        % Sigmai(j,-j);
        Sjc = Sigmai_jcj*inv(Sigmai_cjcj);
        Sigmai_cjj = Sigmai(setdiff(1:p,jj),jj);        % Sigmai(-j,j);
        zi_cj = zi(:,setdiff(1:p,jj));
        V_z(jj) = Sigmai(jj,jj) - Sjc*Sigmai_cjj;
        E_z(:,jj) = zi_cj*Sjc';
        
        % get a and b
        for kk=1:n
            a(kk,jj) = max([-Inf; zi(Y(:,jj)<Y(kk,jj),jj)]);    
            b(kk,jj) = min([Inf;zi(Y(kk,jj)<Y(:,jj),jj)]);
        end      
    end
    u = unifrnd( normcdf((a-E_z)./sqrt(V_z)), normcdf((b-E_z)./sqrt(V_z)) );
    zi = E_z + sqrt(V_z).*norminv(u);
    
    % Gibbs: update Sigma
    Sigmai = iwishrnd(zi'*zi+Sigma0,nu0+n);      % nu0+n is the degree of freedom

    % collect
    z_mcmc{ii} = zi;
    Sigma_mcmc{ii} = Sigmai;
end
toc

%% MCMC analysis
list_of_variables = {'inc', 'deg', 'child', 'pinc', 'pdeg', 'pchild', 'age'};

fprintf('Posterior mean of correlation by Sigma is \n');
rho = cellfun(@corrcov,Sigma_mcmc,'UniformOutput',false);
rho_by_Sigma = mean(cat(3,rho{:}),3);
array2table(round(rho_by_Sigma,2),'VariableNames',list_of_variables,'RowNames',list_of_variables)

fprintf('Posterior mean of correlation by mean of correlation of Z is\n');
rho_by_z = cellfun(@corrcoef,z_mcmc,'UniformOutput',false);
rho_by_z = mean(cat(3,rho_by_z{:}),3);
array2table(round(rho_by_z,2),'VariableNames',list_of_variables,'RowNames',list_of_variables)

%% Regression coefficient (use rho calcuated from Sigma)
dpgraph_mcmc = cell(nmcmc,1);   % initialize dependence graph matrix
for ii=1:nmcmc
    rhoi = rho{ii};
    dpgraphi = nan(p,p);     % dependence graphi (not symmetric matrix)
    for jj=1:p    
        dpgraphi(jj, setdiff(1:p,jj) ) = rhoi(jj,setdiff(1:p,jj)) / ( rhoi(setdiff(1:p,jj),setdiff(1:p,jj)));
    end
    dpgraph_mcmc{ii} = dpgraphi;
end

dpgraph = cat(3,dpgraph_mcmc{:});
dpgraph_mean = mean(dpgraph,3);
dpgraph_std = std(dpgraph,[],3);
dpgraph_significant = double(abs(dpgraph_mean)>1.96*dpgraph_std).*sign(dpgraph_mean);

fprintf('Coefficient signficance of dependence matrix (95% confidence) (Figure 12.4)');
array2table(dpgraph_significant,'VariableNames',list_of_variables,'RowNames',list_of_variables)