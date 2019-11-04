% chapter 10.1
%
% NPQ $2019.11.02$


clear

s0 = readtable(fullfile(pwd,'data','sparrows.csv'),'ReadRowNames',true);

x = s0.age;
y = s0.fledged;

X = [ones(height(s0),1),x,x.^2];
[n,p] = size(X);


%%
figure
boxplot(y,x,'Labels',cellstr(num2str(unique(s0.age))));
xlabel('age');
ylabel('offsprint');
title('Figure 10.1');

%% Grid-based 
% --- prior
mu0 = [0,0,0]; Sigma0 = 100*eye(p);   % prior of beta

% --- grid
b1 = 101; b2 = 102; b3 = 103;
beta1_list = 0.27+linspace(-2.5,2.5,b1);
beta2_list = 0.68+linspace(-2,2,b2);
beta3_list = -0.13+linspace(-0.5,0.5,b3);

% --- posterior
prob_joint = nan(b1,b2,b3);
tic
for ii=1:b1
    for jj=1:b2
        for kk=1:b3
            beta123 = [beta1_list(ii),beta2_list(jj),beta3_list(kk)];
            logtheta = X*beta123(:);  % nx1
           prob_joint(ii,jj,kk) = prod(poisspdf(y,exp(logtheta))) * mvnpdf(beta123,mu0,Sigma0); 
        end
    end
end
toc

%% marginal, and joint posterior
prob_joint = prob_joint/trapz(beta1_list,trapz(beta2_list,trapz(beta3_list,prob_joint,3),2),1);  % normalize
pdf_beta2 = trapz(beta3_list, trapz(beta1_list,prob_joint,1),3);
% pdf_beta2 = pdf_beta2/trapz(beta2_list,pdf_beta2);

pdf_beta3 = trapz(beta2_list, trapz(beta1_list,prob_joint,1),2);
% pdf_beta3 = pdf_beta3/trapz(beta3_list,pdf_beta3);

pdf_beta23 = squeeze(trapz(beta1_list,prob_joint,1));  

figure
subplot(1,3,1);
plot(beta2_list, squeeze(pdf_beta2));
xlabel('\beta_2');
ylabel('p(\beta_2|y');

subplot(1,3,2);
plot(beta3_list, squeeze(pdf_beta3));
xlabel('\beta_3');
ylabel('p(\beta_3|y');

subplot(1,3,3);
imagesc(beta2_list,beta3_list,log10(pdf_beta23'));     % squeez beta2 to row and beta3 to column, so need to transpose
caxis([-10,2]);
set(gca,'ydir','norm');
xlabel('\beta_2');
ylabel('\beta_3');

sgtitle('Figure 10.2');