% Ch4.3
%
% NPQ $2019.11.02$

clear

a = 2; b=1; % prior
n1 = 111; c1 = 217;  % sampling
n2 = 44;  c2 = 66;

A1 = a+c1; B1 = 1/(1/b+n1);
A2 = a+c2; B2 = 1/(1/b+n2);

ns = 10000;

% MC for th
th1_post_mc = random('Gamma',A1,B1,[ns,1]);       % random numbers
th2_post_mc = random('Gamma',A2,B2,[ns,1]);       % random numbers

% MC for y (posterior predictive)
y1_post_mc = random('Poisson',th1_post_mc);
y2_post_mc = random('Poisson',th2_post_mc);

dy = y1_post_mc - y2_post_mc;
[counts,edges] = histcounts(dy,'Normalization','count');
dy_list = (edges(2:end)+edges(1:end-1))/2;

% --- plot
linewidth = 2;
figure
stem(dy_list,counts,'Marker','none','LineWidth',linewidth);
xlabel('$D=\widetilde{Y_1}-\widetilde{Y_2}$','Interpreter','latex')
ylabel('$p(D|\mathbf{y_1},\mathbf{y_2})$','Interpreter','latex')
title('Figure 4.5');