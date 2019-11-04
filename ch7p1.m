% chapter 7.1
%
% NPQ $2019.11.02$

clear
s0 = readtable(fullfile(pwd,'data','reading.csv'),'ReadRowNames',true);

% --- for Theta (score)
Mu = [50, 50];
Lambda1 = [64, -48; -48, 144];
Lambda2 = [64, 0 ; 0, 144];
Lambda3 = [64, 48; 48, 144];

y1_list = linspace(20,80,3000);
y2_list = linspace(20,80,301);
[y1_grid,y2_grid] = meshgrid(y1_list,y2_list);

pdf_Th1 = reshape(mvnpdf([y1_grid(:),y2_grid(:)],Mu,Lambda1),size(y1_grid));
pdf_Th2 = reshape(mvnpdf([y1_grid(:),y2_grid(:)],Mu,Lambda2),size(y1_grid));
pdf_Th3 = reshape(mvnpdf([y1_grid(:),y2_grid(:)],Mu,Lambda3),size(y1_grid));


figure
subplot(1,3,1);
imagesc(y1_list,y2_list,pdf_Th1);
axis image;
set(gca,'Ydir','norm');
xlabel('y1');
ylabel('y2');

subplot(1,3,2);
imagesc(y1_list,y2_list,pdf_Th2);
axis image;
set(gca,'Ydir','norm');
xlabel('y1');
ylabel('y2');

subplot(1,3,3);
imagesc(y1_list,y2_list,pdf_Th3);
axis image;
set(gca,'Ydir','norm');
xlabel('y1');
ylabel('y2');
sgtitle('Figure 7.1')
