% This file contains an example of the method with some instructions. For
% more information and for more functionalities, please refer to the
% readme_BINGO file.

%Load data
% addpath('./BINGO_files/')
% load('./BINGO_files/Example_Data.mat')
% load('./Bingo_files/x.mat')

addpath('./BINGO_files/')
load('./BINGO_files/DREAM100_02_full.mat')
load('./BINGO_files/DREAM100_02_groundtruth_symmterical.mat')
load('./BINGO_files/DREAM100_02_groundtruth_oriented.mat')
%% === EXAMPLE 1: Basic use ===

%The data consist of two time series X1 and X2 with dimension 5. The first
%time series consists of 21 measurements at sampling rate dt1 = 0.5. The 
%second time series consists of 14 measurements at measurement times stored
%in the variable 'times'.

% clear('data')
% %The time series data should be put to field ts:
% data.ts={x};
% 
% %Either the sampling rate or measurement times are put to field Tsam:
% % data.Tsam={ts};
% clear('data')
% %The time series data should be put to field ts:
% data.ts={X3,X4,X5};

%Either the sampling rate or measurement times are put to field Tsam:
data.Tsam={0.05};




clear('data')
data.ts={[X1(:,1),X1],[X2(:,1),X2], [X3(:,1),X3],[X4(:,1),X4],[X5(:,1),X5]};
data.Tsam={0.05};

input1 = [[0,ones(1,11), zeros(1,10)]; zeros(4,22)];
input2 = [zeros(1,22); [0,ones(1,11), zeros(1,10)]; zeros(3,22)];
input3 = [zeros(2,22); [0,ones(1,11), zeros(1,10)]; zeros(2,22)];
input4 = [zeros(3,22); [0,ones(1,11), zeros(1,10)]; zeros(1,22)];
input5 = [zeros(4,22); [0,ones(1,11), zeros(1,10)]];
data.input = {input1, input2, input3, input4, input5};

%Initialization
[data,state,parameters]=BINGO_init(data);

% MCMC Burn-in
[~,chain,~,state,stats]=BINGO(data,state,parameters);
disp_stats(' BURN-IN COMPLETE',stats,chain,parameters.its)
%%
% Actual sampling
parameters.its=10000;
[Plink,chain,xstore,state,stats]=BINGO(data,state,parameters);
disp_stats(' SAMPLING COMPLETE',stats,chain,parameters.its)

%The end result:
confidence_matrix=Plink/chain;
%% Run BINGO

% %Initialization
% [data,state,parameters]=BINGO_init(data);
% 
% % MCMC Burn-in
% [Plink,chain,~,state,stats]=BINGO(data,state,parameters);
% disp_stats(' BURN-IN COMPLETE',stats,chain,parameters.its)
% 
% % Actual sampling
% parameters.its=10000;
% [Plink,chain,xstore,state,stats]=BINGO(data,state,parameters);
% disp_stats(' SAMPLING COMPLETE',stats,chain,parameters.its)
% 
% %The end result:
% confidence_matrix=Plink/chain;


%% Collect more samples

%It is possible to continue collecting MCMC samples to improve the accuracy
%of the method.

chain_old=chain;
Plink_old=Plink;

parameters.its=3000;
xstore_old=xstore;
[Plink,chain,xstore,state,stats]=BINGO(data,state,parameters);

xstore=chain_old/(chain+chain_old)*xstore_old+chain/(chain+chain_old)*xstore;
Plink=Plink_old+Plink;
chain=chain_old+chain;
disp_stats(' SAMPLING COMPLETE',stats,chain,parameters.its)

%The end result:
confidence_matrix=Plink/chain;

%% --- Visualization: Link confidence histogram ---

%It is a good idea to plot a histogram of the confidence matrix to help
%decide on a threshold. Here the ground truth is known, and the links
%corresponding to true links are highlighted in the histogram.

%The ground truth adjacency matrix of the system that generated the data
% n = size(confidence_matrix, 1);
% 
% %% Exclude diagonal (set to 0 instead of NaN so thresholding still works cleanly)
% conf_no_diag = confidence_matrix .* ~eye(n);
% 
% %% Threshold: entries bigger than 0.5 become 1, else 0
% conf_thresholded = conf_no_diag > 0.5;
% 
% %% Make symmetric (an edge counts if it exists in either direction)
% conf_symmetric = conf_thresholded | conf_thresholded';
% 
% %% Compare with Ground_truth
% TP = sum(conf_symmetric(:) == 1 & Ground_truth(:) == 1);
% TN = sum(conf_symmetric(:) == 0 & Ground_truth(:) == 0);
% FP = sum(conf_symmetric(:) == 1 & Ground_truth(:) == 0);
% FN = sum(conf_symmetric(:) == 0 & Ground_truth(:) == 1);
% 
% accuracy = (TP + TN) / (TP + TN + FP + FN);
% precision = TP / (TP + FP);
% recall = TP / (TP + FN);
% 
% fprintf('TP=%d TN=%d FP=%d FN=%d\n', TP, TN, FP, FN);
% fprintf('Accuracy=%.3f Precision=%.3f Recall=%.3f\n', accuracy, precision, recall);
% 
% %% Optional: visualize disagreement
% figure;
% imagesc(conf_symmetric - Ground_truth);
% colorbar;
% title('Predicted - Ground truth (symmetric)');

n = size(Ground_truth_oriented,1);


conf_genes = confidence_matrix(1:n,1:n);


mask = ~eye(n);
scores = conf_genes(mask);
labels = Ground_truth_oriented(mask);


[~,~,~,AUROC] = perfcurve(labels,scores,1);
[~,~,~,AUPR]  = perfcurve(labels,scores,1,'xCrit','reca','yCrit','prec');
fprintf('AUROC = %.3f, AUPR = %.3f\n', AUROC, AUPR);
[Xroc, Yroc, ~, AUROC] = perfcurve(labels, scores, 1);
[Xpr,  Ypr,  ~, AUPR]  = perfcurve(labels, scores, 1, 'xCrit', 'reca', 'yCrit', 'prec');
fprintf('AUROC = %.3f, AUPR = %.3f\n', AUROC, AUPR);

figure;
subplot(1,2,1);
plot(Xroc, Yroc, 'b-', 'LineWidth', 1.5); hold on;
plot([0 1], [0 1], 'k--');  % random-classifier baseline
xlabel('False positive rate'); ylabel('True positive rate');
title(sprintf('ROC curve (AUROC = %.3f)', AUROC));
axis square; grid on;

subplot(1,2,2);
plot(Xpr, Ypr, 'r-', 'LineWidth', 1.5);
xlabel('Recall'); ylabel('Precision');
title(sprintf('PR curve (AUPR = %.3f)', AUPR));
axis square; grid on;
%% Directed graphs
Ground_truth = double(Ground_truth_oriented);
gt_dir = Ground_truth .* mask;
G_true = digraph(gt_dir);

threshold = 0.5;
conf_thresh = conf_genes .* (conf_genes > threshold) .* mask;
G_conf = digraph(conf_thresh);

figure;
subplot(1,2,1); plot(G_true,'Layout','circle'); title('Ground Truth Network (directed)');
subplot(1,2,2); plot(G_conf,'Layout','circle'); title('Inferred Network (directed)');
%%
% symmetrical comparison
n = size(Ground_truth_symmetric, 1);   %  number of genes

conf_genes = confidence_matrix(1:n, 1:n);

mask = ~eye(n);
conf_no_diag = conf_genes .* mask;
conf_symmetric = max(conf_no_diag, conf_no_diag');

scores = conf_symmetric(mask);
labels = Ground_truth_symmetric(mask);

[~, ~, ~, AUROC] = perfcurve(labels, scores, 1);
[~, ~, ~, AUPR] = perfcurve(labels, scores, 1, 'xCrit', 'reca', 'yCrit', 'prec');

fprintf('AUROC = %.3f, AUPR = %.3f\n', AUROC, AUPR);


figure; hold on; grid on;
plot(AUROC, AUPR, 'o', 'MarkerSize', 10, 'MarkerFaceColor', [.2 .4 .8], 'MarkerEdgeColor', 'k');
xlim([0 1]); ylim([0 1]);
xlabel('AUROC');
ylabel('AUPR');
title('AUPR vs AUROC');


Ground_truth = double(Ground_truth_symmetric);
gt_symmetric = max(Ground_truth .* mask, (Ground_truth .* mask)');
G_true = graph(gt_symmetric, 'omitselfloops');
threshold = 0.5;
conf_thresh = conf_symmetric .* (conf_symmetric > threshold);
G_conf = graph(conf_thresh, 'omitselfloops');
[Xroc, Yroc, ~, AUROC] = perfcurve(labels, scores, 1);
[Xpr,  Ypr,  ~, AUPR]  = perfcurve(labels, scores, 1, 'xCrit', 'reca', 'yCrit', 'prec');
fprintf('AUROC = %.3f, AUPR = %.3f\n', AUROC, AUPR);

figure;
subplot(1,2,1);
plot(Xroc, Yroc, 'b-', 'LineWidth', 1.5); hold on;
plot([0 1], [0 1], 'k--');
xlabel('False positive rate'); ylabel('True positive rate');
title(sprintf('ROC curve (AUROC = %.3f)', AUROC));
axis square; grid on;

subplot(1,2,2);
plot(Xpr, Ypr, 'r-', 'LineWidth', 1.5);
xlabel('Recall'); ylabel('Precision');
title(sprintf('PR curve (AUPR = %.3f)', AUPR));
axis square; grid on;

%% Plot both graphs side by side
figure;
subplot(1,2,1);
plot(G_true, 'Layout', 'circle');
title('Ground Truth Network');

subplot(1,2,2);
plot(G_conf, 'Layout', 'circle');
title('Inferred Network (Confidence)');

%% --- Visualization: The trajectory estimate ---

%The posterior mean of the continuous expression trajectory is stored in
%the 'xstore' variable, which contains all trajectories concatenated. The
%variable data.plot_index{j} contains the indices of the j^th time series,
%and the variable data.fine_times{j} contains the corresponding time points
%where this trajectory is computed. These are generated by BINGO_init.

%NOTE: The example data is from a white noise driven linear system, and 
%therefore the trajectory estimate mainly resembles a piecewise linear 
%function instead of a smooth curve.
% 
% figure('Position',[60 360 1290 420])
% for jx=1:5
%     subplot(2,5,jx); grid on; hold on;
%     plot(data.fine_times{1},xstore(jx,data.plot_index{1}),'LineWidth',1)
%     plot(data.Tsam{1},data.ts{1}(jx,:),'ok','MarkerFaceColor','k','MarkerSize',3)
%     axis([0 10 0 inf])
%     title(['Time series 1, variable ' num2str(jx)'])
% 
%     subplot(2,5,5+jx); grid on; hold on;
%     plot(data.fine_times{2},xstore(jx,data.plot_index{2}),'LineWidth',1)
%     plot(data.Tsam{2},data.ts{2}(jx,:),'ok','MarkerFaceColor','k','MarkerSize',3)
%     axis([0 10 -inf inf])    
%     title(['Time series 2, variable ' num2str(jx)'])
% end




%% === EXAMPLE 2: perturbation data series ===

%In the third time series X3, a static perturbation has been applied on the
%system. This can be taken into account in the input-field. The confidence 
%matrix is now 5-by-6 where the sixth column corresponds to the 
%perturbation targets. The sampling rate is the same as in time series 1.

%Form the data structure
clear('data')
data.ts={X1,X3};
data.Tsam={dt1};
data.input={zeros(1,size(X1,2)),ones(1,size(X3,2))};

%Initialization
[data,state,parameters]=BINGO_init(data);

% MCMC Burn-in
[~,chain,~,state,stats]=BINGO(data,state,parameters);
disp_stats(' BURN-IN COMPLETE',stats,chain,parameters.its)

% Actual sampling
parameters.its=20000;
[Plink,chain,xstore,state,stats]=BINGO(data,state,parameters);
disp_stats(' SAMPLING COMPLETE',stats,chain,parameters.its)

%The end result:
confidence_matrix=Plink/chain;



%% === EXAMPLE 3: missing measurements ===

Xmiss=X2;

%Throw away a couple of time points
Xmiss(1,12)=0;
Xmiss(3,4)=0;

clear('data')
data.ts={X1,Xmiss};
data.Tsam={dt1,times};

%Missing measurements:
data.missing={[],[1 12; 3 4]};

%NOTE: the initialization file replaces the missing values by interpolated
%values. The whole procedure of handling missing measurements is described
%in the readme-file.


%Then run normally
[data,state,parameters]=BINGO_init(data);
[~,chain,~,state,stats]=BINGO(data,state,parameters);
disp_stats(' BURN-IN COMPLETE',stats,chain,parameters.its)
parameters.its=20000;
[Plink,chain,xstore,state,stats]=BINGO(data,state,parameters);
disp_stats(' SAMPLING COMPLETE',stats,chain,parameters.its)
confidence_matrix=Plink/chain;



%% === EXAMPLE 4: knockout data, prior information ===

%Time series X4 (sampling times in variable 'times') is formed by
%artificially setting the fourth variable to zero during simulation,
%corresponding to a gene knockout experiment.

clear('data')
data.ts={X1,X4};
data.Tsam={dt1,times};

%Give the indices of the knocked-out genes in each time series
data.ko={[],[4]};

%If it is known a priori that gene 1 is regulated by gene 5 and that gene 2
%is certainly not regulated by gene 3, it can be given like this:
%data.sure=zeros(5,5);
%data.sure(1,5)=1;
%data.sure(2,3)=-1;


%Then run normally
[data,state,parameters]=BINGO_init(data);
[~,chain,~,state,stats]=BINGO(data,state,parameters);
disp_stats(' BURN-IN COMPLETE',stats,chain,parameters.its)
parameters.its=20000;
[Plink,chain,xstore,state,stats]=BINGO(data,state,parameters);
disp_stats(' SAMPLING COMPLETE',stats,chain,parameters.its)
confidence_matrix=Plink/chain;

