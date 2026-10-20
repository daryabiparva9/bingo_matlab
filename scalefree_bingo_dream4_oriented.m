% This file contains an example of the method with some instructions. For
% more information and for more functionalities, please refer to the
% readme_BINGO file.

%Load data
addpath('./BINGO_files/')
load('./BINGO_files/DREAM100_05_full.mat')
load('./BINGO_files/DREAM100_05_groundtruth_symmterical.mat')
load('./BINGO_files/DREAM100_05_groundtruth_oriented.mat')

clear('data')
data.ts = {[X1(:,1),X1],[X2(:,1),X2],[X3(:,1),X3],[X4(:,1),X4],[X5(:,1),X5]};
data.Tsam = {0.05};
input1 = [[0,ones(1,11),zeros(1,10)]; zeros(4,22)];
input2 = [zeros(1,22); [0,ones(1,11),zeros(1,10)]; zeros(3,22)];
input3 = [zeros(2,22); [0,ones(1,11),zeros(1,10)]; zeros(2,22)];
input4 = [zeros(3,22); [0,ones(1,11),zeros(1,10)]; zeros(1,22)];
input5 = [zeros(4,22); [0,ones(1,11),zeros(1,10)]];
data.input = {input1,input2,input3,input4,input5};

n = size(X1,1);
outdeg_opts = struct('mean_k',2.0,'k0',4,'gamma',2.5,'k_max',50, 'edge_penalty',-7);  % prior guess, not from the ground truth

for prior = {'C'}                 % add 'B' to compare all three
    rng(1)                                   % same random numbers for every prior
    [data_p,state,parameters] = BINGO_init(data);
    if ~strcmp(prior{1},'original')
        parameters.outdeg_prior = prior{1};
        parameters.outdeg_opts  = outdeg_opts;
    end

    % Burn-in
    [~,chain,~,state,stats] = BINGO_outdeg(data_p,state,parameters);
    disp_stats([' BURN-IN COMPLETE (' prior{1} ')'],stats,chain,parameters.its)
    fprintf('edges: %d, mean out-degree %.2f, max %d\n', sum(sum(state.S(1:n,1:n))), ...
        mean(sum(state.S(1:n,1:n),1)), max(sum(state.S(1:n,1:n),1)));
    % Actual sampling
    parameters.its = 30000;
    [Plink,chain,xstore,state,stats] = BINGO_outdeg(data_p,state,parameters);
    disp_stats([' SAMPLING COMPLETE (' prior{1} ')'],stats,chain,parameters.its)
    fprintf('edges: %d, mean out-degree %.2f, max %d\n', sum(sum(state.S(1:n,1:n))), ...
        mean(sum(state.S(1:n,1:n),1)), max(sum(state.S(1:n,1:n),1)));
    results.(prior{1}).Plink = Plink/chain;  % posterior link probabilities
    confidence_matrix = Plink/chain;
    results.(prior{1}).state = state;
end





% --- Visualization: Link confidence histogram ---


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
% Directed graphs
Ground_truth = double(Ground_truth_oriented);
gt_dir = Ground_truth .* mask;
G_true = digraph(gt_dir);

threshold = 0.5;
conf_thresh = conf_genes .* (conf_genes > threshold) .* mask;
G_conf = digraph(conf_thresh);

figure;
subplot(1,2,1); plot(G_true,'Layout','circle'); title('Ground Truth Network (directed)');
subplot(1,2,2); plot(G_conf,'Layout','circle'); title('Inferred Network (directed)');

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

%Plot both graphs side by side
figure;
subplot(1,2,1);
plot(G_true, 'Layout', 'circle');
title('Ground Truth Network');

subplot(1,2,2);
plot(G_conf, 'Layout', 'circle');
title('Inferred Network (Confidence)');
