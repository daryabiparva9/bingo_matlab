% Figures for the prior comparison, made from saved results (no BINGO run).
% Needs the results file written by run_compare_priors.m (set results_file below).
%
% Matrix layout (BINGO's, also used for the ground truth here):
%   M(i,j) = 1  means  gene j regulates gene i
%   -> out-degree of gene j = column sum,  in-degree of gene i = row sum

addpath('./BINGO_files/')
load('./BINGO_files/DREAM100_05_groundtruth_symmterical.mat')
load('./BINGO_files/DREAM100_05_groundtruth_oriented.mat')
results_file = 'results_DREAM100_05_v2.mat';   % new name, so the earlier results are kept
load(results_file, 'results');
priors = fieldnames(results)';           % e.g. {'original','C'}
n = size(results.(priors{1}).P, 1);

%% Ground truth (BINGO layout, self-loops removed)
mask  = ~eye(n);
GT    = (Ground_truth_oriented(1:n,1:n) ~= 0) & mask;
GTsym = (Ground_truth_symmetric(1:n,1:n) ~= 0) & mask;
GTsym = GTsym | GTsym';
upper_tri = triu(true(n),1);                 % each undirected gene pair once
fprintf('\nGround truth: %d directed links, %d undirected pairs\n', nnz(GT), nnz(GTsym & upper_tri));

%% ROC and PR curves, both priors in the same axes
colors = lines(numel(priors));
figure('Name','ROC and PR curves','Position',[100 100 1000 850]);
leg = cell(4, numel(priors));
for p = 1:numel(priors)
    prior = priors{p};
    P = results.(prior).P .* mask;
    Psym = max(P, P');

    % directed: (i,j) scored separately from (j,i)
    [fpr,tpr,~,auc]     = perfcurve(GT(mask), P(mask), true);
    [rec,prc,~,aupr]    = perfcurve(GT(mask), P(mask), true, 'xCrit','reca','yCrit','prec');
    % undirected: each pair once, scored by the larger of the two directions
    [fpr_u,tpr_u,~,auc_u]  = perfcurve(GTsym(upper_tri), Psym(upper_tri), true);
    [rec_u,prc_u,~,aupr_u] = perfcurve(GTsym(upper_tri), Psym(upper_tri), true, 'xCrit','reca','yCrit','prec');
    results.(prior).auroc = auc;   results.(prior).aupr = aupr;
    results.(prior).auroc_undirected = auc_u;  results.(prior).aupr_undirected = aupr_u;

    subplot(2,2,1); hold on; plot(fpr,tpr,'-','Color',colors(p,:),'LineWidth',1.5);
    subplot(2,2,2); hold on; plot(rec,prc,'-','Color',colors(p,:),'LineWidth',1.5);
    subplot(2,2,3); hold on; plot(fpr_u,tpr_u,'-','Color',colors(p,:),'LineWidth',1.5);
    subplot(2,2,4); hold on; plot(rec_u,prc_u,'-','Color',colors(p,:),'LineWidth',1.5);
    leg{1,p} = sprintf('%s (AUROC = %.3f)', prior, auc);
    leg{2,p} = sprintf('%s (AUPR = %.3f)',  prior, aupr);
    leg{3,p} = sprintf('%s (AUROC = %.3f)', prior, auc_u);
    leg{4,p} = sprintf('%s (AUPR = %.3f)',  prior, aupr_u);
    fprintf('%-10s directed: AUROC %.3f  AUPR %.3f | undirected: AUROC %.3f  AUPR %.3f\n', ...
            prior, auc, aupr, auc_u, aupr_u);
end
ttl = {'ROC, directed','PR, directed','ROC, undirected','PR, undirected'};
chance_pr = [mean(GT(mask)), mean(GTsym(upper_tri))];
for s = 1:4
    subplot(2,2,s);
    if mod(s,2) == 1
        plot([0 1],[0 1],'k--'); xlabel('False positive rate'); ylabel('True positive rate');
        legend([leg(s,:), {'random'}],'Location','southeast');
    else
        yline(chance_pr(s/2),'k--'); xlabel('Recall'); ylabel('Precision');
        legend([leg(s,:), {'random'}],'Location','northeast');
    end
    title(ttl{s}); axis square; grid on; box on;
end

%% Out-degree histograms: ground truth vs inferred networks at several thresholds
thresholds = [0.5 0.3 0.1];
BINS   = [0 0; 1 1; 2 2; 3 3; 4 5; 6 9; 10 19; 20 Inf];
blabel = {'0','1','2','3','4-5','6-9','10-19','20+'};

k_gt = sum(GT,1);                        % out-degree of each gene = column sum
names = [{'ground truth'}, arrayfun(@(t) sprintf('inferred, P > %.1f', t), thresholds, 'UniformOutput', false)];

figure('Name','Out-degree histograms','Position',[100 100 600*numel(priors) 450]);
ax = gobjects(1,numel(priors));
fprintf('\n%-10s %-9s %6s %8s %9s %5s\n','prior','threshold','edges','correct','precision','max');
for p = 1:numel(priors)
    prior = priors{p};
    P = results.(prior).P .* mask;
    counts = bin_counts(k_gt, BINS);
    for t = thresholds
        A = P > t;
        k = sum(A,1);
        counts = [counts, bin_counts(k, BINS)]; %#ok<AGROW>
        tp = nnz(A & GT);
        fprintf('%-10s %9.1f %6d %8d %9.3f %5d\n', prior, t, nnz(A), tp, tp/max(nnz(A),1), max(k));
    end
    ax(p) = subplot(1,numel(priors),p);
    hb = bar(counts,'grouped');
    hb(1).FaceColor = [.25 .25 .25];                     % ground truth in dark grey
    for h = hb
        h.BaseValue = 0.5;                               % so bars show on the log axis
    end
    set(gca,'YScale','log','XTick',1:numel(blabel),'XTickLabel',blabel);
    xlabel('out-degree k (number of genes regulated)'); ylabel('number of genes');
    title(sprintf('%s prior', prior)); legend(names,'Location','northeast');
    grid on; box off;
end
linkaxes(ax,'y');
ylim(ax(1), [0.5, n]);
fprintf('ground truth: %d edges, max out-degree %d\n', nnz(GT), max(k_gt));


% -------------------------------------------------------------------------
function c = bin_counts(k, BINS)
% number of genes whose out-degree falls in each bin (column vector)
c = zeros(size(BINS,1),1);
for b = 1:size(BINS,1)
    c(b) = sum(k >= BINS(b,1) & k <= BINS(b,2));
end
end

function print_edges(state, n)
kout = sum(state.S(1:n,1:n),1);
fprintf('edges: %d, mean out-degree %.2f, max %d\n', sum(kout), mean(kout), max(kout));
end
