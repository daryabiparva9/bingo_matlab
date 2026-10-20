function logT = outdegree_logT(n_targets, prior, opts)
%OUTDEGREE_LOGT  Log-probability table for an out-degree prior on a directed network.
%
%   logT = outdegree_logT(n_targets, prior)
%   logT = outdegree_logT(n_targets, prior, opts)
%
%   n_targets : number of genes a single gene can regulate. In BINGO, self-loops
%               S(i,i) are allowed, so use n_targets = n (number of genes).
%   prior     : 'C' (exact power law, maximum entropy on the out-degree histogram)
%               'B' (hidden fields, kappa ~ kappa^-gamma)
%   opts      : optional struct with fields (defaults in brackets)
%                 gamma      power-law exponent                         [2.5]
%                 k0         power law imposed for k >= k0              [4]
%                 mean_k     mean out-degree = expected #edges / n      [2]
%                 kappa_max  Option B only: largest expected out-degree [50]
%                 edge_penalty  extra log-prior per edge, <= 0           [0]
%                            Adds edge_penalty*k to logT(k+1): every edge
%                            costs the same extra amount, so the network gets
%                            sparser while the relative preference for adding
%                            edges to hubs is unchanged. E.g. -7 makes the first
%                            edge of a gene cost about as much as link_pr = 1e-5.
%                 k_max      largest allowed out-degree    [floor((n_targets-1)/2)]
%                            Above k_max every extra edge costs 50 (a near-hard
%                            cap). Needed for 'C': without it the per-row prior is
%                            U-shaped in k (a row with ALL targets is almost as
%                            probable as a row with one), and the MCMC drifts to
%                            very dense columns. Below (n_targets-1)/2 the
%                            prior always decreases with every added edge.
%
%   logT is a column vector of length n_targets+1, with
%       logT(k+1) = log P(a gene's targets | it has out-degree k)
%                 = log P(out-degree = k) - log nchoosek(n_targets, k)
%   so that  log P(S) = sum over genes j of logT(outdeg_j + 1)  (+ const).
%   In BINGO the out-degree of gene j is the COLUMN sum sum(S(1:n, j)).

if nargin < 3, opts = struct(); end
known = {'gamma','k0','mean_k','kappa_max','edge_penalty','k_max'};
bad = setdiff(fieldnames(opts), known);
if ~isempty(bad)
    error('outdegree_logT: unknown option(s) %s. Valid (case-sensitive): %s', ...
          strjoin(bad', ', '), strjoin(known, ', '));
end
gamma     = getopt(opts, 'gamma', 2.5);
k0        = getopt(opts, 'k0', 4);
mean_k    = getopt(opts, 'mean_k', 2);
kappa_max = getopt(opts, 'kappa_max', 50);
edge_penalty = getopt(opts, 'edge_penalty', 0);
if edge_penalty > 0, error('edge_penalty must be <= 0'); end

N  = n_targets;
k_max = getopt(opts, 'k_max', floor((N-1)/2));
if k_max < k0 || k_max > N, error('k_max must satisfy k0 <= k_max <= n_targets'); end
k  = (0:N)';
lC = gammaln(N+1) - gammaln(k+1) - gammaln(N-k+1);       % log nchoosek(N, k)

switch upper(prior)
    case 'C'
        % Power law A k^-gamma for k >= k0-1, binomial-shaped C(N,k) x^k below,
        % joined smoothly; x is fixed by the mean out-degree.
        pl    = zeros(N+1,1);
        idx   = k >= k0-1;
        pl(idx) = k(idx).^(-gamma);
        pl(k > k_max) = 0;                             % truncate the power law at k_max
        Spl   = sum(pl);
        below = k < k0-1;

        f = @(lx) pmfC(lx, k, lC, pl, Spl, below, k0) - mean_k;
        lx = bisect(f, -40, 10);
        [~, pk] = pmfC(lx, k, lC, pl, Spl, below, k0);
        logT = log(pk) - lC;

    case 'B'
        % kappa ~ kappa^-gamma on [kappa_min, kappa_max]; kappa_min fixed by the mean.
        if kappa_max >= N, error('kappa_max must be smaller than n_targets'); end
        a = gamma - 1;
        pmean = @(km) ((km^(1-a) - kappa_max^(1-a))/(a-1)) / ((km^(-a) - kappa_max^(-a))/a);
        kappa_min = bisect(@(km) pmean(km) - mean_k, 1e-6, 0.999*kappa_max);

        nq  = 20000;                                       % quadrature nodes = quantiles of rho
        u   = ((0:nq-1)' + 0.5)/nq;
        kap = (kappa_min^(-a) - u*(kappa_min^(-a) - kappa_max^(-a))).^(-1/a);
        q   = kap/N;
        L   = log(q)*k' + log1p(-q)*(N-k)';                % nq x (N+1): log q^k (1-q)^(N-k)
        mx  = max(L, [], 1);
        logT = (mx + log(mean(exp(L - mx), 1)))';          % log E_rho[ q^k (1-q)^(N-k) ]

    otherwise
        error('prior must be ''B'' or ''C''');
end

logT = logT + edge_penalty*k;                     % uniform extra cost per edge

% near-hard cap above k_max (finite, so a state that starts above the cap is
% pulled down instead of producing NaN in BINGO's acceptance ratio)
cap_slope = 50;
above = k > k_max;
logT(above) = logT(k_max+1) - cap_slope*(k(above) - k_max);
end

% -------------------------------------------------------------------------
function [m, pk] = pmfC(lx, k, lC, pl, Spl, below, k0)
b     = exp(lC(below) + k(below)*lx);
Bs    = sum(b);
ratio = exp(lC(k0) - lC(k0-1) + lx);          % binomial ratio pi_{k0-1}/pi_{k0-2} (1-based: lC(k0) is k = k0-1)
c     = ratio*b(end)/Bs;
d     = pl(k0)/Spl;                           % pl at k = k0-1
frac  = c/(d + c);                            % probability mass on the power-law part
pk    = zeros(size(k));
pk(below) = (1-frac)*b/Bs;
pk(~below) = frac*pl(~below)/Spl;
m = sum(k.*pk);
end

function x = bisect(f, lo, hi)
flo = f(lo);
if sign(flo) == sign(f(hi)), error('outdegree_logT: no root in bracket'); end
for it = 1:200
    mid = 0.5*(lo + hi);
    fm  = f(mid);
    if sign(fm) == sign(flo), lo = mid; flo = fm; else, hi = mid; end
    if hi - lo < 1e-13, break; end
end
x = 0.5*(lo + hi);
end

function v = getopt(s, name, default)
if isfield(s, name), v = s.(name); else, v = default; end
end
