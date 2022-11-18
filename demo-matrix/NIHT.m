function [X, para] = NIHT(X, r, opts)
%#ok<*NASGU>

% matrix completion
% optional opts fields defined are below (with defaults)
% to use defaults simply call apg with opts = []

MaxIter = 1e3; % maximum iterations before termination
eps = 1e-10; % tolerance for termination
quadratic_zone = 1e-2;
Problem='MS'; % 'MC'
% step-size throughout, only useful if good
% STEP_SIZE set

if (~isempty(opts))
    Xstar = opts.Xstar;
    Xob = opts.Xob;
    Theta = opts.Theta;
    r = opts.r;

    if isfield(opts,'MaxIter');MaxIter = opts.MaxIter;end
    if isfield(opts,'eps');eps = opts.eps;end
    if isfield(opts,'set_mu');set_mu = opts.set_mu;end
    if isfield(opts,'set_eta');set_eta = opts.set_eta;end
    if isfield(opts,'quadratic_zone');quadratic_zone = opts.quadratic_zone;end
    if isfield(opts,'eta_den');eta_den = opts.eta_den;end
end

err=nan(MaxIter,1);
time=nan(MaxIter,1);
mu=nan(MaxIter,1);
breakiter=0;
Y=X;
for iter=1:MaxIter
    tic;
    err(iter)=norm(X-Xstar,'fro');
    if err(iter) < quadratic_zone && ~breakiter
        breakiter=iter;
    end
    if err(iter) < eps
        break;
    end
    try
        [U,Sig,V] = svds(Y, r);
    catch
        warning('fullSVD: Input to SVD must not contain NaN or Inf');
        break;
    end
    X = U*Sig*V';
    Pu=U*U';
    G = gradient(X, Xob, opts);
    PuG=Pu*G;
    mu(iter)=(PuG(:)'*PuG(:))/(PuG(:)'*Theta*PuG(:));
    Y = X-mu(iter)*G;
    time(iter) = toc;
end
para.err=err;
para.time=time;
para.mu=mu;
para.breakiter=breakiter;
para.iter=iter-1;

end

function G = gradient(X, Xob, opts)
G=zeros(size(X));
if strcmp(opts.Problem, 'MC')
    G(opts.S)=X(opts.S)-Xob(opts.S);
elseif strcmp(opts.Problem, 'MS')
    for l = 1:length(Xob)
        G = G + (opts.As{l}(:)'*X(:) - Xob(l))*opts.As{l};
    end
end
end

