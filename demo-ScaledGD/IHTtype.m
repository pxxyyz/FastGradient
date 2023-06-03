function [X, para] = IHTtype(X, U, V, r, opts)

MaxIter = 1e3; % maximum iterations before termination
eps = 1e-10; % tolerance for termination
quadratic_zone = 1e-2;
UseNAG = false; % if true uses UN-accelerated proximal gradient descent (typically slower)
UseRestart = false; % use adaptive restart scheme
% UseRandom = false; % use Random scheme for QR or SVD
eta_den = 10; % eta(k)=(k-1)/(k+den);
Geometric='Riemannian'; % 'Euclidean'
Retract='Orth'; % Proj
approx = 1;
% scale = 0; % test scale
if (~isempty(opts))
    Xstar = opts.Xstar;
    Xob = opts.Xob;
    r = opts.r;
    if isfield(opts,'Geometric');Geometric = opts.Geometric;end
    if isfield(opts,'Retract');Retract = opts.Retract;end
    if isfield(opts,'UseNAG');UseNAG = opts.UseNAG;end
    if isfield(opts,'UseRestart');UseRestart = opts.UseRestart;end
    if isfield(opts,'MaxIter');MaxIter = opts.MaxIter;end
    if isfield(opts,'eps');eps = opts.eps;end
    if isfield(opts,'set_mu');set_mu = opts.set_mu;end
    if isfield(opts,'set_eta');set_eta = opts.set_eta;end
    if isfield(opts,'quadratic_zone');quadratic_zone = opts.quadratic_zone;end
    if isfield(opts,'eta_den');eta_den = opts.eta_den;end
    if isfield(opts,'approx');approx = opts.approx;end
    %     if isfield(opts,'scale');scale = opts.scale;end % test scale
end

if ~UseNAG && UseRestart
    error('Parameter conflict');
end

if UseNAG && strcmp(Retract, 'Proj')
    error('Undefined inverse of Retract operation');
end

err=nan(MaxIter,1);
% obj=nan(MaxIter,1);
time=nan(MaxIter,1);
mu=nan(MaxIter,1);
eta=nan(MaxIter,1);
breakiter=0;
k=1;
Xold=X;

for iter=1:MaxIter
    tic;
    err(iter)=norm(X-Xstar,'fro');
    %     obj(iter)=(X(:)-Xstar(:))'*Theta*(X(:)-Xstar(:));
    if err(iter) < quadratic_zone && ~breakiter
        breakiter=iter;
        if UseRestart; k=1; end
    end
    if err(iter) < eps
        break;
    end
    if exist('set_eta','var')% fix mu
        eta(iter) = set_eta;
    else %
        if UseNAG && iter>1 % Nesterov
            if ~ UseRestart
                k=k+1;
            else % Adaptive Restart
                if (Y(:)-X(:))'*(X(:)-Xold(:))>0
                    k=1;
                else
                    k=k+1;
                end
            end
        end
        eta(iter) = (k-1)/(k+eta_den);
    end
    if ~UseNAG
        Xold = X;
    elseif strcmp(Geometric, 'Riemannian') && ~approx
        Xold = ProjSubspace(Xold, U, V); % inverse of OrthRetract
    end
    Y = X-eta(iter)*(Xold-X);
    Xold = X;
    if UseNAG && eta(iter)>0
        if strcmp(Geometric, 'Euclidean')
            [U,~,V] = svds(Y, r);
        elseif strcmp(Geometric, 'Riemannian') && strcmp(Retract, 'Orth') && ~approx
            [Y, UY, VY] = OrthRetract(Y, U, V);
        end
    else
        UY = U; VY = V;
    end
    G = gradient(Y, Xob, opts);
    RieG = ProjSubspace(G, U, V);
    if exist('set_mu','var') % fix mu
        mu(iter) = set_mu;
    else % exact line search
        %         mu(iter)=(RieG(:)'*RieG(:))/(RieG(:)'*Theta*RieG(:));
        if strcmp(opts.Problem, 'MC')
            ARieG = RieG(opts.A);
        elseif strcmp(opts.Problem, 'MS')
            ARieG = zeros(length(Xob), 1);
            for l = 1:length(Xob)
                ARieG(l) = opts.A{l}(:)'*RieG(:);
            end
        end
        mu(iter)=(RieG(:)'*RieG(:))/(ARieG(:)'*ARieG(:));
    end
    %     X=Y-(1-scale^iter)*mu(iter)*RieG;
    X=Y-mu(iter)*RieG;
    if strcmp(Geometric, 'Euclidean')
        [U,Sig,V] = svds(X, r);
        X = U*Sig*V';
    elseif strcmp(Geometric, 'Riemannian')
        if strcmp(Retract, 'Orth')
            [X, U, V] = OrthRetract(X, UY, VY);
        elseif strcmp(Retract, 'Proj')
            [X, U, V] = ProjRetract(X, UY, VY, r);
        end
    end
    time(iter) = toc;
end
para.err=err;
% para.obj=obj;
para.time=time;
para.mu=mu;
para.breakiter=breakiter;
para.iter=iter-1;

end

function G = gradient(X, Xob, opts)
G=zeros(size(X));
if strcmp(opts.Problem, 'MC')
    G(opts.A)=X(opts.A)-Xob(opts.A);
elseif strcmp(opts.Problem, 'MS')
    for l = 1:length(Xob)
        G = G + (opts.A{l}(:)'*X(:) - Xob(l))*opts.A{l};
    end
end
end

function [R, U, V] = ProjRetract(X, U, V, r) %#ok<*DEFNU>
Z = X;
[Q1, R1] = qr(Z' * U - V * ((Z * V)' * U), 0);
[Q2, R2] = qr(Z  * V - U * ( U' * Z  * V), 0);
M = [U'*Z*V, R1'; R2, zeros(size(R2))];
[UM, S, VM] = svd(M,'econ');
U = [U, Q2] * UM(:,1:r);
V = [V, Q1] * VM(:,1:r);
R = U * S(1:r,1:r) * V';
end

function [R, U, V] = OrthRetract(P, U, V)
PV = P*V; PU = P'*U; UPV = U'*PV;
[Q1, R1] = qr(PV, 0);
[Q2, R2] = qr(PU, 0);
[U, S, V] = svd(R1/UPV*R2','econ');
U = Q1*U;
V = Q2*V;
R = U * S * V'; % R = X*V/(U'*X*V)*U'*X;
end

function P = ProjSubspace(X, U, V)
% PV = V*V'; PU = U*U';
% P = X*PV+PU*X-PU*X*PV;
% P = X*V*V'+U*U'*X-U*U'*X*V*V';
XV = X*V; UX = U'*X; UXV = U'*XV;
P = XV*V'+U*UX-U*UXV*V';
end
