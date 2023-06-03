function [X, para] = factoredGD(L, R, opts)
%#ok<*NASGU>

% matrix completion
% optional opts fields defined are below (with defaults)
% to use defaults simply call apg with opts = []

MaxIter = 1e3; % maximum iterations before termination
eps = 1e-10; % tolerance for termination
quadratic_zone = 1e-1;
method = 'Scaled'; % Vanilla, Balanced and Scaled
OutPutObj = 0;
% set_mu = 1;
% step-size throughout, only useful if good
% STEP_SIZE set

if (~isempty(opts))
    Xstar = opts.Xstar;
    Xob = opts.Xob;
    r = opts.r;
    if isfield(opts,'MaxIter');MaxIter = opts.MaxIter;end
    if isfield(opts,'eps');eps = opts.eps;end
    if isfield(opts,'set_mu');set_mu = opts.set_mu;end
    if isfield(opts,'quadratic_zone');quadratic_zone = opts.quadratic_zone;end
    if isfield(opts,'OutPutObj');OutPutObj = opts.OutPutObj;end
end

err=nan(MaxIter,1);
obj=nan(MaxIter,1);
time=nan(MaxIter,1);
mu=nan(MaxIter,1);
breakiter=0;


for iter=1:MaxIter
    tic;
    X = L*R';
    err(iter)=norm(X-Xstar,'fro');
%     if OutPutObj
%         if strcmp(opts.Problem, 'MC')
%             obj(iter)=norm(X(opts.A)-Xstar(opts.A),'fro')^2/2;
%         elseif strcmp(opts.Problem, 'MS')
%             er = zeros(length(Xob), 1);
%             for l = 1:length(Xob)
%                 er(l) = opts.A{l}(:)'*(X(:)-Xstar(:));
%             end
%             obj(iter)=norm(er)^2/2;
%         end
%     end

    if err(iter) < quadratic_zone && ~breakiter
        breakiter=iter;
    end
    if err(iter) < eps
        break;
    end
    G = gradient(X, Xob, opts);
    invL = (L'*L)\L';
    invR = R/(R'*R);
    [RieG,PU_RieG,RieG_PV] = ProjSubspace2(G, L, R, invL, invR);
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
        mu(iter)=1*(PU_RieG(:)'*PU_RieG(:)+RieG_PV(:)'*RieG_PV(:))/(ARieG(:)'*ARieG(:));
        %         mu(iter)=(RieG(:)'*RieG(:))/(ARieG(:)'*ARieG(:));
    end

    if strcmp(method, 'Vanilla')
        L = L - mu(iter)*G*R;
        R = R - mu(iter)*G'*L;
    elseif strcmp(method, 'Balanced')
        L = L - mu(iter)*G*R-0.5*L*(L'*L-R'*R);
        R = R - mu(iter)*G'*L-0.5*R*(R'*R-L'*L);
    elseif strcmp(method, 'Scaled')
        %         L = L - mu(iter)*G*R/(R'*R);
        %         R = R - mu(iter)*G'*L/(L'*L);
        L = L - mu(iter)*G*invR;
        R = R - mu(iter)*G'*invL';
    end
    time(iter) = toc;
end
para.err=err;
para.obj=obj;
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

function [P,PUX,XPV] = ProjSubspace2(X, L, R, invL, invR)
% PV = V*V'; PU = U*U';
% P = X*PV+PU*X-PU*X*PV;
% P = X*V*V'+U*U'*X-U*U'*X*V*V';
% PU = L/(L'*L)*L'; PV = R/(R'*R)*R';
% PUX = PU*X;
% XPV = X*PV;
% P = X*PV+PU*X;

% invL = (L'*L)\L';
% invR = R/(R'*R);
LX = invL*X; XR = X*invR; LXR = LX*invR;
PUX = L*LX;
XPV = XR*R';
P = XPV+PUX;
end