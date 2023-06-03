clear; close all; clc;
% rng(101);

ErrBound = 1e-8;
MaxIter = 1e3;
nlist = 40;
savename=sprintf('fig_Scaled_n=%s',num2str(nlist));


%% MC
Problem = 'MC';
plist = 0.5;
rlist = [0.05, 0.1];
null = cell(length(nlist),length(plist),length(rlist));
errs=null;
errshat=null;
times=null;
mus=null;
names=null;
rhos=null;
lambda=null;

for in = 1:length(nlist)
    for ip = 1:length(plist)
        for ir = 1:length(rlist)
            n = nlist(in);
            r = floor(rlist(ir)*n);
            p = plist(ip);
            m = n;
            s=floor(p*m*n);
            Xstar=randn(m,r)*randn(r,n);
            R=randperm(m*n)';
            A=sort(R(1:s));
            Ac=sort(R(s+1:end));
            Xob=Xstar;
            Xob(Ac)=0;
            % Spectral initialization
            [U0, Sigma0, V0] = svds(Xob, r);
            X0 = U0*Sigma0*V0';
            L0 = U0*sqrt(Sigma0);
            R0 = V0*sqrt(Sigma0);

            optinit=[];
            optinit.Xstar=Xstar;
            optinit.Xob=Xob;
            optinit.A=A;
            optinit.r=r;
            optinit.eps=ErrBound;
            optinit.Problem=Problem;
            optinit.MaxIter=MaxIter;
            optinit.quadratic_zone=1e-1;
            [lambda_SGD, lambda_IHT] = GetLambda(optinit);
            lambda{in,ip,ir} = [lambda{in,ip,ir}, lambda_SGD];

            %% IHT
            X = X0; U = U0; V = V0;
            opts=optinit;
            opts.Geometric='Euclidean';
            opts.set_mu=0.5/p;
            opts.UseNAG=false;
            [~, para] = IHTtype(X, U, V, r, opts);
            errs{in,ip,ir}=[errs{in,ip,ir},para.err];
            times{in,ip,ir}=[times{in,ip,ir},para.time];
            mus{in,ip,ir}=[mus{in,ip,ir},para.mu];
            names{in,ip,ir}=cat(2,names{in,ip,ir},{'IHT'});
            errhat = nan(MaxIter,1);
            rhohat = max(abs(1-opts.set_mu*lambda_IHT(2)),abs(opts.set_mu*lambda_IHT(1)-1));
            errhat(para.breakiter:para.iter)=para.err(para.breakiter)*rhohat.^((0:para.iter-para.breakiter)+1);
            errshat{in,ip,ir}=[errshat{in,ip,ir},errhat];
            rhos{in,ip,ir}=[rhos{in,ip,ir},rhohat];
            %% Scaled
            L = L0; R = R0;
            opts=optinit;
            opts.method='Scaled';
            opts.set_mu=0.5/p;
            [~, para] = factoredGD(L, R, opts);
            errs{in,ip,ir}=[errs{in,ip,ir},para.err];
            times{in,ip,ir}=[times{in,ip,ir},para.time];
            mus{in,ip,ir}=[mus{in,ip,ir},para.mu];
            names{in,ip,ir}=cat(2,names{in,ip,ir},{'ScaledGD'});
            errhat = nan(MaxIter,1);
            rhohat = max(abs(1-opts.set_mu*lambda_SGD(2)),abs(opts.set_mu*lambda_SGD(1)-1));
            errhat(para.breakiter:para.iter)=para.err(para.breakiter)*rhohat.^((0:para.iter-para.breakiter)+1);
            errshat{in,ip,ir}=[errshat{in,ip,ir},errhat];
            rhos{in,ip,ir}=[rhos{in,ip,ir},rhohat];
            %% Scaled
            L = L0; R = R0;
            opts=optinit;
            opts.method='Scaled';
            opts.set_mu=1/mean(lambda_SGD);
            [~, para] = factoredGD(L, R, opts);
            errs{in,ip,ir}=[errs{in,ip,ir},para.err];
            times{in,ip,ir}=[times{in,ip,ir},para.time];
            mus{in,ip,ir}=[mus{in,ip,ir},para.mu];
            names{in,ip,ir}=cat(2,names{in,ip,ir},{'ScaledGD ($\mu_\star$)'});
            errhat = nan(MaxIter,1);
            rhohat = max(abs(1-opts.set_mu*lambda_SGD(2)),abs(opts.set_mu*lambda_SGD(1)-1));
            errhat(para.breakiter:para.iter)=para.err(para.breakiter)*rhohat.^((0:para.iter-para.breakiter)+1);
            errshat{in,ip,ir}=[errshat{in,ip,ir},errhat];
            rhos{in,ip,ir}=[rhos{in,ip,ir},rhohat];
            %% Scaled exact line search
            L = L0; R = R0;
            opts=optinit;
            opts.method='Scaled';
            [~, para] = factoredGD(L, R, opts);
            errs{in,ip,ir}=[errs{in,ip,ir},para.err];
            times{in,ip,ir}=[times{in,ip,ir},para.time];
            mus{in,ip,ir}=[mus{in,ip,ir},para.mu];
            names{in,ip,ir}=cat(2,names{in,ip,ir},{'ScaledGD + Exact Line Search'});
            errhat = nan(MaxIter,1);
            muhat = para.mu(para.breakiter);
            rhohat = sqrt(1-(muhat^2*lambda_SGD(1)*lambda_SGD(2))/(muhat*(lambda_SGD(1)+lambda_SGD(2))-1));
            %             rhohat = (lambda_SGD(1)/lambda_SGD(2)-1)/(lambda_SGD(1)/lambda_SGD(2)+1);
            errhat(para.breakiter:para.iter)=para.err(para.breakiter)*rhohat.^((0:para.iter-para.breakiter)+1);
            errshat{in,ip,ir}=[errshat{in,ip,ir},errhat];
            rhos{in,ip,ir}=[rhos{in,ip,ir},rhohat];
        end
    end
end
FontSize = 12;
savefig{1} = figure;
num = max(length(nlist),length(rlist));
set(gcf,'unit','centimeters','position',1*[10 5 14 8.6].*[1 1 num 2*length(plist)]);
set(gcf, 'DefaultAxesFontSize', FontSize);
out = tight_subplot(2*length(plist), num, [.05 .075], [.05 .05], [.075 .025]);
clrs = {'#DB3A34','#3A34DB','#34DB3A','#FFC857','#C857FF','#084C61','#4CC3D9'};% color scheme
mks = {'o', '^', 's', 'h'}; % marker scheme

FaceAlpha = 0.65;
legname=null;
sem=null;
for in = 1:length(nlist)
    for ip = 1:length(plist)
        for ir = 1:length(rlist)
            %         axes(out((in-1)*length(plist)+ip));
            axes(out((ip-1)*num+max(in, ir))); %#ok<LAXES>
            errnum = size(errs{in,ip,ir}, 2);
            %             clrs = num2cell(hsv(errnum+1), 2); %clrs{1} = [0 0 0];
            legname{in,ip,ir} = cell(1,errnum);
            sem{in,ip,ir} = gobjects(1,2*errnum);
            iterall=sum(errs{in,ip,ir}>0);
            for i = 1 : errnum
                iter = iterall(i);
                sem{in,ip,ir}(2*i-1) = semilogy(errs{in,ip,ir}(: ,i),'-','LineWidth',1.2,'Color',clrs{i},'Marker',mks{i});hold on;
                sem{in,ip,ir}(2*i-1).MarkerIndices = floor(linspace(1,iter,4));%1:floor(iter/2):iter;
                sem{in,ip,ir}(2*i-1).MarkerFaceColor = 'none';
                sem{in,ip,ir}(2*i-1).MarkerSize = 8;
                sem{in,ip,ir}(2*i) = semilogy(errshat{in,ip,ir}(: ,i),'--','LineWidth',2.5,'Color',clrs{i},'Marker',mks{i});hold on;
                sem{in,ip,ir}(2*i).Color(4) = FaceAlpha;
                sem{in,ip,ir}(2*i).MarkerIndices = floor(linspace(1,iter,3));%1:floor(iter/3):iter;
                sem{in,ip,ir}(2*i).MarkerFaceColor = clrs{i};
                sem{in,ip,ir}(2*i).MarkerEdgeColor = clrs{i};
                sem{in,ip,ir}(2*i).MarkerSize = 6;
                legname{in,ip,ir}{i} = sprintf('$\\rho=%s$', num2str(sum(rhos{in,ip,ir}(i)),'%1.4f'));
            end
            xlabel('Iteration'); ylabel('$\|X_t-X_\star\|_F$','Interpreter','latex');
            ylim([ErrBound max(max(errs{in,ip,ir}(:)))]);
            leg = legend(sem{in,ip,ir}(2:2:end), legname{in,ip,ir},'Location','southoutside','NumColumns',4,...
                'Interpreter','latex', 'Orientation','horizontal','FontSize', 10);
            set(leg,'AutoUpdate','off')
            kappa = lambda{in,ip,ir}(1)/lambda{in,ip,ir}(2);
            txt = sprintf('$\\frac{\\kappa_{ScaledGD}-1}{\\kappa_{ScaledGD}+1}=%s$', num2str((kappa-1)/(kappa+1),'%1.4f'));
            text(10,1e2*ErrBound,txt,'Interpreter','latex','Color','r');
            pause(1e-10);
            expname=sprintf('%s~($n=%s,~r=%s,~p=%s$)', ...
                optinit.Problem,num2str(nlist(in)),num2str(floor(rlist(ir)*n)),num2str(plist(ip)));
            title(expname,'Interpreter','latex');
            ax = gca;
            ax.FontName = 'Times New Roman';
        end
    end
end

%% MS
Problem = 'MS';
plist = 5;
rlist = [0.05, 0.1];
null = cell(length(nlist),length(plist),length(rlist));
errs=null;
errshat=null;
times=null;
mus=null;
names=null;
rhos=null;
lambda=null;

for in = 1:length(nlist)
    for ip = 1:length(plist)
        for ir = 1:length(rlist)
            n = nlist(in);
            r = floor(rlist(ir)*n);
            p = plist(ip);
            m = floor(p*n*r); %p = 1;
            Xstar = randn(n,r)*randn(r,n);
            Xob = zeros(m, 1);
            A = cell(m, 1);
            for l = 1:m
                A{l} = randn(n, n)/sqrt(m);
                Xob(l) = A{l}(:)'*Xstar(:);
            end
            % Spectral initialization
            Y = zeros(n, n);
            for l = 1:m
                Y = Y + Xob(l)*A{l};
            end
            [U0, Sigma0, V0] = svds(Y, r);
            X0 = U0*Sigma0*V0';
            L0 = U0*sqrt(Sigma0);
            R0 = V0*sqrt(Sigma0);

            optinit=[];
            optinit.Xstar=Xstar;
            optinit.Xob=Xob;
            optinit.A=A;
            optinit.r=r;
            optinit.eps=ErrBound;
            optinit.Problem=Problem;
            optinit.MaxIter=MaxIter;
            optinit.quadratic_zone=1e-1;
            [lambda_SGD, lambda_IHT] = GetLambda(optinit);
            lambda{in,ip,ir} = [lambda{in,ip,ir}, lambda_SGD];

            %% IHT
            X = X0; U = U0; V = V0;
            opts=optinit;
            opts.Geometric='Euclidean';
            opts.set_mu=0.5;
            opts.UseNAG=false;
            [~, para] = IHTtype(X, U, V, r, opts);
            errs{in,ip,ir}=[errs{in,ip,ir},para.err];
            times{in,ip,ir}=[times{in,ip,ir},para.time];
            mus{in,ip,ir}=[mus{in,ip,ir},para.mu];
            names{in,ip,ir}=cat(2,names{in,ip,ir},{'IHT'});
            errhat = nan(MaxIter,1);
            rhohat = max(abs(1-opts.set_mu*lambda_IHT(2)),abs(opts.set_mu*lambda_IHT(1)-1));
            errhat(para.breakiter:para.iter)=para.err(para.breakiter)*rhohat.^((0:para.iter-para.breakiter)+1);
            errshat{in,ip,ir}=[errshat{in,ip,ir},errhat];
            rhos{in,ip,ir}=[rhos{in,ip,ir},rhohat];
            %% Scaled
            L = L0; R = R0;
            opts=optinit;
            opts.method='Scaled';
            opts.set_mu=0.5;
            [~, para] = factoredGD(L, R, opts);
            errs{in,ip,ir}=[errs{in,ip,ir},para.err];
            times{in,ip,ir}=[times{in,ip,ir},para.time];
            mus{in,ip,ir}=[mus{in,ip,ir},para.mu];
            names{in,ip,ir}=cat(2,names{in,ip,ir},{'ScaledGD'});
            errhat = nan(MaxIter,1);
            rhohat = max(abs(1-opts.set_mu*lambda_SGD(2)),abs(opts.set_mu*lambda_SGD(1)-1));
            errhat(para.breakiter:para.iter)=para.err(para.breakiter)*rhohat.^((0:para.iter-para.breakiter)+1);
            errshat{in,ip,ir}=[errshat{in,ip,ir},errhat];
            rhos{in,ip,ir}=[rhos{in,ip,ir},rhohat];
            %% Scaled
            L = L0; R = R0;
            opts=optinit;
            opts.method='Scaled';
            opts.set_mu=1/mean(lambda_SGD);
            [~, para] = factoredGD(L, R, opts);
            errs{in,ip,ir}=[errs{in,ip,ir},para.err];
            times{in,ip,ir}=[times{in,ip,ir},para.time];
            mus{in,ip,ir}=[mus{in,ip,ir},para.mu];
            names{in,ip,ir}=cat(2,names{in,ip,ir},{'ScaledGD ($\mu_\star$)'});
            errhat = nan(MaxIter,1);
            rhohat = max(abs(1-opts.set_mu*lambda_SGD(2)),abs(opts.set_mu*lambda_SGD(1)-1));
            errhat(para.breakiter:para.iter)=para.err(para.breakiter)*rhohat.^((0:para.iter-para.breakiter)+1);
            errshat{in,ip,ir}=[errshat{in,ip,ir},errhat];
            rhos{in,ip,ir}=[rhos{in,ip,ir},rhohat];
            %% Scaled exact line search
            L = L0; R = R0;
            opts=optinit;
            opts.method='Scaled';
            [~, para] = factoredGD(L, R, opts);
            errs{in,ip,ir}=[errs{in,ip,ir},para.err];
            times{in,ip,ir}=[times{in,ip,ir},para.time];
            mus{in,ip,ir}=[mus{in,ip,ir},para.mu];
            names{in,ip,ir}=cat(2,names{in,ip,ir},{'ScaledGD + Exact Line Search'});
            errhat = nan(MaxIter,1);

            muhat = para.mu(para.breakiter);
            rhohat = sqrt(1-(muhat^2*lambda_SGD(1)*lambda_SGD(2))/(muhat*(lambda_SGD(1)+lambda_SGD(2))-1));
            %             rhohat = (lambda_SGD(1)/lambda_SGD(2)-1)/(lambda_SGD(1)/lambda_SGD(2)+1);
            errhat(para.breakiter:para.iter)=para.err(para.breakiter)*rhohat.^((0:para.iter-para.breakiter)+1);
            errshat{in,ip,ir}=[errshat{in,ip,ir},errhat];
            rhos{in,ip,ir}=[rhos{in,ip,ir},rhohat];
        end
    end
end

num = max(length(nlist),length(rlist));
legname=null;
sem=null;
for in = 1:length(nlist)
    for ip = 1:length(plist)
        for ir = 1:length(rlist)
            axes(out((ip)*num+max(in, ir))); %#ok<LAXES>
            errnum = size(errs{in,ip,ir}, 2);
            %             clrs = num2cell(hsv(errnum+1), 2); %clrs{1} = [0 0 0];
            legname{in,ip,ir} = cell(1,errnum);
            sem{in,ip,ir} = gobjects(1,2*errnum);
            iterall=sum(errs{in,ip,ir}>0);
            for i = 1 : errnum
                iter = iterall(i);
                sem{in,ip,ir}(2*i-1) = semilogy(errs{in,ip,ir}(: ,i),'-','LineWidth',1.2,'Color',clrs{i},'Marker',mks{i});hold on;
                sem{in,ip,ir}(2*i-1).MarkerIndices = floor(linspace(1,iter,4));%1:floor(iter/2):iter;
                sem{in,ip,ir}(2*i-1).MarkerFaceColor = 'none';
                sem{in,ip,ir}(2*i-1).MarkerSize = 8;
                sem{in,ip,ir}(2*i) = semilogy(errshat{in,ip,ir}(: ,i),'--','LineWidth',2.5,'Color',clrs{i},'Marker',mks{i});hold on;
                sem{in,ip,ir}(2*i).Color(4) = FaceAlpha;
                sem{in,ip,ir}(2*i).MarkerIndices = floor(linspace(1,iter,3));%1:floor(iter/3):iter;
                sem{in,ip,ir}(2*i).MarkerFaceColor = clrs{i};
                sem{in,ip,ir}(2*i).MarkerEdgeColor = clrs{i};
                sem{in,ip,ir}(2*i).MarkerSize = 6;
                legname{in,ip,ir}{i} = sprintf('$\\rho=%s$', num2str(sum(rhos{in,ip,ir}(i)),'%1.4f'));
            end
            xlabel('Iteration'); ylabel('$\|X_t-X_\star\|_F$','Interpreter','latex');
            ylim([ErrBound max(max(errs{in,ip,ir}(:)))]);
            leg = legend(sem{in,ip,ir}(2:2:end), legname{in,ip,ir},'Location','southoutside','NumColumns',4,...
                'Interpreter','latex', 'Orientation','horizontal','FontSize', 10);
            set(leg,'AutoUpdate','off')
            kappa = lambda{in,ip,ir}(1)/lambda{in,ip,ir}(2);
            txt = sprintf('$\\frac{\\kappa_{ScaledGD}-1}{\\kappa_{ScaledGD}+1}=%s$', num2str((kappa-1)/(kappa+1),'%1.4f'));
            text(10,1e2*ErrBound,txt,'Interpreter','latex','Color','r');
            pause(1e-10);
            expname=sprintf('%s~($n=%s,~r=%s,~p=%s$)', ...
                optinit.Problem,num2str(nlist(in)),num2str(floor(rlist(ir)*n)),num2str(plist(ip)));
            title(expname,'Interpreter','latex');
            ax = gca;
            ax.FontName = 'Times New Roman';
        end
    end
end

lin = gobjects(1,errnum);
for i = 1 : errnum
    lin(i)=plot(nan,'-','LineWidth',1.2,'Color',sem{in,ip,ir}(2*i-1).Color,'Marker',mks{i});hold on;
end
hold off
Ah=axes('Position',get(gca,'Position'),'Visible','off');
pos=[0.23423153662749,0.01293315959468,0.551396301558392,0.032738462536152];
legend(Ah,lin,names{in,ip,ir},'Interpreter','latex','Position',pos,'Orientation','horizontal','NumColumns',4);
ax = gca;
ax.FontName = 'Times New Roman';
ax.FontSize = FontSize;

%% save fig
setprint = true;
if setprint
    [path,name]=fileparts(matlab.desktop.editor.getActiveFilename);
    %     cd(path);
    %     expname = name;
    chemin = '../fig';
    if ~exist(chemin, 'dir')
        mkdir(chemin);
    end
    namefun = @(i,type) sprintf('%s/%s-%d.%s', chemin,savename,i,type);
    for i = 1: length(savefig)
        saveas(savefig{i}, namefun(i,'eps'), 'epsc');
        print(savefig{i},'-dpng',namefun(i,'png'));
    end
end