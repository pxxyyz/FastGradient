clear; close all; clc; %rng(101);
%#ok<*NASGU>

nlist = 50; % [40,60];
plist = [0.4, 0.6];
rlist = [0.06, 0.12];
ErrBound=1e-10;
null = cell(length(nlist),length(plist),length(rlist));
errs=null;
times=null;
mus=null;
names=null;
rhos=null;
breaks=null;

for in = 1:length(nlist)
    for ip = 1:length(plist)
        for ir = 1:length(rlist)
            n = nlist(in);
            m = n;
            r = floor(rlist(ir)*n);
            p = plist(ip);
            s=floor(p*m*n);
            Xstar=randn(m,r)*randn(r,n);
            R=randperm(m*n)';
            S=sort(R(1:s));
            Sc=sort(R(s+1:end));
            Xob=Xstar;
            Xob(Sc)=0;

            %% DRCC
            [Ustar,Sigstar,Vstar] = svds(Xstar, r);
            quadratic_zone = 1e-2*abs((1-1/sqrt(2))*Sigstar(r,r));
            PU = eye(m)-Ustar*Ustar';
            PV = eye(n)-Vstar*Vstar';
            P=eye(m*n)-kron(PV,PU);
            T=zeros(m*n,1);
            T(S)=1;
            Theta=diag(T);
            ee0 = sort(abs(eig(P*Theta)));
            ee = ee0(ee0>1e-10);
            lambda_min = ee(1);
            lambda_max = ee(end);
            muopt=2/(lambda_min+lambda_max);
            muadd=4/(lambda_min+3*lambda_max);
            etaopt = (sqrt(lambda_max/lambda_min)-1)/(sqrt(lambda_max/lambda_min)+1);

            %% Spectral initialization
            [U0, Sigma0, V0] = svds(Xob, r);
            X0 = U0*Sigma0*V0';
            optinit=[];
            optinit.Xstar=Xstar;
            optinit.Xob=Xob;
            optinit.quadratic_zone=quadratic_zone;
            optinit.S=S;
            optinit.Theta=Theta;
            optinit.r=r;
            optinit.Problem='MC';

            %% IHT
            X = X0; U = U0; V = V0;
            opts=optinit;
            opts.Geometric='Euclidean';
            opts.set_mu=muopt;
            opts.UseNAG=false;
            [~, para] = IHTtype(X, U, V, r, opts);

            hatrho = 1-muopt*lambda_min;

            errs{in,ip,ir}=[errs{in,ip,ir},para.err];
            times{in,ip,ir}=[times{in,ip,ir},para.time];
            mus{in,ip,ir}=[mus{in,ip,ir},para.mu];
            breaks{in,ip,ir}=[breaks{in,ip,ir},para.breakiter];
            names{in,ip,ir}=cat(2,names{in,ip,ir},{'IHT($\mu_\star$)'});
            rhos{in,ip,ir}=[rhos{in,ip,ir},hatrho];

            %% NIHT
            [~, para] = NIHT(X0, r, optinit);

            muhat = para.mu(para.iter);
            hatrho = 1-muhat*lambda_min;

            errs{in,ip,ir}=[errs{in,ip,ir},para.err];
            times{in,ip,ir}=[times{in,ip,ir},para.time];
            mus{in,ip,ir}=[mus{in,ip,ir},para.mu];
            breaks{in,ip,ir}=[breaks{in,ip,ir},para.breakiter];
            names{in,ip,ir}=cat(2,names{in,ip,ir},{'NIHT'});
            rhos{in,ip,ir}=[rhos{in,ip,ir},hatrho];

            %% Grad
            X = X0; U = U0; V = V0;
            opts=optinit;
            opts.Geometric='Euclidean';
            opts.UseNAG=false;
            [~, para] = IHTtype(X, U, V, r, opts);

            muhat = para.mu(para.breakiter);
            hatrho = sqrt(1-(muhat^2*lambda_max*lambda_min)/(muhat*(lambda_max+lambda_min)-1));

            errs{in,ip,ir}=[errs{in,ip,ir},para.err];
            times{in,ip,ir}=[times{in,ip,ir},para.time];
            mus{in,ip,ir}=[mus{in,ip,ir},para.mu];
            breaks{in,ip,ir}=[breaks{in,ip,ir},para.breakiter];
            names{in,ip,ir}=cat(2,names{in,ip,ir},{'Grad'});
            rhos{in,ip,ir}=[rhos{in,ip,ir},hatrho];

            %% RGrad-Proj
            X = X0; U = U0; V = V0;
            opts=optinit;
            opts.Geometric='Riemannian';
            opts.UseNAG=false;
            [~, para] = IHTtype(X, U, V, r, opts);

            muhat = para.mu(para.breakiter);
            hatrho = sqrt(1-(muhat^2*lambda_max*lambda_min)/(muhat*(lambda_max+lambda_min)-1));

            errs{in,ip,ir}=[errs{in,ip,ir},para.err];
            times{in,ip,ir}=[times{in,ip,ir},para.time];
            mus{in,ip,ir}=[mus{in,ip,ir},para.mu];
            breaks{in,ip,ir}=[breaks{in,ip,ir},para.breakiter];
            names{in,ip,ir}=cat(2,names{in,ip,ir},{'RGrad-Proj'});
            rhos{in,ip,ir}=[rhos{in,ip,ir},hatrho];

            %% RGrad-Orth
            X = X0; U = U0; V = V0;
            opts=optinit;
            opts.Geometric='Riemannian';
            opts.UseNAG=false;
            [~, para] = IHTtype(X, U, V, r, opts);

            muhat = para.mu(para.breakiter);
            hatrho = sqrt(1-(muhat^2*lambda_max*lambda_min)/(muhat*(lambda_max+lambda_min)-1));

            errs{in,ip,ir}=[errs{in,ip,ir},para.err];
            times{in,ip,ir}=[times{in,ip,ir},para.time];
            mus{in,ip,ir}=[mus{in,ip,ir},para.mu];
            breaks{in,ip,ir}=[breaks{in,ip,ir},para.breakiter];
            names{in,ip,ir}=cat(2,names{in,ip,ir},{'RGrad-Orth'});
            rhos{in,ip,ir}=[rhos{in,ip,ir},hatrho];

            %% NAG-Grad
            X = X0; U = U0; V = V0;
            opts=optinit;
            opts.Geometric='Euclidean';
            opts.UseNAG=true;
            [~, para] = IHTtype(X, U, V, r, opts);

            muhat = para.mu(para.breakiter);
            hatrho = sqrt(1-muhat*lambda_min)*(((para.breakiter-1)*para.breakiter*(para.breakiter+1))/(para.iter*(para.iter+1)*(para.iter+2)))^(1/(para.iter-para.breakiter+1)/2);

            errs{in,ip,ir}=[errs{in,ip,ir},para.err];
            times{in,ip,ir}=[times{in,ip,ir},para.time];
            mus{in,ip,ir}=[mus{in,ip,ir},para.mu];
            breaks{in,ip,ir}=[breaks{in,ip,ir},para.breakiter];
            names{in,ip,ir}=cat(2,names{in,ip,ir},{'NAG'});
            rhos{in,ip,ir}=[rhos{in,ip,ir},hatrho];

            %% NAG-RGrad
            X = X0; U = U0; V = V0;
            opts=optinit;
            opts.Geometric='Riemannian';
            opts.UseNAG=true;
            [~, para] = IHTtype(X, U, V, r, opts);

            muhat = para.mu(para.breakiter);
            hatrho = sqrt(1-muhat*lambda_min)*(((para.breakiter-1)*para.breakiter*(para.breakiter+1))/(para.iter*(para.iter+1)*(para.iter+2)))^(1/(para.iter-para.breakiter+1)/2);

            errs{in,ip,ir}=[errs{in,ip,ir},para.err];
            times{in,ip,ir}=[times{in,ip,ir},para.time];
            mus{in,ip,ir}=[mus{in,ip,ir},para.mu];
            breaks{in,ip,ir}=[breaks{in,ip,ir},para.breakiter];
            names{in,ip,ir}=cat(2,names{in,ip,ir},{'NARG'});
            rhos{in,ip,ir}=[rhos{in,ip,ir},hatrho];

            %% Re-NAG-RGrad
            X = X0; U = U0; V = V0;
            opts=optinit;
            opts.Geometric='Riemannian';
            opts.UseNAG=true;
            opts.UseRestart=true;
            [~, para] = IHTtype(X, U, V, r, opts);

            muhat = 4/(lambda_min+3*lambda_max);
            hatrho = 1-sqrt(4*lambda_min/(lambda_min+3*lambda_max));

            errs{in,ip,ir}=[errs{in,ip,ir},para.err];
            times{in,ip,ir}=[times{in,ip,ir},para.time];
            mus{in,ip,ir}=[mus{in,ip,ir},para.mu];
            breaks{in,ip,ir}=[breaks{in,ip,ir},para.breakiter];
            names{in,ip,ir}=cat(2,names{in,ip,ir},{'NARG+R'});
            rhos{in,ip,ir}=[rhos{in,ip,ir},hatrho];

        end
    end
end


%%
close all;
FontSize = 16;
savefig{1} = figure;
num = max(length(nlist),length(rlist));
set(gcf,'unit','centimeters','position',[10 5 14 8.6].*[1 1 num length(plist)]);
set(gcf, 'DefaultAxesFontSize', FontSize);
out = tight_subplot(length(plist), num, [.05 .075], [.075 .05], [.075 .025]);
legname=null;
sem=null;
mks = {'o', '+', '*', 'x', 'p', 's', 'd','h'};

for in = 1:length(nlist)
    for ip = 1:length(plist)
        for ir = 1:length(rlist)
            %         axes(out((in-1)*length(plist)+ip));
            axes(out((ip-1)*num+max(in, ir))); %#ok<LAXES>
            errnum = size(errs{in,ip,ir}, 2);
            clrs = num2cell(hsv(errnum+1), 2);
            legname{in,ip,ir} = cell(1,2*errnum);
            sem{in,ip,ir} = gobjects(1,2*errnum);
            iterall=sum(errs{in,ip,ir}>0);
            for i = 1 : errnum
                iter = iterall(i);
                sem{in,ip,ir}(2*i-1) = semilogy(errs{in,ip,ir}(: ,i),'--','LineWidth',1.2,'Color',clrs{i},'Marker', mks{i});hold on;
                sem{in,ip,ir}(2*i-1).MarkerIndices = floor(iter/5*rand(1))+1:floor(iter/10):iter;
                legname{in,ip,ir}{2*i-1} = sprintf('($%s$ sec)', num2str(sum(times{in,ip,ir}(1:iter-1, i)),'%1.3f'));
                errhat = nan(iter,1);
                errhat(breaks{in,ip,ir}(i):end)=errs{in,ip,ir}(breaks{in,ip,ir}(i) ,i)*rhos{in,ip,ir}(i).^(0:iter-breaks{in,ip,ir}(i));
                sem{in,ip,ir}(2*i) = semilogy(errhat,':','LineWidth',0.8,'Color',sem{in,ip,ir}(2*i-1).Color);hold on;
                legname{in,ip,ir}{2*i} = sprintf('$\\rho=%s$', num2str(sum(rhos{in,ip,ir}(i)),'%1.4f'));
            end
            xlabel('Iteration'); ylabel('$\|X_t-X_\star\|_F$','Interpreter','latex');
            ylim([ErrBound max(max(errs{in,ip,ir}(:)))]);
            leg = legend(sem{in,ip,ir}, legname{in,ip,ir},'Location','southoutside','NumColumns',4,...
                'Interpreter','latex', 'Orientation','horizontal','FontSize', 10);
            set(leg,'AutoUpdate','off')
            pause(1e-10);
            expname=sprintf('%s~($n=%s,~r=%s,~p=%s$)', ...
                optinit.Problem,num2str(nlist(in)),num2str(floor(rlist(ir)*n)),num2str(plist(ip)));
            title(expname,'Interpreter','latex');
            ax = gca;
            ax.FontName = 'Times New Roman';

            ax = gca;
            axpos = ax.InnerPosition;
            zoomind = {1:5,6:7};

            for iz = 1:length(zoomind)
                ind = cell2mat(zoomind(iz));
                panpos = [.25 .6 .2 .4];
                if range(rhos{in,ip,ir}(ind)) < 0.05
                    rx=min(iterall(ind)); lx=floor(0.9*rx);
                    ly=max(ErrBound,min(min(errs{in,ip,ir}(:,ind))));
                    ry=max(max(errs{in,ip,ir}(lx:rx,ind)));
                    area = [lx ly rx ry];
                    panpos = panpos+iz*[.25 0 0 0];
                    panpos = [axpos(1)+axpos(3)*panpos(1), axpos(2)+axpos(4)*panpos(2), axpos(3:4).*panpos(3:4)];
                    zoomin(ax,area,panpos);
                    pause(1e-10);
                end
            end
        end
    end
end

lin = gobjects(1,errnum);
for i = 1 : errnum
    lin(i) = plot(nan,'-','LineWidth',1.2,'Color',sem{in,ip,ir}(2*i-1).Color,'Marker',sem{in,ip,ir}(2*i-1).Marker);hold on;
end
hold off
Ah=axes('Position',get(gca,'Position'),'Visible','off');
pos=[0.199498289944448,0.005760339750507,0.578340473495905,0.077849606100467];
len=legend(Ah,lin,names{in,ip,ir},'Interpreter','latex','Position',pos,'Orientation','vertical','NumColumns',floor(errnum/2));
ax = gca;
ax.FontName = 'Times New Roman';
ax.FontSize = FontSize;
%% save fig
setprint = true; %true;
if setprint
    [path,name]=fileparts(matlab.desktop.editor.getActiveFilename);
    chemin = '../fig';
    if ~exist(chemin, 'dir')
        mkdir(chemin);
    end
    namefun = @(i,type) sprintf('%s/%s-%d.%s', chemin,name,i,type);
    for i = 1: length(savefig)
        saveas(savefig{i}, namefun(i,'eps'), 'epsc');
        print(savefig{i},'-dpng',namefun(i,'png'));
    end
end
