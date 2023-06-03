function [lambda_SGD, lambda_IHT] = GetLambda(opt)
[n1,n2] = size(opt.Xstar);
[Ustar,~,Vstar] = svds(opt.Xstar, opt.r);
PU = Ustar*Ustar'; PV = Vstar*Vstar';
PUperp = eye(n1)-PU; PVperp = eye(n2)-PV;
Potimes = kron(PVperp,PUperp);
Poplus = kron(Vstar*Vstar',eye(n1))+kron(eye(n2),Ustar*Ustar');
if strcmp(opt.Problem, 'MC')
    T=zeros(n1*n2,1);
    T(opt.A)=1;
    Theta=diag(T);
elseif strcmp(opt.Problem, 'MS')
    Theta = zeros(n1*n2);
    for l = 1:length(opt.Xob)
        Theta=Theta+kron(opt.A{l}(:)',opt.A{l}(:));
    end
end
ee0 = sort(abs(eig(Poplus*Theta)));
ee = ee0(ee0>1e-10);
lambda_SGD = [ee(end), ee(1)];
ee0 = sort(abs(eig((eye(n1*n2)-Potimes)*Theta)));
ee = ee0(ee0>1e-10);
lambda_IHT = [ee(end), ee(1)];
end