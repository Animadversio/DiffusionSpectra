gridN = 2001;
xgrid = linspace(-10,10,gridN);
delta = xgrid(2) - xgrid(1);
%
Diffmat = zeros(gridN) + 1/2 * diag(ones(gridN-1,1),1) - 1/2 * diag(ones(gridN-1,1),-1);
DDiffmat = -2 * diag(ones(gridN,1)) + diag(ones(gridN-1,1),1) + diag(ones(gridN-1,1),-1);
%%
sigma = 0.5;
scorefun = @(x) - (x - 1) ./ sigma^2;
score_vec = scorefun(xgrid);
%%
pi1 = 0.3;
pi2 = 1 - pi1;
mu1 = - 2; sd1 = 1;
mu2 =   3; sd2 = 2;
densityfunc = @(x) pi1 * exp(- (x - mu1).^2 / sd1^2) / sd1 + ...
                   pi2 * exp(- (x - mu2).^2 / sd2^2) / sd2;
logdensityfunc = @(x) log(densityfunc(x));
scorefun = @(x) (logdensityfunc(x + delta / 2) - logdensityfunc(x - delta / 2) ) / delta ;
score_vec = scorefun(xgrid);
figure;
yyaxis left
plot(xgrid,densityfunc(xgrid));hold on 
yyaxis right
plot(xgrid,scorefun(xgrid));hold on
hline(0)

%%
H = -Diffmat .* score_vec / delta + DDiffmat / delta.^2;
[V, D] = eig(H);
eigvals = diag(D);
[eigsort, sortid] = sort(real(eigvals),'descend');

%%
figure('pos',[200,400,1200,350]);
subplot(121)
plot(eigvals,'.')
subplot(122)
for i = 1:10
plot(xgrid,V(:,sortid(i)));hold on
end
%%
figure('pos',[200,100,400,800]);
tiledlayout(15,1,'pad','compact','tilespacing','none')
for i = 1:15
nexttile
plot(xgrid,V(:,sortid(i)));hold on
if i ~= 15, xticklabels([]); end
end

