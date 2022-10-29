%% Spectrum of Time dependent Diffusion operator 


%% Set up coordinate grid
Npnts = 101;
xvec = linspace(-5,5,Npnts);
yvec = linspace(-5,5,Npnts);
[xx, yy] = meshgrid(xvec,yvec);
dx = xvec(2) - xvec(1);

%% Set up index grid to define differential operators
[xxid, yyid] = meshgrid(1:Npnts,1:Npnts);
colidv = reshape(xxid,[],1);
rowidv = reshape(yyid,[],1);
nelem = numel(xxid);
% sub2ind([Npnts,Npnts],rowidv(valmsk),colidv(valmsk)+1)
%% Define discrete Differential operator as sparse matrices 
cyclic_bc = false;
[mapidv,origidv] = rolcol2linear_idx(Npnts,rowidv,colidv + 1, cyclic_bc);
S_ijpls = sparse(origidv,mapidv,1,nelem,nelem);
[mapidv,origidv] = rolcol2linear_idx(Npnts,rowidv,colidv - 1, cyclic_bc);
S_ijmns = sparse(origidv,mapidv,1,nelem,nelem);
[mapidv,origidv] = rolcol2linear_idx(Npnts,rowidv + 1,colidv, cyclic_bc);
S_iplsj = sparse(origidv,mapidv,1,nelem,nelem);
[mapidv,origidv] = rolcol2linear_idx(Npnts,rowidv - 1,colidv, cyclic_bc);
S_imnsj = sparse(origidv,mapidv,1,nelem,nelem);
[mapidv,origidv] = rolcol2linear_idx(Npnts,rowidv,colidv, cyclic_bc);
S_ij = sparse(origidv,mapidv,1,nelem,nelem);
DDLpls = S_ijpls + S_ijmns + S_iplsj + S_imnsj - 4 * S_ij;
Dx     = (S_ijpls - S_ijmns) / 2;
Dy     = (S_iplsj - S_imnsj) / 2;
I      = speye(nelem);
% S = sparse(i,j,v,nelem,nelem);
%%
%% Define Potential field and score function
gaussfun2d = @(x,y,mu,prec) sqrt(prec(1,1) * prec(2,2) - prec(2,1)^2) *...
                                exp( -1/2*(   (x - mu(1)).^2 * prec(1,1) + ...
                                          2 * (x - mu(1)).*(y - mu(2)) * prec(1,2) + ...
                                              (y - mu(2)).^2 * prec(2,2)));
mu1 = [ 1, 1];
mu2 = [-1,-1];
% prec1 = [[1,0];[0.5,0.5]];
% prec2 = [[1,-1];[-1,2]];
% sigma = 0.5;
figdir = "E:\OneDrive - Harvard University\DiffusionSpectralTheory\Simulation\TDDiffSpectra_2";
mkdir(figdir)
for T = [0:0.05:3]
% T = 0;
sigma = 0.2 * exp(T); % too small is unstable
beta = 1/sigma^2;
% density = @(x,y) 0.5 * gaussfun2d(x,y,[2,0],beta*eye(2)) + ...
%                  0.5 * gaussfun2d(x,y,[0,2],beta*eye(2)) + ...
%                  0.5 * gaussfun2d(x,y,[-2,0],beta*eye(2)) + ...
%                  0.5 * gaussfun2d(x,y,[0,-2],beta*eye(2)) + ...
%                  0.5 * gaussfun2d(x,y,[1.4,1.4],beta*eye(2)) + ...
%                  0.5 * gaussfun2d(x,y,[-1.4,-1.4],beta*eye(2));
                    
% density = @(x,y) 0.5 * gaussfun2d(x,y,[2,0],beta*eye(2)) + ...
%                  0.5 * gaussfun2d(x,y,[0,2],beta*eye(2)) + ...
%                  0.5 * gaussfun2d(x,y,[-2,0],beta*eye(2)) + ...
%                  0.5 * gaussfun2d(x,y,[0,-2],beta*eye(2)) + ...
%                  0.5 * gaussfun2d(x,y,[.6,.6],beta*eye(2)) + ...
%                  0.5 * gaussfun2d(x,y,[-.6,-.6],beta*eye(2));
density = @(x,y) 0.5 * gaussfun2d(x,y,[2,0],beta*eye(2)) + ...
                 0.5 * gaussfun2d(x,y,[0,2],beta*eye(2)) + ...
                 0.5 * gaussfun2d(x,y,[-2,0],beta*eye(2)) + ...
                 0.5 * gaussfun2d(x,y,[0,-2],beta*eye(2)) + ...
                 0.5 * gaussfun2d(x,y,[1,1],beta*eye(2)) + ...
                 0.5 * gaussfun2d(x,y,[-1,-1],beta*eye(2));
% density = @(x,y) 0.3 * gaussfun2d(x,y,mu1,prec1);% + ...
% %                         0.7 * gaussfun2d(x,y,mu2,prec2);
logdensity = @(x,y) log(density(x,y));
ddx = 0.0001;
Dxlogdens = @(x,y) (logdensity(x+ddx/2,y) - logdensity(x-ddx/2,y)) / ddx;
Dylogdens = @(x,y) (logdensity(x,y+ddx/2) - logdensity(x,y-ddx/2)) / ddx;
Dxmap = Dxlogdens(xx,yy);
Dymap = Dylogdens(xx,yy);
Ux = reshape(Dxmap,[],1);
Uy = reshape(Dymap,[],1);

%% Diffusion operator 
% sigma = 1.0;
HDiff = 1 * (1/2 / dx ^2 * DDLpls - 1/2/ dx * (Dx * I .* Ux + Dy * I .* Uy)); 

figure(10);set(gcf,'pos',[200,200,800,400])
T2 = tiledlayout(1,2,'padding','none','tilesp','none');
nexttile(1)
quiver(xx(1:4:end,1:4:end),yy(1:4:end,1:4:end),Dxmap(1:4:end,1:4:end),Dymap(1:4:end,1:4:end))
set(gca,"ydir",'reverse')
axis image
nexttile(2)
imagesc(xvec,yvec,density(xx,yy));
axis image
title(T2,compose("T=%.1f sigma=%.1f",T,sigma));

[V,D] = eigs(HDiff,80,0.1);
eigvals = diag(D);
save(fullfile(figdir,compose("eigenmodes_T%.2f.mat",T)),'V','eigvals')
figure(17);set(17,'pos',[10          32        1080         955])
T1 = tiledlayout(4,5,'padding','none','tilespacing','none');
for i = 1:20
nexttile(T1,i);
eigmode = real(V(:,i));
sgn = sign(abs(max(eigmode))-abs(min(eigmode)));
imagesc(sgn * reshape(eigmode,Npnts,Npnts));
title(compose("%.5f",D(i,i)))
axis equal;axis off
end
title(T1,compose("T=%.1f sigma=%.1f",T,sigma))

figure(18);clf
plot(real(diag(D)))
hline(real(diag(D)), 'r:')
ylabel("eigenvalue")
xlabel("eigen id")
title(compose("T=%.1f sigma=%.1f",T,sigma))

saveas(10,fullfile(figdir, compose("potentialField_%.2f.png",T)))
saveas(17,fullfile(figdir, compose("eigenspectra_%.2f.png",T)))
saveas(18,fullfile(figdir, compose("eigenmodes_%.2f.png",T)))
end


function [mapidv,origidv] = rolcol2linear_idx(Npnts,rowidv,colidv,do_wrap)
% input a Npnts x Npnts length vectors rowidv and colidv
% 
if nargin==3, do_wrap=false; end
if do_wrap
rowidv = mod(rowidv - 1, Npnts) + 1;
colidv = mod(colidv - 1, Npnts) + 1;
valmsk = ones(numel(rowidv),1,'logical');
origidv = 1:numel(rowidv);
else
valmsk = (rowidv > 0) & (rowidv <= Npnts) & ...
         (colidv > 0) & (colidv <= Npnts); 
origidv  = find(valmsk); % linear id of i, j 
end
mapidv = sub2ind([Npnts,Npnts],rowidv(valmsk),colidv(valmsk)); % linear id of i, j + 1
end