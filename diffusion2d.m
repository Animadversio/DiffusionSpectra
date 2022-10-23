%% Diffusion 2D

%% Set up coordinate grid
Npnts = 201;
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
%% Constant velocity field 
sigma = 0.3;
Ux = 0.5; Uy = 0.5;
HDiff = 1/2 * sigma^2 / dx ^2 * DDLpls + 1 / dx * (Dx * I .* Ux + Dy * I .* Uy); 
%%
[Ulpls,Dlpls] = eigs(DDLpls, 10);
%% Define Potential field and score function
gaussfun2d = @(x,y,mu,prec) sqrt(prec(1,1) * prec(2,2) - prec(2,1)^2) *...
                                exp( -1/2*(   (x - mu(1)).^2 * prec(1,1) + ...
                                          2 * (x - mu(1)).*(y - mu(2)) * prec(1,2) + ...
                                              (y - mu(2)).^2 * prec(2,2)));
mu1 = [ 1, 1];
mu2 = [-1,-1];
prec1 = [[5,0.5];[0.5,0.5]];
prec2 = [[1,-1];[-1,2]];
density = @(x,y) 0.5 * gaussfun2d(x,y,mu1,prec1) + ...
                 0.5 * gaussfun2d(x,y,mu2,prec2);
                    
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
sigma = 1.0;
HDiff = sigma^2 * (1/2 / dx ^2 * DDLpls - 1/2/ dx * (Dx * I .* Ux + Dy * I .* Uy)); 
%%
figure(10);set(gcf,'pos',[200,200,800,400])
tiledlayout(1,2,'padding','none','tilesp','none')
nexttile(1)
quiver(xx(1:4:end,1:4:end),yy(1:4:end,1:4:end),Dxmap(1:4:end,1:4:end),Dymap(1:4:end,1:4:end))
set(gca,"ydir",'reverse')
axis image
nexttile(2)
imagesc(xvec,yvec,density(xx,yy));
axis image
% imagesc(density(xx,yy));axis image
% imagesc(reshape(logdensity(xx(:),yy(:)),Npnts,Npnts))

%%
[V,D] = eigs(HDiff,100,0);
% diag(D)
%%
figure(18)
plot(real(diag(D)))
ylabel("eigenvalue")
xlabel("eigen id")
%%
figure(17);set(gcf,'pos',[-122          42        1080         955])
tiledlayout(4,5,'padding','none','tilesp','none')
for i = 1:20
nexttile;
sgn = sign(abs(max(V(:,i)))-abs(min(V(:,i))));
imagesc(sgn * reshape(V(:,i),Npnts,Npnts));
title(D(i,i))
axis equal;axis off
end

%% Naive Convection-Diffusion Equation Solver
dt = 0.0005;
% Sparse points initial condition
% xinit = zeros(nelem,1);%
% xinit(randi(nelem,[50,1])) = 1;

% Random field initial condition
% xinit = rand(nelem,1);

% Gaussian field initial condition
xinit = gaussfun2d(xx(:),yy(:),[0,0],[[0.1,0];[0,0.1]]);
xcur = gpuArray(xinit);
figure(1)
for iT = 0:2000
    xcur = xcur + dt * HDiff * xcur; %Dx / dx
    imagesc(reshape(xcur,Npnts,Npnts))
    title(iT);axis image
    colorbar
    pause(0.001)
end
%%

%% Naive Stupid Wave Equation Solver
dt = 0.02;
% xinit = zeros(nelem,1);%
% xinit(randi(nelem,[10,1])) = 1;
xinit = rand(nelem,1);
xcur = xinit;
vcur = zeros(nelem,1);
figure(1);axis image
for iT = 0:1000
    vcur = vcur + dt * HDiff * xcur;
    xcur = xcur + dt * vcur; %Dx / dx
    imagesc(reshape(xcur,Npnts,Npnts))
    title(iT)
    colorbar
    pause(0.05)
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

function [mapidv,origidv] = rolcol2modlinear_idx(Npnts,rowidv,colidv)
% input a Npnts x Npnts length vectors rowidv and colidv
% 
valmsk = (rowidv > 0) & (rowidv <= Npnts) & ...
         (colidv > 0) & (colidv <= Npnts); 
origidv  = find(valmsk); % linear id of i, j 
mapidv = sub2ind([Npnts,Npnts],rowidv(valmsk),colidv(valmsk)); % linear id of i, j + 1
end