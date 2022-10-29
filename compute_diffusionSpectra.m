function compute_diffusionSpectra(xs,sigma_seq,xylim,Npnts,suffix)

figdir = "E:\OneDrive - Harvard University\DiffusionSpectralTheory\Simulation\TDDiffSpectra_"+suffix;
mkdir(figdir)

xvec = linspace(-xylim,xylim,Npnts);
yvec = linspace(-xylim,xylim,Npnts);
[xx, yy] = meshgrid(xvec,yvec);
dx = xvec(2) - xvec(1);

[xxid, yyid] = meshgrid(1:Npnts,1:Npnts);
colidv = reshape(xxid,[],1);
rowidv = reshape(yyid,[],1);
nelem = numel(xxid);
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
%%
for iT = 1:numel(sigma_seq)
sigma = sigma_seq(iT);
prec = 1 / sigma^2;
GM = GaussianMixture(xs,prec);
densityfun = GM.get_density();
scorefun = GM.get_scorefun();
scorevecs = scorefun(xx(:),yy(:));
Ux = scorevecs(:,1);
Uy = scorevecs(:,2);
HDiff = 1/2 / dx^2 * DDLpls - 1/2/ dx * (Dx * I .* Ux + Dy * I .* Uy);

Uxmap = reshape(Ux,size(xx));
Uymap = reshape(Uy,size(xx));

figure(10);set(gcf,'pos',[200,200,800,400])
T2 = tiledlayout(1,2,'padding','none','tilesp','none');
nexttile(1)
quiver(xx(1:4:end,1:4:end),yy(1:4:end,1:4:end),Uxmap(1:4:end,1:4:end),Uymap(1:4:end,1:4:end),1)
set(gca,"ydir",'reverse')
axis image
nexttile(2)
imagesc(xvec,yvec,densityfun(xx,yy));
axis image
title(T2,compose("time i=%d sigma=%.1f",iT,sigma));


[V,D] = eigs(HDiff,80,0.1);
eigvals = diag(D);
save(fullfile(figdir,compose("eigenmodes_T%.2f.mat",iT)),'V','eigvals')

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
title(T1,compose("time i=%d sigma=%.1f",iT,sigma));

figure(18);clf
plot(real(diag(D)))
hline(real(diag(D)), 'r:')
ylabel("eigenvalue")
xlabel("eigen id")
title(compose("time i=%d sigma=%.1f",iT,sigma));

saveas(10,fullfile(figdir, compose("potentialField_%.2f.png",iT)))
saveas(17,fullfile(figdir, compose("eigenspectra_%.2f.png",iT)))
saveas(18,fullfile(figdir, compose("eigenmodes_%.2f.png",iT)))

end
end