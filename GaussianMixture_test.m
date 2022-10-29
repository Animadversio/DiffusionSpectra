GM = GaussianMixture([1,1;-1,-1;2,2],5);
%
densityfun = GM.get_density();
scorefun = GM.get_scorefun();
%
Npnts = 201;
xvec = linspace(-5,5,Npnts);
yvec = linspace(-5,5,Npnts);
[xx, yy] = meshgrid(xvec,yvec);
scorevecs = scorefun(xx(:),yy(:));
%%
Ux = scorevecs(:,1);
Uy = scorevecs(:,2);
Ux = reshape(Ux,size(xx));
Uy = reshape(Uy,size(xx));
%%
subsamp = 7;
figure;
quiver(xx(1:subsamp:end,1:subsamp:end),yy(1:subsamp:end,1:subsamp:end),....
    Ux(1:subsamp:end,1:subsamp:end),Uy(1:subsamp:end,1:subsamp:end),1)
%%
figure;
imagesc(densityfun(xx,yy))
%%

