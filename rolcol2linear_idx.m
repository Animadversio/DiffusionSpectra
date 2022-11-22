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