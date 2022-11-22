classdef GaussianMixture
properties
  nDim
  nBranch
  mus
  precs
  weights
end
methods
  function obj = GaussianMixture(mus,precs,weights)
     obj.nDim = size(mus,2);
     obj.nBranch = size(mus,1);
     obj.mus = mus;
     if numel(precs) == 1
         obj.precs = repmat(precs * eye(obj.nDim),1,1,obj.nBranch);
     elseif all(size(precs) == [obj.nBranch,1]) || all(size(precs) == [1,obj.nBranch])
         obj.precs = repmat(eye(obj.nDim),1,1,obj.nBranch) * reshape(precs,1,1,[]);
     elseif all(size(precs) == [obj.nDim,obj.nDim])
         obj.precs = repmat(precs,1,1,obj.nBranch);
     elseif all(size(precs) == [obj.nDim,obj.nDim,obj.nBranch])
         obj.precs = precs; 
     else
         error("error setting sigmas")
     end
     
     if nargin <=2 
        weights = ones(obj.nBranch, 1); % default to be equal weights
     end
     obj.weights = weights;
  end
  
  function df = get_density(obj)
    df = @densityFun;
    function val = densityFun(x,y)
        val = 0;
        for i = 1:obj.nBranch
            val = val + obj.weights(1) * gaussfun2d(x,y,obj.mus(i,:),obj.precs(:,:,i));
        end
    end
  end
  
  function ldf = get_logdensity(obj)
    df = obj.get_density();
    ldf = @(x,y) log(df(x,y));
  end
  
  function scf = get_scorefun(obj)
    scf = @scoreFun;
    df = obj.get_density();
    function scorevec = scoreFun(x,y)
        probmat = [];
        for i = 1:obj.nBranch
            prob_brc = obj.weights(i) * gaussfun2d(x,y,obj.mus(i,:),obj.precs(:,:,i));
            probmat = [probmat,prob_brc];
        end
        prob = sum(probmat,2);
        scorevec = 0;
        for i = 1:obj.nBranch
            scorevec_brc = -[x-obj.mus(i,1), y-obj.mus(i,2)] * obj.precs(:,:,i);
            scorevec = scorevec + scorevec_brc.*(probmat(:,i)./prob);
        end
    end
  end
%       density = @(x,y) 0.5 * gaussfun2d(x,y,[2,0],beta*eye(2)) + ...
%                  0.5 * gaussfun2d(x,y,[0,2],beta*eye(2)) + ...
%                  0.5 * gaussfun2d(x,y,[-2,0],beta*eye(2)) + ...
%                  0.5 * gaussfun2d(x,y,[0,-2],beta*eye(2)) + ...
%                  0.5 * gaussfun2d(x,y,[1,1],beta*eye(2)) + ...
%                  0.5 * gaussfun2d(x,y,[-1,-1],beta*eye(2));
% % density = @(x,y) 0.3 * gaussfun2d(x,y,mu1,prec1);% + ...
% % %                         0.7 * gaussfun2d(x,y,mu2,prec2);
% logdensity = @(x,y) log(density(x,y));
%   end
  
  
end

end

function val = gaussfun2d(x,y,mu,prec)
val = sqrt(prec(1,1) * prec(2,2) - prec(2,1)^2) *...
                                exp( -1/2*(   (x - mu(1)).^2 * prec(1,1) + ...
                                          2 * (x - mu(1)).*(y - mu(2)) * prec(1,2) + ...
                                              (y - mu(2)).^2 * prec(2,2)));
end                                         