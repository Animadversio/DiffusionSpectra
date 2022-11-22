%% experiments
sigma_seq = 0.2*exp([0.5,1,1.5,2.0]);
xs = [[5,0];[-5,0];[0,6];[0,-5];[1,2]];
compute_diffusionSpectra(xs,sigma_seq,10,201,"3")
%%
sigma_seq = 0.2*exp([0.5,1,1.5,2.0]);
xs = [[5,0];[-5,0];[0,6];[0,-5];[1,2];[6,-6]];
compute_diffusionSpectra(xs,sigma_seq,10,201,"4")
%%
sigma_seq = 0.2*exp([0.5,1,1.5,2.0]);
xs = [[5,0];[-5,0];[0,6];[0,-5];[1,2];[-6,-6]];
compute_diffusionSpectra(xs,sigma_seq,10,201,"5")
%%
sigma_seq = 0.2*exp([0.5,1,1.5,2.0]);
xs = [[5,0];[-5,0];[0,6];[0,-5];[-6,-6]];
compute_diffusionSpectra(xs,sigma_seq,10,201,"6")
%%
sigma_seq = 0.2*exp([0.5,1,1.5,2.0]);
xs = [[5,0];[-5,0];[0,6];[0,-5];[-6,-6];[5,7]];
compute_diffusionSpectra(xs,sigma_seq,10,201,"7")
