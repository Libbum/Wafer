%% Wavefunction
fid = fopen('output/wavefunction_0.csv');

data = textscan(fid, '%u, %u, %u, %f');
fclose(fid);

dx = data{1};
dy = data{2};
dz = data{3};

gs = reshape(data{4},max(dx)+1,max(dy)+1,max(dz)+1);

clear data fid;

mid = floor(max(dx)/2);
pcolor(squeeze(abs(gs(:,mid,:)))); 
shading flat

%% Potential
fid = fopen('output/potential.csv');

data = textscan(fid, '%u, %u, %u, %f');
fclose(fid);

dx = data{1};
dy = data{2};
dz = data{3};

potential = reshape(data{4},max(dx)+1,max(dy)+1,max(dz)+1);

clear data fid;

mid = floor(max(dx)/2);
pcolor(squeeze(potential(:,mid,:))); 
shading flat
