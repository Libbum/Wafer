%% Load configuration file

%addpath('/path/to/matlab-json/')
json.startup
config = json.read('wafer.cfg');

num = config.grid.size;
dn = config.grid.dn;
dt = config.grid.dt;

x = (dn.*num.x-dn)./2;
y = (dn.*num.y-dn)./2;
z = (dn.*num.z-dn)./2;

grx = linspace(-x,x,num.x);
gry = linspace(-y,y,num.y);
grz = linspace(-z,z,num.z);

%% Wavefunction - Load

number = '0';
partial = false;

if partial
    file = ['wavefunction_' number '_partial.csv']; %#ok<UNRCH>
else
    file = ['wavefunction_' number '.csv'];
end

fid = fopen(file);

data = textscan(fid, '%*u, %*u, %*u, %f');
fclose(fid);

wfn = reshape(data{:},num.x,num.y,num.z);

clear data fid;

%% Wavefunction - Slice in center

mid = floor(num.y/2);
X = linspace(-x,x,num.x);
Y = linspace(-y,y,num.y);
pcolor(X,Y,squeeze(abs(wfn(:,mid,:)))); 
shading flat

%% Wavefunction - Isosurfaces

cmap = parula(256);

hold on;
grid on;
[ox,oy,oz] = meshgrid(grx, gry, grz);
switch number
    case '0'
        alph = [0 0 0.2 0 0.4 0 0.6 0 0.8];
        % alph = [1 0.8 0.6 0.2 0.0 0.2 0.6 0.8 1]; %P-type
        %alph = [0.2 0 0.2 0 0.2 0 0.7 0 1]; %S-type
    case '1'
        %alph = [0.2 0 0.2 0 0.2 0 0.7 0 1]; %S-type
        alph = [1 0.8 0.6 0.2 0.0 0.2 0.6 0.8 1]; %P-type
    case '2'
        %alph = fliplr([0.2 0 0.2 0 0.2 0 0.7 0 1]); %S-type
        alph = [1 0.8 0.6 0.2 0.0 0.2 0.6 0.8 1]; %P-type
    case '3'
        %alph = [0.2 0 0.2 0 0.2 0 0.7 0 1]; %S-type
        alph = [1 0.8 0.6 0.2 0.0 0.2 0.6 0.8 1]; %P-type
    case '4'
        % alph = [1 0 0.4 0 0.3 0 0.2 0.8 0]; %S-type
        alph = [1 0.8 0.6 0.2 0.0 0.2 0.6 0.8 1]; %P-type
    case '5'
        alph = [0.2 0 0.2 0 0.2 0 0.7 0 1]; %S-type
        %alph = [1 0.8 0.6 0.2 0.0 0.2 0.6 0.8 1]; %P-type
    otherwise
        %etc
        alph = [1 0.8 0.6 0.2 0.0 0.2 0.6 0.8 1];
end


cmm = [min(wfn(:)) max(wfn(:))]; %caxis is unreliable

surfs = linspace(min(wfn(:)), max(wfn(:)), 11);
surfs([1 end]) = [];


for jj = 1:length(surfs)
    sh(jj) = patch(isosurface(ox,oy,oz,wfn,surfs(jj)));
    isonormals(ox,oy,oz,wfn,sh(jj))
    %this is a bit cray, but it normalises the color value and finds an
    %appropreate value from the given map.
    cidx = floor(length(cmap).*((surfs(jj) - cmm(1))./(cmm(2)-cmm(1))));
    set(sh(jj),'FaceColor',cmap(cidx,:),'EdgeColor','none','FaceAlpha',alph(jj));
    
end
daspect([1,1,1])
view([43,12]);
axis tight
%xlim([-x x]);
%ylim([-y y]);
%zlim([-z z]);
camlight
lighting gouraud

%% Potential - Load

fid = fopen('potential.csv');

data = textscan(fid, '%*u, %*u, %*u, %f');
fclose(fid);

potential = reshape(data{:},num.x,num.y,num.z);

clear data fid;

%% Potential - Slice in center

mid = floor(num.y/2);
X = linspace(-x,x,num.x);
Y = linspace(-y,y,num.y);
pcolor(X,Y,squeeze(potential(:,mid,:))); 
shading flat


%% Potential - Isosurface

isoV = min(potential(:))*0.55; %The value at which the isosurface will be calculated.

[ox,oy,oz] = meshgrid(grx, gry, grz);
ph = patch(isosurface(ox,oy,oz,potential, isoV)); 
n = isonormals(ox,oy,oz,potential,ph);
set(ph,'FaceColor','red','EdgeColor','none','facealpha',0.5);
daspect([1,1,1])
view(3); axis equal
camlight 
lighting gouraud
hold on

axis tight
%ylim([-x x]);
%zlim([-y y]);
%xlim([-z z]);
