
% Given position values in 3D, we are constructing the matrix Sigma that 
% gives the probability of a tracer being at j given that it was previously at i.
% The 3D space is discretized as a cube with sides N, and i and j are the 
% indices representing each one of the N³ cells of this cube.
it0 = 2000;
itf = 3001;
N = 100;
x_max = 1;
x_min = 0;
dx = (x_max - x_min)/N;

filename = '/home/morsoni/dataset/evol1D/tracer.mat';
Sigma = sparse(N, N);
Norm = sparse(N, 1);

load(filename, 'Y');
X = Y(it0,:);    
I0 = floor(X(1,:)*N/(x_max - x_min))+1

Norm = Norm+sparse(I0,ones(size(I0)),ones(size(I0)));
clear X;
	

X = Y(it0,:);
    
I0 = floor(mod(X(1,:),2*pi)*N/(2*pi))+1; 
 
X = Y(itf, :);
    
If = floor(X(1,:)*N/(x_max - x_min))+1
    
    
Sigma = Sigma+sparse(I0',If',1./Norm(I0'),N,N); % ajouter N,N pour forcer S


S = full(Sigma);