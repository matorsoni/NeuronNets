function S = constr_sigma(it0, itf, N)
% Given position values in 3D, we are constructing the matrix Sigma that 
% gives the probability of a tracer being at j given that it was previously at i.
% The 3D space is discretized as a cube with sides N, and i and j are the 
% indices representing each one of the N³ cells of this cube.

DIR = '/data-MD3860f-08--FLUID/csiewert/NS_r3_4096/particles_Cu/tracerSorted_';

Sigma = sparse(N^3, N^3);
Norm = sparse(N^3, 1);
Ntracer = 0;

FamilyAll = 0:9; % 0 à 9, tirar a 10 por enquanto

for ifamily = FamilyAll %runs through tracerSorted_0,1,2,...,10
    disp(['loop 1, ' int2str(ifamily)])
	DIRn = [DIR int2str(ifamily) '/']; %string concatenation
	fname = [DIRn 'tS_' int2str(it0)];
    
	fid = fopen(fname,'r'); 
	X = fread(fid,[3 inf],'single'); 
	fclose(fid);
    
	
    Ntracer = Ntracer+size(X,2);
    
    x1 = floor(mod(X(1,:),2*pi)*N/(2*pi))+1; 
    x2 = floor(mod(X(2,:),2*pi)*N/(2*pi))+1; 
    x3 = floor(mod(X(3,:),2*pi)*N/(2*pi))+1; 
	I0 = sub2ind([N N N], x1, x2, x3);
    Norm = Norm+sparse(I0,ones(size(I0)),ones(size(I0)));
    
	clear X x1 x2 x3;
end

for ifamily = FamilyAll
    disp(['loop 2, ' int2str(ifamily)])
	DIRn = [DIR int2str(ifamily) '/'];
	fname = [DIRn 'tS_' int2str(it0)];
	fid = fopen(fname,'r');
	X = fread(fid,[3 inf],'single');
	fclose(fid);
    
	x1 = floor(mod(X(1,:),2*pi)*N/(2*pi))+1; 
    x2 = floor(mod(X(2,:),2*pi)*N/(2*pi))+1; 
    x3 = floor(mod(X(3,:),2*pi)*N/(2*pi))+1; 
	I0 = sub2ind([N N N], x1, x2, x3);
	
    clear X x1 x2 x3;
    
	fname = [DIRn 'tS_' int2str(itf)];
	fid = fopen(fname,'r');
	X = fread(fid,[3 inf],'single');
	fclose(fid);
    
	x1 = floor(mod(X(1,:),2*pi)*N/(2*pi))+1; 
    x2 = floor(mod(X(2,:),2*pi)*N/(2*pi))+1; 
    x3 = floor(mod(X(3,:),2*pi)*N/(2*pi))+1; 
	If = sub2ind([N N N], x1, x2, x3);
    
	clear X x1 x2 x3;
    
	Sigma = Sigma+sparse(I0,If,1./Norm(I0));
end

S = full(Sigma);

end