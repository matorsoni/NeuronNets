dt = 2e-2;
Nstep = 5000;
Nreal = 1000;
x_min = 0;
x_max = 1;
X = zeros(Nstep,Nreal);
V = zeros(Nstep,Nreal);

%gamma = 0.01; %ballistic - uniform
gamma = 0.04; %non trivial - fractal
%gamma = 1; %trivial - atomic
X(1,:) = 0.5;
V(1,:) = -1;
W = sqrt(dt)*randn(Nstep,Nreal);
for it = 2:Nstep
    X(it,:) = X(it-1,:)+dt*V(it-1,:);
    V(it,:) = V(it-1,:)-gamma*dt*V(it-1,:)+gamma*X(it-1,:).*W(it-1,:);
    I = find(X(it,:)>x_max);
    %X(it,I) = x_min + X(it,I) - x_max; % ciercle
    X(it,I) = 2*x_max - X(it,I);
    V(it,I) = -V(it,I);
    I = find(X(it,:)<x_min);
    %X(it,I) = x_max + X(it,I) - x_min;  % ciercle
    X(it,I) = 2*x_min - X(it,I);
    V(it,I) = -V(it,I);
end
%%
figure(1)
clf
semilogy(mean(abs(X),2))
%figure(2)
%ttherm = 20000;
%Xstat = X(ttherm:end,:);
%[h,b] = histcounts(abs(Xstat(:)),100,'Normalization','pdf');
%loglog(.5*(b(2:end)+b(1:end-1)),h)
%ylim([1e-1,10])


%% save to file
Y = X;
save('/home/morsoni/dataset/evol1D/tracer.mat','Y');
clear Y;
%% save to csv
csvwrite('/home/morsoni/dataset/evol1D/input/x.csv', X);
csvwrite('/home/morsoni/dataset/evol1D/input/v.csv', V);
csvwrite('/home/morsoni/dataset/evol1D/input/w.csv', W);




