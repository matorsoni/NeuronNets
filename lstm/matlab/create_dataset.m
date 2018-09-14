clear variables;

load_DIR = '/data-MD3860f-08--FLUID/csiewert/NS_r3_4096/particles_Cu/tracerSorted_';

N = 3;
t0 = 50000; % tracerSorted_10 starts at 57000, while all the others start at 26000
family_range = 0:9;
n_times = 100;
dt = 100; % tracerSorted_2 has dt = 100, while other have dt = 20
x_inputs = zeros(N^6, n_times); %preallocate to gain speed

save_DIR = ['/home/morsoni/dataset/n' int2str(N) '/'];

for i = 0 : n_times-1
    % if some file doesn't exist, skip to next time step until they all exist
    t = t0+i*dt;
    while check_files(family_range(1), family_range(end), t, load_DIR) < 0
        i=i+1;
    end
    
    s = constr_sigma(t, t+dt, N);
    
    filename = [save_DIR 'sigma_' int2str(t) '_' int2str(t+dt) '_' int2str(N)];
    csvwrite(filename, s);
    %s = s(:); % matrix -> column vector
    %x_inputs(:, i+1) = s;
end

%fazer pca e alguma analise aqui
%coeff = pca(x_inputs);

