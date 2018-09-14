dir = '/home/morsoni/dataset/evol1D/output/';
load([dir 'pos_in.csv']);
load([dir 'pos_out.csv']);
load([dir 'vel_in.csv']);
load([dir 'vel_out.csv']);
load([dir 'error.csv']);

figure(1)
plot(pos_in, 'b');
hold on;
plot(pos_out, 'r');
hold off;

figure(2)
plot(vel_in, 'b');
hold on;
plot(vel_out, 'r');
hold off;

figure(3)
plot(error);