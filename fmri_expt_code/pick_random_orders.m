function pick_random_orders()

% choose a random number 1-100 to assign to each subject
% this defines the random sequence that is used for their experiment.

rndseed = 324545;

rng(rndseed, 'twister')

n_subs = 100;

rndorder = randperm(n_subs);

filename_save = fullfile(pwd, 'random_orders.mat');

save(filename_save, "rndorder");
    