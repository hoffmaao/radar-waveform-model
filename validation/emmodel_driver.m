% Run the exported candidate interface models through CReSIS/OPR em_model.
%
% Usage (from the fabric_anisotropy MATLAB container, which mounts the OPR
% toolbox at /home/matlab/opr):
%
%   matlab -batch "run('/path/to/validation/emmodel_driver.m')"
%
% Reads  figures/validation/emmodel_candidates.mat  (written by
% emmodel_export.py) and writes  figures/validation/emmodel_results.mat
% holding |R(f)| per candidate and eigenpolarisation.
%
% Each candidate is evaluated twice, with er_x(z) and er_y(z): em_model is
% isotropic ("birefringence not supported"), but at nadir each
% eigenpolarisation propagates through an isotropic profile of its own tensor
% component -- the substitution validated against gprMax and the analytic
% birefringent split in this repository's validation suite.

here = fileparts(mfilename('fullpath'));
opr_guess = {'/home/matlab/opr/matlab/em_model', ...
             fullfile(getenv('HOME'), 'projects', 'opr', 'matlab', 'em_model')};
for k = 1:numel(opr_guess)
  if exist(opr_guess{k}, 'dir'), addpath(opr_guess{k}); break; end
end
assert(exist('specularNadir', 'file') == 2, ...
  'em_model not on path: mount/clone gitlab.com/openpolarradar/opr');

in = load(fullfile(here, '..', 'figures', 'validation', 'emmodel_candidates.mat'));
freq = double(in.freq);
names = in.names;
results = struct('freq', freq);

for k = 1:numel(names)
  name = char(names{k});
  depth = double(in.([name '__depth']));
  for pol = {'x', 'y'}
    er = double(in.([name '__er_' pol{1}]));
    % specularNadir wants er as (n_layers, n_freq) complex, e' - j e''.
    H = specularNadir(depth, er, freq);
    results.([name '__R_' pol{1}]) = abs(H(:)).';
  end
  fprintf('  %s done\n', name);
end

save(fullfile(here, '..', 'figures', 'validation', 'emmodel_results.mat'), ...
     '-struct', 'results');
fprintf('wrote emmodel_results.mat\n');
