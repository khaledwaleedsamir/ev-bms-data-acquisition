% csv_to_hppc_mat.m
% ------------------
% Loads an HPPC CSV file collected from the BMS + hoverboard setup and
% saves it as a .mat file containing a proper MATLAB table named hppcData,
% matching the structure of the reference BAK cell dataset:
%
%   hppcData  —  Nx3 table with columns:
%       time_s_      (double)
%       voltage_V_   (double)
%       current_A_   (double)
%
% Usage:
%   Run as-is with the default paths, or change INPUT_CSV / OUTPUT_MAT below.

% ── Configuration ────────────────────────────────────────────────────────────
INPUT_CSV  = 'hppc_full.csv';
OUTPUT_MAT = 'hppc_full.mat';

% Column names in your CSV
TIME_COL    = 'time_s';
VOLTAGE_COL = 'bms_voltage';      % MATLAB replaces / with _ on import
CURRENT_COL = 'bms_current';
% ─────────────────────────────────────────────────────────────────────────────

fprintf('Reading: %s\n', INPUT_CSV);
raw = readtable(INPUT_CSV, 'VariableNamingRule', 'modify');
% Note: readtable replaces special characters (/ space) with underscores,
% so "bms/voltage" becomes "bms_voltage", "bms/current" becomes "bms_current".
% If your column names differ, print them with: disp(raw.Properties.VariableNames)

% Extract the three columns
t = raw.(TIME_COL);
v = raw.(VOLTAGE_COL);
i = raw.(CURRENT_COL);

% Build the table with matching column names
hppcData = table(t, v, i, ...
    'VariableNames', {'time_s_', 'voltage_V_', 'current_A_'});

fprintf('Rows    : %d\n', height(hppcData));
fprintf('Duration: %.0f s (%.2f hrs)\n', t(end), t(end)/3600);
fprintf('Voltage : %.3f – %.3f V\n', min(v), max(v));
fprintf('Current : %.3f – %.3f A\n', min(i), max(i));

% Save
save(OUTPUT_MAT, 'hppcData');
fprintf('Saved to: %s\n', OUTPUT_MAT);