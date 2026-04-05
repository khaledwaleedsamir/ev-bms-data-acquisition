%% SOC% Prediction Analysis
% Plots Actual vs Predicted SOC curves and computes MAE, RMSE, R² scores
% -----------------------------------------------------------------------

clc; clear; close all;

%% ── CONFIGURATION ──────────────────────────────────────────────────────

% List CSV file(s) here.
csv_files = { ...
    'C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\all_data\csv_files\test1_pred.csv', ...
    'C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\all_data\csv_files\test2_pred.csv',  ...
    'C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\all_data\csv_files\test4_pred.csv',...
    'C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\all_data\csv_files\test5_pred.csv'...
};

% Column indices for actual and predicted SOC (1-based)
col_actual    = 1;
col_predicted = 2;

% Does the CSV have a header row? true / false
has_header = true;

% X-axis label
x_label = 'Sample Index';

%% ── COLOURS ─────────────────────────────────────────────────────────────
col_actual_line    = [0.85, 0.11, 0.11];   % red   – Actual SOC
col_predicted_line = [0.00, 0.35, 0.78];   % blue  – Predicted SOC

%% ── FONT SIZES ───────────────────────────────────────────────────────────
fs_axis   = 16;   % axis tick labels
fs_label  = 18;   % x/y axis labels
fs_title  = 19;   % figure title
fs_legend = 16;   % legend text

%%
n_files = numel(csv_files);
results = struct();

%% ── FIGURE PER FILE : SOC Curves ────────────────────────────────────────
for k = 1 : n_files

    % --- load data -------------------------------------------------------
    fname = csv_files{k};
    if has_header
        T    = readtable(fname);
        data = T{:, [col_actual, col_predicted]};
    else
        data = readmatrix(fname);
        data = data(:, [col_actual, col_predicted]);
    end

    actual    = data(:, 1);
    predicted = data(:, 2);
    N         = numel(actual);
    x         = (1:N)';

    % --- metrics ---------------------------------------------------------
    mae    = mean(abs(actual - predicted));
    rmse   = sqrt(mean((actual - predicted).^2));
    ss_res = sum((actual - predicted).^2);
    ss_tot = sum((actual - mean(actual)).^2);
    r2     = 1 - ss_res / ss_tot;

    results(k).file = fname;
    results(k).MAE  = mae;
    results(k).RMSE = rmse;
    results(k).R2   = r2;

    % --- individual SOC curve plot ---------------------------------------
    figure('Name', sprintf('SOC Curves - Test %d', k), ...
           'NumberTitle', 'off', 'Color', 'w', 'Position', [80+k*30, 80+k*30, 1100, 500]);
    hold on; grid on; box on;

    plot(x, actual,    '-',  'Color', col_actual_line,    'LineWidth', 2.2, 'DisplayName', 'Actual SOC');
    plot(x, predicted, '--', 'Color', col_predicted_line, 'LineWidth', 2.0, 'DisplayName', 'Predicted SOC');
    
    xlabel(x_label,   'FontSize', fs_label, 'FontWeight', 'bold');
    ylabel('SOC (%)', 'FontSize', fs_label, 'FontWeight', 'bold');
    title(sprintf('Actual vs Predicted SOC%% - Test %d   (MAE=%.4f | RMSE=%.4f | R^2=%.4f)', ...
          k, mae, rmse, r2), 'FontSize', fs_title, 'FontWeight', 'bold');

    lgd = legend('Location', 'best', 'FontSize', fs_legend);
    lgd.Box = 'on';

    set(gca, 'FontSize', fs_axis, 'LineWidth', 1.2);
end

%% ── FIGURE : Error Distribution (all files) ────────────────────
err_colors = [ ...
    0.85, 0.11, 0.11;   % red
    0.00, 0.35, 0.78;   % blue
    0.13, 0.63, 0.30;   % green
    0.49, 0.18, 0.56;   % purple
    0.93, 0.69, 0.13 ]; % yellow

figure('Name', 'Prediction Error Distribution', 'NumberTitle', 'off', ...
       'Color', 'w', 'Position', [200 200 850 480]);
hold on; grid on; box on;

for k = 1 : n_files
    fname = csv_files{k};
    if has_header
        T    = readtable(fname);
        data = T{:, [col_actual, col_predicted]};
    else
        data = readmatrix(fname);
        data = data(:, [col_actual, col_predicted]);
    end

    errors = data(:,2) - data(:,1);
    c = err_colors(mod(k-1, size(err_colors,1)) + 1, :);

    histogram(errors, 30, 'FaceColor', c, 'FaceAlpha', 0.55, ...
              'EdgeColor', 'none', 'DisplayName', sprintf('Test %d', k));
end

xline(0, 'k--', 'LineWidth', 2.0, 'DisplayName', 'Zero Error');
xlabel('Prediction Error (Predicted - Actual) [%]', 'FontSize', fs_label, 'FontWeight', 'bold');
ylabel('Count', 'FontSize', fs_label, 'FontWeight', 'bold');
title('Distribution of Prediction Errors', 'FontSize', fs_title, 'FontWeight', 'bold');

lgd = legend('Location', 'best', 'FontSize', fs_legend);
lgd.Box = 'on';

set(gca, 'FontSize', fs_axis, 'LineWidth', 1.2);

%% ── METRICS TABLE ───────────────────────────────────────────────────────
fprintf('\n%s\n', repmat('-', 1, 50));
fprintf('  %-10s  %8s  %8s  %8s\n', 'File', 'MAE', 'RMSE', 'R2');
fprintf('%s\n', repmat('-', 1, 50));
for k = 1 : n_files
    fprintf('  %-10s  %8.4f  %8.4f  %8.4f\n', ...
        sprintf('Test %d', k), results(k).MAE, results(k).RMSE, results(k).R2);
end
fprintf('%s\n\n', repmat('-', 1, 50));
