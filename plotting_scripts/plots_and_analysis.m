% TODO: THIS FILE HAS MULTIPLE BUGS THAT NEEEDS TO BE FIXED.
%% battery_analysis.m
%  Reusable MATLAB functions for battery CSV data analysis.
%  Each function accepts a CSV filepath and operates independently.
%
%  USAGE EXAMPLE:
%   csvFile = 'your_data.csv';
%   plot_battery_overview(csvFile);
%   avg_inf = calc_avg_inference(csvFile);
%   plot_clean_esp_temp(csvFile);
%   capacity = calc_actual_capacity(csvFile, 2.0);   % 2.0 Ah initial cap.

% =========================================================================
%  MAIN — Demonstration (comment out when using as function library)
% =========================================================================
csvFile = 'C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\discharge_run-001_80pct_speed_esp.csv';
plot_battery_overview(csvFile);
%avg_us = calc_avg_inference(csvFile);
%[t, T] = plot_clean_esp_temp(csvFile);
[Ah, Wh] = calc_actual_capacity(csvFile, 2.0);  % 2.0 Ah rated capacity

% =========================================================================
%  Load and parse the CSV
% =========================================================================
function data = load_csv(csvFile)
    % Reads the CSV and returns a table with correct types.
    opts = detectImportOptions(csvFile, 'PreserveVariableNames', true);
 
    % Force datetime column to be read as string first, then converted
    opts = setvartype(opts, 'datetime_utc', 'char');
 
    data = readtable(csvFile, opts);
 
    % Convert datetime string to MATLAB datetime
    % Format: 2026-04-22T09:36:14+00:00  (ISO 8601 with timezone offset)
    data.datetime_utc = datetime(data.datetime_utc, ...
        'InputFormat', 'yyyy-MM-dd''T''HH:mm:ssXXX', ...
        'TimeZone', 'UTC');
end

% =========================================================================
%  FUNCTION 1: 2×2 Overview Grid
%  Plots Voltage, Current, Temperature, and SOC (BMS + Predicted)
% =========================================================================
function plot_battery_overview(csvFile)
% PLOT_BATTERY_OVERVIEW  Publication-quality 2×2 overview figure.
%   plot_battery_overview(csvFile) reads the CSV and produces a 2×2 grid:
%     [Voltage]      [Current]
%     [Temperature]  [SOC (BMS solid, Pred dashed)]

    data = load_csv(csvFile);
    t    = data.datetime_utc;

    % ---- Figure & Layout -----------------------------------------------
    fig = figure('Name', 'Battery Overview', ...
                 'Color', 'white', ...
                 'Units', 'centimeters', ...
                 'Position', [2 2 20 15]);

    % Shared style parameters
    lw_main = 1.5;    % line width for primary traces
    lw_sec  = 1.5;    % line width for secondary traces
    fs_ax   = 10;     % axis label font size
    fs_ti   = 11;     % title font size
    fs_leg  = 9;      % legend font size
    gridAlpha = 0.25;

    % Color palette (color-blind friendly)
    cVoltage = [0.122 0.471 0.706];   % blue
    cCurrent = [0.839 0.153 0.157];   % red
    cTemp    = [0.173 0.627 0.173];   % green
    cBMS     = [0.580 0.404 0.741];   % purple
    cPred    = [1.000 0.498 0.055];   % orange

    % ---- Subplot 1: Voltage --------------------------------------------
    ax1 = subplot(2, 2, 1);
    plot(t, data.voltage_V, '-', 'Color', cVoltage, 'LineWidth', lw_main);
    ylabel('Voltage (V)', 'FontSize', fs_ax);
    title('Terminal Voltage', 'FontSize', fs_ti, 'FontWeight', 'bold');
    grid on; ax1.GridAlpha = gridAlpha;
    ax1.XTickLabelRotation = 30;
    set(ax1, 'FontSize', fs_ax, 'Box', 'on', 'TickDir', 'out');
    xlim([t(1) t(end)]);
    ax1.XAxis.TickLabelFormat = 'HH:mm';

    % ---- Subplot 2: Current --------------------------------------------
    ax2 = subplot(2, 2, 2);
    plot(t, data.current_A, '-', 'Color', cCurrent, 'LineWidth', lw_main);
    ylabel('Current (A)', 'FontSize', fs_ax);
    title('Load Current', 'FontSize', fs_ti, 'FontWeight', 'bold');
    grid on; ax2.GridAlpha = gridAlpha;
    ax2.XTickLabelRotation = 30;
    set(ax2, 'FontSize', fs_ax, 'Box', 'on', 'TickDir', 'out');
    xlim([t(1) t(end)]);
    ax2.XAxis.TickLabelFormat = 'HH:mm';

    % ---- Subplot 3: Temperature ----------------------------------------
    ax3 = subplot(2, 2, 3);
    plot(t, data.temperature_degC, '-', 'Color', cTemp, 'LineWidth', lw_main);
    ylabel('Temperature (°C)', 'FontSize', fs_ax);
    xlabel('Time (UTC)', 'FontSize', fs_ax);
    title('Battery Temperature', 'FontSize', fs_ti, 'FontWeight', 'bold');
    grid on; ax3.GridAlpha = gridAlpha;
    ax3.XTickLabelRotation = 30;
    set(ax3, 'FontSize', fs_ax, 'Box', 'on', 'TickDir', 'out');
    xlim([t(1) t(end)]);
    ax3.XAxis.TickLabelFormat = 'HH:mm';

    % ---- Subplot 4: SOC ------------------------------------------------
    ax4 = subplot(2, 2, 4);
    hold on;
    h1 = plot(t, data.bms_soc_pct,  '-',  'Color', cBMS,  'LineWidth', lw_main);
    h2 = plot(t, data.pred_soc_pct, '--', 'Color', cPred, 'LineWidth', lw_sec);
    hold off;
    ylabel('State of Charge (%)', 'FontSize', fs_ax);
    xlabel('Time (UTC)', 'FontSize', fs_ax);
    title('State of Charge Comparison', 'FontSize', fs_ti, 'FontWeight', 'bold');
    legend([h1 h2], {'BMS SoC', 'Predicted SoC'}, ...
           'Location', 'best', 'FontSize', fs_leg, 'Box', 'on');
    ylim([0 105]);
    grid on; ax4.GridAlpha = gridAlpha;
    ax4.XTickLabelRotation = 30;
    set(ax4, 'FontSize', fs_ax, 'Box', 'on', 'TickDir', 'out');
    xlim([t(1) t(end)]);
    ax4.XAxis.TickLabelFormat = 'HH:mm';

    % ---- Super-title ---------------------------------------------------
    sgtitle('Battery Monitoring — Session Overview', ...
            'FontSize', 13, 'FontWeight', 'bold');

    % ---- Export --------------------------------------------------------
    exportgraphics(fig, 'battery_overview.pdf', 'ContentType', 'vector');
    fprintf('[plot_battery_overview] Figure saved → battery_overview.pdf\n');
end

% =========================================================================
%  FUNCTION 2: Average Inference Time
%  Computes and reports mean inference time per sample (µs and ms).
% =========================================================================
function avg_us = calc_avg_inference(csvFile)
% CALC_AVG_INFERENCE  Returns average on-device inference time.
%   avg_us = calc_avg_inference(csvFile)
%
%   Returns:
%     avg_us  – mean inference time in microseconds (scalar double)
%
%   Also prints a formatted summary and saves a bar/distribution figure.

    data   = load_csv(csvFile);
    inf_us = data.inference_us;

    % Remove non-positive values (sensor errors / missing)
    inf_us = inf_us(inf_us > 0);

    avg_us  = mean(inf_us);
    med_us  = median(inf_us);
    std_us  = std(inf_us);
    min_us  = min(inf_us);
    max_us  = max(inf_us);
    n       = numel(inf_us);

    fprintf('\n===== Inference Time Statistics =====\n');
    fprintf('  Samples analysed : %d\n',        n);
    fprintf('  Mean             : %.2f µs  (%.4f ms)\n', avg_us, avg_us/1e3);
    fprintf('  Median           : %.2f µs  (%.4f ms)\n', med_us, med_us/1e3);
    fprintf('  Std Dev          : %.2f µs\n',   std_us);
    fprintf('  Min              : %.2f µs\n',   min_us);
    fprintf('  Max              : %.2f µs\n',   max_us);
    fprintf('=====================================\n\n');

    % ---- Distribution Figure -------------------------------------------
    fig = figure('Name', 'Inference Time Distribution', ...
                 'Color', 'white', ...
                 'Units', 'centimeters', ...
                 'Position', [2 2 16 10]);

    histogram(inf_us, 40, ...
              'FaceColor', [0.122 0.471 0.706], ...
              'EdgeColor', 'white', ...
              'FaceAlpha', 0.85);
    hold on;
    xl = xlim;
    xline(avg_us, '--r', sprintf('Mean = %.1f µs', avg_us), ...
          'LineWidth', 1.8, 'LabelFontSize', 9, 'LabelHorizontalAlignment', 'left');
    xline(med_us, ':k', sprintf('Median = %.1f µs', med_us), ...
          'LineWidth', 1.6, 'LabelFontSize', 9, 'LabelHorizontalAlignment', 'left');
    hold off;

    xlabel('Inference Time (µs)', 'FontSize', 11);
    ylabel('Count', 'FontSize', 11);
    title('On-Device Inference Time Distribution', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 10, 'Box', 'on', 'TickDir', 'out', 'GridAlpha', 0.25);

    exportgraphics(fig, 'inference_time_distribution.pdf', 'ContentType', 'vector');
    fprintf('[calc_avg_inference] Figure saved → inference_time_distribution.pdf\n');
end

% =========================================================================
%  FUNCTION 3: ESP Temperature Cleaning & Interpolation
%  Detects and removes abrupt spikes, then interpolates the clean curve.
% =========================================================================
function [t_clean, temp_clean] = plot_clean_esp_temp(csvFile)
% PLOT_CLEAN_ESP_TEMP  Removes outlier spikes from esp_temp_degC.
%   [t_clean, temp_clean] = plot_clean_esp_temp(csvFile)
%
%   Strategy:
%     1. Z-score outlier detection on the first-difference of the signal
%        (catches abrupt jumps regardless of their absolute value).
%     2. Marks any sample whose delta-temperature exceeds 3× the robust
%        standard deviation as an outlier.
%     3. Replaces outliers with NaN and pchip-interpolates the gaps.
%
%   Returns cleaned time vector and temperature vector.

    data     = load_csv(csvFile);
    t        = data.datetime_utc;
    esp_temp = data.esp_temp_degC;
    n        = numel(esp_temp);

    % ---- Step 1: Detect outliers via differential signal ---------------
    dt_temp  = [0; diff(esp_temp)];          % first difference
    med_d    = median(abs(dt_temp));          % median absolute deviation base
    mad_d    = 1.4826 * median(abs(dt_temp - median(dt_temp))); % MAD

    % Threshold: any jump larger than 4 × MAD (robust, non-Gaussian)
    thresh   = max(4 * mad_d, 2.0);          % at least 2 °C jump

    is_outlier = abs(dt_temp) > thresh;

    % Also mark the sample immediately after a large jump as suspect
    is_outlier_ext = is_outlier | [false; is_outlier(1:end-1)];

    % ---- Step 2: Replace with NaN and interpolate ----------------------
    t_sec        = seconds(t - t(1));        % numeric time axis (seconds)
    temp_clean   = esp_temp;
    temp_clean(is_outlier_ext) = NaN;

    % pchip interpolation over valid samples
    valid        = ~isnan(temp_clean);
    if sum(valid) > 3
        temp_clean = interp1(t_sec(valid), temp_clean(valid), t_sec, 'pchip');
    end
    t_clean = t;

    n_removed = sum(is_outlier_ext);
    fprintf('[plot_clean_esp_temp] %d / %d samples identified as outliers and interpolated.\n', ...
            n_removed, n);

    % ---- Figure --------------------------------------------------------
    fig = figure('Name', 'ESP Temperature Cleaning', ...
                 'Color', 'white', ...
                 'Units', 'centimeters', ...
                 'Position', [2 2 18 10]);

    hold on;
    % Raw signal (light, behind)
    h_raw = plot(t, esp_temp, '-', ...
                 'Color', [0.7 0.7 0.7], 'LineWidth', 1.0);
    % Outlier markers
    h_out = plot(t(is_outlier_ext), esp_temp(is_outlier_ext), 'x', ...
                 'Color', [0.839 0.153 0.157], 'MarkerSize', 8, 'LineWidth', 1.5);
    % Cleaned signal (front)
    h_cln = plot(t_clean, temp_clean, '-', ...
                 'Color', [0.122 0.471 0.706], 'LineWidth', 1.8);
    hold off;

    legend([h_raw h_out h_cln], ...
           {'Raw Signal', 'Detected Outliers', 'Cleaned & Interpolated'}, ...
           'Location', 'best', 'FontSize', 9, 'Box', 'on');
    ylabel('ESP Temperature (°C)', 'FontSize', 11);
    xlabel('Time (UTC)', 'FontSize', 11);
    title('ESP Module Temperature — Outlier Removal & Interpolation', ...
          'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 10, 'Box', 'on', 'TickDir', 'out', ...
             'GridAlpha', 0.25, 'XTickLabelRotation', 30);
    gca.XAxis.TickLabelFormat = 'HH:mm';
    xlim([t(1) t(end)]);

    exportgraphics(fig, 'esp_temp_cleaned.pdf', 'ContentType', 'vector');
    fprintf('[plot_clean_esp_temp] Figure saved → esp_temp_cleaned.pdf\n');
end

% =========================================================================
%  FUNCTION 4: Actual Battery Capacity Calculation
%  Integrates current over time to derive true discharged capacity.
% =========================================================================
function [capacity_Ah, capacity_Wh] = calc_actual_capacity(csvFile, initial_capacity_Ah)
% CALC_ACTUAL_CAPACITY  Computes battery capacity from current integration.
%   [capacity_Ah, capacity_Wh] = calc_actual_capacity(csvFile, initial_capacity_Ah)
%
%   Arguments:
%     csvFile              – path to the CSV file
%     initial_capacity_Ah  – rated/initial capacity in Ah (e.g. 2.0)
%
%   Returns:
%     capacity_Ah  – actual discharged capacity in Ah
%     capacity_Wh  – actual discharged capacity in Wh
%
%   Method: Trapezoidal integration of |current| × dt over the session.
%   Capacity utilisation % is reported relative to initial_capacity_Ah.

    if nargin < 2 || isempty(initial_capacity_Ah)
        initial_capacity_Ah = 1.0;           % default 1 Ah if not supplied
        warning('calc_actual_capacity: initial_capacity_Ah not supplied, defaulting to 1.0 Ah');
    end

    data    = load_csv(csvFile);
    t       = data.datetime_utc;
    I       = data.current_A;     % positive = discharge convention assumed
    V       = data.voltage_V;

    % Time vector in hours
    t_h     = hours(t - t(1));

    % Duration
    total_duration_h = t_h(end);
    total_duration_s = seconds(t(end) - t(1));

    % Trapezoidal integration  ∫ |I| dt  [Ah]
    capacity_Ah = trapz(t_h, abs(I));

    % Energy  ∫ |I| × V dt  [Wh]
    capacity_Wh = trapz(t_h, abs(I) .* V);

    util_pct = (capacity_Ah / initial_capacity_Ah) * 100;

    fprintf('\n===== Battery Capacity Analysis =====\n');
    fprintf('  Rated / Initial Capacity : %.4f Ah\n',   initial_capacity_Ah);
    fprintf('  Session Duration         : %.2f h  (%.0f s)\n', total_duration_h, total_duration_s);
    fprintf('  Discharged Capacity      : %.4f Ah\n',   capacity_Ah);
    fprintf('  Discharged Energy        : %.4f Wh\n',   capacity_Wh);
    fprintf('  Capacity Utilisation     : %.2f %%\n',   util_pct);
    fprintf('=====================================\n\n');

    % ---- Cumulative Capacity Figure ------------------------------------
    cumCap_Ah = cumtrapz(t_h, abs(I));
    cumEne_Wh = cumtrapz(t_h, abs(I) .* V);

    fig = figure('Name', 'Capacity Analysis', ...
                 'Color', 'white', ...
                 'Units', 'centimeters', ...
                 'Position', [2 2 18 11]);

    yyaxis left;
    plot(t, cumCap_Ah, '-', 'Color', [0.122 0.471 0.706], 'LineWidth', 1.8);
    ylabel('Cumulative Capacity (Ah)', 'FontSize', 11);
    yline(initial_capacity_Ah, '--', sprintf('Rated: %.2f Ah', initial_capacity_Ah), ...
          'Color', [0.122 0.471 0.706], 'LineWidth', 1.2, 'LabelFontSize', 8, ...
          'LabelHorizontalAlignment', 'right');

    yyaxis right;
    plot(t, cumEne_Wh, '-', 'Color', [0.839 0.153 0.157], 'LineWidth', 1.8);
    ylabel('Cumulative Energy (Wh)', 'FontSize', 11);

    xlabel('Time (UTC)', 'FontSize', 11);
    title('Cumulative Discharged Capacity & Energy', 'FontSize', 12, 'FontWeight', 'bold');

    legend({'Capacity (Ah)', '', 'Energy (Wh)'}, ...
           'Location', 'northwest', 'FontSize', 9, 'Box', 'on');
    grid on;
    set(gca, 'FontSize', 10, 'Box', 'on', 'TickDir', 'out', ...
             'GridAlpha', 0.25, 'XTickLabelRotation', 30);
    gca.XAxis.TickLabelFormat = 'HH:mm';
    xlim([t(1) t(end)]);

    % Annotation box
    ann_str = sprintf('Q_{disch} = %.3f Ah\nE_{disch} = %.3f Wh\nUtil. = %.1f %%', ...
                      capacity_Ah, capacity_Wh, util_pct);
    annotation('textbox', [0.14 0.60 0.22 0.18], ...
               'String', ann_str, ...
               'FontSize', 9, 'BackgroundColor', [1 1 1 0.85], ...
               'EdgeColor', [0.4 0.4 0.4], 'LineWidth', 0.8, ...
               'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');

    exportgraphics(fig, 'battery_capacity.pdf', 'ContentType', 'vector');
    fprintf('[calc_actual_capacity] Figure saved → battery_capacity.pdf\n');
end