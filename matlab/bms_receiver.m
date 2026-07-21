%% bms_receiver.m
%
% Connects to the Python BMS TCP server (bms_to_matlab.py) and runs a
% background timer that receives one JSON sample per second and writes
% every BMS field into the MATLAB base workspace.
%
% Because the receiver uses a timer (non-blocking), you can start a
% Simulink simulation in the same MATLAB instance while this keeps updating
% the workspace variables in the background.
%
% ── QUICK START ──────────────────────────────────────────────────────────
%   1. Start the Python server first:
%        python -m dataset.run_scripts.bms_to_matlab
%   2. Run this script in MATLAB:
%        run('bms_receiver.m')   or just open and press Run
%   3. Workspace variables update live.  In Simulink use a
%      MATLAB Function block with  evalin('base','bms_voltage')  etc.
%   4. Stop the receiver:
%        bms_stop_receiver
% ─────────────────────────────────────────────────────────────────────────

%% ── CONFIG ───────────────────────────────────────────────────────────────
PYTHON_HOST = '127.0.0.1';   % Python server address (loop back address PC → localhost)
PYTHON_PORT = 5005;
CONNECT_TIMEOUT = 10;        % seconds to wait for connection
TIMER_PERIOD    = 1.0;       % match LOG_HZ in bms_to_matlab.py
%% ─────────────────────────────────────────────────────────────────────────

% Stop any previously running receiver to avoid duplicate timers
existing = timerfind('Name', 'bms_timer');
if ~isempty(existing)
    stop(existing);
    delete(existing);
    fprintf('Previous BMS receiver stopped.\n');
end

% ── Connect ───────────────────────────────────────────────────────────────
fprintf('Connecting to Python BMS server at %s:%d …\n', PYTHON_HOST, PYTHON_PORT);
try
    client = tcpclient(PYTHON_HOST, PYTHON_PORT, 'Timeout', CONNECT_TIMEOUT);
catch ME
    error('Could not connect: %s\nMake sure bms_to_matlab.py is running.', ME.message);
end
configureTerminator(client, 'LF');
fprintf('Connected.\n');

% ── Initialise workspace with safe defaults ───────────────────────────────
% Simulink blocks will read these before the first real packet arrives.
bms_defaults = struct( ...
    'timestamp',     0,            ...
    'voltage',       0,            ...
    'current',       0,            ...
    'battery_level', 0,            ...
    'power',         0,            ...
    'temp_values',   zeros(1, 3),  ...
    'temperature',   0,            ...
    'cycle_charge',  0,            ...
    'cycle_capacity',0,            ...
    'cycles',        0,            ...
    'delta_voltage', 0,            ...
    'cell_count',    0,            ...
    'cell_voltages', zeros(1, 10), ...
    'temp_sensors',  0,            ...
    'battery_charging', 0          ...
);
assignin('base', 'bms',              bms_defaults);
assignin('base', 'bms_voltage',      0);
assignin('base', 'bms_current',      0);
assignin('base', 'bms_soc',          0);
assignin('base', 'bms_power',        0);
assignin('base', 'bms_temp',         0);
assignin('base', 'bms_cycle_charge', 0);
assignin('base', 'bms_cell_voltages',zeros(1, 10));

% ── Start background timer ────────────────────────────────────────────────
t = timer( ...
    'Name',          'bms_timer',                        ...
    'ExecutionMode', 'fixedRate',                        ...
    'Period',        TIMER_PERIOD,                       ...
    'TimerFcn',      @(~,~) bms_timer_cb(client),       ...
    'ErrorFcn',      @(~,e) fprintf('[BMS timer error] %s\n', e.Data.message), ...
    'StopFcn',       @(~,~) bms_on_stop(client)         ...
);
start(t);

fprintf('\nBMS receiver running in the background at %.0f Hz.\n', 1/TIMER_PERIOD);
fprintf('Available workspace variables:\n');
fprintf('  bms              — full struct with all fields\n');
fprintf('  bms_voltage      — pack voltage (V)\n');
fprintf('  bms_current      — pack current (A, + = discharge)\n');
fprintf('  bms_soc          — battery level (%%)\n');
fprintf('  bms_power        — power (W)\n');
fprintf('  bms_temp         — mean temperature (°C)\n');
fprintf('  bms_cycle_charge — cycle charge (Ah)\n');
fprintf('  bms_cell_voltages— 1×10 cell voltages (V)\n');
fprintf('\nTo stop: bms_stop_receiver\n\n');
fprintf('%-10s  %-10s  %-10s  %-8s  %-8s\n', ...
        'Time', 'V (V)', 'I (A)', 'SOC (%)', 'P (W)');
fprintf('%s\n', repmat('-', 1, 52));

% ─────────────────────────────────────────────────────────────────────────
%% Local functions
% ─────────────────────────────────────────────────────────────────────────

function bms_timer_cb(client)
% Called every TIMER_PERIOD seconds by the timer.
    if client.NumBytesAvailable == 0
        return          % nothing buffered yet — keep last values
    end
    try
        raw = readline(client);          % read one newline-terminated JSON line
        bms  = jsondecode(raw);          % parse into struct

        % Flatten list fields that jsondecode may give as Nx1 → ensure row vectors
        bms.temp_values   = reshape(bms.temp_values,   1, []);
        bms.cell_voltages = reshape(bms.cell_voltages, 1, []);

        % Derived scalar: mean temperature
        temp_mean = mean(bms.temp_values);

        % Push full struct and individual convenience variables
        assignin('base', 'bms',              bms);
        assignin('base', 'bms_voltage',      bms.voltage);
        assignin('base', 'bms_current',      bms.current);
        assignin('base', 'bms_soc',          bms.battery_level);
        assignin('base', 'bms_power',        bms.power);
        assignin('base', 'bms_temp',         temp_mean);
        assignin('base', 'bms_cycle_charge', bms.cycle_charge);
        assignin('base', 'bms_cell_voltages',bms.cell_voltages);

        fprintf('%s    %-10.3f  %-10.3f  %-8.1f  %-8.2f\n', ...
            datestr(now, 'HH:MM:SS'), ...
            bms.voltage, bms.current, bms.battery_level, bms.power);

    catch ME
        fprintf('[bms_receiver] Parse/read error: %s\n', ME.message);
    end
end

function bms_on_stop(client)
    clear client
    fprintf('BMS receiver stopped and TCP connection closed.\n');
end
