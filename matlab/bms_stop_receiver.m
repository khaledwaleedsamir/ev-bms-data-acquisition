%% bms_stop_receiver.m
% Stops the background BMS receiver timer and closes the TCP connection.
t = timerfind('Name', 'bms_timer');
if isempty(t)
    fprintf('No BMS receiver is currently running.\n');
else
    stop(t);
    delete(t);
end
