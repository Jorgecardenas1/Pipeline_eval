% ----------------------------
% File paths
% ----------------------------
dut_file = 'absorber1.s2p';        % S2P file for the DUT (absorber)
ref_file = 'reflect1.s2p';   % S2P file for the reference (e.g., metal)

% ----------------------------
% Load S-parameter data
% ----------------------------
dut_data = sparameters(dut_file);
ref_data = sparameters(ref_file);

% Frequency vector (assumes both have same frequency points)
frequency = dut_data.Frequencies;  % in Hz

% Extract S11 (assumes 2-port file)
s11_dut = rfparam(dut_data, 2, 2);
s11_ref = rfparam(ref_data, 2, 2);

% ----------------------------
% Compute Reflected Power
% ----------------------------
P_dut = (-abs(s11_dut)+abs(s11_ref)).^2;
P_ref = abs(s11_dut).^2;

% Prevent division by zero
P_ref(P_ref == 0) = eps;

% ----------------------------
% Compute Absorption
% ----------------------------
absorption =1- P_dut./P_ref;
normalized_absorption = (absorption - 0) / (80000 - 0);


% Clip absorption to range [0, 1]
%absorption = min(max(absorption, 0), 1);

% ----------------------------
% Plot Absorption
% ----------------------------
figure;
plot(frequency/1e9, normalized_absorption, 'b-', 'LineWidth', 2);
xlabel('Frequency (GHz)');
ylabel('Absorption (Normalized)');
title('Absorption Profile from S11 Comparison');
grid on;
ylim([0 1.05]);
