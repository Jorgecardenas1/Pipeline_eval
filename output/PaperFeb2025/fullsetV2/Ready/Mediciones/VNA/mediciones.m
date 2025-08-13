% ----------------------------
% File paths
% ----------------------------
ref_file = 'reflector2.s2p';   % Reference (e.g., metal plate)
dut_file = 'absorber2.s2p';    % DUT (absorber)

% ----------------------------
% Load S-parameter data
% ----------------------------
ref_data = sparameters(ref_file);
dut_data = sparameters(dut_file);

% Frequency vector
freq = ref_data.Frequencies;  % in Hz
assert(isequal(freq, dut_data.Frequencies), 'Frequencies must match');

% Extract S22 from both
S22_ref = rfparam(ref_data, 2, 2);  % Reference S22
S22_dut = rfparam(dut_data, 2, 2);  % DUT S22

% ----------------------------
% Convert to magnitudes
% ----------------------------
mag_ref = abs(S22_ref);
mag_dut = abs(S22_dut);

% Convert to dB
S22_ref_dB = 20 * log10(mag_ref);
S22_dut_dB = 20 * log10(mag_dut);

% ----------------------------
% Absorption (natural units)
% ----------------------------
Absorption = 1 - mag_dut.^2;

% ----------------------------
% Normalized absorption (vs reference)
% ----------------------------
reflectance_ratio = (mag_dut ./ mag_ref).^2;
Absorption_norm = 1 - reflectance_ratio;

% ----------------------------
% dB Difference (DUT - Reference)
% ----------------------------
diff_dB = S22_dut_dB - S22_ref_dB;

% ----------------------------
% Plot 1: S22 in dB
% ----------------------------
figure;
plot(freq/1e9, S22_ref_dB, 'b--', 'LineWidth', 2); hold on;
plot(freq/1e9, S22_dut_dB, 'r-', 'LineWidth', 2);
xlabel('Frequency (GHz)');
ylabel('|S_{22}| (dB)');
title('Reflection Coefficient |S_{22}| in dB');
legend('Reference (Metal Plate)', 'DUT (Absorber)', 'Location', 'SouthWest');
grid on;
ylim([-60 0]);

% ----------------------------
% Plot 2: Absorption (1 - |S22|^2)
% ----------------------------
figure;
plot(freq/1e9, Absorption, 'k', 'LineWidth', 2);
xlabel('Frequency (GHz)');
ylabel('Absorption');
title('Absorption (DUT only) = 1 - |S_{22,DUT}|^2');
grid on;
ylim([0 1.1]);

% ----------------------------
% Plot 3: Normalized Absorption (vs Ref)
% ----------------------------
figure;
plot(freq/1e9, Absorption_norm, 'g', 'LineWidth', 2);
xlabel('Frequency (GHz)');
ylabel('Normalized Absorption');
title('Normalized Absorption = 1 - (|S_{22,DUT}| / |S_{22,Ref}|)^2');
grid on;
ylim([0 1.1]);

% ----------------------------
% Plot 4: Difference in dB (DUT - Ref)
% ----------------------------
figure;
plot(freq/1e9, diff_dB, 'm', 'LineWidth', 2);
xlabel('Frequency (GHz)');
ylabel('Difference [dB]');
title('Difference in |S_{22}| (DUT - Ref) in dB');
grid on;
