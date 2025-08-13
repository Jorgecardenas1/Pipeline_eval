% ------------------------------------------------------------
% S11 Absorption Analysis for Multiple Incidence Angles
% Author: Jorge Cardenas
% Description: Loads S-parameter data, calculates absorption
%              for 0°, 10°, and 20° incidence, and exports CSV.
% ------------------------------------------------------------

% ---------- Input Files ----------

files = struct(...
    'angle0', struct('absorber', 'abs2.s2p', 'reflect', 'Reflection0Deg.s2p'), ...
    'angle10', struct('absorber', 'abs10degband.s2p', 'reflect', 'Reflectioncontrol10Deg.s2p'), ...
    'angle20', struct('absorber', 'abs20degband.s2p', 'reflect', 'Reflectioncontrol20Deg.s2p'),'angle30', struct('absorber', 'abs30degband.s2p', 'reflect', 'Reflectioncontrol20Deg.s2p') ...
    );

% ---------- Load and Process ----------
[abs0, freq]  = processAngle(files.angle0.absorber, files.angle0.reflect, 0);
abs10         = processAngle(files.angle10.absorber, files.angle10.reflect, 10);
abs20         = processAngle(files.angle20.absorber, files.angle20.reflect, 20);

% ---------- Export to CSV ----------
T = table(freq, abs0, abs10, abs20, ...
    'VariableNames', {'Frequency_GHz', 'Abs0deg', 'Abs10deg', 'Abs20deg'});

writetable(T, 'absorption_all_angles.csv');
disp('CSV saved: absorption_all_angles.csv');

% ============================================================
% Function: Process and Plot for One Angle
% ============================================================
function [P_absorbed_normalized, freq_GHz] = processAngle(absorber_file, reflect_file, angle_deg)

    % Load S-parameters
    data_absorber = sparameters(absorber_file);
    data_reflect = sparameters(reflect_file);
    freq_GHz = data_absorber.Frequencies / 1e9;
    
    % Extract S11
    s11_absorber = rfparam(data_absorber, 2, 2);
    s11_reflect = rfparam(data_reflect, 2, 2);

    plotS11Comparison(freq_GHz, data_absorber, data_reflect, s11_absorber, s11_reflect, angle_deg);
    diff_mag_db = mag2db((abs(s11_absorber)+(abs(s11_reflect)).^2)./abs(s11_absorber)) ;
    diff_mag = db2mag(diff_mag_db)
    % Calculate Absorption (custom domain-specific formula)
    P_in =  abs(diff_mag).^2;  % normalized
    P_absorbed_normalized =1-( P_in);

    % Plot
    figure('Name', sprintf('Absorption @ %d°', angle_deg));
    plot(freq_GHz, P_absorbed_normalized, 'b-', 'LineWidth', 2);
    xlabel('Frequency (GHz)', 'FontWeight', 'bold');
    ylabel('Absorption (Normalized)', 'FontWeight', 'bold');
    title(sprintf('Normalized Absorption vs Frequency @ %d°', angle_deg), 'FontWeight', 'bold');
    xlim([min(freq_GHz) max(freq_GHz)]);
    ylim([0 1]);
    grid on;
    set(gca, 'FontSize', 12);
end


% ============================================================
% Function: Plot S11 Comparison
% ============================================================
function plotS11Comparison(freq, data_dut, data_ref, s11_dut, s11_ref, angle_deg)
    diff_mag_db = mag2db(abs(s11_dut)) - mag2db(abs(s11_ref));

    figure('Name', sprintf('S11 Comparison @ %d°', angle_deg));
    subplot(2,1,1)
    rfplot(data_dut, 2, 2); hold on;
    rfplot(data_ref, 2, 2);
    xlabel('Frequency (GHz)');
    ylabel('|S_{11}| (dB)');
    title(sprintf('S11 Comparison @ %d°', angle_deg));
    legend('DUT', 'Reference');
    grid on;

    subplot(2,1,2)
    plot(freq, diff_mag_db, 'k', 'LineWidth', 2);
    xlabel('Frequency (GHz)');
    ylabel('Difference |S_{11}| (dB)');
    title('Difference in Reflection Magnitude');
    grid on;
    set(gca, 'FontSize', 12);
end
