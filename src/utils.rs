/* ==================================================================================================
 *                           This file is part of the bachelor thesis project
 *                  Implementation and Analysis of Selected Noise Reduction Methods
 *                                Weronika Tarnawska (Index No. 331171)
 *                                  Supervisor:  dr hab. Paweł Woźny
 *                                  University of Wrocław, June 2025
 * ================================================================================================== */
use rand::Rng;

/// Mix clean signal with noise, scaling noise by `noise_level`.
pub fn add_noise(signal: &Vec<f64>, noise: &Vec<f64>, noise_level: f64) -> Vec<f64> {
    signal
        .iter()
        .zip(noise.iter())
        .map(|(s, n)| s + n * noise_level)
        .collect()
}

/// Apply variable scaling to the input signal: every `CHUNK_SIZE` samples,
/// draw a random scaling factor in [0.0, 1.0] and multiply the next chunk by it.
///
/// # Arguments
/// * `signal` - reference to the input signal vector
///
/// # Returns
/// A new `Vec<f64>` where each block of `CHUNK_SIZE` samples has been multiplied
/// by an independently drawn random factor in [0.0, 1.0].
pub fn apply_variable_scaling(signal: &Vec<f64>) -> Vec<f64> {
    const CHUNK_SIZE: usize = 44_100;
    let mut rng = rand::thread_rng();
    let len = signal.len();
    let mut output = Vec::with_capacity(len);

    // Process in chunks of CHUNK_SIZE
    let mut idx = 0;
    while idx < len {
        // Determine end of this chunk (may be smaller at the end)
        let end = if idx + CHUNK_SIZE > len { len } else { idx + CHUNK_SIZE };
        // Draw a random scale factor in [0.0, 1.0]
        let scale: f64 = rng.gen_range(0.0..1.0);
        // Multiply each sample in this chunk by the scale
        for &sample in &signal[idx..end] {
            output.push(sample * scale);
        }
        idx = end;
    }
    output
}

/// Computes the mean squared error (MSE) between two signals.
pub fn mean_square_error(signal1: &Vec<f64>, signal2: &Vec<f64>) -> f64 {
    signal1
        .iter()
        .zip(signal2.iter())
        .map(|(s1, s2)| (s1 - s2) * (s1 - s2))
        .sum::<f64>()
        / signal1.len() as f64
}

/// Compute the linear signal-to-noise ratio between a clean reference and a processed signal.
/// SNR = P_clean / P_noise, where noise = clean - processed.
///
/// # Arguments
/// * `clean` - Vector of clean (reference) samples.
/// * `processed` - Vector of signal samples (input or processed-output).
///
/// If input vectors have different lengths, cut's both to the length of the shorter one.
///
/// # Returns
/// * Linear SNR.
pub fn sig_to_noise_ratio(clean: &Vec<f64>, processed: &Vec<f64>) -> f64 {
    // assert_eq!(clean.len(), processed.len(), "Input vectors must have same length");
    let len = clean.len().min(processed.len());
    let clean = &clean[..len].to_vec();
    let processed = &processed[..len].to_vec();

    let pow_signal = clean.iter().map(|&x| x * x).sum::<f64>();
    let pow_error = clean
        .iter()
        .zip(processed.iter())
        .map(|(&d, &pd)| (d - pd).powi(2))
        .sum::<f64>();
    if pow_error == 0.0 {
        return f64::INFINITY;
    }
    pow_signal / pow_error
}

/// Compute the SNR in decibels: 10 * log10(linear SNR).
///
/// # Arguments
/// * `clean` - Vector of clean samples.
/// * `processed` - Vector of processed samples.
///
/// # Returns
/// * SNR in dB.
fn sig_to_noise_ratio_db(clean: &Vec<f64>, processed: &Vec<f64>) -> f64 {
    let snr = sig_to_noise_ratio(clean, processed);
    10.0 * snr.log10()
}

/// Compute the improvement in SNR (in dB) from a noisy input to a processed output,
/// relative to a clean reference.
///
/// improvement_dB = SNR_db(clean, processed) - SNR_db(clean, noisy)
///
/// # Arguments
/// * `clean` - Vector of clean reference samples.
/// * `noisy` - Vector of noisy input samples.
/// * `processed` - Vector of processed output samples.
///
/// # Returns
/// * SNR improvement in dB.
pub fn snr_improvement_db(clean: &Vec<f64>, noisy: &Vec<f64>, processed: &Vec<f64>) -> f64 {
    let snr_in = sig_to_noise_ratio_db(clean, noisy);
    let snr_out = sig_to_noise_ratio_db(clean, processed);
    snr_out - snr_in
}

/// Compute the linear improvement in SNR (ratio) from a noisy input to a processed output.
///
/// improvement = SNR(clean, processed) / SNR(clean, noisy)
///
/// # Returns
/// * Linear SNR improvement.
pub fn snr_improvement(clean: &Vec<f64>, noisy: &Vec<f64>, processed: &Vec<f64>) -> f64 {
    let snr_in = sig_to_noise_ratio(clean, noisy);
    let snr_out = sig_to_noise_ratio(clean, processed);
    if snr_in == 0.0 {
        return f64::INFINITY;
    }
    snr_out / snr_in
}
