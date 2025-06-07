/* ==================================================================================================
 *                           This file is part of the bachelor thesis project
 *                  Implementation and Analysis of Selected Noise Reduction Methods
 *                                Weronika Tarnawska (Index No. 331171)
 *                                  Supervisor:  dr hab. Paweł Woźny
 *                                  University of Wrocław, June 2025
 * ================================================================================================== */

/// Performs the core LMS algorithm on preprocessed signals.
///
/// # Arguments
/// * `x` - Input noise signal with `p - 1` leading zeros (length must be `n + p - 1`)
/// * `d` - Desired signal (typically the noisy signal), length `n`
/// * `p` - Filter length
/// * `n` - Number of samples to process
/// * `mu` - Step size (learning rate)
///
/// # Returns
/// A tuple containing:
/// * `y` - Output of the adaptive filter
/// * `e` - Error signal (difference between desired and output)
/// * `w` - Final filter coefficients
fn lms_inner(x: &Vec<f64>, d: &Vec<f64>, p: usize, n: usize, mu: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut e = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut w = vec![0.0; p];

    for k in 0..n {
        let l = k + p - 1;

        // one-sample convolution
        let mut y_k = 0.0;
        for i in 0..p {
            y_k += w[i] * x[l - i];
        }

        // error between desired and output
        let e_k = d[k] - y_k;

        // update weights
        for i in 0..p {
            w[i] += mu * e_k * x[l - i];
        }

        e[k] = e_k;
        y[k] = y_k;
    }

    (y, e, w)
}

/// Applies the Least Mean Squares (LMS) for denoising.
///
/// If the input signals do not have equal lengths,
/// both will be cut to the length of the shorter one.
///
/// # Arguments
/// * `noise_only` - Reference noise signal
/// * `noisy_signal` - Observed signal with noise
/// * `filter_size` - Number of filter coefficients
/// * `step_size` - Learning rate (mu): controls adaptation speed
///
/// # Returns
/// A tuple containing:
/// * The estimate of how `noise_only` appears in `noisy_signal`
/// * Estimated clean signal
/// * Final filter coefficients
pub fn lms(
    noise_only: &Vec<f64>,
    noisy_signal: &Vec<f64>,
    filter_size: usize,
    step_size: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // assert_eq!(noisy_signal.len(), noise_only.len());
    let len = noise_only.len().min(noisy_signal.len());
    let noise_only = &noise_only[..len].to_vec();
    let noisy_signal = &noisy_signal[..len].to_vec();
    let n = noisy_signal.len();

    // prepend p-1 zeros to noise for initial conditions
    let mut noise_padded = Vec::with_capacity(n + filter_size - 1);
    noise_padded.extend(std::iter::repeat(0.0).take(filter_size - 1));
    noise_padded.extend_from_slice(noise_only);

    lms_inner(&noise_padded, noisy_signal, filter_size, n, step_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::*;
    use crate::utils::*;
    use float_cmp::approx_eq;

    #[test]
    fn test_lms_roundtrip_when_noise_zero() {
        let clean = vec![0.0, 0.5, -1.0, 2.0, -0.3];
        let noisy = clean.clone();
        let noise = vec![0.0; noisy.len()];
        let (_, out, _) = lms(&noise, &noisy, 2, 0.1);
        assert!(out.len() == noisy.len(), "out: {}, orig: {}", out.len(), noisy.len());
        for (o, &c) in out.iter().zip(&clean) {
            assert!(approx_eq!(f64, *o, c, epsilon = 1e-8), "Failed, out: {:?}", out);
        }
    }

    #[test]
    fn test_lms_denoise() {
        let len = 4000;
        let sample_rate = 4000.0; // arbitrary
        let clean_freq = 500.0;
        let noise_freq = 1200.0;
        // Generate clean and noise
        let clean = generate_signal(len, SignalType::Sinusoidal(clean_freq), sample_rate);
        let noise = generate_signal(len, SignalType::Sinusoidal(noise_freq), sample_rate);
        let noisy_sig = add_noise(&clean, &noise, 1.0);
        let mse_noisy = mean_square_error(&clean, &noisy_sig);
        for filt_size in [32, 64, 128] {
            let (_, out, _) = lms(&noise, &noisy_sig, filt_size, 0.001);
            let improvement = snr_improvement(&clean, &noisy_sig, &out);
            assert!(
                improvement >= 25.0,
                "Failed for fs={}: snr improvement={}",
                filt_size,
                improvement
            );

            let mse = mean_square_error(&clean, &out);
            println!(
                "Passed for fs={}: snr improvement={}; mse={} < mse_noisy={}",
                filt_size, improvement, mse, mse_noisy
            );
        }
    }
}
