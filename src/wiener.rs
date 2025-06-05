/* ==========================================================================================
 *                          This file is part of the Bachelor Thesis project
 *                                   University of Wrocław
 *                         Author: Weronika Tarnawska (Index No. 331171)
 *                                         June 2025
 * ========================================================================================== */
use crate::fft::{freq2time, time2freq, Window};
use num_complex::Complex;

/// Compute the periodogram PSD of a single complex spectrum frame:
/// S_xx[k] = |X[k]|^2 (1/N scaling is omitted, as we'll divide S_{nn}/S_{xx} later)
fn periodogram_psd(frame: &Vec<Complex<f64>>) -> Vec<f64> {
    frame.iter().map(|c| c.norm_sqr()).collect()
}

/// Estimate the noise PSD S_nn using average periodograms method:
/// 1. Compute periodogram of each frame
/// 2. Average periodograms across frames
pub fn average_psd(noise_spectra: &Vec<Vec<Complex<f64>>>, frame_size: usize) -> Vec<f64> {
    let mut sum_psd = vec![0.0; frame_size];
    let n_frames = noise_spectra.len() as f64;

    for frame in noise_spectra.iter() {
        let psd = periodogram_psd(frame);
        for (k, &val) in psd.iter().enumerate() {
            sum_psd[k] += val;
        }
    }
    sum_psd.iter_mut().for_each(|v| *v /= n_frames);
    sum_psd
}

/// Apply Wiener filtering to a single noisy spectrum frame X[k]:
/// 1. Estimate S_xx
/// 2. Estimate signal PSD: S_dd = max(S_xx - S_nn, 0)
/// 3. Compute Wiener gain: W = S_dd / (S_dd + S_nn + epsilon)
/// 4. Multiply: D_hat = W * X
fn wiener_frame(x: &Vec<Complex<f64>>, s_nn: &Vec<f64>, epsilon: f64) -> Vec<Complex<f64>> {
    let s_xx = periodogram_psd(x);
    x.iter()
        .zip(s_xx.iter().zip(s_nn.iter()))
        .map(|(&x_k, (&s_xx_k, &s_nn_k))| {
            let s_dd = (s_xx_k - s_nn_k).max(0.0);
            let w_k = s_dd / (s_dd + s_nn_k + epsilon);
            x_k * w_k
        })
        .collect()
}

#[derive(Clone, Copy)]
pub enum WienerVariant {
    AvgNoise,
    InstNoise,
}

/// Perform Wiener denoising:
/// 1. Depending on variant, estimate noise PSD (global or per-frame)
/// 2. STFT noisy signal
/// 3. Wiener-filter each frame
/// 4. ISTFT back
///
/// # Arguments
/// * `noisy`       - Reference to the time-domain noisy signal samples.
/// * `noise`       - Reference to the time-domain noise-only signal samples used for PSD estimation.
/// * `frame_size`  - Number of samples per frame (FFT size, must be power of two).
/// * `window`      - Window function to apply to each frame.
/// * `epsilon`     - Small constant to avoid division by zero in gain computation.
/// * `variant`     - Choose between AvgNoise (global average) or InstNoise (per-frame).
///
/// # Returns
/// A Vec<f64> containing the denoised time-domain signal.
pub fn wiener_denoise(
    noisy: &Vec<f64>,
    noise: &Vec<f64>,
    frame_size: usize,
    window: Window,
    epsilon: f64,
    variant: WienerVariant,
) -> Vec<f64> {
    let noise_spectra: Vec<Vec<Complex<f64>>> = time2freq(noise, frame_size, window);
    let noisy_spectra: Vec<Vec<Complex<f64>>> = time2freq(noisy, frame_size, window);

    // Prepare noise PSD depending on variant
    let (s_nn_global, noise_psd_frames): (Option<Vec<f64>>, Option<Vec<Vec<f64>>>) = match variant {
        WienerVariant::AvgNoise => {
            let avg = average_psd(&noise_spectra, frame_size);
            (Some(avg), None)
        }
        WienerVariant::InstNoise => {
            let per_frame: Vec<Vec<f64>> = noise_spectra.iter().map(|frame| periodogram_psd(frame)).collect();
            (None, Some(per_frame))
        }
    };

    // Wiener filter each frame
    let filtered_spectra: Vec<Vec<Complex<f64>>> = noisy_spectra
        .iter()
        .enumerate()
        .map(|(i, x)| {
            match variant {
                WienerVariant::AvgNoise => {
                    // Use global average noise PSD
                    let s_nn = s_nn_global.as_ref().unwrap();
                    wiener_frame(x, s_nn, epsilon)
                }
                WienerVariant::InstNoise => {
                    // Use noise PSD from the corresponding frame
                    // If noise has fewer frames than noisy, clamp index to last available
                    let n_frames_noise = noise_psd_frames.as_ref().unwrap().len();
                    let idx = if i < n_frames_noise { i } else { n_frames_noise - 1 };
                    let s_nn_frame = &noise_psd_frames.as_ref().unwrap()[idx];
                    wiener_frame(x, s_nn_frame, epsilon)
                }
            }
        })
        .collect();

    // ISTFT, overlap-add
    freq2time(&filtered_spectra, window)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::*;
    use crate::utils::*;
    use crate::_EPSILON;
    use float_cmp::approx_eq;
    use num_complex::Complex;

    #[test]
    fn test_periodogram_psd() {
        // For frame [1+0j, 0+1j, 2+2j]
        let frame = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0), Complex::new(2.0, 2.0)];
        let psd = periodogram_psd(&frame);
        // we expect [1^2+0^2, 0^2+1^2, 2^2+2^2] = [1,1,8]
        assert_eq!(psd.len(), 3);
        assert!(approx_eq!(f64, psd[0], 1.0, epsilon = _EPSILON));
        assert!(approx_eq!(f64, psd[1], 1.0, epsilon = _EPSILON));
        assert!(approx_eq!(f64, psd[2], 8.0, epsilon = _EPSILON));
    }

    #[test]
    fn test_average_psd_constant_noise() {
        // Noise is cost: [1.0, 1.0, 1.0, 1.0]
        let noise = vec![1.0; 4];
        let frame_size = 2;
        let noise_spectra: Vec<Vec<Complex<f64>>> = time2freq(&noise, frame_size, Window::Rectangle);
        let psd = average_psd(&noise_spectra, frame_size);
        // Frames: [1,1], [1,1], [1,1]
        // fft[1,1] is [2,0]
        // Then periodogram is [2^2,0^2] = [4,0]
        assert_eq!(psd, vec![4., 0.]);
    }

    #[test]
    fn test_wiener_frame_identity_when_no_noise() {
        // If S_nn = 0, W = S_xx/(S_xx + 0 + eps) ~ 1 => D_hat ~ X
        let x = vec![Complex::new(1.0, -1.0), Complex::new(2.0, 0.5)];
        let s_nn = vec![0.0, 0.0];
        let eps = 1e-8;
        let out = wiener_frame(&x, &s_nn, eps);
        for (o, &x) in out.iter().zip(&x) {
            assert!(approx_eq!(f64, o.re, x.re, epsilon = eps));
            assert!(approx_eq!(f64, o.im, x.im, epsilon = eps));
        }
    }

    #[test]
    fn test_wiener_denoise_roundtrip_when_noise_zero() {
        // If noise = zero, the result should be `noisy` unchanged
        let clean = vec![0.0, 0.5, -1.0, 2.0, -0.3];
        let noisy = clean.clone();
        let noise = vec![0.0; noisy.len()];
        let out = wiener_denoise(&noisy, &noise, 4, Window::Rectangle, 1e-10, WienerVariant::AvgNoise);
        assert!(out.len() >= noisy.len() && out.len() < noisy.len() + 4);
        for (o, &c) in out.iter().zip(&clean) {
            assert!(approx_eq!(f64, *o, c, epsilon = 1e-8), "Failed, out: {:?}", out);
        }
    }

    #[test]
    fn test_wiener_denoise() {
        let len = 4000;
        let sample_rate = 4000.0; // arbitrary
        let clean_freq = 500.0;
        let noise_freq = 1200.0;
        // Generate clean and noise
        let clean = generate_signal(len, SignalType::Sinusoidal(clean_freq), sample_rate);
        let noise = generate_signal(len, SignalType::Sinusoidal(noise_freq), sample_rate);
        let noisy_sig = add_noise(&clean, &noise, 1.0);
        let mse_noisy = mean_square_error(&clean, &noisy_sig);
        // Parameter grids
        let frame_sizes = [256, 512];
        let windows = [Window::Hamming, Window::Hanning];
        for &frame_size in &frame_sizes {
            for &window in &windows {
                let out = wiener_denoise(&noisy_sig, &noise, frame_size, window, 1e-10, WienerVariant::AvgNoise);

                let improvement = snr_improvement(&clean, &noisy_sig, &out[..clean.len()].to_vec());
                assert!(
                    improvement >= 25.0,
                    "Failed for fs={}, window={:?}: snr improvement={}",
                    frame_size,
                    window,
                    improvement
                );

                let mse_sub = mean_square_error(&clean, &out);
                println!(
                    "Passed for fs={}, window={:?}: snr improvement={}; mse_sub={} < mse_noisy={}",
                    frame_size, window, improvement, mse_sub, mse_noisy
                );
            }
        }
    }
}
