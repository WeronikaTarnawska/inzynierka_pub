/* ==================================================================================================
 *                           This file is part of the bachelor thesis project
 *                  Implementation and Analysis of Selected Noise Reduction Methods
 *                                Weronika Tarnawska (Index No. 331171)
 *                                  Supervisor:  dr hab. Paweł Woźny
 *                                  University of Wrocław, June 2025
 * ================================================================================================== */
use crate::fft::{freq2time, time2freq, Window};
use num_complex::Complex;

#[derive(Clone, Copy, Debug)]
pub enum SsVariant {
    Power,
    Magnitude,
}

/// Average noise power/magnitude across frames
///
/// # Arguments
///
/// * `noise_frames` - A vector of frames containing noise in frequency domain.
/// * `ssv` - Variant specifying whether to use magnitude or power for noise estimation.
///
/// # Returns
/// A vector of f64 values representing the noise estimate per frequency bin.
fn noise_estimate(noise_frames: Vec<Vec<Complex<f64>>>, ssv: SsVariant) -> Vec<f64> {
    let num_frames = noise_frames.len();
    assert!(num_frames > 0);
    let frame_len = noise_frames[0].len();
    noise_frames
        .into_iter()
        .fold(vec![0.0; frame_len], |acc, frame| {
            acc.iter()
                .zip(frame)
                .map(|(&acc_n, frame_n)| {
                    acc_n
                        + match ssv {
                            SsVariant::Magnitude => frame_n.norm(),
                            SsVariant::Power => frame_n.norm_sqr(),
                        }
                })
                .collect()
        })
        .iter()
        .map(|n| n / num_frames as f64)
        .collect()
}

/// Perform spectral subtraction on a single frame.
///
/// # Arguments
///
/// * `x_frame` - A reference to a vector of Complex<f64> representing a single noisy frame in frequency domain.
/// * `noise_est` - A reference to a vector of f64 representing the noise estimate per frequency bin.

///
/// # Returns
/// A vector of f64 values representing the subtracted amplitude spectrum for the frame.
fn spectral_sub_frame(
    x_frame: &Vec<Complex<f64>>,
    noise_est: &Vec<f64>,
    alpha: f64,
    beta: f64,
    variant: SsVariant,
) -> Vec<f64> {
    x_frame
        .iter()
        .zip(noise_est.iter())
        .map(|(&x, &n)| {
            match variant {
                SsVariant::Magnitude => {
                    // y = |X|, sub = y - alpha⋅N, floor = beta⋅y
                    let y = x.norm();
                    let res = y - alpha * n;
                    if res < 0.0 {
                        beta * y
                    } else {
                        res
                    }
                }
                SsVariant::Power => {
                    // y2 = |X|^2, sub2 = y2 - alpha⋅N, floor2 = beta⋅y2
                    let y2 = x.norm_sqr();
                    let res = y2 - alpha * n;
                    let clipped = if res < 0.0 { beta * y2 } else { res };
                    clipped.sqrt() // Restore amplitude from power
                }
            }
        })
        .collect()
}

/// Restore the complex spectrum by combining amplitudes with original phases.
///
/// # Arguments
/// * `amplitudes` - A vector of f64 representing the amplitude spectrum after spectral subtraction.
/// * `x_frame` - A vector of Complex<f64> representing the original noisy frame (to extract phase).
///
/// # Returns
/// A vector of Complex<f64> representing the spectrum with restored phase information.
fn restore_phase(amplitudes: Vec<f64>, x_frame: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    assert!(amplitudes.len() == x_frame.len());
    amplitudes
        .into_iter()
        .zip(x_frame.into_iter())
        .map(|(a, x)| Complex::from_polar(a, x.arg()))
        .collect()
}

/// Perform spectral subtraction noise reduction.
///
/// 1. Divide the input signals into overlapping frames
/// 2. Transform them to the frequency domain
/// 3. Estimate the noise spectrum
/// 4. Performs spectral subtraction on each frame
/// 5. Reconstructs the time-domain signal.
///
/// # Arguments
/// * `noisy_sig` - A reference to a vector of f64 containing the noisy input signal samples in time domain.
/// * `noise` - A reference to a vector of f64 containing a noise-only segment of the signal in time domain.
/// * `frame_size` - The size (number of samples) of each analysis frame, must be power of two.
/// * `alpha` - Over-subtraction factor (controls amount of noise reduction).
/// * `beta` - Spectral floor factor (prevents negative or too-small values).
/// * `variant` - Variant specifying whether to use magnitude or power spectral subtraction (`SsVariant::Magnitude` or `SsVariant::Power`).
/// * `window_type` - The type of window to apply to each frame before FFT
///
/// # Returns
/// A vector of f64 containing the filtered output signal in time domain.
pub fn spectral_subtraction(
    noisy_sig: &Vec<f64>,
    noise: &Vec<f64>,
    frame_size: usize,
    alpha: f64,
    beta: f64,
    variant: SsVariant,
    window_type: Window,
) -> Vec<f64> {
    // Convert noise and signal to frequency domain
    let noise_frames = time2freq(&noise, frame_size, window_type);
    let x_frames = time2freq(&noisy_sig, frame_size, window_type);

    // Get noise estimation
    let noise_est = noise_estimate(noise_frames, variant);

    // Process each frame
    let mut processed_frames: Vec<Vec<Complex<f64>>> = Vec::with_capacity(x_frames.len());
    for frame in x_frames.into_iter() {
        let amps = spectral_sub_frame(&frame, &noise_est, alpha, beta, variant);
        let restored = restore_phase(amps, frame);
        processed_frames.push(restored);
    }

    // Back to time domain
    freq2time(&processed_frames, window_type)
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
    fn test_noise_estimate_magnitude() {
        let frames = vec![
            vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)],
            vec![Complex::new(3.0, 0.0), Complex::new(4.0, 0.0)],
        ];
        let est = noise_estimate(frames, SsVariant::Magnitude);
        let expected = vec![2.0, 3.0];
        let cmp: Vec<bool> = est
            .iter()
            .zip(expected.iter())
            .map(|(&a, &b)| approx_eq!(f64, a, b, epsilon = _EPSILON))
            .collect();
        assert!(cmp.iter().all(|&x| x));
    }

    #[test]
    fn test_noise_estimate_power() {
        let frames = vec![
            vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)],
            vec![Complex::new(3.0, 0.0), Complex::new(4.0, 0.0)],
        ];
        let est = noise_estimate(frames, SsVariant::Power);
        let expected = vec![(1.0_f64 + 9.0) / 2.0, (4.0 + 16.0) / 2.0];
        let cmp: Vec<bool> = est
            .iter()
            .zip(expected.iter())
            .map(|(&a, &b)| approx_eq!(f64, a, b, epsilon = _EPSILON))
            .collect();
        assert!(cmp.iter().all(|&x| x));
    }

    #[test]
    fn test_spectral_sub_frame_magnitude() {
        // First element triggers flooring, second uses subtraction
        let x = vec![Complex::new(1.0, 0.0), Complex::new(10.0, 0.0)];
        let noise = vec![2.0, 1.0];
        // y1=1, sub1=-1->floor=0.5 => 0.5; y2=10, sub2=9>floor=5 =>9
        let out = spectral_sub_frame(&x, &noise, 1.0, 0.5, SsVariant::Magnitude);
        let expected = vec![0.5, 9.0];
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!(approx_eq!(f64, *o, *e, epsilon = _EPSILON));
        }
    }

    #[test]
    fn test_spectral_sub_frame_power() {
        // First element triggers flooring, second uses subtraction
        let x = vec![Complex::new(1.0, 0.0), Complex::new(3.0, 4.0)];
        let noise = vec![5.0, 10.0];
        // y2_1=1, sub2_1=-4->floor2=0.5 => sqrt(0.5)
        // y2_2=25, sub2_2=15>floor2=12.5 => sqrt(15)
        let out = spectral_sub_frame(&x, &noise, 1.0, 0.5, SsVariant::Power);
        let expected = vec![0.5_f64.sqrt(), 15.0_f64.sqrt()];
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!(approx_eq!(f64, *o, *e, epsilon = _EPSILON));
        }
    }

    #[test]
    fn test_restore_phase() {
        let amps = vec![1.0, 2.0];
        // angles 90 and 180
        let x = vec![Complex::new(0.0, 1.0), Complex::new(-1.0, 0.0)];
        let out = restore_phase(amps.clone(), x.clone());
        // expected: [(1,90), (2,180)] => [(0,1),(-2,0)]
        assert!(approx_eq!(f64, out[0].re, 0.0, epsilon = _EPSILON));
        assert!(approx_eq!(f64, out[0].im, 1.0, epsilon = _EPSILON));
        assert!(approx_eq!(f64, out[1].re, -2.0, epsilon = _EPSILON));
        assert!(approx_eq!(f64, out[1].im, 0.0, epsilon = _EPSILON));
    }

    #[test]
    fn test_ss_roundtrip_when_noise_zero() {
        let clean = vec![0.0, 0.5, -1.0, 2.0, -0.3];
        let noisy = clean.clone();
        let noise = vec![0.0; noisy.len()];
        for ssv in [SsVariant::Magnitude, SsVariant::Power] {
            let out = spectral_subtraction(&noisy, &noise, 4, 1., 0., ssv, Window::Rectangle);
            assert!(out.len() >= noisy.len() && out.len() < noisy.len() + 4);
            for (o, &c) in out.iter().zip(&clean) {
                assert!(approx_eq!(f64, *o, c, epsilon = 1e-8), "Failed for variant {:?}", ssv);
            }
        }
    }

    #[test]
    fn test_spectral_subtraction() {
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
        let alphas = [1.0, 1.5];
        let betas = [0.001, 0.0001, 0.0];
        let variants = [SsVariant::Magnitude, SsVariant::Power];
        let windows = [Window::Hamming, Window::Hanning];
        for &frame_size in &frame_sizes {
            for &alpha in &alphas {
                for &beta in &betas {
                    for &variant in &variants {
                        for &window in &windows {
                            let overlap = frame_size / 2;
                            let out =
                                spectral_subtraction(&noisy_sig, &noise, frame_size, alpha, beta, variant, window);

                            let improvement = snr_improvement(&clean, &noisy_sig, &out[..clean.len()].to_vec());
                            assert!(improvement>=25.0, // >=3 is ok actually...
                                "Failed for fs={}, ov={}, alpha={}, beta={}, variant={:?}, window={:?}: snr improvement={}"
                                , frame_size, overlap, alpha, beta, variant, window, improvement);

                            let mse_sub = mean_square_error(&clean, &out);
                            println!("Passed for fs={}, ov={}, alpha={}, beta={}, variant={:?}, window={:?}: snr improvement={}; mse_sub={} < mse_noisy={}"
                                    , frame_size, overlap, alpha, beta, variant, window, improvement, mse_sub, mse_noisy);
                        }
                    }
                }
            }
        }
    }
}
