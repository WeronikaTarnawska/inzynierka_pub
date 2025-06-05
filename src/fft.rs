/* ==========================================================================================
 *                          This file is part of the Bachelor Thesis project
 *                                   University of Wrocław
 *                         Author: Weronika Tarnawska (Index No. 331171)
 *                                         June 2025
 * ========================================================================================== */
use num_complex::Complex;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub enum Window {
    Hamming,
    Hanning,
    Blackman,
    Rectangle,
}

/// Handle coefficients and overlap for various window types
impl Window {
    /// Determine overlap size based on window type and frame size.
    fn get_overlap(&self, frame_size: usize) -> usize {
        match self {
            Window::Hamming | Window::Hanning => frame_size / 2,
            Window::Blackman => (frame_size * 2) / 3,
            Window::Rectangle => 0,
        }
    }

    /// Generate window coefficients for a given window type and frame size.
    fn get_coefficients(&self, frame_size: usize) -> Vec<f64> {
        match self {
            Window::Hamming => (0..frame_size)
                .map(|n| 0.54 - 0.46 * (2.0 * PI * n as f64 / (frame_size - 1) as f64).cos())
                .collect(),
            Window::Hanning => (0..frame_size)
                .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f64 / (frame_size - 1) as f64).cos())
                .collect(),
            Window::Blackman => (0..frame_size)
                .map(|n| {
                    let ratio = 2.0 * PI * n as f64 / (frame_size - 1) as f64;
                    0.42 - 0.50 * ratio.cos() + 0.08 * (2.0 * ratio).cos()
                })
                .collect(),
            Window::Rectangle => vec![1.0; frame_size],
        }
    }
}

/// Compute the discrete Fourier transform (DFT) of a sequence using radix-2 Cooley-Tukey FFT.
fn _recursive_fft(xs: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let n = xs.len();
    if n == 1 {
        return vec![xs[0]];
    }
    assert!(n % 2 == 0, "Input length must be a power of two");
    // Split into even and odd indices
    let even = xs.iter().step_by(2).cloned().collect::<Vec<_>>();
    let odd = xs.iter().skip(1).step_by(2).cloned().collect::<Vec<_>>();
    // Recursively compute smaller fft's
    let even_fft = fft(&even);
    let odd_fft = fft(&odd);
    // Combine
    let mut y = vec![Complex::new(0.0, 0.0); n];
    for k in 0..n / 2 {
        let twiddle = Complex::from_polar(1.0, -2.0 * PI * k as f64 / n as f64);
        y[k] = even_fft[k] + twiddle * odd_fft[k];
        y[k + n / 2] = even_fft[k] - twiddle * odd_fft[k];
    }
    y
}

/// Cooley-Tukey iterative FFT implementation
fn fft(xs: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let n = xs.len();
    assert!(n.is_power_of_two(), "Input length must be a power of two");

    // Bit-reverse permutation
    let mut output = xs.clone();
    let mut j = 0;
    for i in 1..n - 1 {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            output.swap(i, j);
        }
    }

    // Iterative FFT
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let phase_step = -2.0 * PI / len as f64;
        for start in (0..n).step_by(len) {
            for i in 0..half {
                let angle = phase_step * i as f64;
                let w = Complex::from_polar(1.0, angle);
                let u = output[start + i];
                let v = output[start + i + half] * w;
                output[start + i] = u + v;
                output[start + i + half] = u - v;
            }
        }
        len <<= 1;
    }
    output
}

/// Compute the inverse FFT (IFFT) by conjugation and scaling.
fn ifft(ys: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let n = ys.len() as f64;
    let conj: Vec<Complex<f64>> = ys.iter().map(|y| Complex::new(y.re, -y.im)).collect();
    let inv = fft(&conj);
    inv.iter().map(|y| Complex::new(y.re / n, -y.im / n)).collect()
}

/// Compute the short-time Fourier transform (STFT).
fn stft(samples: &Vec<Complex<f64>>, frame_size: usize, window: Window) -> Vec<Vec<Complex<f64>>> {
    // Determine overlap based on window type
    let overlap = window.get_overlap(frame_size);
    let step = frame_size - overlap;

    // Compute window coefficients
    let window = window.get_coefficients(frame_size);

    let mut frames: Vec<Vec<Complex<f64>>> = Vec::new();
    let mut start = 0;
    // Extract and window each full frame
    while start + frame_size <= samples.len() {
        let mut frame = Vec::with_capacity(frame_size);
        for i in 0..frame_size {
            frame.push(samples[start + i] * window[i]);
        }
        frames.push(fft(&frame));
        start += step;
    }
    // Handle last partial frame with zero-padding
    if start < samples.len() {
        let mut frame = Vec::with_capacity(frame_size);
        let remaining = samples.len() - start;
        for i in 0..remaining {
            frame.push(samples[start + i] * window[i]);
        }
        for _ in remaining..frame_size {
            frame.push(Complex::new(0.0, 0.0));
        }
        frames.push(fft(&frame));
    }
    frames
}

/// Reconstruct a time-domain signal by overlap-add of frames with correct windowing and normalization.
fn istft(frames: &Vec<Vec<Complex<f64>>>, frame_size: usize, window: Window) -> Vec<Complex<f64>> {
    let overlap = window.get_overlap(frame_size);
    let hop = frame_size - overlap;
    let num_frames = frames.len();
    let output_len = hop * (num_frames - 1) + frame_size;
    let mut output = vec![Complex::new(0.0, 0.0); output_len];

    for (i, freq_frame) in frames.iter().enumerate() {
        // Inverse FFT -> time-domain frame (still unwindowed)
        let time_frame = ifft(freq_frame);

        // Overlap-add into output;
        let offset = i * hop;
        for n in 0..frame_size {
            output[offset + n] += time_frame[n];
        }
    }
    output
}

/// Convert a real-valued time-domain signal to its STFT representation.
///
/// This performs framing, windowing, and FFT on each frame with overlap determined by the window type:
/// - Hamming, Hanning: overlap = frame_size / 2
/// - Blackman: overlap = 2 * frame_size / 3
/// - Rectangle: overlap = 0
///
/// # Arguments
/// * `samples`    - Reference to a vector of real time-domain samples.
/// * `frame_size` - Number of samples per frame (must be power-of-two).
/// * `window_type`- Window function to apply
///
/// # Returns
/// A vector of frames, each a Vec<Complex<f64>> representing frequency bins.
pub fn time2freq(samples: &Vec<f64>, frame_size: usize, window_type: Window) -> Vec<Vec<Complex<f64>>> {
    let complex_samples: Vec<Complex<f64>> = samples.iter().map(|&x| Complex::new(x, 0.0)).collect();
    stft(&complex_samples, frame_size, window_type)
}

/// Convert a series of complex frequency-domain frames back to a real time-domain signal.
///
/// # Arguments
/// * `frames`     - Vector of complex frequency-domain frames (each of same length that is power-of-two).
/// * `window_type`- Window function that was applied during STFT (determines overlap and window shape).
///
/// # Returns
/// A vector of real-valued time-domain samples.
pub fn freq2time(frames: &Vec<Vec<Complex<f64>>>, window_type: Window) -> Vec<f64> {
    let frame_size = frames[0].len();
    let complex_time = istft(frames, frame_size, window_type);
    complex_time.into_iter().map(|c| c.re).collect()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::signal::*;
    use crate::utils::*;
    use crate::_EPSILON;
    use float_cmp::approx_eq;
    use num_complex::Complex;
    use rustfft::FftPlanner;

    #[test]
    fn test_fft_vs_rustfft() {
        let size = 4096;
        let samples: Vec<Complex<f64>> = generate_signal(size, SignalType::Chord(440.0), 44100.0)
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        let mut planner = FftPlanner::<f64>::new();
        let r_fft = planner.plan_fft_forward(size);
        let mut rustfft_result: Vec<Complex<f64>> = samples.clone();
        r_fft.process(&mut rustfft_result);

        let my_result = fft(&samples);

        let cmp: Vec<bool> = my_result
            .iter()
            .zip(rustfft_result)
            .map(|(a, b)| {
                approx_eq!(f64, a.re, b.re, epsilon = _EPSILON) && approx_eq!(f64, a.im, b.im, epsilon = _EPSILON)
            })
            .collect();

        assert!(cmp.iter().all(|&x| x));
    }

    #[test]
    fn test_ifft_vs_rustfft() {
        let size = 4096;
        let samples: Vec<Complex<f64>> = generate_signal(size, SignalType::Chord(440.0), 44100.0)
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        let samples = fft(&samples);

        let mut planner = FftPlanner::<f64>::new();
        let r_fft = planner.plan_fft_inverse(size);
        let mut rustfft_result: Vec<Complex<f64>> = samples.clone();
        r_fft.process(&mut rustfft_result);
        let rustfft_result: Vec<Complex<f64>> = rustfft_result
            .iter()
            .map(|y| Complex::new(y.re / size as f64, y.im / size as f64))
            .collect(); // rustfft expects scaling to be performed outside

        let my_result = ifft(&samples);

        let cmp: Vec<bool> = my_result
            .iter()
            .zip(rustfft_result)
            .map(|(a, b)| {
                approx_eq!(f64, a.re, b.re, epsilon = _EPSILON) && approx_eq!(f64, a.im, b.im, epsilon = _EPSILON)
            })
            .collect();

        assert!(cmp.iter().all(|&x| x));
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let samples: Vec<Complex<f64>> = generate_signal(8192, SignalType::Chord(440.0), 44100.0)
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        let spectral = fft(&samples);
        let sapmles_after = ifft(&spectral);

        let cmp: Vec<bool> = samples
            .iter()
            .zip(sapmles_after)
            .map(|(a, b)| {
                approx_eq!(f64, a.re, b.re, epsilon = _EPSILON) && approx_eq!(f64, a.im, b.im, epsilon = _EPSILON)
            })
            .collect();

        assert!(cmp.iter().all(|&x| x));
    }

    #[test]
    fn test_fft_symmetry_check() {
        let size = 1024;
        let samples: Vec<Complex<f64>> = generate_signal(size, SignalType::Chord(440.0), 44100.0)
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        let mut first_half = fft(&samples);

        let mut second_half = first_half.split_off(size / 2)[1..].to_vec();
        second_half.reverse();

        // Symmetry property: Y[k] == complex_conjugate(Y[N-k])
        // Special Cases for k=0 and k=N/2
        let cmp: Vec<bool> = first_half[1..]
            .iter()
            .zip(second_half)
            .map(|(a, b)| {
                approx_eq!(f64, a.re, b.re, epsilon = _EPSILON) && approx_eq!(f64, a.im, -b.im, epsilon = _EPSILON)
            })
            .collect();

        assert!(cmp.iter().all(|&x| x));
    }

    #[test]
    fn test_stft_istft_roundtrip() {
        let signal_types = vec![
            // SignalType::WhiteNoise,
            SignalType::Sinusoidal(9375.0),
            SignalType::Siren(9375.0, 18750.0),
            SignalType::Chirp(9375.0, 18750.0),
        ];
        let frame_sizes = vec![128, 256];
        let sample_rate = 48000.0;

        for sig_type in signal_types {
            let original: Vec<f64> = generate_signal(10000, sig_type, sample_rate);

            for &frame_size in &frame_sizes {
                for &win in &[Window::Hamming, Window::Hanning /*, Window::Rectangle*/] {
                    let overlap = frame_size / 2;
                    let freq_dom = time2freq(&original, frame_size, win);
                    let reconstructed = freq2time(&freq_dom, win);
                    assert!(
                        reconstructed.len() >= original.len()
                            && reconstructed.len() <= original.len() + frame_size - overlap,
                        "Length mismatch for {:?}, frame {}, window {:?}: original={} vs reconstructed={}",
                        sig_type,
                        frame_size,
                        win,
                        original.len(),
                        reconstructed.len()
                    );
                    let reconstructed = reconstructed[..original.len()].to_vec();
                    let snr = sig_to_noise_ratio(&original, &reconstructed);
                    assert!(
                        snr > 100.0,
                        "Signal too corrupted for {:?}, frame {}, overlap {}, window {:?} : SNR = {}",
                        sig_type,
                        frame_size,
                        overlap,
                        win,
                        snr
                    );
                }
            }
        }
    }
}
