/* ==========================================================================================
 *                          This file is part of the Bachelor Thesis project
 *                                   University of Wrocław
 *                         Author: Weronika Tarnawska (Index No. 331171)
 *                                         June 2025
 * ========================================================================================== */
use rand::{distributions::Uniform, thread_rng, Rng};
use std::f64::consts::PI;

/// Signal types
#[derive(Clone, Copy, Debug)]
pub enum SignalType {
    WhiteNoise,      // White noise
    Sinusoidal(f64), // Sinusoidal signal with given frequency (Hz)
    Siren(f64, f64), // Sinusoidal signal alternating between two tones (f1, f2)
    RandomTone(f64), // Sinusoidal signal with random frequency change every given time (in seconds)
    Chirp(f64, f64), // Linear chirp from f1 to f2
    Chord(f64),      // Major chord based on root frequency
    Melody,          // Random melody of major chords, changing every 1/2 second
}

/// Generates white noise signal
fn generate_white_noise(len: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    let uniform = Uniform::from(-1.0..1.0);
    (0..len).map(|_| rng.sample(uniform)).collect()
}

/// Generates a sinusoidal signal at the given frequency
fn generate_sinusoidal(len: usize, frequency: f64, sr: f64) -> Vec<f64> {
    (0..len)
        .map(|i| {
            let t = i as f64 / sr;
            (2.0 * PI * frequency * t).sin()
        })
        .collect()
}

/// Generates a siren-like alternating tones signal
fn generate_siren(len: usize, f1: f64, f2: f64, sr: f64) -> Vec<f64> {
    let period = (sr / 2.0) as usize; // 1/4 second
    (0..len)
        .map(|i| {
            let t = i as f64 / sr;
            let freq = if (i / period) % 2 == 0 { f1 } else { f2 };
            (2.0 * PI * freq * t).sin()
        })
        .collect()
}

/// Generates a signal with random tone changes every given interval
fn generate_random_tone(len: usize, change_secs: f64, sr: f64) -> Vec<f64> {
    let mut rng = thread_rng();
    let change_period = (sr * change_secs) as usize;
    let mut current_freq = rng.gen_range(200.0..600.0);

    (0..len)
        .map(|i| {
            if i % change_period == 0 {
                current_freq = rng.gen_range(200.0..600.0);
            }
            let t = i as f64 / sr;
            (2.0 * PI * current_freq * t).sin()
        })
        .collect()
}

/// Generates a linear chirp from f1 to f2 over the signal duration
fn generate_chirp(len: usize, f1: f64, f2: f64, sr: f64) -> Vec<f64> {
    let duration = len as f64 / sr;
    let k = (f2 - f1) / duration;

    (0..len)
        .map(|i| {
            let t = i as f64 / sr;
            (2.0 * PI * (f1 * t + 0.5 * k * t * t)).sin()
        })
        .collect()
}

/// Generates a major chord (triad) based on the root frequency
fn generate_major_chord(root_freq: f64, len: usize, sample_rate: f64) -> Vec<f64> {
    // Calculate semitone ratio = 2^(1/12)
    let semitone = 2f64.powf(1.0 / 12.0);
    let third_freq = root_freq * semitone.powf(4.0);
    let fifth_freq = root_freq * semitone.powf(7.0);

    // Generate individual sinusoidal signals
    let root_sig = generate_sinusoidal(len, root_freq, sample_rate);
    let third_sig = generate_sinusoidal(len, third_freq, sample_rate);
    let fifth_sig = generate_sinusoidal(len, fifth_freq, sample_rate);

    // Sum signals sample-wise
    root_sig
        .into_iter()
        .zip(third_sig.into_iter())
        .zip(fifth_sig.into_iter())
        .map(|((r, t), f)| r + t + f)
        .collect()
}

/// Generates a random melody of major chords, switching every 1/4 second
fn generate_melody(len: usize, sr: f64) -> Vec<f64> {
    let mut rng = thread_rng();
    let change_period = (sr * 0.5) as usize; // 1/4 second
    let mut current_root = rng.gen_range(100.0..350.0);

    (0..len)
        .map(|i| {
            if i % change_period == 0 {
                current_root = rng.gen_range(200.0..600.0);
            }
            // compute chord sample by summing triad
            let t = i as f64 / sr;
            let semitone = 2f64.powf(1.0 / 12.0);
            let third = current_root * semitone.powf(4.0);
            let fifth = current_root * semitone.powf(7.0);
            (2.0 * PI * current_root * t).sin() + (2.0 * PI * third * t).sin() + (2.0 * PI * fifth * t).sin()
        })
        .collect()
}

/// Generates a signal vector of given length and type.
pub fn generate_signal(len: usize, sig_type: SignalType, sample_rate: f64) -> Vec<f64> {
    match sig_type {
        SignalType::WhiteNoise => generate_white_noise(len),
        SignalType::Sinusoidal(freq) => generate_sinusoidal(len, freq, sample_rate),
        SignalType::Siren(f1, f2) => generate_siren(len, f1, f2, sample_rate),
        SignalType::RandomTone(ch) => generate_random_tone(len, ch, sample_rate),
        SignalType::Chirp(f1, f2) => generate_chirp(len, f1, f2, sample_rate),
        SignalType::Chord(root) => generate_major_chord(root, len, sample_rate),
        SignalType::Melody => generate_melody(len, sample_rate),
    }
}
