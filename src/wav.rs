/* ==========================================================================================
 *                          This file is part of the Bachelor Thesis project
 *                                   University of Wrocław
 *                         Author: Weronika Tarnawska (Index No. 331171)
 *                                         June 2025
 * ========================================================================================== */
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

const SAMPLE_MAX: f64 = 32767.0;

/// Normalize a vector of samples to the range [-1.0, 1.0]
fn normalize_samples(samples: &Vec<f64>) -> Vec<f64> {
    let max_sample = samples.iter().fold(0.0_f64, |max, &s| max.max(s.abs()));
    if max_sample > 1.0 {
        return samples.into_iter().map(|s| s / max_sample).collect();
    }
    return samples.clone();
}

/// Reads a single-channel 16-bit PCM WAV file at 44.1 kHz and returns a Vec<f64> of samples normalized to [-1.0, 1.0].
/// Panics if the file cannot be opened, if it is has more than one channel, not 44.1 kHz, not 16-bit, or not PCM Int.
pub fn read_wav(path: &str) -> Vec<f64> {
    let mut reader = WavReader::open(path).unwrap_or_else(|e| {
        panic!("Failed to open WAV file '{}': {}", path, e);
    });
    let spec = reader.spec();

    // Verify WAV format matches expected: mono, 44.1 kHz, 16-bit PCM Int
    if spec.channels != 1 {
        panic!("Expected 1 channel, found {}", spec.channels);
    }
    if spec.sample_rate != 44100 {
        panic!("Expected sample rate 44100 Hz, found {}", spec.sample_rate);
    }
    if spec.bits_per_sample != 16 {
        panic!("Expected 16 bits per sample, found {}", spec.bits_per_sample);
    }
    if spec.sample_format != SampleFormat::Int {
        panic!("Expected PCM Integer format, found {:?}", spec.sample_format);
    }

    let mut samples = Vec::new();
    for sample in reader.samples::<i16>() {
        samples.push((sample.unwrap() as f64) / SAMPLE_MAX);
    }

    samples
}

/// Writes a vector of samples (in any range) to a single-channel 16-bit PCM WAV file at 44.1 kHz.
/// Panics if the file cannot be created.
/// The entire signal is first normalized to [-1.0, 1.0], then scaled to i16.
pub fn save_wav(sig: &Vec<f64>, path: &str) {
    let normalized = normalize_samples(sig);

    let spec = WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec).unwrap_or_else(|e| {
        panic!("Failed to create WAV file '{}': {}", path, e);
    });

    for s in normalized {
        writer.write_sample((s * SAMPLE_MAX) as i16).unwrap();
    }

    writer.finalize().unwrap();
}
