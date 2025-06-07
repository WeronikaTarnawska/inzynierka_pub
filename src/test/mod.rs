/* ==================================================================================================
 *                           This file is part of the bachelor thesis project
 *                  Implementation and Analysis of Selected Noise Reduction Methods
 *                                Weronika Tarnawska (Index No. 331171)
 *                                  Supervisor:  dr hab. Paweł Woźny
 *                                  University of Wrocław, June 2025
 * ================================================================================================== */
use my_dsp::fft::Window;
use my_dsp::lms::lms;
use my_dsp::ss::{spectral_subtraction, SsVariant};
use my_dsp::utils::{snr_improvement, snr_improvement_db};
use my_dsp::wav::{read_wav, save_wav};
use my_dsp::wiener::{wiener_denoise, WienerVariant};

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use csv::Writer;

/// Helper: find the first file in `case_path` whose stem starts with `prefix`
fn find_file(case_path: &Path, prefix: &str) -> Option<PathBuf> {
    fs::read_dir(case_path)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .map(|s| s.starts_with(prefix))
                .unwrap_or(false)
        })
}

/// Load the noisy, noise-only, and clean WAV signals for a given test case
fn load_case_signals(case_dir: &Path) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let noisy_path = find_file(case_dir, "mixed").unwrap_or_else(|| panic!("No 'mixed*.wav' file in {:?}", case_dir));
    let noise_path = find_file(case_dir, "noise").unwrap_or_else(|| panic!("No 'noise*.wav' file in {:?}", case_dir));
    let clean_path = find_file(case_dir, "clean").unwrap_or_else(|| panic!("No 'clean*.wav' file in {:?}", case_dir));

    let noisy_signal = read_wav(noisy_path.to_str().unwrap());
    let noise_signal = read_wav(noise_path.to_str().unwrap());
    let clean_signal = read_wav(clean_path.to_str().unwrap());

    // Make sure signals have equal lengths
    let len = noisy_signal.len().min(clean_signal.len().min(noise_signal.len()));
    (
        noisy_signal[..len].to_vec(),
        noise_signal[..len].to_vec(),
        clean_signal[..len].to_vec(),
    )
}

/// Process Spectral Subtraction for one test case
fn process_spectral_subtraction(
    case_name: &str,
    noisy: &Vec<f64>,
    noise: &Vec<f64>,
    clean: &Vec<f64>,
    workdir: &Path,
    csv_writer: &mut Writer<std::fs::File>,
    ss_frame_alpha: &[(usize, f64)],
    beta: f64,
) -> Result<(), Box<dyn Error>> {
    for (frame, alpha) in ss_frame_alpha {
        for &variant in &[SsVariant::Magnitude, SsVariant::Power] {
            let start = Instant::now();
            let denoised = spectral_subtraction(noisy, noise, *frame, *alpha, beta, variant, Window::Hanning);
            let duration_sec = start.elapsed().as_secs_f64();

            let snr_lin = snr_improvement(clean, noisy, &denoised);
            let snr_db = snr_improvement_db(clean, noisy, &denoised);

            let alg_str = match variant {
                SsVariant::Magnitude => "SS-Magnitude",
                SsVariant::Power => "SS-Power",
            };
            let filename = format!("{}_{}_a{:.2}_f{}.wav", case_name, alg_str.to_lowercase(), alpha, frame);
            let out_path = workdir.join(&filename);
            save_wav(&denoised, out_path.to_str().unwrap());

            csv_writer.write_record(&[
                case_name,
                alg_str,
                frame.to_string().as_str(),
                "",
                format!("{:.2}", alpha).as_str(),
                "",
                &snr_lin.to_string(),
                &snr_db.to_string(),
                &duration_sec.to_string(),
            ])?;
        }
    }

    Ok(())
}

/// Process Wiener denoising for one test case
fn process_wiener_denoising(
    case_name: &str,
    noisy: &Vec<f64>,
    noise: &Vec<f64>,
    clean: &Vec<f64>,
    workdir: &Path,
    csv_writer: &mut Writer<std::fs::File>,
    frame_sizes: &[usize],
    epsilon: f64,
) -> Result<(), Box<dyn Error>> {
    for &variant in &[WienerVariant::AvgNoise, WienerVariant::InstNoise] {
        for &frame_size in frame_sizes {
            // AvgNoise variant
            let start = Instant::now();
            let denoised = wiener_denoise(noisy, noise, frame_size, Window::Hanning, epsilon, variant);
            let duration = start.elapsed().as_secs_f64();

            let snr_lin = snr_improvement(clean, noisy, &denoised);
            let snr_db = snr_improvement_db(clean, noisy, &denoised);

            let alg_str = match variant {
                WienerVariant::AvgNoise => "Wiener-Avg",
                WienerVariant::InstNoise => "Wiener-Inst",
            };
            let filename = format!("{}_{}_f{}.wav", case_name, alg_str.to_lowercase(), frame_size);
            let out_path = workdir.join(&filename);
            save_wav(&denoised, out_path.to_str().unwrap());
            csv_writer.write_record(&[
                case_name,
                alg_str,
                frame_size.to_string().as_str(),
                "",
                "",
                "",
                &snr_lin.to_string(),
                &snr_db.to_string(),
                &duration.to_string(),
            ])?;
        }
    }

    Ok(())
}

/// Process LMS denoising for one test case
fn process_lms_denoising(
    case_name: &str,
    noisy: &Vec<f64>,
    noise: &Vec<f64>,
    clean: &Vec<f64>,
    workdir: &Path,
    csv_writer: &mut Writer<std::fs::File>,
    filter_step: &[(usize, f64)],
) -> Result<(), Box<dyn Error>> {
    // 3a) Sweep over filter sizes
    for (filter_size, step_size) in filter_step {
        let start = Instant::now();
        let (_y, denoised_lms, _w) = lms(noise, noisy, *filter_size, *step_size);
        let duration_sec = start.elapsed().as_secs_f64();

        let snr_lin = snr_improvement(clean, noisy, &denoised_lms);
        let snr_db = snr_improvement_db(clean, noisy, &denoised_lms);

        let filename = format!("{}_lms_s{:.4}_f{}.wav", case_name, step_size, filter_size);
        let out_path = workdir.join(&filename);
        save_wav(&denoised_lms, out_path.to_str().unwrap());
        csv_writer.write_record(&[
            case_name,
            "LMS",
            "",
            filter_size.to_string().as_str(),
            "",
            format!("{:.4}", step_size).as_str(),
            &snr_lin.to_string(),
            &snr_db.to_string(),
            &duration_sec.to_string(),
        ])?;
    }

    Ok(())
}

pub fn run() -> Result<(), Box<dyn Error>> {
    // === Configuration ===

    // List of subdirectories under "./test"
    let cases = vec![
        // "chirp_chirp",
        // "chirpdown_sine",
        // "chirpup_sine",
        "chord_chord",
        "guitar_speech",
        // "melody_chord", //?
        // "sine_varsine",
        // "sine_sine",
        "sine_white",
        // "siren_siren", //?
        "speech_guitar",
        "speech_vartones",
        // "speech_varnoise",
        // "melody_hairdrier",//?
        "tones_hairdrier",
        // "tones_white", //?

        // "speech_hairdrier",
        // "speech_microwave",
        // "guitar_oven",
    ];

    // Paths
    let test_dir = Path::new("./test");
    let workdir = Path::new("./workdir");

    // Spectral Subtraction (SS) parameters
    let ss_frame_alpha = vec![
        (1024, 1.0),
        (1024, 2.0),
        (4096, 1.0),
        (4096, 2.0),
        (16384, 1.0),
        (16384, 2.0),
    ];
    let beta = 1e-5;

    // Wiener parameters
    let wiener_frame_sizes = vec![1024, 4096, 16384];
    let epsilon = 1e-5;

    // LMS parameters
    let lms_filter_step = vec![(64, 0.0001), (32, 0.0005), (16, 0.005)];

    // Prepare CSV writer for results (add "snr_linear", "snr_db", and "time_sec" columns)
    let mut csv_writer = Writer::from_path("results.csv")?;
    csv_writer.write_record(&[
        "case",
        "algorithm",
        "frame",
        "filter",
        "alpha",
        "step",
        "snr_linear",
        "snr_db",
        "time_sec",
    ])?;

    // === Loop over all test cases ===
    for case in &cases {
        println!("Running case {}", case);
        let case_path = test_dir.join(case);
        let case_out = workdir.join(Path::new(case));
        fs::create_dir_all(&case_out)?;

        // Load noisy, noise-only, and clean signals
        let (noisy_signal, noise_signal, clean_signal) = load_case_signals(&case_path);

        // 1) Spectral Subtraction
        process_spectral_subtraction(
            case,
            &noisy_signal,
            &noise_signal,
            &clean_signal,
            &case_out,
            &mut csv_writer,
            &ss_frame_alpha,
            beta,
        )?;

        // 2) Wiener Denoising
        process_wiener_denoising(
            case,
            &noisy_signal,
            &noise_signal,
            &clean_signal,
            &case_out,
            &mut csv_writer,
            &wiener_frame_sizes,
            epsilon,
        )?;

        // 3) LMS Denoising
        process_lms_denoising(
            case,
            &noisy_signal,
            &noise_signal,
            &clean_signal,
            &case_out,
            &mut csv_writer,
            &lms_filter_step,
        )?;
    }

    // Finalize CSV
    csv_writer.flush()?;
    println!("All tests completed. Outputs in './workdir', SNR results (linear & dB, with timing) in 'results.csv'.");

    Ok(())
}
