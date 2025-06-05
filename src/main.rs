/* ==========================================================================================
 *                          This file is part of the Bachelor Thesis project
 *                                   University of Wrocław
 *                         Author: Weronika Tarnawska (Index No. 331171)
 *                                         June 2025
 * ========================================================================================== */
use clap::{arg, Command};

use my_dsp::fft::Window;
use my_dsp::lms::lms;
use my_dsp::signal::{generate_signal, SignalType};
use my_dsp::ss::{spectral_subtraction, SsVariant};
use my_dsp::utils::{add_noise, apply_variable_scaling};
use my_dsp::wav::{read_wav, save_wav};
use my_dsp::wiener::{wiener_denoise, WienerVariant};

mod test;
use test::run;

fn main() {
    let matches = Command::new("Audio Processing CLI")
        .version("1.0")
        .author("Weronika")
        .about("CLI for signal generation and denoising")
        .subcommand(
            Command::new("sig-gen")
                .about("Generate signal and save to WAV")
                .arg(arg!(-t --"type" <TYPE> "Noise type, e.g., white|sine,440.0|siren,300,600|randtone,0.5|chirp,200,800|chord,440.0|melody").required(true))
                .arg(arg!(-d --"duration" <DUR> "Duration in seconds").required(true))
                .arg(arg!(-o --"out-file" <FILE> "Output WAV path").default_value("output.wav")),
        )
        .subcommand(
            Command::new("lms")
                .about("LMS adaptive filtering")
                .arg(arg!(-x --"noisy" <FILE> "Noisy WAV").required(true))
                .arg(arg!(-y --"noise-only" <FILE> "Noise-only WAV").required(true))
                .arg(arg!(-f --"filt-size" <N> "Filter size").default_value("32"))
                .arg(arg!(-s --"step-size" <MU> "Step size").default_value("0.0005"))
                .arg(arg!(-o --"out-file" <OUT> "Output WAV path").default_value("lms.wav")),
        )
        .subcommand(
            Command::new("ss")
                .about("Spectral subtraction")
                .arg(arg!(-x --"noisy" <FILE> "Noisy WAV").required(true))
                .arg(arg!(-y --"noise-only" <FILE> "Noise-only WAV").required(true))
                .arg(arg!(-f --"frame-size" <M> "Frame size").default_value("4096"))
                .arg(arg!(-a --"alpha" <ALPHA> "Subtraction level alpha").default_value("1.0"))
                .arg(arg!(-b --"beta" <BETA> "Spectral floor beta").default_value("1e-5"))
                .arg(arg!(-v --"variant" <VAR> "Power|Magnitude").default_value("Magnitude"))
                .arg(arg!(-w --"window" <W> "Hamming|Hanning|Blackman|Rectangle").default_value("Hanning"))
                .arg(arg!(-o --"out-file" <OUT> "Output WAV path").default_value("ss.wav")),
        )
        .subcommand(
            Command::new("wiener")
                .about("Frequency-domain Wiener filter")
                .arg(arg!(-x --"noisy" <FILE> "Noisy WAV").required(true))
                .arg(arg!(-y --"noise-only" <FILE> "Noise-only WAV").required(true))
                .arg(arg!(-f --"frame-size" <M> "Frame size").default_value("4096"))
                .arg(arg!(-w --"window" <W> "Hanning|Hamming|Blackman|Rectangle").default_value("Hanning"))
                .arg(arg!(-e --"eps" <EPS> "Regularization epsilon").default_value("1e-5"))
                .arg(arg!(-v --"variant" <VAR> "InstNoise|AvgNoise").default_value("AvgNoise"))
                .arg(arg!(-o --"out-file" <OUT> "Output WAV path").default_value("wiener.wav")),
        )
        .subcommand(
            Command::new("mix")
                .about("Mix clean and noise signals")
                .arg(arg!(-c --"clean" <FILE> "Path to clean WAV").required(true))
                .arg(arg!(-n --"noise" <FILE> "Path to noise WAV").required(true))
                .arg(arg!(-l --"noise-level" <VAL> "Noise level multiplier").default_value("1.0"))
                .arg(arg!(-v --"varying-noise" "Enable varying noise level"))
                .arg(arg!(-o --"out-file" <FILE> "Output WAV path").default_value("mixed.wav")),
        )
        .subcommand(
            Command::new("test")
                .about("Run final tests"),
        )
        .subcommand(
            Command::new("tmp")
                .about("Temporary: run to see what it does ;)"),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("sig-gen", m)) => handle_sig_gen(m),
        Some(("lms", m)) => handle_lms(m),
        Some(("ss", m)) => handle_ss(m),
        Some(("wiener", m)) => handle_wiener(m),
        Some(("mix", m)) => handle_mix(m),
        Some(("test", _)) => run().unwrap(),
        Some(("tmp", _)) => tmp(),
        _ => eprintln!("Unknown command. Use --help."),
    }
}

fn handle_sig_gen(m: &clap::ArgMatches) {
    let noise_type = m.get_one::<String>("type").unwrap();
    let duration: f64 = m.get_one::<String>("duration").unwrap().parse().unwrap();
    let out_file = m.get_one::<String>("out-file").unwrap();
    let nt = parse_signal_type(noise_type);
    let len = (duration * 44100.0) as usize;
    let sig = generate_signal(len, nt, 44100.0);
    save_wav(&sig, out_file);
    println!("Generated {}-second {} -> {}", duration, noise_type, out_file);
}

fn handle_lms(m: &clap::ArgMatches) {
    let noise = read_wav(m.get_one::<String>("noise-only").unwrap());
    let noisy = read_wav(m.get_one::<String>("noisy").unwrap());
    let filt_size: usize = m.get_one::<String>("filt-size").unwrap().parse().unwrap();
    let step_size: f64 = m.get_one::<String>("step-size").unwrap().parse().unwrap();
    let out_path = m.get_one::<String>("out-file").unwrap();
    let (_, out, _) = lms(&noise, &noisy, filt_size, step_size);
    save_wav(&out, out_path);
    println!("LMS done -> {}", out_path);
}

fn handle_ss(m: &clap::ArgMatches) {
    let noisy = read_wav(m.get_one::<String>("noisy").unwrap());
    let noise = read_wav(m.get_one::<String>("noise-only").unwrap());
    let frame_size: usize = m.get_one::<String>("frame-size").unwrap().parse().unwrap();
    let alpha: f64 = m.get_one::<String>("alpha").unwrap().parse().unwrap();
    let beta: f64 = m.get_one::<String>("beta").unwrap().parse().unwrap();
    let out_path = m.get_one::<String>("out-file").unwrap();
    let variant = match m.get_one::<String>("variant").unwrap().to_lowercase().as_str() {
        "magnitude" => SsVariant::Magnitude,
        "power" => SsVariant::Power,
        _ => panic!("Unknown SS variant"),
    };
    let window = match m.get_one::<String>("window").unwrap().to_lowercase().as_str() {
        "hamming" => Window::Hamming,
        "hanning" => Window::Hanning,
        "blackman" => Window::Blackman,
        "rectangle" => Window::Rectangle,
        _ => panic!("Unknown window type"),
    };
    let out = spectral_subtraction(&noisy, &noise, frame_size, alpha, beta, variant, window);
    save_wav(&out, out_path);
    println!("Spectral subtraction done -> {}", out_path);
}

fn handle_wiener(m: &clap::ArgMatches) {
    let noise = read_wav(m.get_one::<String>("noise-only").unwrap());
    let noisy = read_wav(m.get_one::<String>("noisy").unwrap());

    let frame_size: usize = m.get_one::<String>("frame-size").unwrap().parse().unwrap();
    let eps: f64 = m.get_one::<String>("eps").unwrap().parse().unwrap();
    let out_path = m.get_one::<String>("out-file").unwrap();

    let window = match m.get_one::<String>("window").unwrap().to_lowercase().as_str() {
        "hamming" => Window::Hamming,
        "hanning" => Window::Hanning,
        "blackman" => Window::Blackman,
        "rectangle" => Window::Rectangle,
        _ => panic!("Unknown window type"),
    };

    let variant = match m.get_one::<String>("variant").unwrap().to_lowercase().as_str() {
        "instnoise" => WienerVariant::InstNoise,
        "avgnoise" => WienerVariant::AvgNoise,
        _ => panic!("Invalid variant"),
    };

    let denoised: Vec<f64> = wiener_denoise(&noisy, &noise, frame_size, window, eps, variant);

    save_wav(&denoised, out_path);
    println!("Frequency-domain Wiener filtering done -> {}", out_path);
}

fn handle_mix(m: &clap::ArgMatches) {
    // Read arguments
    let clean_file = m.get_one::<String>("clean").unwrap();
    let noise_file = m.get_one::<String>("noise").unwrap();
    let varying = m.get_one::<bool>("varying-noise").unwrap();
    let noise_level: f64 = m
        .get_one::<String>("noise-level")
        .unwrap()
        .parse()
        .expect("Invalid noise-level");
    let out_file = m.get_one::<String>("out-file").unwrap();

    // Load WAV data
    let clean = read_wav(clean_file);
    let noise = read_wav(noise_file);
    let sig_len = clean.len().min(noise.len());
    let clean = clean[..sig_len].to_vec();
    let mut noise = noise[..sig_len].to_vec();
    if *varying {
        noise = apply_variable_scaling(&noise);
    }

    // Mix
    let mixed = add_noise(&clean, &noise, noise_level);

    // Save
    save_wav(&mixed, out_file);
    println!(
        "Mixed {} + {} * {} -> {}",
        clean_file, noise_file, noise_level, out_file
    );
}

fn parse_signal_type(s: &str) -> SignalType {
    let s = s.to_lowercase();
    if s == "white" {
        SignalType::WhiteNoise
    } else if s.starts_with("sine,") {
        let f = s["sine,".len()..].parse().unwrap();
        SignalType::Sinusoidal(f)
    } else if s.starts_with("siren,") {
        let parts: Vec<f64> = s["siren,".len()..].split(',').map(|x| x.parse().unwrap()).collect();
        SignalType::Siren(parts[0], parts[1])
    } else if s.starts_with("randtone,") {
        let p: f64 = s["randtone,".len()..].parse().unwrap();
        SignalType::RandomTone(p)
    } else if s.starts_with("chirp,") {
        let parts: Vec<f64> = s["chirp,".len()..].split(',').map(|x| x.parse().unwrap()).collect();
        SignalType::Chirp(parts[0], parts[1])
    } else if s.starts_with("chord,") {
        let f: f64 = s["chord,".len()..].parse().unwrap();
        SignalType::Chord(f)
    } else if s == "melody" {
        SignalType::Melody
    } else {
        panic!("Unknown type: {}", s)
    }
}

fn tmp() {
    println!("Hello :)")
}
