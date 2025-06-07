# Thesis project

This repository is my bachelor thesis project  
*Implementation and Analysis of Selected Noise Reduction Methods*  
by Weronika Tarnawska (Index No. 331171),  
Supervisor:  dr hab. Paweł Woźny,  
University of Wrocław, June 2025.

---

## Build and run

```bash
# Build
cargo build --release

# Run (list subcommands)
./target/release/frontend help
# or
cargo run --release -- help

# Print help for given subcommand (here `ss` as an example)
./target/release/frontend ss --help
# or
cargo run --release -- ss --help
```

## Thesis tests and plotting results

```bash
# Run tests (set mentioned in the experiments chapter in the thesis)
./target/release/frontend test

# Prepare python environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install pandas matplotlib seaborn
# Plot results
python plot_results.py
```

## Test signals generation

```bash
# sine_white
cargo r -r -- sig-gen -t white -d 6 -o noise_white.wav
cargo r -r -- sig-gen -t sine,440.0 -d 6 -o clean_sine.wav
cargo r -r -- mix -n noise_white.wav  -c clean_sine.wav -l 1.1

# chord_chord
chord_chord
cargo r -r -- sig-gen -t chord,500.0 -d 6 -o clean_chord_500.wav
cargo r -r -- sig-gen -t chord,600.0 -d 6 -o noise_chord_600.wav
cargo r -r -- mix -n noise_chord_600.wav -c clean_chord_500.wav

# speech_vartones
arecord -f S16_LE -c 1 -r 44100 clean_speech.wav
cargo r -r -- sig-gen -t randtone,1.5 -d 20 -o noise_tones.wav
cargo r -r -- mix -n noise_tones.wav  -c clean_speech.wav -l 0.5 -v

# tones_hairdrier
arecord -f S16_LE -c 1 -r 44100 noise_hairdrier_varying.wav
cargo r -r -- sig-gen -t randtone,0.5 -d 20 -o clean_tones.wav
cargo r -r -- mix -n noise_hairdrier_varying.wav  -c clean_tones.wav 

# speech_guitar
arecord -f S16_LE -c 1 -r 44100 noise_guitar.wav
arecord -f S16_LE -c 1 -r 44100 clean_speech.wav
cargo r -r mix -n noise_guitar.wav -c clean_speech.wav -l 0.4
# guitar_speech
cp ../speech_guitar/noise_guitar.wav clean_guitar.wav
cp ../speech_guitar/clean_speech.wav noise_speech.wav
cargo r -r -- mix -n noise_speech.wav -c clean_guitar.wav  -l 0.4
```

## Example denoising commands

```sh
# spectral subtraction alpha=2.5 frame=2048 variant=Magnitude
cargo r -r -- ss -x mixed.wav -y noise.wav -a 2.5 -f 2048 -v Magnitude -o ss.wav 
# spectral subtraction alpha=3.2 frame=4096 variant=Power
cargo r -r -- ss -x mixed.wav -y noise.wav -a 3.2 -f 4096 -v Power -o ss.wav
# wiener variant=instant-noise frame=1024
cargo r -r wiener -x mixed.wav -y noise.wav -f 1024 -v InstNoise -o wiener.wav
# wiener variant=average-noise frame=8192
cargo r -r wiener -x mixed.wav -y noise.wav -f 8192 -v AvgNoise -o wiener.wav
# lms filter=16 step=0.0001
cargo r -r -- lms -x mixed.wav  -y noise.wav -f 16 -s 0.0001 -o lms.wav

# details
cargo r -r -- --help
cargo r -r -- <ss|lms|wiener> --help
```
