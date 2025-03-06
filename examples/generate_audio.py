"""
Example script for generating synthetic audio data.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import our modules directly
from data_types import audio_data
from utils import exporters

# Check if audio libraries are available
try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    print("Warning: Audio libraries (librosa, soundfile) not available. Some audio functions may not work.")

def plot_waveform(audio, sample_rate, title, file_path):
    """Helper function to plot and save audio waveforms."""
    duration = len(audio) / sample_rate
    time = np.linspace(0, duration, len(audio))
    
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()

def main():
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'audio')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set the sample rate
    sample_rate = 22050
    
    # Example 1: Generate basic waveforms
    print("Generating basic waveforms...")
    
    waveforms = {
        'sine': audio_data.generate_sine_wave(duration=1.0, frequency=440.0, amplitude=0.5, sample_rate=sample_rate),
        'square': audio_data.generate_square_wave(duration=1.0, frequency=440.0, amplitude=0.5, sample_rate=sample_rate),
        'sawtooth': audio_data.generate_sawtooth_wave(duration=1.0, frequency=440.0, amplitude=0.5, sample_rate=sample_rate),
        'triangle': audio_data.generate_triangle_wave(duration=1.0, frequency=440.0, amplitude=0.5, sample_rate=sample_rate)
    }
    
    for name, wave in waveforms.items():
        # Save audio file
        audio_path = os.path.join(output_dir, f'{name}_wave.wav')
        audio_data.save_audio(wave, audio_path, sample_rate)
        
        # Plot waveform
        plot_path = os.path.join(plots_dir, f'{name}_wave.png')
        plot_waveform(wave, sample_rate, f'{name.capitalize()} Wave (440 Hz)', plot_path)
        
        print(f"Generated {name} wave: {audio_path}")
    
    print("\n" + "-"*80 + "\n")
    
    # Example 2: Generate noise signals
    print("Generating noise signals...")
    
    noise_types = {
        'white': audio_data.generate_white_noise(duration=1.0, amplitude=0.5, sample_rate=sample_rate),
        'pink': audio_data.generate_pink_noise(duration=1.0, amplitude=0.5, sample_rate=sample_rate),
        'brown': audio_data.generate_brown_noise(duration=1.0, amplitude=0.5, sample_rate=sample_rate)
    }
    
    for name, noise in noise_types.items():
        # Save audio file
        audio_path = os.path.join(output_dir, f'{name}_noise.wav')
        audio_data.save_audio(noise, audio_path, sample_rate)
        
        # Plot waveform
        plot_path = os.path.join(plots_dir, f'{name}_noise.png')
        plot_waveform(noise, sample_rate, f'{name.capitalize()} Noise', plot_path)
        
        print(f"Generated {name} noise: {audio_path}")
    
    print("\n" + "-"*80 + "\n")
    
    # Example 3: Apply envelope to a sine wave
    print("Applying envelope to a sine wave...")
    
    # Generate a sine wave
    sine_wave = audio_data.generate_sine_wave(duration=2.0, frequency=440.0, amplitude=0.5, sample_rate=sample_rate)
    
    # Apply ADSR envelope
    sine_with_envelope = audio_data.apply_envelope(
        sine_wave,
        attack=0.1,
        decay=0.2,
        sustain=0.7,
        release=0.5,
        sample_rate=sample_rate
    )
    
    # Save audio file
    audio_path = os.path.join(output_dir, 'sine_with_envelope.wav')
    audio_data.save_audio(sine_with_envelope, audio_path, sample_rate)
    
    # Plot waveform
    plot_path = os.path.join(plots_dir, 'sine_with_envelope.png')
    plot_waveform(sine_with_envelope, sample_rate, 'Sine Wave with ADSR Envelope', plot_path)
    
    print(f"Generated sine wave with envelope: {audio_path}")
    print("\n" + "-"*80 + "\n")
    
    # Example 4: Apply filters to a noise signal
    print("Applying filters to a noise signal...")
    
    # Generate white noise
    white_noise = audio_data.generate_white_noise(duration=1.0, amplitude=0.5, sample_rate=sample_rate)
    
    filter_types = {
        'lowpass': audio_data.apply_filter(white_noise, filter_type='lowpass', cutoff_frequency=1000.0, sample_rate=sample_rate),
        'highpass': audio_data.apply_filter(white_noise, filter_type='highpass', cutoff_frequency=2000.0, sample_rate=sample_rate),
        'bandpass': audio_data.apply_filter(white_noise, filter_type='bandpass', cutoff_frequency=(500.0, 2000.0), sample_rate=sample_rate)
    }
    
    for name, filtered in filter_types.items():
        # Save audio file
        audio_path = os.path.join(output_dir, f'{name}_filtered_noise.wav')
        audio_data.save_audio(filtered, audio_path, sample_rate)
        
        # Plot waveform
        plot_path = os.path.join(plots_dir, f'{name}_filtered_noise.png')
        plot_waveform(filtered, sample_rate, f'White Noise with {name.capitalize()} Filter', plot_path)
        
        print(f"Generated {name} filtered noise: {audio_path}")
    
    print("\n" + "-"*80 + "\n")
    
    # Example 5: Generate a chord
    print("Generating a chord...")
    
    # Generate a major chord (C major: C4, E4, G4)
    c_major = audio_data.generate_chord(
        notes=[261.63, 329.63, 392.00],  # C4, E4, G4
        duration=2.0,
        waveform='sine',
        amplitude=0.5,
        sample_rate=sample_rate
    )
    
    # Save audio file
    audio_path = os.path.join(output_dir, 'c_major_chord.wav')
    audio_data.save_audio(c_major, audio_path, sample_rate)
    
    # Plot waveform
    plot_path = os.path.join(plots_dir, 'c_major_chord.png')
    plot_waveform(c_major, sample_rate, 'C Major Chord', plot_path)
    
    print(f"Generated C major chord: {audio_path}")
    print("\n" + "-"*80 + "\n")
    
    # Example 6: Generate a melody
    print("Generating a melody...")
    
    # Generate a simple melody (C major scale)
    c_scale_notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C4 to C5
    c_scale_durations = [0.25] * 8  # Each note is 0.25 seconds
    
    c_scale = audio_data.generate_melody(
        notes=c_scale_notes,
        durations=c_scale_durations,
        waveform='sine',
        amplitude=0.5,
        sample_rate=sample_rate
    )
    
    # Save audio file
    audio_path = os.path.join(output_dir, 'c_scale_melody.wav')
    audio_data.save_audio(c_scale, audio_path, sample_rate)
    
    # Plot waveform
    plot_path = os.path.join(plots_dir, 'c_scale_melody.png')
    plot_waveform(c_scale, sample_rate, 'C Major Scale Melody', plot_path)
    
    print(f"Generated C major scale melody: {audio_path}")
    print("\n" + "-"*80 + "\n")
    
    # Example 7: Generate speech-like audio
    print("Generating speech-like audio...")
    
    speech = audio_data.generate_speech_like_audio(
        duration=3.0,
        amplitude=0.5,
        formant_frequencies=[500, 1500, 2500],
        sample_rate=sample_rate
    )
    
    # Save audio file
    audio_path = os.path.join(output_dir, 'speech_like.wav')
    audio_data.save_audio(speech, audio_path, sample_rate)
    
    # Plot waveform
    plot_path = os.path.join(plots_dir, 'speech_like.png')
    plot_waveform(speech, sample_rate, 'Speech-like Audio', plot_path)
    
    print(f"Generated speech-like audio: {audio_path}")
    print("\n" + "-"*80 + "\n")
    
    # Example 8: Generate a drum pattern
    print("Generating a drum pattern...")
    
    # Generate a simple drum pattern
    drum_pattern = [1, 0, 0, 1, 0, 1, 0, 0]  # Basic rhythm
    
    drums = audio_data.generate_drum_pattern(
        pattern=drum_pattern,
        tempo=120,
        sample_rate=sample_rate
    )
    
    # Save audio file
    audio_path = os.path.join(output_dir, 'drum_pattern.wav')
    audio_data.save_audio(drums, audio_path, sample_rate)
    
    # Plot waveform
    plot_path = os.path.join(plots_dir, 'drum_pattern.png')
    plot_waveform(drums, sample_rate, 'Drum Pattern', plot_path)
    
    print(f"Generated drum pattern: {audio_path}")
    print("\n" + "-"*80 + "\n")
    
    # Example 9: Mix multiple audio signals
    print("Mixing multiple audio signals...")
    
    # Generate a chord and a drum pattern
    chord = audio_data.generate_chord(
        notes=[261.63, 329.63, 392.00],  # C4, E4, G4
        duration=2.0,
        waveform='sine',
        amplitude=0.3,
        sample_rate=sample_rate
    )
    
    drum_pattern = [1, 0, 1, 0, 1, 0, 1, 0] * 4  # Repeated pattern
    drums = audio_data.generate_drum_pattern(
        pattern=drum_pattern,
        tempo=120,
        sample_rate=sample_rate
    )
    
    # Mix the signals
    mixed = audio_data.mix_audio(
        audio_signals=[chord, drums],
        weights=[0.7, 0.3]
    )
    
    # Save audio file
    audio_path = os.path.join(output_dir, 'mixed_audio.wav')
    audio_data.save_audio(mixed, audio_path, sample_rate)
    
    # Plot waveform
    plot_path = os.path.join(plots_dir, 'mixed_audio.png')
    plot_waveform(mixed, sample_rate, 'Mixed Audio (Chord + Drums)', plot_path)
    
    print(f"Generated mixed audio: {audio_path}")
    print("\n" + "-"*80 + "\n")
    
    # Example 10: Generate an audio dataset
    print("Generating an audio dataset...")
    
    dataset = audio_data.generate_audio_dataset(
        num_samples=5,
        duration=1.0,
        audio_types=['sine', 'square', 'sawtooth', 'triangle', 'white_noise'],
        sample_rate=sample_rate,
        output_dir=os.path.join(output_dir, 'dataset')
    )
    
    print(f"Generated an audio dataset with {len(dataset)} samples")
    print(f"Dataset saved to: {os.path.join(output_dir, 'dataset')}")
    
    print("\nAudio data generation examples completed successfully!")

if __name__ == "__main__":
    main()
