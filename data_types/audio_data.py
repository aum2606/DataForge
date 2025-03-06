"""
Module for generating synthetic audio data.
"""

import numpy as np
import random
import math
import os
import sys
from typing import List, Dict, Tuple, Any, Optional, Union

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import our modules
from utils import distributions
from config import config

# Try to import audio libraries, but don't fail if they're not available
try:
    import librosa
    import soundfile as sf
    from scipy import signal
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    print("Warning: Audio libraries (librosa, soundfile) not available. Some audio functions may not work.")

def generate_sine_wave(duration: float = 1.0, frequency: float = 440.0, 
                      amplitude: float = 0.5, sample_rate: int = 22050) -> np.ndarray:
    """
    Generate a sine wave audio signal.
    
    Args:
        duration (float): Duration of the audio in seconds
        frequency (float): Frequency of the sine wave in Hz
        amplitude (float): Amplitude of the sine wave (0.0 to 1.0)
        sample_rate (int): Sample rate of the audio in Hz
        
    Returns:
        numpy.ndarray: Generated sine wave audio signal
    """
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate sine wave
    audio = amplitude * np.sin(2 * np.pi * frequency * t)
    
    return audio

def generate_square_wave(duration: float = 1.0, frequency: float = 440.0, 
                        amplitude: float = 0.5, duty_cycle: float = 0.5,
                        sample_rate: int = 22050) -> np.ndarray:
    """
    Generate a square wave audio signal.
    
    Args:
        duration (float): Duration of the audio in seconds
        frequency (float): Frequency of the square wave in Hz
        amplitude (float): Amplitude of the square wave (0.0 to 1.0)
        duty_cycle (float): Duty cycle of the square wave (0.0 to 1.0)
        sample_rate (int): Sample rate of the audio in Hz
        
    Returns:
        numpy.ndarray: Generated square wave audio signal
    """
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate square wave
    audio = amplitude * ((t * frequency) % 1.0 < duty_cycle) * 2 - 1
    
    return audio

def generate_sawtooth_wave(duration: float = 1.0, frequency: float = 440.0, 
                          amplitude: float = 0.5, sample_rate: int = 22050) -> np.ndarray:
    """
    Generate a sawtooth wave audio signal.
    
    Args:
        duration (float): Duration of the audio in seconds
        frequency (float): Frequency of the sawtooth wave in Hz
        amplitude (float): Amplitude of the sawtooth wave (0.0 to 1.0)
        sample_rate (int): Sample rate of the audio in Hz
        
    Returns:
        numpy.ndarray: Generated sawtooth wave audio signal
    """
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate sawtooth wave
    audio = amplitude * 2 * ((t * frequency) % 1.0) - amplitude
    
    return audio

def generate_triangle_wave(duration: float = 1.0, frequency: float = 440.0, 
                          amplitude: float = 0.5, sample_rate: int = 22050) -> np.ndarray:
    """
    Generate a triangle wave audio signal.
    
    Args:
        duration (float): Duration of the audio in seconds
        frequency (float): Frequency of the triangle wave in Hz
        amplitude (float): Amplitude of the triangle wave (0.0 to 1.0)
        sample_rate (int): Sample rate of the audio in Hz
        
    Returns:
        numpy.ndarray: Generated triangle wave audio signal
    """
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate triangle wave
    audio = amplitude * 2 * np.abs(2 * ((t * frequency) % 1.0) - 1) - amplitude
    
    return audio

def generate_white_noise(duration: float = 1.0, amplitude: float = 0.1, 
                        sample_rate: int = 22050) -> np.ndarray:
    """
    Generate white noise audio signal.
    
    Args:
        duration (float): Duration of the audio in seconds
        amplitude (float): Amplitude of the noise (0.0 to 1.0)
        sample_rate (int): Sample rate of the audio in Hz
        
    Returns:
        numpy.ndarray: Generated white noise audio signal
    """
    # Generate white noise
    audio = amplitude * np.random.uniform(-1, 1, int(sample_rate * duration))
    
    return audio

def generate_pink_noise(duration: float = 1.0, amplitude: float = 0.1, 
                       sample_rate: int = 22050) -> np.ndarray:
    """
    Generate pink noise audio signal.
    
    Args:
        duration (float): Duration of the audio in seconds
        amplitude (float): Amplitude of the noise (0.0 to 1.0)
        sample_rate (int): Sample rate of the audio in Hz
        
    Returns:
        numpy.ndarray: Generated pink noise audio signal
    """
    # Generate white noise
    white_noise = np.random.uniform(-1, 1, int(sample_rate * duration))
    
    # Convert to frequency domain
    X = np.fft.rfft(white_noise)
    
    # Generate pink noise by applying 1/f filter
    f = np.fft.rfftfreq(len(white_noise))
    f[0] = 1  # Avoid division by zero
    X = X / np.sqrt(f)
    
    # Convert back to time domain
    pink_noise = np.fft.irfft(X)
    
    # Normalize and apply amplitude
    pink_noise = pink_noise / np.max(np.abs(pink_noise))
    audio = amplitude * pink_noise
    
    return audio

def generate_brown_noise(duration: float = 1.0, amplitude: float = 0.1, 
                        sample_rate: int = 22050) -> np.ndarray:
    """
    Generate brown noise audio signal.
    
    Args:
        duration (float): Duration of the audio in seconds
        amplitude (float): Amplitude of the noise (0.0 to 1.0)
        sample_rate (int): Sample rate of the audio in Hz
        
    Returns:
        numpy.ndarray: Generated brown noise audio signal
    """
    # Generate white noise
    white_noise = np.random.uniform(-1, 1, int(sample_rate * duration))
    
    # Convert to frequency domain
    X = np.fft.rfft(white_noise)
    
    # Generate brown noise by applying 1/f^2 filter
    f = np.fft.rfftfreq(len(white_noise))
    f[0] = 1  # Avoid division by zero
    X = X / f
    
    # Convert back to time domain
    brown_noise = np.fft.irfft(X)
    
    # Normalize and apply amplitude
    brown_noise = brown_noise / np.max(np.abs(brown_noise))
    audio = amplitude * brown_noise
    
    return audio

def apply_envelope(audio: np.ndarray, attack: float = 0.1, decay: float = 0.1, 
                  sustain: float = 0.7, release: float = 0.2,
                  sample_rate: int = 22050) -> np.ndarray:
    """
    Apply an ADSR (Attack, Decay, Sustain, Release) envelope to an audio signal.
    
    Args:
        audio (numpy.ndarray): Input audio signal
        attack (float): Attack time in seconds
        decay (float): Decay time in seconds
        sustain (float): Sustain level (0.0 to 1.0)
        release (float): Release time in seconds
        sample_rate (int): Sample rate of the audio in Hz
        
    Returns:
        numpy.ndarray: Audio signal with the envelope applied
    """
    # Convert times to samples
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    release_samples = int(release * sample_rate)
    
    # Calculate the total length of the audio
    total_samples = len(audio)
    
    # Calculate the sustain samples
    sustain_samples = total_samples - attack_samples - decay_samples - release_samples
    
    # If the envelope is longer than the audio, adjust the segments
    if sustain_samples < 0:
        # Reduce each segment proportionally
        total_envelope_samples = attack_samples + decay_samples + release_samples
        ratio = total_samples / total_envelope_samples
        
        attack_samples = int(attack_samples * ratio)
        decay_samples = int(decay_samples * ratio)
        release_samples = int(release_samples * ratio)
        sustain_samples = 0
    
    # Create the envelope
    envelope = np.zeros(total_samples)
    
    # Attack segment (linear ramp from 0 to 1)
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    # Decay segment (linear ramp from 1 to sustain level)
    if decay_samples > 0:
        envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain, decay_samples)
    
    # Sustain segment (constant at sustain level)
    if sustain_samples > 0:
        envelope[attack_samples + decay_samples:attack_samples + decay_samples + sustain_samples] = sustain
    
    # Release segment (linear ramp from sustain level to 0)
    if release_samples > 0:
        release_start = attack_samples + decay_samples + sustain_samples
        release_end = min(total_samples, release_start + release_samples)
        actual_release_samples = release_end - release_start
        
        if actual_release_samples > 0:
            envelope[release_start:release_end] = np.linspace(sustain, 0, actual_release_samples)
    
    # Apply the envelope to the audio
    audio_with_envelope = audio * envelope
    
    return audio_with_envelope

def apply_filter(audio: np.ndarray, filter_type: str = 'lowpass', 
                cutoff_frequency: float = 1000.0, order: int = 4,
                sample_rate: int = 22050) -> np.ndarray:
    """
    Apply a filter to an audio signal.
    
    Args:
        audio (numpy.ndarray): Input audio signal
        filter_type (str): Type of filter ('lowpass', 'highpass', 'bandpass')
        cutoff_frequency (float or tuple): Cutoff frequency in Hz. For bandpass, provide a tuple of (low, high)
        order (int): Order of the filter
        sample_rate (int): Sample rate of the audio in Hz
        
    Returns:
        numpy.ndarray: Filtered audio signal
    """
    from scipy import signal
    
    # Normalize the cutoff frequency
    nyquist = 0.5 * sample_rate
    
    # Design the filter
    if filter_type == 'lowpass':
        normalized_cutoff = cutoff_frequency / nyquist
        # Ensure the normalized cutoff is within the valid range
        normalized_cutoff = min(0.99, max(0.01, normalized_cutoff))
        b, a = signal.butter(order, normalized_cutoff, btype='low')
    elif filter_type == 'highpass':
        normalized_cutoff = cutoff_frequency / nyquist
        # Ensure the normalized cutoff is within the valid range
        normalized_cutoff = min(0.99, max(0.01, normalized_cutoff))
        b, a = signal.butter(order, normalized_cutoff, btype='high')
    elif filter_type == 'bandpass':
        # For bandpass, cutoff_frequency should be a tuple (low, high)
        if isinstance(cutoff_frequency, (list, tuple)) and len(cutoff_frequency) == 2:
            low, high = cutoff_frequency
            normalized_low = low / nyquist
            normalized_high = high / nyquist
            # Ensure the normalized cutoffs are within the valid range
            normalized_low = min(0.99, max(0.01, normalized_low))
            normalized_high = min(0.99, max(0.01, normalized_high))
            b, a = signal.butter(order, [normalized_low, normalized_high], btype='band')
        else:
            # Default to a bandpass around the cutoff frequency
            normalized_cutoff = cutoff_frequency / nyquist
            normalized_cutoff = min(0.99, max(0.01, normalized_cutoff))
            bandwidth = 0.2 * normalized_cutoff
            b, a = signal.butter(order, [normalized_cutoff - bandwidth, normalized_cutoff + bandwidth], btype='band')
    else:
        # Default to lowpass
        normalized_cutoff = cutoff_frequency / nyquist
        normalized_cutoff = min(0.99, max(0.01, normalized_cutoff))
        b, a = signal.butter(order, normalized_cutoff, btype='low')
    
    # Apply the filter
    filtered_audio = signal.lfilter(b, a, audio)
    
    return filtered_audio

def apply_reverb(audio: np.ndarray, room_size: float = 0.5, damping: float = 0.5, 
                wet_level: float = 0.3, dry_level: float = 0.7,
                sample_rate: int = 22050) -> np.ndarray:
    """
    Apply a simple reverb effect to an audio signal.
    
    Args:
        audio (numpy.ndarray): Input audio signal
        room_size (float): Size of the room (0.0 to 1.0)
        damping (float): Damping factor (0.0 to 1.0)
        wet_level (float): Level of the reverb signal (0.0 to 1.0)
        dry_level (float): Level of the original signal (0.0 to 1.0)
        sample_rate (int): Sample rate of the audio in Hz
        
    Returns:
        numpy.ndarray: Audio signal with reverb applied
    """
    # Calculate the delay time based on room size
    delay_samples = int(room_size * sample_rate * 0.1)  # 0.1 seconds max delay
    
    # Create the impulse response
    impulse_response = np.zeros(delay_samples)
    
    # Add some early reflections
    num_reflections = 5
    for i in range(num_reflections):
        reflection_time = int((i + 1) * delay_samples / (num_reflections + 1))
        reflection_amplitude = (1 - damping) ** (i + 1)
        impulse_response[reflection_time] = reflection_amplitude
    
    # Add the late reverberation (exponential decay)
    t = np.arange(delay_samples)
    late_reverb = np.exp(-damping * t / delay_samples)
    impulse_response += late_reverb * 0.2  # Scale down the late reverb
    
    # Normalize the impulse response
    impulse_response = impulse_response / np.sum(impulse_response)
    
    # Apply the reverb using convolution
    from scipy import signal
    reverb_signal = signal.convolve(audio, impulse_response, mode='full')[:len(audio)]
    
    # Mix the dry and wet signals
    output = dry_level * audio + wet_level * reverb_signal
    
    # Normalize to avoid clipping
    if np.max(np.abs(output)) > 1.0:
        output = output / np.max(np.abs(output))
    
    return output

def generate_chord(notes: List[float], duration: float = 1.0, 
                  waveform: str = 'sine', amplitude: float = 0.5,
                  sample_rate: int = 22050) -> np.ndarray:
    """
    Generate a chord from a list of frequencies.
    
    Args:
        notes (list): List of frequencies in Hz
        duration (float): Duration of the chord in seconds
        waveform (str): Type of waveform ('sine', 'square', 'sawtooth', 'triangle')
        amplitude (float): Amplitude of the chord (0.0 to 1.0)
        sample_rate (int): Sample rate of the audio in Hz
        
    Returns:
        numpy.ndarray: Generated chord audio signal
    """
    # Initialize the chord signal
    chord = np.zeros(int(sample_rate * duration))
    
    # Generate each note and add to the chord
    for note in notes:
        if waveform == 'sine':
            note_signal = generate_sine_wave(duration, note, amplitude / len(notes), sample_rate)
        elif waveform == 'square':
            note_signal = generate_square_wave(duration, note, amplitude / len(notes), 0.5, sample_rate)
        elif waveform == 'sawtooth':
            note_signal = generate_sawtooth_wave(duration, note, amplitude / len(notes), sample_rate)
        elif waveform == 'triangle':
            note_signal = generate_triangle_wave(duration, note, amplitude / len(notes), sample_rate)
        else:
            # Default to sine wave
            note_signal = generate_sine_wave(duration, note, amplitude / len(notes), sample_rate)
        
        chord += note_signal
    
    # Apply a simple envelope to avoid clicks
    chord = apply_envelope(chord, attack=0.01, decay=0.1, sustain=0.7, release=0.1, sample_rate=sample_rate)
    
    return chord

def generate_melody(notes: List[float], durations: List[float], 
                   waveform: str = 'sine', amplitude: float = 0.5,
                   sample_rate: int = 22050) -> np.ndarray:
    """
    Generate a melody from a list of frequencies and durations.
    
    Args:
        notes (list): List of frequencies in Hz
        durations (list): List of durations in seconds for each note
        waveform (str): Type of waveform ('sine', 'square', 'sawtooth', 'triangle')
        amplitude (float): Amplitude of the melody (0.0 to 1.0)
        sample_rate (int): Sample rate of the audio in Hz
        
    Returns:
        numpy.ndarray: Generated melody audio signal
    """
    # Ensure notes and durations have the same length
    if len(notes) != len(durations):
        raise ValueError("Notes and durations must have the same length")
    
    # Calculate the total duration
    total_duration = sum(durations)
    
    # Initialize the melody signal
    melody = np.array([])
    
    # Generate each note and concatenate
    for note, duration in zip(notes, durations):
        if waveform == 'sine':
            note_signal = generate_sine_wave(duration, note, amplitude, sample_rate)
        elif waveform == 'square':
            note_signal = generate_square_wave(duration, note, amplitude, 0.5, sample_rate)
        elif waveform == 'sawtooth':
            note_signal = generate_sawtooth_wave(duration, note, amplitude, sample_rate)
        elif waveform == 'triangle':
            note_signal = generate_triangle_wave(duration, note, amplitude, sample_rate)
        else:
            # Default to sine wave
            note_signal = generate_sine_wave(duration, note, amplitude, sample_rate)
        
        # Apply a simple envelope to avoid clicks
        note_signal = apply_envelope(note_signal, attack=0.01, decay=0.1, sustain=0.7, release=0.1, sample_rate=sample_rate)
        
        # Concatenate to the melody
        melody = np.concatenate((melody, note_signal))
    
    return melody

def generate_speech_like_audio(duration: float = 5.0, amplitude: float = 0.5,
                             formant_frequencies: List[float] = None,
                             sample_rate: int = 22050) -> np.ndarray:
    """
    Generate a speech-like audio signal using formant synthesis.
    
    Args:
        duration (float): Duration of the audio in seconds
        amplitude (float): Amplitude of the audio (0.0 to 1.0)
        formant_frequencies (list): List of formant frequencies in Hz
        sample_rate (int): Sample rate of the audio in Hz
        
    Returns:
        numpy.ndarray: Generated speech-like audio signal
    """
    # Default formant frequencies if not provided
    if formant_frequencies is None:
        formant_frequencies = [500, 1500, 2500]  # Typical formant frequencies
    
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate the fundamental frequency (pitch) contour
    # Vary the pitch to make it sound more natural
    f0_mean = 120  # Average fundamental frequency (Hz)
    f0_var = 20    # Variance in fundamental frequency
    f0 = f0_mean + f0_var * np.sin(2 * np.pi * 0.5 * t)
    
    # Generate the source signal (glottal pulse)
    source = np.sin(2 * np.pi * f0 * t)
    
    # Add some noise to the source
    source += 0.1 * np.random.normal(0, 1, len(source))
    
    # Initialize the output signal
    output = np.zeros_like(source)
    
    # Apply formant filtering
    for formant in formant_frequencies:
        # Create a bandpass filter around the formant frequency
        from scipy import signal
        nyquist = 0.5 * sample_rate
        bandwidth = 100  # Bandwidth of the formant (Hz)
        low = (formant - bandwidth / 2) / nyquist
        high = (formant + bandwidth / 2) / nyquist
        b, a = signal.butter(2, [low, high], btype='band')
        
        # Apply the filter and add to the output
        formant_signal = signal.lfilter(b, a, source)
        output += formant_signal
    
    # Normalize and apply amplitude
    output = output / np.max(np.abs(output))
    output = amplitude * output
    
    # Apply an envelope to simulate syllables
    syllable_rate = 4  # Syllables per second
    syllable_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * t)**2
    output = output * syllable_envelope
    
    return output

def generate_drum_pattern(pattern: List[int], tempo: int = 120, 
                         sample_rate: int = 22050) -> np.ndarray:
    """
    Generate a drum pattern.
    
    Args:
        pattern (list): List of 1s and 0s representing the drum hits
        tempo (int): Tempo in beats per minute
        sample_rate (int): Sample rate of the audio in Hz
        
    Returns:
        numpy.ndarray: Generated drum pattern audio signal
    """
    # Calculate the duration of each step
    step_duration = 60 / tempo  # Duration in seconds
    
    # Calculate the total duration
    total_duration = step_duration * len(pattern)
    
    # Initialize the drum pattern signal
    drum_pattern = np.zeros(int(sample_rate * total_duration))
    
    # Generate a drum sound (simple sine wave with fast decay)
    drum_sound_duration = 0.1  # 100 ms
    drum_sound = generate_sine_wave(drum_sound_duration, 100, 0.8, sample_rate)
    drum_sound = apply_envelope(drum_sound, attack=0.001, decay=0.1, sustain=0.0, release=0.0, sample_rate=sample_rate)
    
    # Place the drum sound at each hit
    for i, hit in enumerate(pattern):
        if hit == 1:
            start_sample = int(i * step_duration * sample_rate)
            end_sample = start_sample + len(drum_sound)
            if end_sample <= len(drum_pattern):
                drum_pattern[start_sample:end_sample] += drum_sound
    
    return drum_pattern

def mix_audio(audio_signals: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
    """
    Mix multiple audio signals together.
    
    Args:
        audio_signals (list): List of audio signals to mix
        weights (list): List of weights for each audio signal
        
    Returns:
        numpy.ndarray: Mixed audio signal
    """
    # Ensure all audio signals have the same length
    max_length = max(len(signal) for signal in audio_signals)
    padded_signals = []
    
    for signal in audio_signals:
        # Pad the signal to the maximum length
        padded_signal = np.pad(signal, (0, max_length - len(signal)), 'constant')
        padded_signals.append(padded_signal)
    
    # Use equal weights if not provided
    if weights is None:
        weights = [1.0 / len(audio_signals)] * len(audio_signals)
    
    # Ensure weights sum to 1
    weights = np.array(weights) / np.sum(weights)
    
    # Mix the signals
    mixed_signal = np.zeros(max_length)
    
    for signal, weight in zip(padded_signals, weights):
        mixed_signal += weight * signal
    
    # Normalize to avoid clipping
    if np.max(np.abs(mixed_signal)) > 1.0:
        mixed_signal = mixed_signal / np.max(np.abs(mixed_signal))
    
    return mixed_signal

def add_background_noise(audio: np.ndarray, noise_level: float = 0.1,
                        noise_type: str = 'white', sample_rate: int = 22050) -> np.ndarray:
    """
    Add background noise to an audio signal.
    
    Args:
        audio (numpy.ndarray): Input audio signal
        noise_level (float): Level of noise to add (0.0 to 1.0)
        noise_type (str): Type of noise ('white', 'pink', 'brown')
        sample_rate (int): Sample rate of the audio in Hz
        
    Returns:
        numpy.ndarray: Audio signal with background noise
    """
    # Generate noise
    if noise_type == 'white':
        noise = generate_white_noise(len(audio) / sample_rate, noise_level, sample_rate)
    elif noise_type == 'pink':
        noise = generate_pink_noise(len(audio) / sample_rate, noise_level, sample_rate)
    elif noise_type == 'brown':
        noise = generate_brown_noise(len(audio) / sample_rate, noise_level, sample_rate)
    else:
        # Default to white noise
        noise = generate_white_noise(len(audio) / sample_rate, noise_level, sample_rate)
    
    # Add noise to the audio
    noisy_audio = audio + noise
    
    # Normalize to avoid clipping
    if np.max(np.abs(noisy_audio)) > 1.0:
        noisy_audio = noisy_audio / np.max(np.abs(noisy_audio))
    
    return noisy_audio

def generate_audio(duration: float = 1.0, audio_type: str = 'sine',
                  frequency: float = 440.0, amplitude: float = 0.5,
                  sample_rate: int = 22050, **kwargs) -> np.ndarray:
    """
    Generate an audio signal of the specified type.
    
    Args:
        duration (float): Duration of the audio in seconds
        audio_type (str): Type of audio to generate
            ('sine', 'square', 'sawtooth', 'triangle', 'white_noise', 'pink_noise', 'brown_noise', 'speech', 'chord', 'melody')
        frequency (float): Frequency of the audio in Hz (for waveforms)
        amplitude (float): Amplitude of the audio (0.0 to 1.0)
        sample_rate (int): Sample rate of the audio in Hz
        **kwargs: Additional arguments for specific audio types
        
    Returns:
        numpy.ndarray: Generated audio signal
    """
    if audio_type == 'sine':
        audio = generate_sine_wave(duration, frequency, amplitude, sample_rate)
    
    elif audio_type == 'square':
        duty_cycle = kwargs.get('duty_cycle', 0.5)
        audio = generate_square_wave(duration, frequency, amplitude, duty_cycle, sample_rate)
    
    elif audio_type == 'sawtooth':
        audio = generate_sawtooth_wave(duration, frequency, amplitude, sample_rate)
    
    elif audio_type == 'triangle':
        audio = generate_triangle_wave(duration, frequency, amplitude, sample_rate)
    
    elif audio_type == 'white_noise':
        audio = generate_white_noise(duration, amplitude, sample_rate)
    
    elif audio_type == 'pink_noise':
        audio = generate_pink_noise(duration, amplitude, sample_rate)
    
    elif audio_type == 'brown_noise':
        audio = generate_brown_noise(duration, amplitude, sample_rate)
    
    elif audio_type == 'speech':
        formant_frequencies = kwargs.get('formant_frequencies', None)
        audio = generate_speech_like_audio(duration, amplitude, formant_frequencies, sample_rate)
    
    elif audio_type == 'chord':
        notes = kwargs.get('notes', [frequency, frequency * 1.25, frequency * 1.5])  # Default to a major chord
        waveform = kwargs.get('waveform', 'sine')
        audio = generate_chord(notes, duration, waveform, amplitude, sample_rate)
    
    elif audio_type == 'melody':
        notes = kwargs.get('notes', [frequency, frequency * 1.125, frequency * 1.25, frequency * 1.5])
        durations = kwargs.get('durations', [duration / 4] * 4)
        waveform = kwargs.get('waveform', 'sine')
        audio = generate_melody(notes, durations, waveform, amplitude, sample_rate)
    
    elif audio_type == 'drum':
        pattern = kwargs.get('pattern', [1, 0, 1, 0, 1, 0, 1, 0])
        tempo = kwargs.get('tempo', 120)
        audio = generate_drum_pattern(pattern, tempo, sample_rate)
    
    else:
        # Default to sine wave
        audio = generate_sine_wave(duration, frequency, amplitude, sample_rate)
    
    # Apply effects if specified
    if kwargs.get('apply_envelope', False):
        attack = kwargs.get('attack', 0.1)
        decay = kwargs.get('decay', 0.1)
        sustain = kwargs.get('sustain', 0.7)
        release = kwargs.get('release', 0.2)
        audio = apply_envelope(audio, attack, decay, sustain, release, sample_rate)
    
    if kwargs.get('apply_filter', False):
        filter_type = kwargs.get('filter_type', 'lowpass')
        cutoff_frequency = kwargs.get('cutoff_frequency', 1000.0)
        order = kwargs.get('filter_order', 4)
        audio = apply_filter(audio, filter_type, cutoff_frequency, order, sample_rate)
    
    if kwargs.get('apply_reverb', False):
        room_size = kwargs.get('room_size', 0.5)
        damping = kwargs.get('damping', 0.5)
        wet_level = kwargs.get('wet_level', 0.3)
        dry_level = kwargs.get('dry_level', 0.7)
        audio = apply_reverb(audio, room_size, damping, wet_level, dry_level, sample_rate)
    
    if kwargs.get('add_noise', False):
        noise_level = kwargs.get('noise_level', 0.1)
        noise_type = kwargs.get('noise_type', 'white')
        audio = add_background_noise(audio, noise_level, noise_type, sample_rate)
    
    return audio

def save_audio(audio: np.ndarray, file_path: str, sample_rate: int = 22050, format: str = 'wav') -> None:
    """
    Save an audio signal to a file.
    
    Args:
        audio (numpy.ndarray): Audio signal to save
        file_path (str): Path to save the audio file
        sample_rate (int): Sample rate of the audio in Hz
        format (str): Format of the audio file ('wav', 'flac', 'ogg')
        
    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Save the audio file
    sf.write(file_path, audio, sample_rate, format=format)

def generate_audio_dataset(num_samples: int = 10, duration: float = 1.0,
                          audio_types: List[str] = None, sample_rate: int = 22050,
                          output_dir: str = None, **kwargs) -> Dict[str, np.ndarray]:
    """
    Generate a dataset of synthetic audio samples.
    
    Args:
        num_samples (int): Number of audio samples to generate
        duration (float): Duration of each audio sample in seconds
        audio_types (list): List of audio types to generate
        sample_rate (int): Sample rate of the audio in Hz
        output_dir (str): Directory to save the audio files
        **kwargs: Additional arguments for audio generation
        
    Returns:
        dict: Dictionary of generated audio samples
    """
    # Set random seed if specified in config
    random_seed = config.get('random_seed')
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Default audio types if not provided
    if audio_types is None:
        audio_types = ['sine', 'square', 'sawtooth', 'triangle', 'white_noise', 'pink_noise']
    
    # Generate audio samples
    dataset = {}
    
    for i in range(num_samples):
        # Randomly select an audio type
        audio_type = random.choice(audio_types)
        
        # Generate random parameters
        frequency = random.uniform(100, 1000)
        amplitude = random.uniform(0.3, 0.8)
        
        # Additional parameters for specific audio types
        additional_params = {}
        
        if audio_type == 'chord':
            # Generate a random chord
            base_note = random.uniform(200, 500)
            if random.random() < 0.5:
                # Major chord
                additional_params['notes'] = [base_note, base_note * 1.25, base_note * 1.5]
            else:
                # Minor chord
                additional_params['notes'] = [base_note, base_note * 1.2, base_note * 1.5]
        
        elif audio_type == 'melody':
            # Generate a random melody
            base_note = random.uniform(200, 500)
            num_notes = random.randint(4, 8)
            
            # Generate random notes and durations
            notes = [base_note * random.choice([1, 1.125, 1.25, 1.333, 1.5, 1.667, 1.875, 2]) for _ in range(num_notes)]
            durations = [duration / num_notes] * num_notes
            
            additional_params['notes'] = notes
            additional_params['durations'] = durations
        
        # Randomly apply effects
        if random.random() < 0.3:
            additional_params['apply_envelope'] = True
            additional_params['attack'] = random.uniform(0.01, 0.2)
            additional_params['release'] = random.uniform(0.1, 0.3)
        
        if random.random() < 0.3:
            additional_params['apply_filter'] = True
            additional_params['filter_type'] = random.choice(['lowpass', 'highpass', 'bandpass'])
            additional_params['cutoff_frequency'] = random.uniform(500, 2000)
        
        if random.random() < 0.3:
            additional_params['apply_reverb'] = True
            additional_params['room_size'] = random.uniform(0.3, 0.7)
        
        if random.random() < 0.2:
            additional_params['add_noise'] = True
            additional_params['noise_level'] = random.uniform(0.05, 0.2)
            additional_params['noise_type'] = random.choice(['white', 'pink', 'brown'])
        
        # Generate the audio
        audio = generate_audio(
            duration=duration,
            audio_type=audio_type,
            frequency=frequency,
            amplitude=amplitude,
            sample_rate=sample_rate,
            **additional_params,
            **kwargs
        )
        
        # Add to the dataset
        dataset[f'sample_{i+1}'] = audio
        
        # Save the audio file if output directory is specified
        if output_dir:
            file_path = os.path.join(output_dir, f'sample_{i+1}.wav')
            save_audio(audio, file_path, sample_rate)
    
    return dataset
