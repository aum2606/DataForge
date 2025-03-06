"""
Example script for generating all types of synthetic data.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import our modules directly
from data_types import tabular_data, image_data, text_data, time_series_data, audio_data
from utils import exporters
from config import config

def main():
    # Create main output directory
    base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
    os.makedirs(base_output_dir, exist_ok=True)
    
    print("=" * 80)
    print("SYNTHETIC DATA GENERATOR - COMPREHENSIVE EXAMPLE")
    print("=" * 80)
    print("\nThis example will generate samples of all data types supported by the library.\n")
    
    # Get configuration settings
    print(f"Using random seed: {config.get('random_seed')}")
    
    # Set the number of samples to generate for each type
    num_samples = 5
    
    # Part 1: Generate Tabular Data
    print("\n" + "=" * 40)
    print("PART 1: GENERATING TABULAR DATA")
    print("=" * 40)
    
    tabular_output_dir = os.path.join(base_output_dir, 'tabular')
    os.makedirs(tabular_output_dir, exist_ok=True)
    
    print(f"\nGenerating {num_samples} tabular data samples...")
    
    # Define a schema for customer data
    customer_schema = {
        'customer_id': {'type': 'id', 'start': 1000},
        'name': {'type': 'name'},
        'age': {'type': 'int', 'min': 18, 'max': 80},
        'income': {'type': 'float', 'min': 20000, 'max': 150000, 'distribution': 'normal', 'mean': 60000, 'std': 20000},
        'credit_score': {'type': 'int', 'min': 300, 'max': 850, 'distribution': 'normal', 'mean': 700, 'std': 100},
        'is_active': {'type': 'boolean', 'true_probability': 0.8},
        'customer_segment': {'type': 'category', 'categories': ['Premium', 'Standard', 'Basic'], 'weights': [0.2, 0.5, 0.3]},
        'signup_date': {'type': 'date', 'start': '2020-01-01', 'end': '2023-12-31'},
        'last_purchase': {'type': 'date', 'start': '2022-01-01', 'end': '2023-12-31'},
        'email': {'type': 'email'}
    }
    
    # Define correlations
    correlations = {
        'age': {'income': 0.6, 'credit_score': 0.4},
        'income': {'credit_score': 0.7}
    }
    
    # Generate the customer dataset
    customer_df = tabular_data.generate_dataset(
        rows=100,
        schema=customer_schema,
        correlations=correlations
    )
    
    print("\nCustomer Dataset Preview:")
    print(customer_df.head())
    
    # Save the dataset in multiple formats
    csv_path = os.path.join(tabular_output_dir, 'customers.csv')
    exporters.to_csv(customer_df, csv_path)
    
    excel_path = os.path.join(tabular_output_dir, 'customers.xlsx')
    exporters.to_excel(customer_df, excel_path)
    
    json_path = os.path.join(tabular_output_dir, 'customers.json')
    exporters.to_json(customer_df, json_path)
    
    print(f"\nTabular data saved to: {tabular_output_dir}")
    
    # Part 2: Generate Image Data
    print("\n" + "=" * 40)
    print("PART 2: GENERATING IMAGE DATA")
    print("=" * 40)
    
    image_output_dir = os.path.join(base_output_dir, 'images')
    os.makedirs(image_output_dir, exist_ok=True)
    
    print(f"\nGenerating {num_samples} image samples...")
    
    # Generate different types of images
    image_types = ['noise', 'gradient', 'pattern', 'geometric']
    
    for i, image_type in enumerate(image_types):
        if image_type == 'noise':
            img = image_data.generate_noise_image(
                width=256, 
                height=256, 
                noise_type='gaussian'
            )
        elif image_type == 'gradient':
            img = image_data.generate_gradient_image(
                width=256, 
                height=256, 
                gradient_type='radial',
                start_color=(255, 0, 0),
                end_color=(0, 0, 255)
            )
        elif image_type == 'pattern':
            img = image_data.generate_pattern_image(
                width=256, 
                height=256, 
                pattern_type='checkerboard',
                primary_color=(0, 0, 0),
                secondary_color=(255, 255, 255)
            )
        elif image_type == 'geometric':
            img = image_data.generate_geometric_image(
                width=256, 
                height=256, 
                shape_type='mixed',
                num_shapes=10
            )
        
        # Save the image
        img_path = os.path.join(image_output_dir, f'sample_{image_type}.png')
        exporters.save_image(img, img_path)
        print(f"Generated {image_type} image: {img_path}")
    
    # Generate an image dataset
    dataset_dir = os.path.join(image_output_dir, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    
    image_dataset = image_data.generate_image_dataset(
        num_images=num_samples,
        width=128,
        height=128,
        image_types=image_types,
        output_dir=dataset_dir
    )
    
    print(f"\nImage dataset saved to: {dataset_dir}")
    
    # Part 3: Generate Text Data
    print("\n" + "=" * 40)
    print("PART 3: GENERATING TEXT DATA")
    print("=" * 40)
    
    text_output_dir = os.path.join(base_output_dir, 'text')
    os.makedirs(text_output_dir, exist_ok=True)
    
    print(f"\nGenerating {num_samples} text samples...")
    
    # Generate an article
    article = text_data.generate_article(
        title="The Future of Synthetic Data",
        num_paragraphs=3,
        sentences_per_paragraph=(3, 5)
    )
    
    article_path = os.path.join(text_output_dir, 'article.txt')
    with open(article_path, 'w') as f:
        f.write(article)
    print(f"Generated article: {article_path}")
    
    # Generate a conversation
    conversation = text_data.generate_conversation(
        participants=["Customer", "Support Agent"],
        num_exchanges=5
    )
    
    conversation_path = os.path.join(text_output_dir, 'conversation.txt')
    with open(conversation_path, 'w') as f:
        for speaker, message in conversation:
            f.write(f"{speaker}: {message}\n")
    print(f"Generated conversation: {conversation_path}")
    
    # Generate structured text
    structured_texts = text_data.generate_structured_text(
        template="Product Review: {product_name}\n\nRating: {rating}/5\n\nReview: {review}\n\nBy: {reviewer_name} on {review_date}",
        num_samples=num_samples,
        fields={
            'product_name': {'type': 'product'},
            'rating': {'type': 'int', 'min': 1, 'max': 5},
            'review': {'type': 'paragraph', 'min_sentences': 2, 'max_sentences': 4},
            'reviewer_name': {'type': 'name'},
            'review_date': {'type': 'date', 'format': '%B %d, %Y', 'start': '2023-01-01', 'end': '2023-12-31'}
        }
    )
    
    for i, text in enumerate(structured_texts):
        text_path = os.path.join(text_output_dir, f'review_{i+1}.txt')
        with open(text_path, 'w') as f:
            f.write(text)
        print(f"Generated structured text: {text_path}")
    
    # Generate a text dataset
    text_dataset = text_data.generate_text_dataset(
        num_samples=num_samples,
        text_type='paragraph',
        min_sentences=3,
        max_sentences=6,
        output_format='json',
        output_file=os.path.join(text_output_dir, 'text_dataset.json')
    )
    
    print(f"\nText dataset saved to: {os.path.join(text_output_dir, 'text_dataset.json')}")
    
    # Part 4: Generate Time Series Data
    print("\n" + "=" * 40)
    print("PART 4: GENERATING TIME SERIES DATA")
    print("=" * 40)
    
    ts_output_dir = os.path.join(base_output_dir, 'time_series')
    os.makedirs(ts_output_dir, exist_ok=True)
    
    print(f"\nGenerating {num_samples} time series samples...")
    
    # Generate a univariate time series
    time_index, ts = time_series_data.generate_time_series(
        length=365,
        components=['trend', 'seasonality', 'noise'],
        start_date='2023-01-01',
        freq='D',
        trend_type='linear',
        seasonality_period=30,  # Monthly seasonality
        noise_level=0.3
    )
    
    # Plot the time series
    plt.figure(figsize=(10, 6))
    plt.plot(time_index, ts)
    plt.title('Synthetic Time Series')
    plt.grid(True)
    plt.tight_layout()
    
    ts_plot_path = os.path.join(ts_output_dir, 'time_series.png')
    plt.savefig(ts_plot_path)
    plt.close()
    
    # Save the time series data
    ts_df = pd.DataFrame({'date': time_index, 'value': ts})
    ts_path = os.path.join(ts_output_dir, 'time_series.csv')
    ts_df.to_csv(ts_path, index=False)
    print(f"Generated time series: {ts_path}")
    
    # Generate a multivariate time series
    correlation_matrix = np.array([
        [1.0, 0.7, -0.3],
        [0.7, 1.0, 0.2],
        [-0.3, 0.2, 1.0]
    ])
    
    time_index, multivariate_ts = time_series_data.generate_multivariate_time_series(
        length=365,
        num_series=3,
        correlation_matrix=correlation_matrix,
        start_date='2023-01-01',
        freq='D'
    )
    
    # Save the multivariate time series
    multi_ts_df = pd.DataFrame(
        multivariate_ts,
        index=time_index,
        columns=[f'series_{i+1}' for i in range(multivariate_ts.shape[1])]
    )
    
    multi_ts_path = os.path.join(ts_output_dir, 'multivariate_time_series.csv')
    multi_ts_df.to_csv(multi_ts_path)
    print(f"Generated multivariate time series: {multi_ts_path}")
    
    # Generate a time series dataset
    ts_dataset = time_series_data.generate_time_series_dataset(
        num_series=num_samples,
        length=100,
        output_format='dataframe',
        output_file=os.path.join(ts_output_dir, 'time_series_dataset.csv')
    )
    
    print(f"\nTime series dataset saved to: {os.path.join(ts_output_dir, 'time_series_dataset.csv')}")
    
    # Part 5: Generate Audio Data
    print("\n" + "=" * 40)
    print("PART 5: GENERATING AUDIO DATA")
    print("=" * 40)
    
    audio_output_dir = os.path.join(base_output_dir, 'audio')
    os.makedirs(audio_output_dir, exist_ok=True)
    
    # Set the sample rate
    sample_rate = 22050
    
    print(f"\nGenerating {num_samples} audio samples...")
    
    # Generate a melody
    c_major_scale = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C4 to C5
    durations = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5]  # Quarter notes with final half note
    
    melody = audio_data.generate_melody(
        notes=c_major_scale,
        durations=durations,
        waveform='sine',
        amplitude=0.7,
        sample_rate=sample_rate
    )
    
    melody_path = os.path.join(audio_output_dir, 'melody.wav')
    audio_data.save_audio(melody, melody_path, sample_rate)
    print(f"Generated melody: {melody_path}")
    
    # Generate a chord progression (C-F-G-C)
    chord_progression = [
        [261.63, 329.63, 392.00],  # C major
        [349.23, 440.00, 523.25],  # F major
        [392.00, 493.88, 587.33],  # G major
        [261.63, 329.63, 392.00]   # C major
    ]
    
    chord_durations = [1.0, 1.0, 1.0, 1.0]
    
    progression = np.array([])
    for chord, duration in zip(chord_progression, chord_durations):
        chord_audio = audio_data.generate_chord(
            notes=chord,
            duration=duration,
            waveform='sine',
            amplitude=0.5,
            sample_rate=sample_rate
        )
        progression = np.append(progression, chord_audio)
    
    progression_path = os.path.join(audio_output_dir, 'chord_progression.wav')
    audio_data.save_audio(progression, progression_path, sample_rate)
    print(f"Generated chord progression: {progression_path}")
    
    # Generate speech-like audio
    speech = audio_data.generate_speech_like_audio(
        duration=3.0,
        amplitude=0.5,
        formant_frequencies=[500, 1500, 2500],
        sample_rate=sample_rate
    )
    
    speech_path = os.path.join(audio_output_dir, 'speech.wav')
    audio_data.save_audio(speech, speech_path, sample_rate)
    print(f"Generated speech-like audio: {speech_path}")
    
    # Generate a drum pattern
    drum_pattern = [1, 0, 0, 1, 0, 1, 0, 0] * 4  # Basic rhythm repeated 4 times
    
    drums = audio_data.generate_drum_pattern(
        pattern=drum_pattern,
        tempo=120,
        sample_rate=sample_rate
    )
    
    drums_path = os.path.join(audio_output_dir, 'drums.wav')
    audio_data.save_audio(drums, drums_path, sample_rate)
    print(f"Generated drum pattern: {drums_path}")
    
    # Generate an audio dataset
    audio_dataset = audio_data.generate_audio_dataset(
        num_samples=num_samples,
        duration=1.0,
        audio_types=['sine', 'square', 'sawtooth', 'triangle', 'white_noise'],
        sample_rate=sample_rate,
        output_dir=os.path.join(audio_output_dir, 'dataset')
    )
    
    print(f"\nAudio dataset saved to: {os.path.join(audio_output_dir, 'dataset')}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 80)
    
    print(f"\nAll synthetic data has been generated and saved to: {base_output_dir}")
    print("\nData types generated:")
    print(f"  - Tabular data: {tabular_output_dir}")
    print(f"  - Image data: {image_output_dir}")
    print(f"  - Text data: {text_output_dir}")
    print(f"  - Time series data: {ts_output_dir}")
    print(f"  - Audio data: {audio_output_dir}")
    
    print("\nYou can now use this data for testing, development, or demonstration purposes.")

if __name__ == "__main__":
    main()
