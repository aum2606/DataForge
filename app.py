from flask import Flask, render_template, request, jsonify, send_file, url_for, send_from_directory
from werkzeug.utils import safe_join
import os
import json
import time
from PIL import Image, ImageDraw, ImageFont
import io
import pandas as pd
import numpy as np
import datetime
import zipfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server use
import matplotlib.pyplot as plt

# Import our data type modules
from data_types import tabular_data, image_data, text_data, time_series_data, audio_data

app = Flask(__name__)

# Create data directories if they don't exist
os.makedirs('static/generated_data', exist_ok=True)
os.makedirs('static/images', exist_ok=True)

# Helper function to generate placeholder images
def create_placeholder_image(text, size=(512, 256), bg_color=(70, 80, 100), text_color=(255, 255, 255)):
    image = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(image)
    
    # Try to use a system font
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        # Fallback to default
        font = ImageFont.load_default()
    
    # Center the text
    text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (200, 40)
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    
    # Draw the text
    draw.text(position, text, font=font, fill=text_color)
    
    return image

# Route to serve placeholder images when actual images don't exist
@app.route('/static/images/<path:filename>')
def serve_placeholder_image(filename):
    file_path = os.path.join('static/images', filename)
    
    # Check if the actual file exists
    if os.path.exists(file_path):
        return send_from_directory('static/images', filename)
    
    # Generate placeholder based on filename
    image_type = filename.split('.')[0]  # Extract name without extension
    image = create_placeholder_image(image_type.upper())
    
    # Save the image for future use
    image.save(file_path)
    
    # Serve the newly created image
    return send_from_directory('static/images', filename)

# Function to clean up old files
def cleanup_old_files():
    """Delete files older than 1 hour"""
    directory = os.path.join(app.root_path, 'static/generated_data')
    if not os.path.exists(directory):
        return
    
    # Get current time
    now = time.time()
    
    # Check all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Skip if it's not a file
        if not os.path.isfile(file_path):
            continue
            
        # Remove file if older than 1 hour
        if os.stat(file_path).st_mtime < now - 3600:
            try:
                os.remove(file_path)
            except:
                pass

@app.route('/')
def index():
    # Clean up old files on each request to home page
    cleanup_old_files()
    return render_template('index.html')

@app.route('/forge')
def forge():
    return render_template('forge.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/api/generate', methods=['POST'])
def generate_data():
    """API endpoint to generate data based on the provided configuration"""
    try:
        data = request.json
        data_type = data.get('data_type', 'tabular')
        
        # Generate a unique timestamp for filenames
        timestamp = int(time.time())
        output_dir = 'static/generated_data'
        
        if data_type == 'tabular':
            # Extract parameters for tabular data
            num_rows = int(data.get('num_rows', 100))
            output_format = data.get('format', 'csv').lower()
            schema_type = data.get('schema_type', 'customer')
            custom_schema = data.get('schema', None)
            
            # Create a schema dictionary
            if custom_schema:
                schema = custom_schema
            else:
                # You'd need to implement this in the tabular_data module
                # For simplicity, let's create a basic schema by data type
                if schema_type == 'customer':
                    schema = {
                        'customer_id': {'type': 'id'},
                        'name': {'type': 'name'},
                        'email': {'type': 'email'},
                        'address': {'type': 'address'},
                        'phone': {'type': 'phone'},
                        'age': {'type': 'int', 'min': 18, 'max': 90},
                        'joined_date': {'type': 'date'}
                    }
                elif schema_type == 'transaction':
                    schema = {
                        'transaction_id': {'type': 'id'},
                        'customer_id': {'type': 'int', 'min': 1000, 'max': 9999},
                        'amount': {'type': 'float', 'min': 10, 'max': 1000},
                        'date': {'type': 'date'},
                        'status': {'type': 'category', 'categories': ['completed', 'pending', 'failed']}
                    }
                else:
                    # Generic schema with various data types
                    schema = {
                        'id': {'type': 'id'},
                        'name': {'type': 'name'},
                        'value': {'type': 'float', 'min': 0, 'max': 100}
                    }
            
            # Generate the dataset
            df = tabular_data.generate_dataset(rows=num_rows, schema=schema)
            
            # Create filename for the output
            filename = f"tabular_{schema_type}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.{output_format}"
            file_path = os.path.join(output_dir, filename)
            
            # Save based on format
            if output_format == 'csv':
                df.to_csv(file_path, index=False)
            elif output_format == 'json':
                df.to_json(file_path, orient='records')
            elif output_format == 'excel':
                df.to_excel(file_path, index=False)
            elif output_format == 'pkl':
                df.to_pickle(file_path)
            else:
                # Default to CSV
                output_format = 'csv'
                df.to_csv(file_path, index=False)
            
            # Create response
            download_url = url_for('download_file', filename=os.path.basename(file_path))
            
            return jsonify({
                'success': True,
                'message': f'Generated {num_rows} rows of {len(df.columns)} columns in {output_format} format',
                'download_url': download_url,
                'rows': num_rows,
                'columns': len(df.columns),
                'format': output_format,
                'filename': os.path.basename(file_path)
            })
            
        elif data_type == 'image':
            # Extract parameters for image data
            image_type = data.get('image_type', 'noise')
            width = int(data.get('width', 512))
            height = int(data.get('height', 512))
            num_images = int(data.get('num_images', 1))
            output_format = data.get('format', 'png').lower()
            color_mode = data.get('color_mode', 'RGB')
            
            # Set number of channels based on color mode
            channels = 3 if color_mode == 'RGB' else 1
            
            # Generate images based on the requested type
            images = []
            for i in range(num_images):
                if image_type == 'noise':
                    img_array = image_data.generate_noise_image(width, height, channels)
                elif image_type == 'gradient':
                    img_array = image_data.generate_gradient_image(width, height, channels)
                elif image_type == 'pattern':
                    img_array = image_data.generate_pattern_image(width, height, channels)
                elif image_type == 'geometric':
                    img_array = image_data.generate_geometric_image(width, height, channels)
                else:
                    # Default to noise
                    img_array = image_data.generate_noise_image(width, height, channels)
                
                # Convert to PIL Image
                if channels == 1:
                    img = Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
                else:
                    img = Image.fromarray((img_array * 255).astype(np.uint8), mode='RGB')
                
                images.append(img)
            
            # Create filename for the output
            base_filename = f"image_{timestamp}"
            
            if num_images == 1:
                # Single image case
                filename = f"{base_filename}.{output_format}"
                file_path = os.path.join(output_dir, filename)
                
                # Save the image
                images[0].save(file_path)
                
                view_url = url_for('static', filename=f'generated_data/{filename}')
            else:
                # Multiple images - create a ZIP file
                filename = f"{base_filename}.zip"
                file_path = os.path.join(output_dir, filename)
                
                with zipfile.ZipFile(file_path, 'w') as zipf:
                    for i, img in enumerate(images):
                        img_filename = f"image_{i+1}.{output_format}"
                        
                        # Save to a bytes buffer
                        buffer = io.BytesIO()
                        img.save(buffer, format=output_format.upper())
                        buffer.seek(0)
                        
                        # Add to ZIP
                        zipf.writestr(img_filename, buffer.read())
                
                view_url = None
            
            # Create response
            download_url = url_for('download_file', filename=os.path.basename(file_path))
            
            return jsonify({
                'success': True,
                'message': f'Generated {num_images} image(s) of size {width}x{height} in {output_format} format',
                'download_url': download_url,
                'view_url': view_url,
                'width': width,
                'height': height,
                'num_images': num_images,
                'format': output_format,
                'filename': os.path.basename(file_path)
            })
            
        elif data_type == 'text':
            # Extract parameters for text data
            text_type = data.get('text_type', 'paragraph')
            length = data.get('length', 'medium')
            language = data.get('language', 'en')
            include_entities = data.get('include_entities', False)
            
            # Map length to word counts
            length_map = {
                'short': 100,
                'medium': 500,
                'long': 2000
            }
            word_count = length_map.get(length, 500)
            
            # Generate text based on type
            if text_type == 'paragraph':
                num_paragraphs = max(1, word_count // 100)
                paragraphs = []
                for _ in range(num_paragraphs):
                    paragraphs.append(text_data.generate_paragraph(
                        min_sentences=3,
                        max_sentences=10,
                        language=language
                    ))
                generated_text = '\n\n'.join(paragraphs)
            elif text_type == 'article':
                generated_text = text_data.generate_article(
                    min_paragraphs=max(3, word_count // 150),
                    include_title=True,
                    language=language
                )
            elif text_type == 'conversation':
                generated_text = text_data.generate_conversation(
                    num_exchanges=max(3, word_count // 50),
                    language=language
                )
            elif text_type == 'structured':
                # Create a simple structure for sections
                structure = [
                    {'type': 'heading', 'level': 1, 'content': 'Main Title'},
                    {'type': 'paragraph', 'content': None},
                    {'type': 'heading', 'level': 2, 'content': 'Section 1'},
                    {'type': 'paragraph', 'content': None},
                    {'type': 'paragraph', 'content': None},
                    {'type': 'heading', 'level': 2, 'content': 'Section 2'},
                    {'type': 'paragraph', 'content': None},
                    {'type': 'paragraph', 'content': None}
                ]
                generated_text = text_data.generate_structured_text(
                    structure=structure,
                    language=language
                )
            else:
                # Default to paragraph
                num_paragraphs = max(1, word_count // 100)
                paragraphs = []
                for _ in range(num_paragraphs):
                    paragraphs.append(text_data.generate_paragraph(
                        min_sentences=3,
                        max_sentences=10,
                        language=language
                    ))
                generated_text = '\n\n'.join(paragraphs)
            
            # Count words
            actual_word_count = len(generated_text.split())
            
            # Create filename and save the text
            filename = f"text_{timestamp}.txt"
            file_path = os.path.join(output_dir, filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(generated_text)
            
            # Create a preview (first 200 words)
            preview_words = generated_text.split()[:200]
            preview = ' '.join(preview_words)
            if len(preview_words) < len(generated_text.split()):
                preview += '...'
            
            # Create response
            download_url = url_for('download_file', filename=os.path.basename(file_path))
            
            return jsonify({
                'success': True,
                'message': f'Generated {text_type} text in {language} with {actual_word_count} words',
                'download_url': download_url,
                'word_count': actual_word_count,
                'language': language,
                'format': 'txt',
                'filename': os.path.basename(file_path),
                'preview': preview
            })
            
        elif data_type == 'timeseries':
            # Extract parameters for timeseries data
            ts_length = int(data.get('ts_length', 365))
            frequency = data.get('frequency', 'D')
            num_series = int(data.get('num_series', 1))
            components = data.get('components', ['trend', 'seasonality', 'noise'])
            output_format = data.get('format', 'csv').lower()
            
            # Generate time series data
            time_index, ts_values = time_series_data.generate_time_series(
                length=ts_length,
                freq=frequency,
                components=components
            )
            
            # Convert to DataFrame
            series_names = [f'series_{i+1}' for i in range(num_series)]
            if num_series == 1:
                ts_data = pd.DataFrame(ts_values, index=time_index, columns=[series_names[0]])
            else:
                # Generate multiple series with some correlation
                time_index, multi_values = time_series_data.generate_multivariate_time_series(
                    length=ts_length,
                    num_series=num_series,
                    freq=frequency
                )
                ts_data = pd.DataFrame(multi_values, index=time_index, columns=series_names)
            
            # Create filename for the data
            filename = f"timeseries_{timestamp}.{output_format}"
            file_path = os.path.join(output_dir, filename)
            
            # Save based on format
            if output_format == 'csv':
                ts_data.to_csv(file_path, index=True)
            elif output_format == 'json':
                ts_data.to_json(file_path, orient='columns')
            elif output_format == 'excel':
                ts_data.to_excel(file_path, index=True)
            elif output_format == 'pkl':
                ts_data.to_pickle(file_path)
            else:
                # Default to CSV
                output_format = 'csv'
                ts_data.to_csv(file_path, index=True)
            
            # Generate a chart preview
            chart_filename = f"timeseries_{timestamp}_chart.png"
            chart_path = os.path.join(output_dir, chart_filename)
            
            plt.figure(figsize=(10, 6))
            for col in ts_data.columns:
                plt.plot(ts_data.index, ts_data[col], label=col)
            
            plt.title(f"Time Series - {frequency} Frequency")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()
            plt.savefig(chart_path)
            plt.close()
            
            # Create response
            download_url = url_for('download_file', filename=os.path.basename(file_path))
            chart_url = url_for('static', filename=f'generated_data/{chart_filename}')
            
            return jsonify({
                'success': True,
                'message': f'Generated time series with {ts_length} points, {num_series} series, and frequency {frequency}',
                'download_url': download_url,
                'chart_url': chart_url,
                'length': ts_length,
                'frequency': frequency,
                'num_series': num_series,
                'format': output_format,
                'filename': os.path.basename(file_path)
            })
            
        elif data_type == 'audio':
            # Extract parameters for audio data
            audio_type = data.get('audio_type', 'waveform')
            duration = float(data.get('duration', 5.0))
            sample_rate = int(data.get('sample_rate', 44100))
            file_format = data.get('file_format', 'wav').lower()
            
            # Generate audio data based on type
            if audio_type == 'waveform':
                # Get the waveform type from the request, default to sine
                waveform_type = data.get('waveform_type', 'sine')
                if waveform_type == 'sine':
                    samples = audio_data.generate_sine_wave(
                        duration=duration,
                        sample_rate=sample_rate
                    )
                elif waveform_type == 'square':
                    samples = audio_data.generate_square_wave(
                        duration=duration,
                        sample_rate=sample_rate
                    )
                elif waveform_type == 'sawtooth':
                    samples = audio_data.generate_sawtooth_wave(
                        duration=duration,
                        sample_rate=sample_rate
                    )
                elif waveform_type == 'triangle':
                    samples = audio_data.generate_triangle_wave(
                        duration=duration,
                        sample_rate=sample_rate
                    )
                else:
                    # Default to sine wave
                    samples = audio_data.generate_sine_wave(
                        duration=duration,
                        sample_rate=sample_rate
                    )
            elif audio_type == 'noise':
                # Get the noise type from the request, default to white
                noise_type = data.get('noise_type', 'white')
                if noise_type == 'white':
                    samples = audio_data.generate_white_noise(
                        duration=duration,
                        sample_rate=sample_rate
                    )
                elif noise_type == 'pink':
                    samples = audio_data.generate_pink_noise(
                        duration=duration,
                        sample_rate=sample_rate
                    )
                elif noise_type == 'brown':
                    samples = audio_data.generate_brown_noise(
                        duration=duration,
                        sample_rate=sample_rate
                    )
                else:
                    # Default to white noise
                    samples = audio_data.generate_white_noise(
                        duration=duration,
                        sample_rate=sample_rate
                    )
            elif audio_type == 'speech':
                samples = audio_data.generate_speech_like_audio(
                    duration=duration,
                    sample_rate=sample_rate
                )
            elif audio_type == 'musical':
                # Generate a chord with a few notes
                notes = [261.63, 329.63, 392.00]  # C4, E4, G4 (C major chord)
                samples = audio_data.generate_chord(
                    notes=notes,
                    duration=duration,
                    sample_rate=sample_rate
                )
            else:
                # Default to using the generic audio generation function with the appropriate type
                samples = audio_data.generate_audio(
                    duration=duration,
                    audio_type='sine',  # Default type
                    sample_rate=sample_rate
                )
            
            # Create filename for the audio
            filename = f"audio_{timestamp}.{file_format}"
            file_path = os.path.join(output_dir, filename)
            
            # Save the audio file
            audio_data.save_audio(
                audio=samples,
                file_path=file_path,
                sample_rate=sample_rate,
                format=file_format
            )
            
            # Create response
            download_url = url_for('download_file', filename=os.path.basename(file_path))
            view_url = url_for('static', filename=f'generated_data/{filename}')
            
            return jsonify({
                'success': True,
                'message': f'Generated {audio_type} audio of {duration} seconds at {sample_rate}Hz in {file_format} format',
                'download_url': download_url,
                'view_url': view_url,
                'duration': duration,
                'sample_rate': sample_rate,
                'format': file_format,
                'filename': os.path.basename(file_path)
            })
        
        else:
            return jsonify({
                'success': False,
                'message': f'Unsupported data type: {data_type}'
            }), 400
            
    except Exception as e:
        app.logger.error(f"Error generating data: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error generating data: {str(e)}'
        }), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download a generated file"""
    try:
        # Check if the file exists
        file_path = os.path.join('static/generated_data', filename)
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'message': 'File not found'
            }), 404
            
        # Serve the file for download
        return send_from_directory(
            directory='static/generated_data',
            path=filename,
            as_attachment=True
        )
    except Exception as e:
        app.logger.error(f"Error downloading file: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error downloading file: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Ensure the generated_data directory exists
    os.makedirs(os.path.join('static', 'generated_data'), exist_ok=True)
    app.run(debug=True) 