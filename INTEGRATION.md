# Integrating DataForge UI with Synthetic Data Generator

This document explains how to fully integrate the DataForge web interface with the Synthetic Data Generator Python project.

## Overview

Currently, the DataForge web interface provides the UI for configuring and requesting synthetic data, but it doesn't yet connect to the actual data generation functionality in the Python backend. This document outlines the steps needed to complete this integration.

## Integration Steps

### 1. Update the Flask Application

In `app.py`, you'll need to import the appropriate modules from the Synthetic Data Generator project:

```python
# Import Synthetic Data Generator modules
from data_types import tabular_data, image_data, text_data, time_series_data, audio_data
from utils import exporters
```

### 2. Create Route Handlers for Each Data Type

Update the `/api/generate` endpoint to handle different data types:

```python
@app.route('/api/generate', methods=['POST'])
def generate_data():
    data_type = request.json.get('data_type')
    config = request.json.get('config', {})
    
    try:
        # Generate unique file ID
        file_id = f"{data_type}_{int(time.time())}"
        
        if data_type == 'tabular':
            return generate_tabular_data(config, file_id)
        elif data_type == 'image':
            return generate_image_data(config, file_id)
        elif data_type == 'text':
            return generate_text_data(config, file_id)
        elif data_type == 'timeseries':
            return generate_timeseries_data(config, file_id)
        elif data_type == 'audio':
            return generate_audio_data(config, file_id)
        else:
            return jsonify({
                'success': False,
                'message': f'Unsupported data type: {data_type}'
            }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating data: {str(e)}'
        }), 500
```

### 3. Implement Specific Data Generation Functions

Create functions to handle each data type:

#### Tabular Data

```python
def generate_tabular_data(config, file_id):
    rows = int(config.get('rows', 1000))
    output_format = config.get('output_format', 'csv')
    schema_type = config.get('schema_type', 'customer')
    
    # Define schema based on schema_type
    if schema_type == 'customer':
        schema = {
            'customer_id': {'type': 'id', 'start': 1000},
            'name': {'type': 'name'},
            'credit_score': {'type': 'int', 'min': 500, 'max': 850},
            'is_active': {'type': 'boolean'},
            'customer_segment': {'type': 'category', 'categories': ['Basic', 'Standard', 'Premium']},
            'signup_date': {'type': 'date', 'start': '2020-01-01', 'end': '2023-12-31'},
            'last_purchase': {'type': 'date', 'start': '2022-01-01', 'end': '2023-12-31'},
            'email': {'type': 'email'},
            'age': {'type': 'int', 'min': 18, 'max': 90},
            'income': {'type': 'float', 'min': 20000, 'max': 100000}
        }
    # Add other schema types...
    
    # Generate data
    data = tabular_data.generate_dataset(rows=rows, schema=schema)
    
    # Create output directory if it doesn't exist
    os.makedirs('static/generated', exist_ok=True)
    
    # Export data
    filename = f"static/generated/{file_id}"
    if output_format == 'csv':
        file_path = f"{filename}.csv"
        exporters.to_csv(data, file_path)
    elif output_format == 'json':
        file_path = f"{filename}.json"
        exporters.to_json(data, file_path)
    elif output_format == 'excel':
        file_path = f"{filename}.xlsx"
        exporters.to_excel(data, file_path)
    elif output_format == 'pkl':
        file_path = f"{filename}.pkl"
        exporters.to_pickle(data, file_path)
    
    # Return response
    return jsonify({
        'success': True,
        'message': f'Successfully generated tabular data with {rows} rows',
        'download_url': f"/{file_path}"
    })
```

#### Image Data

```python
def generate_image_data(config, file_id):
    width = int(config.get('width', 256))
    height = int(config.get('height', 256))
    image_type = config.get('image_type', 'noise')
    num_images = int(config.get('num_images', 1))
    format = config.get('format', 'png')
    
    # Create output directory
    output_dir = f"static/generated/{file_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate images
    image_paths = []
    for i in range(num_images):
        if image_type == 'noise':
            img = image_data.generate_noise_image(width, height)
        elif image_type == 'pattern':
            pattern_type = config.get('pattern_type', 'checkerboard')
            img = image_data.generate_pattern_image(width, height, pattern_type)
        elif image_type == 'gradient':
            gradient_type = config.get('gradient_type', 'horizontal')
            img = image_data.generate_gradient_image(width, height, gradient_type)
        elif image_type == 'geometric':
            shape_type = config.get('shape_type', 'mixed')
            img = image_data.generate_geometric_image(width, height, shape_type)
        
        # Save image
        img_path = f"{output_dir}/image_{i+1}.{format}"
        exporters.save_image(img, img_path)
        image_paths.append(f"/{img_path}")
    
    # Create ZIP if multiple images
    zip_path = None
    if num_images > 1:
        zip_path = f"{output_dir}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for img_path in image_paths:
                zipf.write(img_path.lstrip('/'))
    
    # Return response
    return jsonify({
        'success': True,
        'message': f'Successfully generated {num_images} images',
        'download_url': f"/{zip_path}" if zip_path else image_paths[0],
        'image_paths': image_paths
    })
```

### 4. Add Download Routes

```python
@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)
```

### 5. Create a Static Directory for Generated Files

```python
os.makedirs('static/generated', exist_ok=True)
```

## Error Handling and Security Considerations

1. **Input Validation**: Validate all user inputs to prevent security issues.
2. **File Management**: Implement a cleanup routine for generated files.
3. **Authentication**: Add user authentication for protected routes.
4. **Rate Limiting**: Implement rate limiting to prevent abuse.

## Example Complete Integration

A complete integration would involve:

1. Adding all the functions for each data type
2. Implementing proper error handling
3. Adding authentication
4. Creating a file management system for generated data
5. Adding server-side validation for form inputs

## Testing the Integration

To test the integration:

1. Start the Flask application
2. Navigate to the Forge Data page
3. Select a data type and configure parameters
4. Submit the form and verify that data is generated correctly

## Deployment Considerations

When deploying the integrated application:

1. Use a production-ready WSGI server (like Gunicorn)
2. Set `debug=False` in the Flask application
3. Configure appropriate logging
4. Consider using a CDN for static assets
5. Set up proper error handling and monitoring

## Next Steps

After completing the basic integration:

1. Add user accounts to save configurations
2. Implement a job queue for long-running generation tasks
3. Add data visualization options
4. Create an API documentation page
5. Implement batch processing for large datasets 