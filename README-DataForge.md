# DataForge Web Interface

A responsive and interactive web interface for the Synthetic Data Generator project.

## Overview

DataForge is a user-friendly web interface for generating various types of synthetic data:

- Tabular Data
- Image Data
- Text Data
- Time Series Data
- Audio Data

## Features

- Modern, responsive design
- Interactive data type selection
- Customizable parameters for each data type
- Real-time form validation
- Mobile-friendly interface

## Setup Instructions

### Prerequisites

- Python 3.7+
- Flask
- All dependencies from the Synthetic Data Generator project

### Installation

1. Ensure you have the Synthetic Data Generator project set up and the virtual environment activated:

```bash
# Activate the virtual environment
# On Windows
synthetic_data_env\Scripts\activate.ps1

# On Unix/MacOS
source synthetic_data_env/bin/activate
```

2. Install Flask if not already installed:

```bash
pip install flask
```

### Running the Web Interface

1. Start the Flask application:

```bash
python app.py
```

2. Open your browser and navigate to:

```
http://localhost:5000
```

## Using DataForge

1. **Home Page**: The home page shows an overview of the DataForge platform with a "Start Forging" button that takes you to the data generation page.

2. **Forge Data Page**: This is where you can:
   - Select a data type (Tabular, Image, Text, Time Series, or Audio)
   - Configure generation parameters specific to the selected data type
   - Generate synthetic data and download the results

### Generating Data

1. Click on one of the data type cards to select it
2. Fill in the configuration parameters for your data
3. Click the "Generate Data" button
4. Once the data is generated, you can download it using the provided link

## Project Structure

```
DataForge/
├── app.py                 # Flask application file
├── static/                # Static assets
│   ├── css/
│   │   └── style.css      # Main stylesheet
│   ├── js/
│   │   └── main.js        # JavaScript for interactive elements
│   └── images/            # Image assets for the website
├── templates/             # HTML templates
│   ├── index.html         # Home page
│   └── forge.html         # Data forge page
```

## Integration with Synthetic Data Generator

The web interface is designed to work with the existing Synthetic Data Generator project. It provides a user-friendly way to configure and generate synthetic data.

In a future update, the Flask application will be fully integrated with the backend synthetic data generation capabilities.

## Future Development

To fully connect this frontend with the Synthetic Data Generator backend:

1. Update the `/api/generate` endpoint in `app.py` to call the appropriate generation functions based on the data type
2. Import the required modules from the Synthetic Data Generator project
3. Create appropriate API routes for each data type
4. Add authentication and error handling

## Future Enhancements

- User authentication and saved configurations
- Data preview before generation
- Enhanced visualization options
- API documentation
- Batch processing capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details. 