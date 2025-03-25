from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forge')
def forge():
    return render_template('forge.html')

# This will be implemented later to connect with the synthetic data generator
@app.route('/api/generate', methods=['POST'])
def generate_data():
    # Placeholder API endpoint that will be implemented later
    data_type = request.json.get('data_type')
    config = request.json.get('config', {})
    
    # For now, just return a success message
    return jsonify({
        'success': True,
        'message': f'Successfully generated {data_type} data',
        'download_url': f'/download/{data_type}_data'
    })

if __name__ == '__main__':
    app.run(debug=True) 