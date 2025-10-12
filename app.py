from flask import Flask, render_template, request, jsonify, send_file
import os
import mlflow
from routes.preprocessing import preprocessing_bp
from routes.training import training_bp
from routes.generating import generate_bp
from routes.status import status_bp
from utils.dependencies import check_dependencies

app = Flask(__name__)

# Register blueprints
app.register_blueprint(preprocessing_bp, url_prefix='/api/preprocessing')
app.register_blueprint(training_bp, url_prefix='/api/training')
app.register_blueprint(generate_bp, url_prefix='/api/generating')
app.register_blueprint(status_bp, url_prefix='/api/status')

@app.route('/')
def home():
    deps = check_dependencies()
    return render_template('home.html', dependencies=deps, active_page='home')

@app.route('/preprocessing')
def preprocessing():
    return render_template('preprocessing.html', active_page='preprocessing')

@app.route('/training')
def training():
    deps = check_dependencies()
    return render_template('training.html', 
                          active_page='training', 
                          training_available=deps['training'])

@app.route('/status')
def status():
    return render_template('status.html', active_page='status')

@app.route('/generating')
def generating():
    return render_template('generating.html', active_page='generating')

@app.route('/download_dataset')
def download_dataset():
    try:
        return send_file('creative_writing_dataset.csv', as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': 'Dataset file not found. Please run preprocessing first.'}), 404

if __name__ == '__main__':
    # Ensure MLflow tracking directory exists
    os.makedirs('mlruns', exist_ok=True)
    
    deps = check_dependencies()
    
    print("🚀 Starting Unified ML Pipeline Web Interface...")
    print("📊 MLflow tracking URI:", mlflow.get_tracking_uri())
    print("🌐 Access the web interface at: http://localhost:5000")
    print("📚 Preprocessing available:", "✅ Yes" if deps['preprocessing'] else "❌ No")
    print("🤖 Training available:", "✅ Yes" if deps['training'] else "❌ No")
    print("🤖 Generation available:", "✅ Yes" if deps['generation'] else "❌ No")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)