from flask import Flask, render_template, request, jsonify, send_file
import os
import mlflow
from routes.preprocessing import preprocessing_bp
from routes.training import training_bp
from routes.generating import generate_bp
from routes.gantraining import gantraining_bp
from routes.status import status_bp
from utils.dependencies import check_dependencies
from utils.ip_filter import init_ip_filtering
import config

app = Flask(__name__)

# Initialize IP filtering
init_ip_filtering(app)

# Register blueprints
app.register_blueprint(preprocessing_bp, url_prefix='/api/preprocessing')
app.register_blueprint(training_bp, url_prefix='/api/training')
app.register_blueprint(generate_bp, url_prefix='/api/generating')
app.register_blueprint(status_bp, url_prefix='/api/status')
app.register_blueprint(gantraining_bp, url_prefix='/api/gantraining')

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

@app.route('/gantraining')
def gantraining():
    deps = check_dependencies()
    return render_template('gantraining.html', active_page='gantraining', training_available=deps['training'])

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
    print("📚 Preprocessing available:", "✅ Yes" if deps['preprocessing'] else "❌ No")
    print("🤖 Training available:", "✅ Yes" if deps['training'] else "❌ No")
    print("🤖 Generation available:", "✅ Yes" if deps['generation'] else "❌ No")
    print()
    print("="*60)
    print("🖥️  SERVER INFORMATION:")
    print(f"   Server IPs: {', '.join(config.SERVER_IPS)}")
    print(f"   Listening on: {config.HOST}:{config.PORT}")
    print("="*60)
    print("🌐 ACCESS URLs:")
    print(f"   Local:        http://localhost:{config.PORT}")
    print(f"   Local:        http://127.0.0.1:{config.PORT}")
    for server_ip in config.SERVER_IPS:
        print(f"   Network:      http://{server_ip}:{config.PORT}")
    print("="*60)
    
    # Print IP filtering status
    if config.ENABLE_IP_FILTERING:
        print("🔒 ACCESS CONTROL: RESTRICTED")
        print(f"   Allowed client IPs: {', '.join(config.ALLOWED_IPS)}")
        if config.ALLOW_LOCAL_NETWORK:
            print("   Local network: ALLOWED")
    else:
        print("✅ ACCESS CONTROL: PUBLIC (Anyone can access)")
    print("="*60)
    
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT, threaded=config.THREADED)