"""
API routes for training functionality
"""
import threading
from flask import Blueprint, jsonify, request
from routes.status import gan_training_status
from gantraining.GAN_pipeline import run_gan_training
from utils.status import get_gan_training_status,gan_training_status

gantraining_bp = Blueprint('gantraining', __name__)

@gantraining_bp.route('/status', methods=['GET'])
def get_status():
    """Get current GAN training status"""
    return jsonify(get_gan_training_status())


@gantraining_bp.route('/start', methods=['POST'])
def start_gan_training():
    """Start GAN training pipeline"""
    if gan_training_status['running']:
        return jsonify({
            'status': 'already_running', 
            'message': 'Training is already running'
        })
    
    # Get configuration from request
    config = request.get_json()
    if not config:
        return jsonify({
            'status': 'error', 
            'message': 'No configuration provided'
        })
    
    # Validate required fields
    required_fields = ['experiment_name', 'gan_epochs']
    for field in required_fields:
        if field not in config:
            return jsonify({
                'status': 'error',
                'message': f'Missing required field: {field}'
            })
    
    # Set default values
    config.setdefault('gan_epochs', 3)
    config.setdefault('learning_rate_d', 1e-5)
    config.setdefault('learning_rate_g', 1e-6)
    config.setdefault('gan_batch_size', 4)
    config.setdefault('discriminator_model', 'distilbert-base-uncased')
    config.setdefault('output_dir', './gpt2_finetuned')
    config.setdefault('experiment_name', 'creative-writing')

    
    # Start GAN training in background thread
    thread = threading.Thread(target=run_gan_training, args=(config,))
    thread.daemon = False # let it run independently
    thread.start()
    
    return jsonify({
        'status': 'started', 
        'message': 'GAN Training started'
    })

