"""
API routes for training functionality
"""
import threading
from flask import Blueprint, jsonify, request
from training.pipeline import run_training_pipeline
from gantraining.GAN_pipeline import run_gan_training
from utils.status import get_training_status, training_status,get_gan_training_status

training_bp = Blueprint('training', __name__)

@training_bp.route('/start', methods=['POST'])
def start_training():
    """Start training pipeline"""
    if training_status['running']:
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
    required_fields = ['input_csv', 'model_name', 'output_dir', 'experiment_name']
    for field in required_fields:
        if field not in config:
            return jsonify({
                'status': 'error',
                'message': f'Missing required field: {field}'
            })
    
    # Set default values
    config.setdefault('num_train_epochs', 1)
    config.setdefault('train_batch_size', 2)
    config.setdefault('eval_batch_size', 2)
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('block_size', 256)
    config.setdefault('test_size', 0.1)
    config.setdefault('logging_steps', 50)
    config.setdefault('seed', 42)
    config.setdefault('fp16', False)
    
    # Start training in background thread
    thread = threading.Thread(target=run_training_pipeline, args=(config,))
    thread.daemon = False # let it run independently
    thread.start()
    
    return jsonify({
        'status': 'started', 
        'message': 'Training started'
    })

@training_bp.route('/status', methods=['GET'])
def get_status():
    """Get current training status"""
    return jsonify(get_training_status())


@training_bp.route('/start_gan', methods=['POST'])
def start_gan_training():
    """Start GAN training pipeline"""
    if training_status['running']:
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

@training_bp.route('/gan_status', methods=['GET'])
def get_gan_status():
    """Get current GAN training status"""
    return jsonify(get_gan_training_status())