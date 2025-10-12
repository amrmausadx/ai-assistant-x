"""
API routes for status monitoring
"""
from flask import Blueprint, jsonify
from utils.status import get_preprocessing_status, get_training_status,get_generation_status,get_gan_training_status
from mlflow.tracking import MlflowClient

status_bp = Blueprint('status', __name__)

@status_bp.route('/preprocessing', methods=['GET'])
def preprocessing_status():
    """Get preprocessing status"""
    return jsonify(get_preprocessing_status())

@status_bp.route('/generating', methods=['GET'])
def generation_status():
    """Get generation status"""
    return jsonify(get_generation_status())

@status_bp.route('/training', methods=['GET'])
def training_status():
    """Get training status"""
    return jsonify(get_training_status())

@status_bp.route('/gan_training', methods=['GET'])
def gan_training_status():
    """Get GAN training status"""
    return jsonify(get_gan_training_status())

@status_bp.route('/all', methods=['GET'])
def all_status():
    """Get all status information"""
    return jsonify({
        'preprocessing': get_preprocessing_status(),
        'training': get_training_status(),
        'generating': get_generation_status()
    })

@status_bp.route('/last_model')
def get_last_registered_model():
    client = MlflowClient()
    try:
        # Get the latest model
        experiments = client.search_experiments(order_by=["creation_time DESC"])
        exp_name = experiments[0]
        model_name = exp_name + "-model"

        return jsonify({
            "name": client.get_registered_model(model_name).name,
            "version": client.get_registered_model(model_name)._latest_version,
            "status": "Ready",
            "creation_time": client.get_registered_model(model_name).last_updated_timestamp
            }) 
    
    except Exception as e:
        print(e)
        return jsonify({
            "error": str(e),
            "name": None,
            "version": None,
            "status": "Error",
            "creation_time": None
        })