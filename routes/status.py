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

@status_bp.route('/gantraining', methods=['GET'])
def gantraining_status():
    """Get GAN training status"""
    return jsonify(get_gan_training_status())

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
        # 1. Fetch all registered models (sorted by last updated)
        registered_models = client.search_registered_models(order_by=["last_updated_timestamp DESC"])

        if not registered_models:
            return jsonify({
                "name": None,
                "version": None,
                "status": "No models found",
                "creation_time": None
            })
        # 2. Get the latest model
        latest_model = registered_models[0]
        latest_version = latest_model.latest_versions[0] if latest_model.latest_versions else None
        if not latest_version:
            return jsonify({
                "name": latest_model.name,
                "version": None,
                "status": "No versions found",
                "creation_time": None
            })
        # 3. Return relevant details
        return jsonify({
            "name": latest_model.name,
            "version": latest_version.version,
            "status": latest_version.current_stage,
            "creation_time": latest_version.creation_timestamp
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