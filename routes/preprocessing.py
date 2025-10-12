"""
API routes for preprocessing functionality
"""
import threading
from flask import Blueprint, jsonify,request
from preprocessing.pipeline import run_preprocessing_pipeline
from utils.status import get_preprocessing_status, preprocessing_status
from utils.dependencies import setup_nltk
import re

preprocessing_bp = Blueprint('preprocessing', __name__)

@preprocessing_bp.route('/start', methods=['POST'])
def start_preprocessing():
    """Start preprocessing pipeline"""
    if preprocessing_status.get('running'):
        return jsonify({'status': 'already_running', 'message': 'Preprocessing is already running'})
    
    setup_nltk()

    config = request.get_json()
    if not config:
        return jsonify({'status': 'error', 'message': 'No configuration provided'})

    required_fields = ['experiment_name','limit_load']
    for field in required_fields:
        if field not in config:
            return jsonify({'status':'error','message': f'Missing required field: {field}'})

    threading.Thread(target=run_preprocessing_pipeline, args=(config,), daemon=True).start()

    return jsonify({'status': 'started', 'message': 'Preprocessing started'})


@preprocessing_bp.route('/status', methods=['GET'])
def get_status():
    """Get current preprocessing status"""
    return jsonify(get_preprocessing_status())