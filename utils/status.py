"""
Status tracking for preprocessing and training tasks
"""
from datetime import datetime

# Global status tracking
preprocessing_status = {
    'running': False,
    'progress': 0,
    'message': 'Ready to start',
    'error': None,
    'start_time': None,
    'end_time': None,
    'run_id': None,
    'experiment_name':None,
}

training_status = {
    'running': False,
    'progress': 0,
    'message': 'Ready to start',
    'error': None,
    'start_time': None,
    'end_time': None,
    'run_id': None,
    'current_epoch': 0,
    'total_epochs': 0,
    'current_loss': None,
    'current_perplexity': None,
    'experiment_name':None,
}

gan_training_status = {
    'running': False,
    'progress': 0,
    'message': 'Ready to start',
    'error': None,
    'start_time': None,
    'end_time': None,
    'run_id': None,
    'current_epoch': 0,
    'total_epochs': 0,
    'current_loss': None,
    'current_perplexity': None,
    'experiment_name':None,
    'quantize': False,
}

generation_status = {
    "running": False,
    "progress": 0,
    "message": "Ready to generate",
    "error": None,
    "start_time": None,
    "end_time": None,
    "run_id": None,
    "input_length": 0,
    "output_length": 0,
    "output_text": "",
    "experiment_name": None,
    "current_perplexity": None,
    "bleu_score": None,
    "rouge1_fmeasure": None,
}

def reset_generation_status():
    global generation_status
    generation_status = {
        "running": False,
        "progress": 0,
        "message": "Ready to generate",
        "error": None,
        "start_time": None,
        "end_time": None,
        "run_id": None,
        "input_length": 0,
        "output_length": 0,
        "output_text": "",
        "experiment_name": None,
        "current_perplexity": None,
        "bleu_score": None,
        "rouge1_fmeasure": None,
    }

def update_preprocessing_status(**kwargs):
    """Update preprocessing status"""
    global preprocessing_status
    preprocessing_status.update(kwargs)

def update_training_status(**kwargs):
    """Update training status"""
    global training_status
    training_status.update(kwargs)

def get_preprocessing_status():
    """Get preprocessing status with formatted timestamps"""
    status_copy = preprocessing_status.copy()
    
    if status_copy['start_time']:
        status_copy['start_time'] = status_copy['start_time'].strftime('%Y-%m-%d %H:%M:%S')
    if status_copy['end_time']:
        status_copy['end_time'] = status_copy['end_time'].strftime('%Y-%m-%d %H:%M:%S')
    
    return status_copy

def get_training_status():
    """Get training status with formatted timestamps"""
    status_copy = training_status.copy()
    
    if status_copy['start_time']:
        status_copy['start_time'] = status_copy['start_time'].strftime('%Y-%m-%d %H:%M:%S')
    if status_copy['end_time']:
        status_copy['end_time'] = status_copy['end_time'].strftime('%Y-%m-%d %H:%M:%S')
    
    return status_copy

def update_generation_status(**kwargs):
    global generation_status
    generation_status.update({k:v for k,v in kwargs.items() if v is not None})

def get_generation_status():
    status_copy = generation_status.copy()
    if status_copy['start_time']:
        status_copy['start_time'] = status_copy['start_time'].strftime('%Y-%m-%d %H:%M:%S')
    if status_copy['end_time']:
        status_copy['end_time'] = status_copy['end_time'].strftime('%Y-%m-%d %H:%M:%S')
    return status_copy

def reset_preprocessing_status():
    """Reset preprocessing status to initial state"""
    global preprocessing_status
    preprocessing_status = {
        'running': False,
        'progress': 0,
        'message': 'Ready to start',
        'error': None,
        'start_time': None,
        'end_time': None,
        'run_id': None,
        'experiment_name':None,
    }

def reset_training_status():
    """Reset training status to initial state"""
    global training_status
    training_status = {
        'running': False,
        'progress': 0,
        'message': 'Ready to start',
        'error': None,
        'start_time': None,
        'end_time': None,
        'run_id': None,
        'current_epoch': 0,
        'total_epochs': 0,
        'current_loss': None,
        'experiment_name':None,
    }

# GAN training status functions
def reset_gan_training_status():
    """Reset GAN training status to initial state"""
    global gan_training_status
    gan_training_status = {
        'running': False,
        'progress': 0,
        'message': 'Ready to start GAN training',
        'error': None,
        'start_time': None,
        'end_time': None,
        'run_id': None,
        'current_epoch': 0,
        'total_epochs': 0,
        'current_loss': None,
        'experiment_name':None,
        'current_perplexity': None,
    }
def get_gan_training_status():
    """Get GAN training status with formatted timestamps"""
    status_copy = gan_training_status.copy()

    if status_copy['start_time']:
        status_copy['start_time'] = status_copy['start_time'].strftime('%Y-%m-%d %H:%M:%S')
    if status_copy['end_time']:
        status_copy['end_time'] = status_copy['end_time'].strftime('%Y-%m-%d %H:%M:%S')
    
    return status_copy
def update_gan_training_status(**kwargs):
    """Update GAN training status"""
    global gan_training_status
    gan_training_status.update({k:v for k,v in kwargs.items() if v is not None})