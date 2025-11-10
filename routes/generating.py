# generate.py
import threading
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from utils.status import get_generation_status, update_generation_status, generation_status, reset_generation_status
# Note the import: import from pipeline (the file above)
from generating.pipeline  import run_generation

generate_bp = Blueprint('generating', __name__)

@generate_bp.route("/start", methods=["POST"])
def start_generation():
    if generation_status.get("running"):
        return jsonify({"status": "already_running", "message": "Generation is already running"}), 200

    payload = request.get_json() or {}

    # Collect all inputs into a config dict
    config = {
        "prompt": payload.get("prompt", ""),
        "max_length": int(payload.get("max_length", 200)),
        "temperature": float(payload.get("temperature", 0.9)),
        "experiment_name": payload.get("experiment_name", "creative-writing"),
        "model_name": payload.get("model_name", "gpt2"),  # optional extra
        "start_time": datetime.utcnow(),
        "current_perplexity": payload.get("current_perplexity", None),
        "bleu_score": payload.get("bleu_score", None),
        "rouge1_fmeasure": payload.get("rouge1_fmeasure", None),
        "style": payload.get("style", "creative").lower().strip(),
    }

    reset_generation_status()
    update_generation_status(
        running=True,
        progress=0,
        message="Starting generation",
        start_time=config["start_time"],
        input_length=len(config["prompt"] or ""),
        experiment_name=config["experiment_name"],
        current_perplexity=config.get("current_perplexity", ""),
        bleu_score=config.get("bleu_score", ""),
        rouge1_fmeasure=config.get("rouge1_fmeasure", ""),
        style=config.get("style", "creative"),
    )

    thread = threading.Thread(target=run_generation, args=(config,))
    thread.daemon = False  # let it run independently
    thread.start()

    return jsonify({"status": "started", "message": "Generation started"}), 200


@generate_bp.route("/status", methods=["GET"])
def get_status():
    return jsonify(get_generation_status()), 200