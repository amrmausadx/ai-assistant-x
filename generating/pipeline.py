# pipeline.py
import time
import mlflow
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from utils.status import update_generation_status,get_generation_status
from flask import jsonify
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from datetime import datetime

def build_creative_prompt(user_prompt: str, style: str) -> str:
    """
    Build contextual prompts based on style
    """
    style_instructions = {
        "creative": "Write engaging, descriptive text with vivid imagery and smooth flow. ",
        "story": "Tell a compelling story with clear narrative structure and character development. ",
        "poem": "Write a moving poem with rhythm, rhyme (or free verse), and profound emotion. " # Added 'poem'
    }
    style = style.lower().strip() if style else "creative"
    instruction = style_instructions.get(style)
    return instruction + user_prompt.strip() if user_prompt else instruction + "Begin your text:"

#    return instruction + user_prompt.strip() if user_prompt else instruction + "Begin your story:"

def run_generation(config: dict):
    """
    Worker function for generation. Takes a config dict with prompt, max_length, temperature,
    model_name, experiment_name, etc.
    """
    prompt = config.get("prompt", "")
    max_length = min(config.get("max_length", 200),512)
    temperature = config.get("temperature", 0.8)
    model_name = config.get("model_name", "gpt2")
    experiment_name = config.get("experiment_name", "generation")
    style = config.get("style","creative").lower().strip() 
    print(f"Backend received style: '{style}'")  # Add debug log

    seed = config.get("seed",42)

    set_seed(seed)

    try:
        update_generation_status(
            running=True,
            message="Enhancing prompt for model", 
            progress=5,
            start_time=datetime.utcnow())
        enhanced_prompt = build_creative_prompt(prompt,style)
        run_name = "text_generation"

        device = None
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except:
            device = "cpu"
        #check if user choose./gpt2_finetuned/
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_params({
                "prompt": prompt[:100],  # Truncate long prompts in logs
                "enhanced_prompt": enhanced_prompt[:100],  # Truncate long prompts in logs
                "prompt_length": len(prompt),
                "max_length": max_length,
                "temperature": temperature,
                "model_name": model_name,
                "style": style,
                "seed": seed,
                "device": device
            })
            
            update_generation_status(
                run_id=run.info.run_id,
                experiment=experiment_name,
                progress=10
            )
            text = ""
            # Check if using fine-tuned model
            if model_name == "./gpt2_finetuned/":
                text = _generate_from_mlflow_model(
                    enhanced_prompt, max_length, temperature, 
                    experiment_name
                )
            else:
                text = _generate_from_pretrained(
                    enhanced_prompt, max_length, temperature,
                    model_name, device
                )
            
            # Calculate quality metrics
            quality_scores = calculate_text_quality(text, prompt)
            #calculate perplexity
            #perplexity = torch.exp(torch.tensor(quality_scores["overall_quality"])) if quality_scores["overall_quality"] < 20 else float("inf")
            perplexity = torch.exp(torch.tensor(-quality_scores["overall_quality"]))
            # Log metrics
            duration = time.time() - run.info.start_time / 1000  # Convert to seconds
            mlflow.log_metric("generation_time_seconds", duration)
            mlflow.log_metric("output_length_chars", len(text))
            mlflow.log_metric("output_length_words", len(text.split()))
            mlflow.log_metric("perplexity", perplexity.item() if isinstance(perplexity, torch.Tensor) else perplexity)
            mlflow.log_metric("bleu_score", quality_scores.get("bleu_score", 0.0))
            mlflow.log_metric("rouge1_fmeasure", quality_scores.get("rouge1_fmeasure", 0.0))

            for metric_name, score in quality_scores.items():
                mlflow.log_metric(f"quality_{metric_name}", score)
            
            # Save output
            mlflow.log_text(text, "generated_text.txt")
            
            update_generation_status(
                running=False,
                progress=100,
                message="✅ Generation complete",
                end_time=datetime.utcnow(),
                run_id=run.info.run_id,
                experiment=experiment_name,
                output_length=len(text),
                current_perplexity=perplexity.item() if isinstance(perplexity, torch.Tensor) else perplexity,
                output_text=text,
                bleu_score=quality_scores.get("bleu_score", 0.0),
                rouge1_fmeasure=quality_scores.get("rouge1_fmeasure", 0.0),
            )
            
            return True

    except Exception as e:
        handle_error(e,config)
        return False


def handle_error(e,config=None):
    print(f"Error during generation: {e}")
    update_generation_status(
        experiment=config.get("experiment_name") if config else None,
        running=False,
        progress=0,
        message=f"Generation failed: {e}",
        error=str(e),
        end_time=datetime.utcnow()
    )
    if config and "experiment_name" in config:
        mlflow.end_run(status="FAILED")

def _generate_from_mlflow_model(prompt, max_length, temperature, experiment_name):
    """Generate from MLflow registered model"""
    update_generation_status(progress=20, message="Loading fine-tuned model from MLflow...")
    
    client = mlflow.tracking.MlflowClient()
    
    # Try to get production model first, fall back to latest
    try:
        model_versions = client.get_latest_versions(
            experiment_name + "-model",
            stages=["Production"]
        )
        if not model_versions:
            model_versions = client.get_latest_versions(
                experiment_name + "-model",
                stages=["None"]
            )
        
        latest_version = model_versions[-1].version
        logged_model = f"models:/{experiment_name}-model/{latest_version}"
        
        mlflow.log_param("model_version", latest_version)
        
    except Exception as e:
        handle_error(e,{"experiment_name": experiment_name})
        return False
    
    update_generation_status(progress=40, message="Generating text...")
    
    # Load the pipeline
    pipeline = mlflow.transformers.load_model(logged_model)
    
    # Generate with enhanced parameters
    start_time = time.time()
    result = pipeline(
        prompt,
        max_length=max_length,
        min_length=max(20, len(prompt.split()) + 10),  # Ensure minimum generation
        temperature=temperature,
        do_sample=True,
        top_k=50,
        top_p=0.92,  # Slightly lower for more coherence
        repetition_penalty=1.3,  # Stronger penalty for repetition
        no_repeat_ngram_size=3,
        num_return_sequences=1,
        pad_token_id=pipeline.tokenizer.pad_token_id,
        eos_token_id=pipeline.tokenizer.eos_token_id
    )
    duration = time.time() - start_time
    
    text = result[0]['generated_text']
    mlflow.log_metric("generation_time", duration)
    
    return text


def _generate_from_pretrained(prompt, max_length, temperature, model_name, device):
    """Generate from pretrained model"""
    update_generation_status(progress=20, message=f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    update_generation_status(progress=40, message="Generating text...")
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=max(20, input_ids.shape[1] + 10),
            temperature=temperature,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    duration = time.time() - start_time
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    mlflow.log_metric("generation_time", duration)
    
    return text


def calculate_text_quality(generated_text: str, original_prompt: str) -> dict:
    """
    Calculate quality metrics for generated text
    """
    # Remove the prompt from generated text if it's included
    if generated_text.startswith(original_prompt):
        actual_generation = generated_text[len(original_prompt):].strip()
    else:
        actual_generation = generated_text
    
    words = actual_generation.split()
    
    if len(words) == 0:
        return {
            "length_score": 0.0,
            "diversity_score": 0.0,
            "repetition_score": 0.0,
            "overall_quality": 0.0
        }
    
    # 1. Length score (prefer 50-200 words)
    word_count = len(words)
    if word_count < 20:
        length_score = word_count / 20.0
    elif word_count > 200:
        length_score = max(0.5, 1.0 - (word_count - 200) / 200.0)
    else:
        length_score = 1.0
    
    # 2. Vocabulary diversity (unique words / total words)
    unique_words = len(set(w.lower() for w in words))
    diversity_score = unique_words / len(words) if len(words) > 0 else 0
    
    # 3. Repetition score (penalize repeated adjacent words)
    repetitions = sum(1 for i in range(len(words) - 1) if words[i].lower() == words[i + 1].lower())
    repetition_score = max(0, 1.0 - (repetitions / len(words)))
    
    # 4. Sentence structure (count sentences)
    sentences = [s.strip() for s in actual_generation.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    structure_score = 1.0 if 5 <= avg_sentence_length <= 25 else 0.7
    reference = [original_prompt.lower().split()] # BLEU expects a list of reference token lists    
    candidate = actual_generation.lower().split()    
    bleu_score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(original_prompt, actual_generation)
    rouge1_fmeasure = scores['rouge1'].fmeasure    
    # Overall quality (weighted average)
    overall_quality = (
        length_score * 0.2 +
        diversity_score * 0.3 +
        repetition_score * 0.3 +
        structure_score * 0.2 +
        (bleu_score * 0.05 + rouge1_fmeasure * 0.05) 
    )


    
    return {
        "length_score": length_score,
        "diversity_score": diversity_score,
        "repetition_score": repetition_score,
        "structure_score": structure_score,
        "overall_quality": overall_quality,
        "word_count": word_count,
        "unique_words": unique_words,
        "sentence_count": len(sentences),
        "bleu_score": bleu_score,
        "rouge1_fmeasure": rouge1_fmeasure
    }

