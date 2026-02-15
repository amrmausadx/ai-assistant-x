# Enhanced pipeline.py - CORRECTED AND EFFICIENT VERSION
import time
import mlflow
import torch
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, BitsAndBytesConfig
from utils.status import update_generation_status, get_generation_status
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from datetime import datetime

def build_creative_prompt(user_prompt: str, style: str, use_instruction: bool = True) -> str:
    """
    Build contextual prompts based on style.
    For fine-tuned models, you may want to skip instructions (use_instruction=False)
    """
    if not use_instruction:
        return user_prompt.strip()
    
    style_instructions = {
        "creative": "Write engaging, descriptive text with vivid imagery and smooth flow.\n\n",
        "story": "Continue this story naturally, maintaining the style and tone:\n\n",
        "poem": "Write a moving poem with rhythm and emotion.\n\n"
        #"code": "Generate clean, efficient, well-documented code for the following task:\n\n"
    }
    style = style.lower().strip() if style else "creative"
    instruction = style_instructions.get(style, "")
    return instruction + user_prompt.strip() if user_prompt else instruction + "Begin:"

def calculate_perplexity(model, tokenizer, text: str, device: str = "cpu") -> float:
    """
    Calculate actual perplexity of generated text using the model's loss.
    Lower perplexity = better (more confident/fluent).
    Memory-optimized version with proper cleanup.
    """
    try:
        model.eval()
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get the actual device the model is on (important for quantized models)
        model_device = next(model.parameters()).device
        input_ids = encodings.input_ids.to(model_device)
        
        # Create labels (shift input_ids by 1)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()  # Convert to float immediately
        
        # Clean up tensors
        del input_ids, encodings, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return perplexity
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return float('inf')

def run_generation(config: dict):
    """
    Enhanced worker function for text generation with proper memory management.
    """
    prompt = config.get("prompt", "")
    max_length = config.get("max_length", 200)  # Don't clamp here, let each function decide
    temperature = config.get("temperature", 0.8)
    model_name = config.get("model_name", "gpt2")
    experiment_name = config.get("experiment_name", "generation")
    style = config.get("style", "creative").lower().strip()
    seed = config.get("seed", 42)
    use_instruction = config.get("use_instruction", True)
    
    set_seed(seed)
    
    # Variables for cleanup
    model = None
    tokenizer = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        update_generation_status(
            running=True,
            message="Preparing prompt...", 
            progress=5,
            start_time=datetime.utcnow())
        
        # For fine-tuned models on specific texts, skip instruction prefix
        is_finetuned = model_name == "./gpt2_finetuned/"
        enhanced_prompt = build_creative_prompt(prompt, style, use_instruction=not is_finetuned)
        
        run_name = "text_generation"
        
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_params({
                "prompt": prompt[:100],
                "enhanced_prompt": enhanced_prompt[:100],
                "prompt_length": len(prompt),
                "max_length": max_length,
                "temperature": temperature,
                "model_name": model_name,
                "style": style,
                "seed": seed,
                "device": device,
                "use_instruction": use_instruction
            })
            
            update_generation_status(
                run_id=run.info.run_id,
                experiment=experiment_name,
                progress=10
            )
            
            # Generate text and get model/tokenizer for perplexity calculation
            if model_name == "./gpt2_finetuned/":
                text, model, tokenizer = _generate_from_mlflow_model(
                    enhanced_prompt, max_length, temperature, 
                    output_dir=model_name, device=device
                )
            elif "Qwen/Qwen3-Coder-Next" in model_name:
                text, model, tokenizer = _generate_from_qwen_coder(
                    enhanced_prompt, max_length, temperature,
                    model_name, device, use_quantization=True
                )
            else:
                text, model, tokenizer = _generate_from_pretrained(
                    enhanced_prompt, max_length, temperature,
                    model_name, device
                )
            
            # Calculate ACTUAL perplexity
            perplexity = calculate_perplexity(model, tokenizer, text, device)
            
            # Clean up model immediately after perplexity calculation
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            model = None  # Mark as cleaned
            
            # Calculate quality metrics (no model needed)
            quality_scores = calculate_text_quality(text, prompt)
            
            # Log metrics
            duration = time.time() - run.info.start_time / 1000
            mlflow.log_metric("generation_time_seconds", duration)
            mlflow.log_metric("output_length_chars", len(text))
            mlflow.log_metric("output_length_words", len(text.split()))
            mlflow.log_metric("perplexity", perplexity)
            mlflow.log_metric("bleu_score", quality_scores.get("bleu_score", 0.0))
            mlflow.log_metric("rouge1_fmeasure", quality_scores.get("rouge1_fmeasure", 0.0))
            mlflow.log_metric("rouge2_fmeasure", quality_scores.get("rouge2_fmeasure", 0.0))
            mlflow.log_metric("rougeL_fmeasure", quality_scores.get("rougeL_fmeasure", 0.0))

            for metric_name, score in quality_scores.items():
                if metric_name not in ["bleu_score", "rouge1_fmeasure", "rouge2_fmeasure", "rougeL_fmeasure"]:
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
                current_perplexity=perplexity,
                output_text=text,
                bleu_score=quality_scores.get("bleu_score", 0.0),
                rouge1_fmeasure=quality_scores.get("rouge1_fmeasure", 0.0),
            )
            
            return True

    except Exception as e:
        handle_error(e, config)
        return False
    finally:
        # Ensure cleanup even on error
        try:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass


def handle_error(e, config=None):
    """Handle generation errors with proper cleanup"""
    print(f"Error during generation: {e}")
    traceback.print_exc()
    
    update_generation_status(
        experiment=config.get("experiment_name") if config else None,
        running=False,
        progress=0,
        message=f"Generation failed: {str(e)}",
        error=str(e),
        end_time=datetime.utcnow()
    )
    if config and "experiment_name" in config:
        try:
            mlflow.end_run(status="FAILED")
        except:
            pass


def _generate_from_mlflow_model(prompt, max_length, temperature, output_dir, device="cpu"):
    """Generate from fine-tuned model with improved parameters and memory cleanup"""
    update_generation_status(progress=20, message="Loading Fine-Tuned Model...")
    
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    
    # Move to device
    if device == "cuda":
        model = model.to(device)
    
    # Fix pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Improved generation parameters with consistent temperature clamping
    max_length = min(max_length, 512)
    min_length = inputs['input_ids'].shape[1] + 20
    temperature = float(max(0.7, min(temperature, 1.2)))
    
    update_generation_status(progress=40, message="Generating text...")
    
    start_time = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            temperature=temperature,
            do_sample=True,
            top_k=40,
            top_p=0.95,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    duration = time.time() - start_time
    mlflow.log_metric("generation_time", duration)

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Clean up generation tensors
    del inputs, output_ids
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return text, model, tokenizer


def _generate_from_pretrained(prompt, max_length, temperature, model_name, device):
    """Generate from pretrained model with improved parameters and memory cleanup"""
    update_generation_status(progress=20, message=f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    # Move to device
    if device == "cuda":
        model = model.to(device)
    
    model.eval()
    
    update_generation_status(progress=40, message="Generating text...")
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    # Consistent temperature clamping
    temperature = float(max(0.7, min(temperature, 1.2)))
    
    # Clamp max_length for pretrained models
    max_length = min(max_length, 512)
    
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=input_ids.shape[1] + 20,
            temperature=temperature,
            do_sample=True,
            top_k=40,
            top_p=0.95,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    duration = time.time() - start_time
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Proper cleanup
    del input_ids, attention_mask, output_ids
    if device == "cuda":
        torch.cuda.empty_cache()
    
    mlflow.log_metric("generation_time", duration)
    
    return text, model, tokenizer


def _generate_from_qwen_coder(prompt, max_length, temperature, model_name, device, use_quantization=True):
    """
    Generate from Qwen Coder models with 4-bit quantization.
    """
    update_generation_status(progress=20, message=f"Loading {model_name} with quantization...")
    
    # Configure model loading with quantization
    if use_quantization and device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model_kwargs = {
            "quantization_config": quantization_config,
            "device_map": "auto"
        }
        mlflow.log_param("quantization_type", "4bit")
    else:
        # Fallback to full precision if no CUDA
        model_kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }
        mlflow.log_param("quantization_type", "none")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Move to device only if not using quantization (device_map handles it)
    if not use_quantization and device == "cuda":
        model = model.to(device)
    
    # Fix pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    
    update_generation_status(progress=40, message="Generating with Qwen Coder...")
    
    # Qwen Coder uses chat template for instruct models
    if "instruct" in model_name.lower():
        messages = [
            {"role": "system", "content": "You are Qwen, a professional software engineer created by Alibaba Cloud. You provide clean, efficient, well-documented code."},
            {"role": "user", "content": prompt}
        ]
        
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Base model - no chat template
        text_input = prompt
    
    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=512)
    
    # FIXED: Get the actual device the model is on (important for quantized models)
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    
    # Qwen supports longer contexts
    max_length = min(max_length, 2048)
    min_length = inputs['input_ids'].shape[1] + 30
    temperature = float(max(0.7, min(temperature, 1.2)))
    
    start_time = time.time()
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            temperature=temperature,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    duration = time.time() - start_time
    mlflow.log_metric("generation_time", duration)
    
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Clean up generation tensors
    del inputs, output_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return text, model, tokenizer


def calculate_text_quality(generated_text: str, original_prompt: str) -> dict:
    """
    Enhanced quality metrics for generated text
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
            "overall_quality": 0.0,
            "bleu_score": 0.0,
            "rouge1_fmeasure": 0.0,
            "rouge2_fmeasure": 0.0,
            "rougeL_fmeasure": 0.0
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
    
    # 5. BLEU score with smoothing (for short texts)
    reference = [original_prompt.lower().split()]
    candidate = actual_generation.lower().split()
    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    # 6. ROUGE scores (multiple variants)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(original_prompt, actual_generation)
    rouge1_f = rouge_scores['rouge1'].fmeasure
    rouge2_f = rouge_scores['rouge2'].fmeasure
    rougeL_f = rouge_scores['rougeL'].fmeasure
    
    # Overall quality (weighted average)
    overall_quality = (
        length_score * 0.15 +
        diversity_score * 0.25 +
        repetition_score * 0.25 +
        structure_score * 0.15 +
        bleu_score * 0.05 +
        rouge1_f * 0.05 +
        rouge2_f * 0.05 +
        rougeL_f * 0.05
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
        "rouge1_fmeasure": rouge1_f,
        "rouge2_fmeasure": rouge2_f,
        "rougeL_fmeasure": rougeL_f
    }
