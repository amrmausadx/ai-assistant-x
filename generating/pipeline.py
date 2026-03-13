# Enhanced pipeline.py - CORRECTED AND EFFICIENT VERSION
import time
import mlflow
import torch
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, BitsAndBytesConfig
from utils.status import update_generation_status, get_generation_status
#from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
#from rouge_score import rouge_scorer
from datetime import datetime

def build_creative_prompt(user_prompt: str, style: str, num_paragraphs: int = 2, num_characters: int = None,
                           theme: str = None, use_instruction: bool = True) -> str:
    
    if not use_instruction:
        return user_prompt.strip()
    
    # Build structured constraint block
    constraints = []
    if num_paragraphs:
        constraints.append(f"- Write exactly {num_paragraphs} paragraphs")
    if num_characters:
        constraints.append(f"- Include exactly {num_characters} characters")
    if theme:
        constraints.append(f"- Central theme: {theme}")
    
    constraint_block = "\n".join(constraints)
    
    style_templates = {
        "story": f"""Write a coherent, engaging short story with the following requirements:
                {constraint_block}
                The story must begin with this exact line:
                \"{user_prompt.strip()}\"
                Story:""",
        "poem": f"""Write a moving poem with the following requirements:
                {constraint_block}
                Opening line:
                \"{user_prompt.strip()}\" 
                Poem:""",
                "creative": f"""Write creative text with the following requirements:
                {constraint_block}
                Starting with:
                \"{user_prompt.strip()}\"
                
        Text:"""
            }
    
    style = style.lower().strip() if style else "story"
    return style_templates.get(style, style_templates["creative"])
                               
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
        enhanced_prompt = build_creative_prompt(prompt, style,
                                                num_paragraphs=2,
                                                num_characters=None,
                                                theme=None,
                                                use_instruction=not is_finetuned)
        
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
            duration = time.time() - (run.info.start_time / 1000)
            
            metric_lines = []
            for metric_name, score in quality_scores.items():
                #if metric_name not in ["bleu_score", "rouge1_fmeasure", "rouge2_fmeasure", "rougeL_fmeasure"]:
                try:
                    mlflow.log_metric(f"quality_{metric_name}", float(score))
                    metric_lines.append(f"{metric_name}: {score}")
                except (TypeError, ValueError):
                    pass
            metrics_display = "\n".join(metric_lines)
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
                output_text=text + "\n\n---📊 Quality Metrics---\n" + metrics_display,
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
    
    if generated_text.startswith(original_prompt):
        actual_generation = generated_text[len(original_prompt):].strip()
    else:
        actual_generation = generated_text
    
    words = actual_generation.lower().split()
    sentences = [s.strip() for s in 
                 actual_generation.replace('!','.').replace('?','.').split('.') 
                 if s.strip()]
    
    # 1. Vocabulary Richness (Type-Token Ratio)
    unique_words = set(words)
    ttr = len(unique_words) / len(words) if words else 0
    
    # 2. Repetition penalty (n-gram repetition)
    def ngram_repetition(words, n=3):
        ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        if not ngrams:
            return 0.0
        unique_ngrams = set(ngrams)
        repetition = 1.0 - (len(unique_ngrams) / len(ngrams))
        return repetition  # lower is better
    
    trigram_rep = ngram_repetition(words, 3)
    
    # 3. Sentence length variation (good stories vary sentence length)
    if len(sentences) > 1:
        sent_lengths = [len(s.split()) for s in sentences]
        avg_len = sum(sent_lengths) / len(sent_lengths)
        variance = sum((l - avg_len)**2 for l in sent_lengths) / len(sent_lengths)
        sentence_variety = min(1.0, variance / 50)  # normalize
    else:
        sentence_variety = 0.0
    
    # 4. Prompt adherence — did it continue from the prompt?
    prompt_words = set(original_prompt.lower().split())
    gen_words = set(words[:50])  # check first 50 words
    prompt_adherence = len(prompt_words & gen_words) / len(prompt_words) if prompt_words else 0
    
    # 5. Coherence proxy — consecutive sentence similarity
    # (high similarity = repetitive, low = incoherent, mid = good)
    def sentence_coherence(sentences):
        if len(sentences) < 2:
            return 0.5
        scores = []
        for i in range(len(sentences)-1):
            s1 = set(sentences[i].lower().split())
            s2 = set(sentences[i+1].lower().split())
            if not s1 or not s2:
                continue
            overlap = len(s1 & s2) / max(len(s1), len(s2))
            scores.append(overlap)
        avg = sum(scores)/len(scores) if scores else 0
        # ideal coherence overlap is 0.1-0.3
        coherence = 1.0 - abs(avg - 0.2) / 0.2
        return max(0.0, min(1.0, coherence))
    
    coherence_score = sentence_coherence(sentences)
    
    # 6. Overall quality — no BLEU/ROUGE
    overall_quality = (
        ttr * 0.30 +                          # vocabulary richness
        (1 - trigram_rep) * 0.25 +            # non-repetitive
        sentence_variety * 0.20 +             # varied sentences
        coherence_score * 0.15 +              # coherent flow
        prompt_adherence * 0.10               # stays on topic
    )
    
    return {
        "vocabulary_richness": round(ttr, 4),
        "repetition_rate": round(trigram_rep, 4),      # lower better
        "sentence_variety": round(sentence_variety, 4),
        "coherence_score": round(coherence_score, 4),
        "prompt_adherence": round(prompt_adherence, 4),
        "overall_quality": round(overall_quality, 4),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "unique_words": len(unique_words),
        # Keep BLEU/ROUGE for compatibility but don't weight them
        "bleu_score": 0.0,
        "rouge1_fmeasure": 0.0,
        "rouge2_fmeasure": 0.0,
        "rougeL_fmeasure": 0.0,
    }
