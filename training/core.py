"""
Core training functions and utilities
"""
import math
import json
import os
import mlflow
import torch
from datetime import datetime
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from utils.status import update_training_status

class MLflowCallback(TrainerCallback):
    """Custom callback for MLflow logging"""
    
    def __init__(self, run_id=None):
        self.run_id = run_id
        self.best_loss = float('inf')
        self.best_perplexity = float('inf')

    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        current_epoch = int(state.epoch) + 1
        total_epochs = int(args.num_train_epochs)

        update_training_status(
            current_epoch=current_epoch,
            message=f'Training epoch {current_epoch}/{total_epochs}'
        )
        
        # Calculate progress (leave room for final evaluation)
        progress = min(90, (current_epoch / total_epochs) * 85)
        update_training_status(progress=progress)

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs:
            # Update current loss if available
            if 'loss' in logs:
                # Calculate perplexity
                perplexity = math.exp(logs['loss']) if logs['loss'] < 20 else None

                update_training_status(current_loss=logs['loss'], current_perplexity=perplexity)
                
                if logs['loss'] < self.best_loss:
                    self.best_loss = logs['loss']
                    self.best_perplexity = perplexity
                    mlflow.log_metric("best_train_loss", self.best_loss, step=state.global_step)
                    # FIXED: Only log perplexity if it's a valid number
                    if self.best_perplexity is not None:
                        mlflow.log_metric("best_train_perplexity", self.best_perplexity, step=state.global_step)
            
            if 'eval_loss' in logs:
                mlflow.log_metric("eval_loss", logs['eval_loss'], step=state.global_step)
                # FIXED: Only log eval perplexity if it's a valid number
                eval_perplexity = math.exp(logs["eval_loss"]) if logs['eval_loss'] < 20 else None
                if eval_perplexity is not None:
                    mlflow.log_metric("eval_perplexity", eval_perplexity, step=state.global_step)
            
            # Log all metrics to MLflow 
            for k, v in logs.items():
                try:
                    mlflow.log_metric(k, float(v), step=state.global_step if state else 0)
                except Exception:
                    pass

def evaluate_model(trainer, tokenizer, config):
    """Evaluate the trained model"""
    try:
        # Basic evaluation metrics
        metrics = trainer.evaluate()
        eval_loss = metrics.get("eval_loss", None)
        
        if eval_loss is not None:
            perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")
            # FIXED: Only log if perplexity is a valid finite number
            if math.isfinite(perplexity):
                mlflow.log_metric("final_perplexity", perplexity)
            mlflow.log_metric("final_eval_loss", eval_loss)
        
        # Generate sample texts
        #generated_samples = generate_samples(trainer.model, tokenizer, config['output_dir'])
        
        return metrics, None
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {}, []
        
def generate_samples(model, tokenizer, output_dir):
    """Generate sample texts from the trained model"""
    gen_inputs = [
        "The moonlight spilled over the old city and",
        "She opened the dusty letter and found",
        "Once upon a time in the small village of",
        "The ancient book contained secrets that",
        "As the storm approached, the lighthouse keeper"
    ]
    
    model.eval()
    generated_texts = []
    device = model.device
    #temperatures
    temperatures = [0.7,0.9,1.0]
    for prompt in gen_inputs:
        try:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            for temp in temperatures:    
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_length=200,
                        min_length=50,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        temperature=temp,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penality=1.2,
                        no_repeat_ngram_size=3,
                    )
            
                text = tokenizer.decode(output[0], skip_special_tokens=True)
                generated_texts.append({"prompt": prompt,"temperature":temp, "generated": text,"length":len(text.split())})
            
        except Exception as e:
            print(f"Generation failed for prompt '{prompt}': {e}")
            generated_texts.append({"prompt": prompt, "generated": f"Generation failed: {e}","error":e})
    
    # Save generation examples
    try:
        gen_file = os.path.join(output_dir, "generation_examples.json")
        with open(gen_file, "w", encoding="utf-8") as f:
            json.dump(generated_texts, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(gen_file)
    except Exception as e:
        print(f"Failed to save generation examples: {e}")
    
    return generated_texts


def prepare_tokenization_function(tokenizer, block_size):
    effective_block_size = min(block_size, tokenizer.model_max_length)

    def tokenize_map(examples):
        texts = [t for t in examples["text"] if isinstance(t, str) and t.strip()]
        
        # FIXED: Always return the expected schema, even if empty
        if not texts:
            return {
                #"input_ids": [],
                "input_ids": [[]],  # List of empty list maintains structure
                "labels": []
            }

        joined = tokenizer.eos_token.join(texts)
        tokenized = tokenizer(joined, add_special_tokens=True, truncation=False)

        input_ids = tokenized["input_ids"]
        chunks = []

        for i in range(0, len(input_ids), effective_block_size):
            chunk = input_ids[i:i + effective_block_size]

            if len(chunk) < effective_block_size // 2:
                continue

            if len(chunk) < effective_block_size:
                chunk += [tokenizer.pad_token_id] * (effective_block_size - len(chunk))

            chunks.append(chunk)

        # FIXED: Return consistent schema even when no chunks
        if not chunks:
            return {
                "input_ids": [],
                "labels": []
            }

        return {
            "input_ids": chunks,
            "labels": chunks.copy()
        }

    return tokenize_map
def calculate_generation_quality(generated_samples):
    """
    Calculate quality metrics for generated text
    """
    metrics = {
        "avg_length": 0,
        "vocab_diversity": 0,
        "repetition_rate": 0
    }
    
    if not generated_samples:
        return metrics
    
    total_words = 0
    all_words = set()
    total_repetitions = 0
    valid_samples = 0
    
    for sample in generated_samples:
        if "error" in sample or not sample.get("generated"):
            continue
        
        valid_samples += 1
        text = sample["generated"]
        words = text.lower().split()
        
        if not words:
            continue
        
        # Average length
        total_words += len(words)
        
        # Vocabulary diversity (unique words)
        all_words.update(words)
        
        # Repetition rate (repeated adjacent words)
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                total_repetitions += 1
    
    if valid_samples > 0:
        metrics["avg_length"] = total_words / valid_samples
        metrics["vocab_diversity"] = len(all_words) / total_words if total_words > 0 else 0
        metrics["repetition_rate"] = total_repetitions / total_words if total_words > 0 else 0
    
    return metrics
