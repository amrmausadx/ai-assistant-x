import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import mlflow
from utils.status import update_gan_training_status
from training.core import MLflowCallback, generate_samples, evaluate_model, prepare_tokenization_function
from datetime import datetime
from training.pipeline import _handle_training_error, _save_and_register
import math
from urllib.parse import urlparse, unquote
import os
import torch.nn.functional as F

# -------------------------
# Discriminator Definition
# -------------------------
class Discriminator(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def resolve_artifact_path(uri: str) -> str:
    """
    Recursively resolve MLflow artifact URIs into a valid local filesystem path.
    Handles file:// prefixes, %20 spaces, and Windows/Linux differences.
    """
    if not uri:
        raise ValueError("Empty artifact URI")

    # Decode URL encodings (e.g. %20 -> space)
    uri = unquote(uri)

    # Case 1: starts with file:// -> strip and recurse
    if uri.startswith("file://"):
        inner = urlparse(uri).path
        # On Windows, strip leading "/" (e.g. /D:/path -> D:/path)
        if os.name == "nt" and inner.startswith("/"):
            inner = inner.lstrip("/")
        return resolve_artifact_path(inner)

    # Case 2: looks like a directory path
    if os.path.isdir(uri):
        return os.path.normpath(uri)

    # If still unresolved, raise
    raise FileNotFoundError(f"Could not resolve artifact path: {uri}")


def load_generator_model(config, device):
    """
    Load generator model from local directory or MLflow registry.
    Tries local path first, then falls back to registry.
    """
    # Option 1: Try local directory first
    model_path = config.get("generator_model_path", "./gpt2_finetuned/")
    
    if os.path.exists(model_path) and os.path.isdir(model_path):
        print(f"Loading generator from local path: {model_path}")
        try:
            generator = AutoModelForCausalLM.from_pretrained(model_path).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(f"✓ Successfully loaded generator from {model_path}")
            return generator, tokenizer
        except Exception as e:
            print(f"Failed to load from local path: {e}")
    
    # Option 2: Try MLflow registry
    print("Attempting to load generator from MLflow registry...")
    try:
        client = mlflow.tracking.MlflowClient()
        model_name = config.get("generator_model_name", config["experiment_name"] + "-model")
        
        latest_versions = client.get_latest_versions(model_name)
        if not latest_versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        
        latest = latest_versions[-1]
        artifact_uri = latest.source
        artifact_path = resolve_artifact_path(artifact_uri)
        
        print(f"Loading from MLflow artifact: {artifact_path}")
        generator = AutoModelForCausalLM.from_pretrained(artifact_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(artifact_path)
        print(f"✓ Successfully loaded generator from registry")
        return generator, tokenizer
        
    except Exception as e:
        raise ValueError(
            f"Could not load generator model from local path '{model_path}' or MLflow registry: {e}\n"
            f"Please ensure the model exists at the specified location or is registered in MLflow."
        )


def calculate_perplexity_on_texts(model, tokenizer, texts, device):
    """
    Calculate perplexity on a set of texts.
    Returns average perplexity across all texts.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for text in texts:
            if not text.strip():
                continue
                
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            count += 1
    
    if count == 0:
        return float('inf')
    
    avg_loss = total_loss / count
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    return perplexity


def compute_generator_loss_reinforce(generator, tokenizer, disc_tokenizer, discriminator,
                                      prompts, device, temperature=0.8):
    generator.train()
    discriminator.eval()

    # Step 1: Generate samples (no grad)
    gen_inputs = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=128
    )
    gen_inputs = {k: v.to(device) for k, v in gen_inputs.items()}

    with torch.no_grad():
        gen_outputs = generator.generate(
            **gen_inputs,
            max_length=200, min_length=50,
            do_sample=True, temperature=temperature,
            top_k=50, top_p=0.95,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

    gen_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in gen_outputs]

    # Step 2: Get discriminator rewards (no grad)
    with torch.no_grad():
        disc_inputs = disc_tokenizer(
            gen_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        )
        disc_inputs = {k: v.to(device) for k, v in disc_inputs.items()}
        d_logits = discriminator(disc_inputs["input_ids"], attention_mask=disc_inputs["attention_mask"])
        rewards = F.softmax(d_logits, dim=-1)[:, 1]

    # Step 3: Compute log probs using original gen_outputs (no re-tokenization!)
    outputs = generator(input_ids=gen_outputs)
    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = gen_outputs[..., 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Mask padding
    pad_id = tokenizer.pad_token_id
    non_pad_mask = (shift_labels != pad_id).float()
    token_log_probs = token_log_probs * non_pad_mask
    seq_lengths = non_pad_mask.sum(dim=1).clamp(min=1)
    avg_log_probs = token_log_probs.sum(dim=1) / seq_lengths

    # Step 4: REINFORCE loss with variance normalization
    baseline = rewards.mean().detach()
    advantages = rewards - baseline

    # Guard: if no variance, skip generator update
    if advantages.abs().max() < 1e-8:
        return torch.tensor(0.0, requires_grad=True, device=device), rewards.mean().item(), gen_texts

    advantages = (advantages / (advantages.std() + 1e-8)).detach()
    policy_loss = -(advantages * avg_log_probs).mean()

    return policy_loss, rewards.mean().item(), gen_texts


# -------------------------
# GAN Training Function
# -------------------------
def run_gan_training(config, real_texts=None):
    """
    GAN training loop using REINFORCE (policy gradient):
    - Generator: pre-finetuned LM (GPT-2, etc.)
    - Discriminator: classification head on transformer encoder
    - Generator is trained to maximize discriminator's "real" score
    """
    device = None
    generator = None
    discriminator = None
    tokenizer = None
    disc_tokenizer = None
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    mlflow.set_experiment(config['experiment_name'])
    mlflow.autolog(disable=True)  # Disable autolog to avoid conflicts
    
    with mlflow.start_run(run_name="GAN_Training") as run:
        try:
            # -------------------------
            # Validate Config
            # -------------------------
            required_keys = ['gan_epochs', 'gan_batch_size', 'learning_rate_d', 'learning_rate_g', 
                           'real_texts', 'prompts', 'experiment_name']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required config key: {key}")
            
            epochs = int(config['gan_epochs'])
            batch_size = int(config['gan_batch_size'])
            lr_d = float(config['learning_rate_d'])
            lr_g = float(config['learning_rate_g'])
            
            # Parse and validate real texts and prompts
            real_texts = config.get("real_texts", "").split(";")
            real_texts = [t.strip() for t in real_texts if t.strip()]
            if not real_texts:
                raise ValueError("No real_texts provided in config. Please provide semicolon-separated texts.")
            
            prompts = config.get("prompts", "").split(";")
            prompts = [t.strip() for t in prompts if t.strip()]
            if not prompts:
                raise ValueError("No prompts provided in config. Please provide semicolon-separated prompts.")
            
            # Log all config params
            mlflow.log_params({
                "gan_epochs": epochs,
                "gan_batch_size": batch_size,
                "learning_rate_d": lr_d,
                "learning_rate_g": lr_g,
                "num_real_texts": len(real_texts),
                "num_prompts": len(prompts),
                "discriminator_model": config.get("discriminator_model", "distilbert-base-uncased"),
                "device": device
            })
            
            update_gan_training_status(
                running=True,
                progress=5,
                message="Loading models...",
                start_time=datetime.utcnow(),
                run_id=run.info.run_id,
                experiment_name=config['experiment_name'],
                total_epochs=epochs
            )

            # -------------------------
            # Load Generator
            # -------------------------
            generator, tokenizer = load_generator_model(config, device)
            
            # Ensure pad token exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                generator.config.pad_token_id = tokenizer.eos_token_id
            
            # Set left padding for decoder-only models (important for batch generation)
            tokenizer.padding_side = "left"
            
            update_gan_training_status(
                running=True,
                progress=15,
                message="Generator loaded successfully",
            )

            # -------------------------
            # Initialize Discriminator
            # -------------------------
            disc_model_name = config.get("discriminator_model", "distilbert-base-uncased")
            disc_tokenizer = AutoTokenizer.from_pretrained(disc_model_name)
            discriminator = Discriminator(model_name=disc_model_name).to(device)
            
            if disc_tokenizer.pad_token is None:
                disc_tokenizer.pad_token = disc_tokenizer.eos_token
            
            # Discriminator uses standard (right) padding
            disc_tokenizer.padding_side = "right"
            
            update_gan_training_status(
                running=True,
                progress=25,
                message="Discriminator initialized",
            )

            # -------------------------
            # Optimizers
            # -------------------------
            d_optimizer = optim.AdamW(discriminator.parameters(), lr=lr_d, weight_decay=0.01)
            g_optimizer = optim.AdamW(generator.parameters(), lr=lr_g, weight_decay=0.01)
            
            # Optional: Learning rate schedulers
            d_scheduler = optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=epochs)
            g_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=epochs)

            # -------------------------
            # Training Loop
            # -------------------------
            temperatures = [0.7, 0.8, 0.9]
            loss_fn = nn.CrossEntropyLoss()
            
            print(f"\nStarting GAN training for {epochs} epochs...")
            print(f"Real texts: {len(real_texts)}, Prompts: {len(prompts)}")
            
            for epoch in range(epochs):
                epoch_start_time = datetime.utcnow()
                
                # -------------------------
                # 1. Generate Fake Samples (for discriminator training)
                # -------------------------
                generator.eval()
                fake_samples = []
                
                with torch.no_grad():
                    for temp in temperatures:
                        # Tokenize prompts
                        inputs = tokenizer(
                            prompts, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True,
                            max_length=128
                        )
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        # Generate samples
                        outputs = generator.generate(
                            **inputs,
                            max_length=200,
                            min_length=50,
                            do_sample=True,
                            top_k=50,
                            top_p=0.95,
                            temperature=temp,
                            num_return_sequences=1,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            repetition_penalty=1.2,
                            no_repeat_ngram_size=3,
                        )
                        
                        # Decode generated texts
                        gen_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
                        fake_samples.extend(gen_texts)
                
                # Clean up fake samples
                fake_samples = [t.strip() for t in fake_samples if t.strip()]
                
                if not fake_samples:
                    print(f"Warning: No fake samples generated at epoch {epoch+1}")
                    continue
                
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"  Generated {len(fake_samples)} fake samples")
                
                # -------------------------
                # 2. Train Discriminator
                # -------------------------
                discriminator.train()
                
                # Combine real and fake samples
                all_texts = real_texts + fake_samples
                labels = [1] * len(real_texts) + [0] * len(fake_samples)
                
                # Shuffle for better training
                combined = list(zip(all_texts, labels))
                import random
                random.shuffle(combined)
                all_texts, labels = zip(*combined)
                all_texts = list(all_texts)
                labels = list(labels)
                
                # Tokenize for discriminator
                enc = disc_tokenizer(
                    all_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
                
                # Forward pass
                d_optimizer.zero_grad()
                logits = discriminator(enc["input_ids"], attention_mask=enc["attention_mask"])
                d_loss = loss_fn(logits, labels_tensor)
                
                # Backward pass
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                d_optimizer.step()
                
                # Calculate discriminator accuracy
                predictions = torch.argmax(logits, dim=-1)
                d_accuracy = (predictions == labels_tensor).float().mean().item()
                
                print(f"  D Loss: {d_loss.item():.4f}, D Accuracy: {d_accuracy:.2%}")
                
                # -------------------------
                # 3. Train Generator (REINFORCE)
                # -------------------------
                g_optimizer.zero_grad()
                
                # Compute generator loss using REINFORCE
                g_loss, avg_reward, gen_texts = compute_generator_loss_reinforce(
                    generator=generator,
                    tokenizer=tokenizer,
                    disc_tokenizer=disc_tokenizer,
                    discriminator=discriminator,
                    prompts=prompts,
                    device=device,
                    temperature=0.8
                )
                
                # Backward pass
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                g_optimizer.step()
                
                # Step schedulers
                d_scheduler.step()
                g_scheduler.step()
                
                print(f"  G Loss: {g_loss.item():.4f}, Avg Reward: {avg_reward:.4f}")
                
                # -------------------------
                # 4. Calculate Perplexity
                # -------------------------
                perplexity = calculate_perplexity_on_texts(generator, tokenizer, gen_texts, device)
                print(f"  Perplexity: {perplexity:.2f}")
                
                # -------------------------
                # 5. Log Metrics & Update Status
                # -------------------------
                progress = int(((epoch + 1) / epochs) * 100)
                
                mlflow.log_metrics({
                    "d_loss": d_loss.item(),
                    "d_accuracy": d_accuracy,
                    "g_loss": g_loss.item(),
                    "avg_reward": avg_reward,
                    "perplexity": perplexity if perplexity != float('inf') else None,
                    "learning_rate_d": d_optimizer.param_groups[0]['lr'],
                    "learning_rate_g": g_optimizer.param_groups[0]['lr'],
                }, step=epoch + 1)
                
                # Log sample generated text
                if epoch % 5 == 0 or epoch == epochs - 1:
                    sample_text = gen_texts[0][:500] if gen_texts else "No text generated"
                    mlflow.log_text(sample_text, f"sample_epoch_{epoch+1}.txt")
                
                update_gan_training_status(
                    running=True,
                    progress=progress,
                    message=f"Epoch {epoch+1}/{epochs} complete",
                    current_epoch=epoch + 1,
                    total_epochs=epochs,
                    current_d_loss=d_loss.item(),
                    current_g_loss=g_loss.item(),
                    current_perplexity=perplexity if perplexity != float('inf') else None,
                    run_id=run.info.run_id,
                    experiment_name=config['experiment_name']
                )
                
                # Clean up tensors to save memory
                del enc, labels_tensor, logits
                if device == "cuda":
                    torch.cuda.empty_cache()

            # -------------------------
            # 6. Save Final Models
            # -------------------------
            print("\nSaving final generator model...")
            _save_and_register(generator, tokenizer, config, run.info.run_id)
            
            # Optionally save discriminator too
            disc_save_path = os.path.join(config.get('output_dir', './output'), 'discriminator')
            os.makedirs(disc_save_path, exist_ok=True)
            discriminator.encoder.save_pretrained(disc_save_path)
            disc_tokenizer.save_pretrained(disc_save_path)
            torch.save(discriminator.classifier.state_dict(), os.path.join(disc_save_path, 'classifier.pt'))
            mlflow.log_artifacts(disc_save_path, artifact_path="discriminator")
            
            print("✓ Models saved successfully")

            update_gan_training_status(
                running=False,
                progress=100,
                message="✅ GAN training complete",
                end_time=datetime.utcnow(),
                run_id=run.info.run_id,
                experiment_name=config['experiment_name'],
            )
            
            return discriminator, generator

        except Exception as e:
            error_msg = f"GAN Training failed: {str(e)}"
            print(f"\n❌ {error_msg}")
            import traceback
            traceback.print_exc()
            
            update_gan_training_status(
                running=False,
                progress=0,
                message=error_msg,
                error=str(e),
                end_time=datetime.utcnow(),
                run_id=run.info.run_id,
                experiment_name=config['experiment_name']
            )
            
            mlflow.end_run(status="FAILED")
            raise
        
        finally:
            # Clean up to free memory
            try:
                if generator is not None:
                    del generator
                if discriminator is not None:
                    del discriminator
                if tokenizer is not None:
                    del tokenizer
                if disc_tokenizer is not None:
                    del disc_tokenizer
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass


def evaluate_model(trainer, tokenizer, config):
    """Evaluate the trained model"""
    try:
        # Basic evaluation metrics
        metrics = trainer.evaluate()
        eval_loss = metrics.get("eval_loss", None)
        
        if eval_loss is not None:
            perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")
            mlflow.log_metric("perplexity", perplexity)
        
        return metrics, None
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {}, []
