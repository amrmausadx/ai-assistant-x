import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import mlflow
from utils.status import update_gan_training_status
from training.core import MLflowCallback, generate_samples, evaluate_model, prepare_tokenization_function
from datetime import datetime
from training.pipeline import _handle_training_error,_save_and_register
import math
from urllib.parse import urlparse,unquote
import os
# -------------------------
# Discriminator Definition
# -------------------------
class Discriminator(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
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
# -------------------------
# GAN Training Function
# -------------------------
def run_gan_training(config, real_texts=None):
    """
    GAN training loop:
    - Generator: pre-finetuned LM (GPT-2, etc.)
    - Discriminator: classification head on top of transformer encoder
    - Config includes learning rates, batch size, epochs, etc.
    """
    device = None
    try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    except:
            device = "cpu"
    mlflow.set_experiment(config['experiment_name'])
    mlflow.autolog()
    with mlflow.start_run(run_name="GAN_Training") as run:
        # log all config params
        for key, value in config.items():
            mlflow.log_param(key, value)
        try:
            epochs = int(config['gan_epochs'])
            batch_size = int(config['gan_batch_size'])
            lr_d = float(config['learning_rate_d'])
            lr_g = float(config['learning_rate_g'])

            update_gan_training_status(
                running=True,
                progress=0,
                message="Starting GAN training",
                start_time=datetime.utcnow(),
                run_id=run.info.run_id,
                experiment_name=config['experiment_name'],
                total_epochs=epochs
            )

            # -------------------------
            # Load Generator (from MLflow registry)
            # -------------------------
            client = mlflow.tracking.MlflowClient()
            latest = client.get_latest_versions(config["experiment_name"]+"-model")[-1]
            artifact_uri = latest.source
            artifact_path = resolve_artifact_path(artifact_uri)
            #print(latest, artifact_uri, artifact_path)

            generator = AutoModelForCausalLM.from_pretrained(artifact_path).to(device)
            tokenizer = AutoTokenizer.from_pretrained(artifact_path)
            # ensure pad token exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token  # safe fallback for decoder-only models

            # set left padding for decoder-only models
            tokenizer.padding_side = "left"
            # -------------------------
            # Init Discriminator
            # -------------------------
            disc_tokenizer = AutoTokenizer.from_pretrained(config["discriminator_model"])
            discriminator = Discriminator(model_name=config["discriminator_model"]).to(device)
            if disc_tokenizer.pad_token is None:
                disc_tokenizer.pad_token = disc_tokenizer.eos_token  # safe fallback for decoder-only models

            # set left padding for decoder-only models
            tokenizer.padding_side = "left"

            d_optimizer = optim.Adam(discriminator.parameters(), lr=lr_d)
            g_optimizer = optim.Adam(generator.parameters(), lr=lr_g)

            # -------------------------
            # Training Loop
            # -------------------------
            real_texts = config["real_texts"].split(";")
            prompts = config.get("prompts").split(";")
            real_texts = [t.strip() for t in real_texts if isinstance(t, str) and t.strip() != ""]
            prompts = [t.strip() for t in prompts if isinstance(t, str) and t.strip() != ""]

            temperatures = [0.7,0.8,0.9]
            fake_samples = []
            for epoch in range(epochs):

                # ---- 1. Generate fake samples ----
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    for temp in temperatures:
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
                        gen_text = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
                        fake_samples.append(gen_text)
                
                # ---- 2. Discriminator step ----
                all_texts = real_texts + fake_samples
                all_texts = [t.strip() for t in all_texts if isinstance(t, str) and t.strip() != ""]
                fake_samples = [t.strip() for t in fake_samples if isinstance(t, str) and t.strip() != ""]

                labels = [1] * len(real_texts) + [0] * len(fake_samples)

                enc = disc_tokenizer(all_texts, return_tensors="pt", padding=True, truncation=True)
                enc = {k: v.to(device) for k,v in enc.items()}
                

                labels = torch.tensor(labels).to(device)

                logits = discriminator(enc["input_ids"], attention_mask=enc["attention_mask"])
                loss_fn = nn.CrossEntropyLoss()
                d_loss = loss_fn(logits, labels)

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()


                # ---- 3. Generator step ----
                generator.train()
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

                outputs = generator(**inputs)
                logits = outputs.logits
                log_probs = torch.log_softmax(logits, dim=-1)
                chosen_log_probs = log_probs.gather(-1, inputs["input_ids"].unsqueeze(-1)).squeeze(-1)
                seq_log_prob = chosen_log_probs.mean(dim=1)

                sample_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in inputs["input_ids"]]
                enc_fake = disc_tokenizer(sample_texts, return_tensors="pt", padding=True, truncation=True)

                with torch.no_grad():
                    d_scores = torch.softmax(discriminator(enc_fake["input_ids"], attention_mask=enc_fake["attention_mask"]), dim=-1)
                    rewards = d_scores[:, 1]  # realness score

                g_loss = -(seq_log_prob * rewards).mean()
                #calculate perplexity
                # calculate perplexity - handle inf by converting to None
                perplexity = None
                try:
                    perplexity = torch.exp(g_loss).item() if g_loss < 20 else None
                except:
                    perplexity = None
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                # ---- 4. Log + Update Status ----
                progress = int(((epoch + 1) / epochs) * 100)
                update_gan_training_status(
                    running=True,
                    progress=progress,
                    message=f"Epoch {epoch+1}/{epochs} complete",
                    current_epoch=epoch+1,
                    total_epochs=epochs,
                    current_d_loss=d_loss.item(),
                    current_g_loss=g_loss.item(),
                    current_perplexity=perplexity,
                    run_id=run.info.run_id,
                    experiment_name=config['experiment_name']
                )

                mlflow.log_metric("d_loss", d_loss.item(), step=epoch+1)
                mlflow.log_metric("g_loss", g_loss.item(), step=epoch+1)
                

            # -------------------------
            # Save final generator
            # -------------------------
            _save_and_register(generator, tokenizer, config, run.info.run_id)

            update_gan_training_status(
                running=False,
                progress=100,
                message="GAN training complete",
                end_time=datetime.utcnow(),
                run_id=run.info.run_id,
                experiment_name=config['experiment_name'],
                
            )
            return discriminator, generator

        except Exception as e:
            update_gan_training_status(
                running=False,
                progress=0,
                message=f"GAN Training failed: {e}",
                error=str(e),
                end_time=datetime.utcnow(),
                run_id=run.info.run_id,
                experiment_name=config['experiment_name']
            )
            _handle_training_error(e)
            mlflow.end_run(status="FAILED")
            raise

def evaluate_model(trainer, tokenizer, config):
    """Evaluate the trained model"""
    try:
        # Basic evaluation metrics
        metrics = trainer.evaluate()
        eval_loss = metrics.get("eval_loss", None)
        
        if eval_loss is not None:
            perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")
            mlflow.log_metric("perplexity", perplexity)
        
        # Generate sample texts
        #generated_samples = generate_samples(trainer.model, tokenizer, config['output_dir'])
        
        return metrics,None
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {}, []

