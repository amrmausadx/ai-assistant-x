"""
Main training pipeline with optional hardware optimization
"""
import os
from unittest import result
import mlflow
import traceback
from datetime import datetime
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .core import MLflowCallback, evaluate_model, prepare_tokenization_function
from utils.status import update_training_status
from utils.dependencies import check_dependencies


# -------------------------------
# Entry point
# -------------------------------
def run_training_pipeline(config, optimize=False):
    """
    Run training pipeline.
    If optimize=True, use hardware-aware optimization, otherwise baseline.
    """
    deps = check_dependencies()
    if not deps["training"]:
        update_training_status(
            running=False,
            error="Training dependencies not available",
            message="❌ Please install transformers, torch, and evaluate packages",
        )
        return False

    if optimize:
        return _run_optimized_training(config)
    else:
        return _run_baseline_training(config)


# -------------------------------
# Baseline training
# -------------------------------
def _run_baseline_training(config):
    update_training_status(
        running=True,
        progress=0,
        message="Starting baseline training...",
        error=None,
        start_time=datetime.now(),
        current_epoch=0,
        total_epochs=config.get("num_train_epochs", 2),
        current_loss=None,
        experiment_name = config["experiment_name"],
    )

    try:
        mlflow.set_experiment(config["experiment_name"])
        mlflow.autolog()

        with mlflow.start_run(run_name="gpt2_finetuning") as run:
            update_training_status(run_id=run.info.run_id)

            # log all config params
            for key, value in config.items():
                mlflow.log_param(key, value)

            # tokenizer & model
            update_training_status(progress=5, message="Loading tokenizer and model...")
            tokenizer = AutoTokenizer.from_pretrained(config["model_name"]).to("cuda")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(config["model_name"]).to("cuda")
            model.resize_token_embeddings(len(tokenizer))

            mlflow.log_param("model_parameters", sum(p.numel() for p in model.parameters()))

            # dataset
            update_training_status(progress=10, message="Loading dataset...")
            if not os.path.exists(config["input_csv"]):
                raise FileNotFoundError(f"{config['input_csv']} not found. Please run preprocessing first.")

            dataset = load_dataset("csv", data_files={"data": config["input_csv"]})["data"]
            dataset = dataset.shuffle(seed=config.get("seed", 42))
            split = dataset.train_test_split(test_size=config.get("test_size", 0.05), seed=config.get("seed", 42))

            dataset_dict = DatasetDict({"train": split["train"], "test": split["test"]})
            mlflow.log_metric("train_samples", dataset_dict["train"].num_rows)
            mlflow.log_metric("eval_samples", dataset_dict["test"].num_rows)

            # tokenization
            update_training_status(progress=20, message="Tokenizing data...")
            tokenize_map = prepare_tokenization_function(tokenizer, config["block_size"])

            tokenized_train = dataset_dict["train"].map(
                tokenize_map, batched=True, batch_size=8, remove_columns=dataset_dict["train"].column_names
            )
            tokenized_eval = dataset_dict["test"].map(
                tokenize_map, batched=True, batch_size=8, remove_columns=dataset_dict["test"].column_names
            )
            tokenized_train.set_format(type="torch", columns=["input_ids"])
            tokenized_eval.set_format(type="torch", columns=["input_ids"])
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

            # trainer
            update_training_status(progress=30, message="Setting up trainer...")
            training_args = TrainingArguments(
                output_dir=config["output_dir"],
                overwrite_output_dir=True,
                num_train_epochs=config["num_train_epochs"],
                per_device_train_batch_size=config["train_batch_size"],
                per_device_eval_batch_size=config["eval_batch_size"],
                gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
                do_eval=True,
                eval_steps=config.get("eval_steps", 200),
                save_strategy="steps",
                save_steps=config.get("save_steps", 200),
                logging_strategy="steps",
                logging_steps=config.get("logging_steps", 50),
                learning_rate=config["learning_rate"],
                weight_decay=0.01,
                warmup_steps=min(
                    100, dataset_dict["train"].num_rows // (config["train_batch_size"] * 4)
                ),
                fp16=config.get("fp16", False),
                dataloader_num_workers=config.get("dataloader_num_workers", 0),
                push_to_hub=False,
                save_total_limit=2,
                report_to="none",
                remove_unused_columns=False,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_eval,
                data_collator=data_collator,
                callbacks=[MLflowCallback(run_id=run.info.run_id)],
            )

            # train
            update_training_status(progress=40, message="Training...")
            trainer.train()

            # save + log + register
            _save_and_register(model, tokenizer, config, run.info.run_id)

            # evaluate
            update_training_status(progress=90, message="Evaluating model...")
            evaluate_model(trainer, tokenizer, config)

            update_training_status(
                progress=100,
                message="✅ Training completed successfully!",
                running=False,
                end_time=datetime.now(),
                experiment_name = config["experiment_name"],
            )
            return True

    except Exception as e:
        return _handle_training_error(e)


# -------------------------------
# Optimized training
# -------------------------------
def _run_optimized_training(config):
    update_training_status(
        running=True,
        progress=0,
        message="Detecting hardware and optimizing configuration...",
        error=None,
        start_time=datetime.now(),
        current_epoch=0,
        total_epochs=config.get("num_train_epochs", 2),
        current_loss=None,
        experiment_name = config["experiment_name"]
    )

    try:
        # detect hardware
        hardware_info = detect_hardware()
        use_optimization = config.get("optimize_for_hardware", True)

        if use_optimization or hardware_info["ram_gb"] < 8:
            optimal_config = get_optimal_config(hardware_info)
            config.update(optimal_config)

        mlflow.set_experiment(config["experiment_name"])

        with mlflow.start_run(run_name="optimized_gpt2_finetuning") as run:
            update_training_status(run_id=run.info.run_id)

            # log config & hardware
            for key, value in config.items():
                mlflow.log_param(key, value)
            for k, v in hardware_info.items():
                mlflow.log_param(f"hardware_{k}", v)

            # tokenizer & model
            tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(config["model_name"])
            model.resize_token_embeddings(len(tokenizer))
            mlflow.log_param("model_parameters", sum(p.numel() for p in model.parameters()))

            # dataset
            if not os.path.exists(config["input_csv"]):
                raise FileNotFoundError(f"{config['input_csv']} not found. Please run preprocessing first.")
            dataset = load_dataset("csv", data_files={"data": config["input_csv"]})["data"]
            if config.get("max_samples") and len(dataset) > config["max_samples"]:
                dataset = create_demo_dataset(dataset, config["max_samples"])
            dataset = dataset.shuffle(seed=config.get("seed", 42))
            split = dataset.train_test_split(test_size=config.get("test_size", 0.05), seed=config.get("seed", 42))
            dataset_dict = DatasetDict({"train": split["train"], "test": split["test"]})

            # tokenize
            tokenize_map = prepare_tokenization_function(tokenizer, config["block_size"])
            tokenized_train = dataset_dict["train"].map(
                tokenize_map, batched=True, batch_size=4, remove_columns=dataset_dict["train"].column_names
            )
            tokenized_eval = dataset_dict["test"].map(
                tokenize_map, batched=True, batch_size=4, remove_columns=dataset_dict["test"].column_names
            )
            tokenized_train.set_format(type="torch", columns=["input_ids"])
            tokenized_eval.set_format(type="torch", columns=["input_ids"])
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

            # trainer
            training_args = TrainingArguments(
                output_dir=config["output_dir"],
                overwrite_output_dir=True,
                num_train_epochs=config["num_train_epochs"],
                per_device_train_batch_size=config["train_batch_size"],
                per_device_eval_batch_size=config["eval_batch_size"],
                gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
                eval_strategy="steps",
                eval_steps=config.get("eval_steps", 200),
                save_strategy="steps",
                save_steps=config.get("save_steps", 200),
                logging_strategy="steps",
                logging_steps=config.get("logging_steps", 50),
                learning_rate=config["learning_rate"],
                fp16=config.get("fp16", False),
                dataloader_num_workers=config.get("dataloader_num_workers", 0),
                push_to_hub=False,
                save_total_limit=2,
                report_to="none",
                remove_unused_columns=False,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_eval,
                data_collator=data_collator,
                callbacks=[MLflowCallback(run_id=run.info.run_id)],
            )

            # train
            trainer.train()

            # save + log + register
            _save_and_register(model, tokenizer, config, run.info.run_id)

            # evaluate
            evaluate_model(trainer, tokenizer, config)

            update_training_status(
                progress=100,
                message="✅ Optimized training completed!",
                running=False,
                end_time=datetime.now(),
                experiment_name = config["experiment_name"],
            )
            return True

    except Exception as e:
        return _handle_training_error(e)
    


# -------------------------------
# Helpers
# -------------------------------
def _save_and_register(model, tokenizer, config, run_id,task="text-generation"):
    """Save locally, log artifacts, and register model in MLflow"""
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    mlflow.log_artifacts(config["output_dir"], artifact_path="model")
    model_uri = f"runs:/{run_id}/model"
    # log model + tokenizer together
    mlflow.transformers.log_model(transformers_model={"model": model, "tokenizer": tokenizer}, artifact_path="model", task=task)
    result = mlflow.register_model(model_uri, config["experiment_name"]+"-model")
    update_training_status(message="Model registered in MLflow Model Registry", progress=95)
    print(f"Registered as {result.name}, version {result.version}")



def _handle_training_error(e):
    error_message = str(e)
    update_training_status(
        running=False,
        error=error_message,
        message=f"❌ Training failed: {error_message}",
        end_time=datetime.now()
    )
    try:
        
        mlflow.log_param("status", "failed")
        mlflow.log_param("error_message", str(e))
        mlflow.log_text(traceback.format_exc(), "error_traceback.txt")
        mlflow.end_run(status="FAILED")
    except:
        pass
    print(f"Training failed with error: {e}")
    traceback.print_exc()
    return False
