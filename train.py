import wandb
import os
import torch
from transformers import TrainingArguments, Trainer
from src.modeling import load_lora_model
from src.dataset import load_datasets
from src.utils import load_config, compute_metrics

def main():
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ WARNING: No GPU detected! Training will be very slow.")
    
    cfg = load_config("configs/model.yaml")

    print("Loaded config.")

    # Initialize W&B
    print("Initializing Weights & Biases...")
    

    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"]
    )

    # Load model with LoRA enabled
    print(f"Loading model: {cfg['model_name']} (with LoRA)")
    model = load_lora_model(
        cfg["model_name"],
        num_labels=7,
        lora_cfg=cfg["lora"]
    )

    # Load train + validation datasets
    print("Loading datasets...")
    train_ds, val_ds = load_datasets(
        cfg["data"]["train_path"],
        cfg["data"]["val_path"]
    )
    print(f"   Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    # Set up HuggingFace training arguments
    print("Preparing training arguments...")
    training_args = TrainingArguments(
        output_dir="models/checkpoints",
        num_train_epochs=cfg["training"]["epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["grad_accum_steps"],
        learning_rate=float(cfg["training"]["learning_rate"]),
        fp16=cfg["training"]["fp16"],
        eval_strategy=cfg["training"]["eval_strategy"],
        save_strategy=cfg["training"]["save_strategy"],
        save_total_limit=cfg["training"]["save_total_limit"],
        load_best_model_at_end=cfg["training"]["load_best_model_at_end"],
        metric_for_best_model="accuracy",
        greater_is_better=True, 
        report_to=["wandb"],
        run_name=cfg["wandb"]["run_name"],
        dataloader_num_workers=cfg["data"].get("num_workers", 4),
        dataloader_pin_memory=cfg["data"].get("pin_memory", True),
        dataloader_persistent_workers=cfg["data"].get("persistent_workers", True),
    )

    # Create trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # Start/resume training
    print("Starting training...")
    checkpoint_dir = "models/checkpoints"
    
    os.makedirs(checkpoint_dir, exist_ok=True)

    has_checkpoint = (
        os.path.exists(checkpoint_dir) and 
        any(d.startswith("checkpoint-") for d in os.listdir(checkpoint_dir))
    )
    
    if has_checkpoint:
        print("Resuming from last checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("Starting fresh training...")
        trainer.train()

    # Final evaluation
    print("Running final evaluation...")
    results = trainer.evaluate()
    print("Evaluation results:", results)

    wandb.log(results)
    wandb.finish()
    print("Training complete. W&B run finished.")


if __name__ == "__main__":
    main()
