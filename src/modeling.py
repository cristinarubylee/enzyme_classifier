import torch
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

def load_lora_model(model_name: str, num_labels: int, lora_cfg: dict):
    """Load a transformer model with LoRA adapters"""
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        trust_remote_code=True
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=lora_cfg.get("dropout", 0.1),
        bias=lora_cfg.get("bias", "none"),
        task_type="SEQ_CLS"  # Sequence classification
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Shows how many params are trainable
    
    # Move to GPU
    model = model.to(device)
    
    return model