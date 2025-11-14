from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

def load_lora_model(model_name: str, num_labels: int, lora_cfg: dict):
    base = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type="SEQ_CLS"
    )
    
    model = get_peft_model(base, config)
    model.print_trainable_parameters()
    return model
