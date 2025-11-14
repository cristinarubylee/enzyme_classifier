import yaml
from evaluate import load

def load_config(path="configs/model.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

accuracy_metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return accuracy_metric.compute(predictions=preds, references=labels)
