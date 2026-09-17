import torch
from torch import nn
from transformers import AutoModel


class TwoHeadClassifier(nn.Module):
    def __init__(self, model_name, num_labels=3, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.stance_head = nn.Linear(hidden, num_labels)
        self.premise_head = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        return self.stance_head(pooled), self.premise_head(pooled)

    @staticmethod
    def loss(logits, labels):
        stance_logits, premise_logits = logits
        criterion = nn.functional.cross_entropy
        return criterion(stance_logits, labels["stance"]) + criterion(premise_logits, labels["premise"])


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
