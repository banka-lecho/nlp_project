import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

TASKS = ("stance", "premise")


def mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def corn_probabilities(logits):
    """Логиты условных вероятностей P(y > k | y > k-1) в распределение по классам.

    cum[:, k] = P(y > k), поэтому P(y = k) = P(y > k-1) - P(y > k).
    """
    cum = torch.cumprod(torch.sigmoid(logits.float()), dim=1)
    edges = torch.cat([torch.ones_like(cum[:, :1]), cum, torch.zeros_like(cum[:, :1])], dim=1)
    return edges[:, :-1] - edges[:, 1:]


def corn_loss(logits, target, weight=None):
    """CORN (Shi et al., 2021): каждый узел k учится на подвыборке с y > k-1.

    Обычный softmax считает ошибки Bad→Neutral и Bad→Good одинаковыми, хотя классы
    упорядочены. В датасете крайние путаницы и так редки (79 пар против 457 соседних),
    и ординальная голова кодирует этот порядок явно.
    """
    sample_weight = None if weight is None else weight[target]
    mask = torch.ones_like(target, dtype=torch.bool)
    total = logits.new_zeros(())
    for k in range(logits.size(1)):
        if not mask.any():
            break
        bce = F.binary_cross_entropy_with_logits(
            logits[mask, k], (target[mask] > k).float(), reduction="none"
        )
        if sample_weight is None:
            total = total + bce.mean()
        else:
            total = total + (bce * sample_weight[mask]).sum() / sample_weight[mask].sum()
        mask = mask & (target > k)
    return total


class TwoHeadClassifier(nn.Module):
    def __init__(self, model_name, num_labels=3, dropout=0.1, pooling="cls", head="softmax",
                 class_weights=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        self.head = head
        hidden = self.encoder.config.hidden_size
        outputs = num_labels if head == "softmax" else num_labels - 1
        self.dropout = nn.Dropout(dropout)
        self.stance_head = nn.Linear(hidden, outputs)
        self.premise_head = nn.Linear(hidden, outputs)
        for task in TASKS:
            weights = None if class_weights is None else torch.as_tensor(class_weights[task], dtype=torch.float)
            self.register_buffer(f"{task}_weights", weights, persistent=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = mean_pool(hidden, attention_mask) if self.pooling == "mean" else hidden[:, 0]
        pooled = self.dropout(pooled)
        return self.stance_head(pooled), self.premise_head(pooled)

    def loss(self, logits, labels):
        total = 0.0
        for task, task_logits in zip(TASKS, logits):
            weight = getattr(self, f"{task}_weights")
            if self.head == "softmax":
                total = total + F.cross_entropy(task_logits, labels[task], weight=weight)
            else:
                total = total + corn_loss(task_logits, labels[task], weight)
        return total

    def probabilities(self, logits):
        if self.head == "softmax":
            return logits.float().softmax(-1)
        return corn_probabilities(logits)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
