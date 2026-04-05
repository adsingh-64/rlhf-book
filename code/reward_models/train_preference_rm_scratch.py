"""Preference Reward Model — Implement from Scratch

Your goal: train a Bradley-Terry preference model that learns to score
"chosen" responses higher than "rejected" responses.

Loss: -log(sigmoid(r_chosen - r_rejected))

Run with:
    cd code && uv run python -m reward_models.train_preference_rm_scratch --no-wandb --samples 200 --epochs 1
"""

import argparse
import random
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Config
# =============================================================================

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME = "argilla/ultrafeedback-binarized-preferences-cleaned"


# =============================================================================
# Model
# =============================================================================


class PreferenceRewardModel(nn.Module):
    """A reward model that outputs a scalar score for a sequence.

    Architecture:
    - Pretrained causal LM as backbone (frozen or fine-tuned)
    - Linear head: hidden_size -> 1 (the "reward")

    For scoring: take the hidden state at the last non-padding token,
    project it to a scalar.
    """

    def __init__(self, model_id: str = MODEL_ID):
        super().__init__()

        # TODO 1: Load the pretrained causal LM
        # - Use AutoModelForCausalLM.from_pretrained()
        # - Use dtype="bfloat16" for efficiency
        # - Set use_cache = False on the config (we don't need KV cache for training)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="bfloat16",
        )
        self.backbone.config.use_cache=False

        # TODO 2: Create the reward head
        # - A single nn.Linear layer
        # - Input dim: self.backbone.config.hidden_size
        # - Output dim: 1 (scalar reward)
        # - Cast to bfloat16 to match backbone
        self.head = nn.Linear(
            in_features=self.backbone.config.hidden_size,
            out_features=1,
            dtype=torch.bfloat16,
        )

    def get_reward(
        self,
        input_ids: torch.Tensor,      # [batch, seq]
        attention_mask: torch.Tensor,  # [batch, seq]
    ) -> torch.Tensor:                 # [batch]
        """Compute a scalar reward for each sequence in the batch.

        Steps:
        1. Run input through backbone with output_hidden_states=True
        2. Get the last layer's hidden states
        3. For each sequence, find the last non-padding position
        4. Take that position's hidden state and project through self.head
        5. Return shape: (batch_size,)
        """
        # TODO 3: Implement reward scoring
        # Hint: attention_mask.sum(dim=1) - 1 gives you the index of the last real token
        output = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states_final = output.hidden_states[-1] # [batch, seq, d_model]
        last_token_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)

        rewards = self.head(hidden_states_final[batch_indices, last_token_indices]).squeeze(-1)
        return rewards


    def forward(
        self,
        chosen_ids: torch.Tensor,      # [batch, seq_c]
        chosen_mask: torch.Tensor,     # [batch, seq_c]
        rejected_ids: torch.Tensor,    # [batch, seq_r]
        rejected_mask: torch.Tensor,   # [batch, seq_r]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # (scalar, [batch], [batch])
        """Compute Bradley-Terry loss.

        Returns: (loss, r_chosen, r_rejected)
        """
        # TODO 4: Implement the Bradley-Terry loss
        # 1. Get rewards for chosen and rejected
        # 2. loss = -log(sigmoid(r_chosen - r_rejected)).mean()
        r_chosen = self.get_reward(chosen_ids, chosen_mask)
        r_rejected = self.get_reward(rejected_ids, rejected_mask)
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()

        return loss, r_chosen, r_rejected


# =============================================================================
# Data
# =============================================================================


def load_tokenizer(model_id: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_conversation(messages: List[Dict]) -> str:
    """Turn a list of {role, content} dicts into a flat string."""
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def build_dataset(
    tokenizer: AutoTokenizer,
    num_samples: int = 200,
    max_length: int = 512,
    seed: int = 42,
) -> Dataset:
    """Load UltraFeedback and tokenize chosen/rejected pairs.

    Each record should have:
        chosen_ids, chosen_mask, rejected_ids, rejected_mask
    """
    # TODO 5: Implement data loading
    # 1. Load the dataset with load_dataset(DATASET_NAME, split="train")
    # 2. Shuffle and select num_samples examples
    # 3. For each example:
    #    - Extract the chosen and rejected responses (they're lists of message dicts)
    #    - Format them as strings with format_conversation()
    #    - Tokenize both with tokenizer(..., max_length=max_length, truncation=True)
    #    - Store input_ids and attention_mask for both
    # 4. Return Dataset.from_list(records)
    ds = load_dataset(DATASET_NAME, split="train")
    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(num_samples))
    records = []
    for item in ds:
        chosen = item["chosen"]
        rejected = item["rejected"]
        chosen_formatted = format_conversation(chosen)
        rejected_formatted = format_conversation(rejected)
        chosen_tokens = tokenizer(chosen_formatted, max_length=max_length, truncation=True)
        rejected_tokens = tokenizer(rejected_formatted, max_length=max_length, truncation=True)
        records.append({
            "chosen_ids": chosen_tokens["input_ids"],
            "chosen_mask": chosen_tokens["attention_mask"],
            "rejected_ids": rejected_tokens["input_ids"],
            "rejected_mask": rejected_tokens["attention_mask"],
        })

    return Dataset.from_list(records)


def pad_and_collate(batch: List[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:  # each value: [batch, seq]
    """Pad sequences in a batch to the same length.

    For input_ids fields: pad with pad_token_id
    For attention_mask fields: pad with 0
    """
    # TODO 6: Implement collation
    # For each field (chosen_ids, chosen_mask, rejected_ids, rejected_mask):
    #   - Find the max length across the batch
    #   - Pad shorter sequences to that length
    #   - Stack into a tensor
    result = {}
    max_len_chosen = max(len(item["chosen_ids"]) for item in batch)
    max_len_rejected = max(len(item["rejected_ids"]) for item in batch)

    for field in ["chosen_ids", "chosen_mask", "rejected_ids", "rejected_mask"]:
        sequences = [item[field] for item in batch]
        pad_value = pad_token_id if field.endswith("_ids") else 0
        # pad each sequence to max length, then stack into tensor
        for sequence in sequences:
            if field.startswith("chosen"):
                sequence += [pad_value] * (max_len_chosen - len(sequence))
            else:
                sequence += [pad_value] * (max_len_rejected - len(sequence))
        result[field] = torch.stack([torch.tensor(seq) for seq in sequences])

    return result

# =============================================================================
# Training
# =============================================================================


def train(
    samples: int = 200,
    batch_size: int = 2,
    epochs: int = 1,
    lr: float = 1e-6,
    seed: int = 42,
):
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(MODEL_ID)
    dataset = build_dataset(tokenizer, num_samples=samples, seed=seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: pad_and_collate(b, tokenizer.pad_token_id),
    )

    model = PreferenceRewardModel(MODEL_ID).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    # TODO 7: Implement the training loop
    # For each epoch:
    #   For each batch:
    #     1. Move batch to device
    #     2. Forward pass: loss, r_chosen, r_rejected = model(**batch)
    #     3. Backward pass: loss.backward()
    #     4. Optimizer step + zero_grad
    #     5. Track accuracy: (r_chosen > r_rejected).float().mean()
    #     6. Print loss and accuracy every 10 steps
    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, r_chosen, r_rejected = model(**batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            accuracy = (r_chosen > r_rejected).float().mean()
            if step % 10 == 0:
                print(f"Epoch: {epoch} | Loss: {loss} | Accuracy: {accuracy}")

    # --- Demo: score two responses after training ---
    model.eval()
    prompt = "Explain quantum computing in simple terms."
    good = f"user: {prompt}\nassistant: Quantum computers use qubits that can be 0 and 1 simultaneously via superposition, enabling parallel exploration of solutions."
    bad = f"user: {prompt}\nassistant: Computers are electronic. Quantum is physics. I don't know."

    with torch.no_grad():
        for label, text in [("Good", good), ("Bad", bad)]:
            tokens = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            r = model.get_reward(tokens["input_ids"].to(device), tokens["attention_mask"].to(device))
            print(f"{label} response reward: {r.item():.4f}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    train(samples=args.samples, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)
