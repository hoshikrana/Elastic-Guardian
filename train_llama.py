#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EGX LLaMA Training Script (FULL DATASET TRAINING)
"""

import logging
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from egx.api.trainer import EGXTrainer
from egx.api.config import EGXConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger("train_llama")


def format_dataset(dataset, tokenizer):
    """
    Format the ENTIRE Alpaca dataset for causal language model training.
    """

    logger.info("Formatting FULL dataset for training...")

    def format_batch(batch):

        texts = []

        for instruction, inp, output in zip(
            batch["instruction"], batch["input"], batch["output"]
        ):

            if inp.strip() != "":
                text = f"""Below is an instruction that describes a task, paired with an input.

### Instruction:
{instruction}

### Input:
{inp}

### Response:
{output}{tokenizer.eos_token}"""

            else:
                text = f"""Below is an instruction that describes a task.

### Instruction:
{instruction}

### Response:
{output}{tokenizer.eos_token}"""

            texts.append(text)

        tokens = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=512,
        )

        tokens["labels"] = tokens["input_ids"].copy()

        return tokens

    tokenized_dataset = dataset.map(
        format_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing full dataset",
    )

    # Convert to standard list of dicts format that EGX expects
    final_data = []
    for item in tokenized_dataset:
        final_data.append({
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["labels"])
        })
        
    return final_data


def main():

    logger.info("Starting EGX LLaMA Training Pipeline")

    # 1️⃣ Download dataset
    logger.info("Downloading Alpaca dataset...")

    dataset = load_dataset(
        "yahma/alpaca-cleaned",
        split="train"
    )

    logger.info(f"Dataset size: {len(dataset)} samples")

    # 2️⃣ Load tokenizer + model

    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

    logger.info(f"Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="cpu"
    )

    # 3️⃣ Format FULL dataset
    formatted_dataset = format_dataset(dataset, tokenizer)

    # 4️⃣ EGX Configuration

    logger.info("Configuring EGX")

    config = EGXConfig(
        learning_rate=2e-4,
        num_epochs=3,   # full dataset -> fewer epochs needed
        lora_rank=8,
        lora_alpha=16,
        lora_targets=["q_proj", "v_proj"],
        output_dir="./egx_llama_output",

        overrides={
            "batch_size": 2,
            "grad_accum_steps": 8,
            "gradient_checkpointing": True,
        }
    )

    trainer = EGXTrainer(config=config)

    # 5️⃣ Train model

    logger.info("Starting training...")

    result = trainer.train(
        model=model,
        dataset=formatted_dataset
    )

    logger.info("Training finished")

    logger.info(f"Final Loss: {result.get('final_loss', 0):.4f}")
    logger.info(f"Training Time: {result.get('duration_s', 0):.2f}s")


if __name__ == "__main__":
    main()