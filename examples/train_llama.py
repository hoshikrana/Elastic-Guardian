#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EGX LLaMA Training Script — Complete Showcase

Demonstrates the full EGX v2.0 flexible training framework:
- HuggingFace dataset integration (lazy, memory-efficient)
- Callback system (logging, early stopping, throughput)
- Evaluation during training
- Text generation after training
- Custom metrics
- All state-of-the-art training patterns
"""

import logging
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from egx.api.trainer import EGXTrainer
from egx.api.config import EGXConfig
from egx.api.callbacks import (
    EarlyStoppingCallback,
    LoggingCallback,
    NaNDetectionCallback,
    ThroughputCallback,
    CheckpointCallback,
)
from egx.data.hf_adapter import HFDatasetAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger("train_llama")


def format_alpaca_dataset(dataset, tokenizer, max_length=512):
    """
    Format the Alpaca dataset for causal language model training.
    Uses the lazy HFDatasetAdapter — no memory-heavy list conversion.
    """

    logger.info("Formatting dataset for training...")

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
            max_length=max_length,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(
        format_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    # Use lazy adapter — converts to tensor on-the-fly (no 51K list in memory)
    return HFDatasetAdapter(tokenized_dataset)


def main():
    logger.info("Starting EGX LLaMA Training Pipeline (v2.0)")

    # ─── 1. Download dataset ───
    logger.info("Downloading Alpaca dataset...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    logger.info(f"Dataset size: {len(dataset)} samples")

    # Split into train / eval
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_data = split["train"]
    eval_data = split["test"]
    logger.info(f"Train: {len(train_data)} | Eval: {len(eval_data)}")

    # ─── 2. Load tokenizer + model ───
    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    logger.info(f"Loading model via EGX Component: {model_id}")

    from egx.models.loader import AutoModelLoader

    model, tokenizer = AutoModelLoader.from_pretrained(
        pretrained_model_name_or_path=model_id,
        dtype=torch.float16,
        device_map="auto",
    )

    # ─── 3. Format datasets (lazy, memory-efficient) ───
    train_dataset = format_alpaca_dataset(train_data, tokenizer)
    eval_dataset = format_alpaca_dataset(eval_data, tokenizer)

    # ─── 4. EGX Configuration ───
    logger.info("Configuring EGX v2.0")

    config = EGXConfig(
        # Training
        learning_rate=2e-4,
        num_epochs=3,
        batch_size=2,
        gradient_accumulation_steps=8,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        # PEFT
        lora_rank=8,
        lora_alpha=16,
        lora_targets=["q_proj", "v_proj"],
        # Evaluation (during training)
        eval_strategy="epoch",
        compute_perplexity=True,
        # Logging
        logging_steps=50,
        # Early stopping
        early_stopping_patience=2,
        early_stopping_threshold=0.01,
        # Checkpointing
        output_dir="./egx_llama_output",
        save_steps=500,
    )

    # ─── 5. Create trainer with callbacks ───
    trainer = EGXTrainer(
        config=config,
        callbacks=[
            LoggingCallback(log_every_n_steps=50),
            NaNDetectionCallback(halt_on_nan=False, max_nan_count=10),
            ThroughputCallback(log_every_n_steps=100),
            CheckpointCallback(save_every_n_steps=500, save_best=True),
            EarlyStoppingCallback(patience=2, min_delta=0.01),
        ],
    )

    # ─── 6. Train model ───
    logger.info("Starting training...")
    result = trainer.train(
        model=model,
        dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("Training finished!")
    logger.info(f"Final Loss: {result.get('final_loss', 0):.4f}")
    logger.info(f"Best Eval Loss: {result.get('best_eval_loss', 'N/A')}")
    logger.info(f"Training Time: {result.get('duration_s', 0):.2f}s")
    logger.info(f"Global Steps: {result.get('global_steps', 0)}")

    # ─── 7. Standalone evaluation ───
    logger.info("Running standalone evaluation...")
    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    logger.info(f"Eval Loss: {eval_metrics.get('eval_loss', 0):.4f}")
    logger.info(f"Eval Perplexity: {eval_metrics.get('eval_perplexity', 0):.2f}")

    # ─── 8. Generation demo ───
    logger.info("Running generation demo...")
    prompts = [
        "Below is an instruction that describes a task.\n\n### Instruction:\nExplain what machine learning is in simple terms.\n\n### Response:",
        "Below is an instruction that describes a task.\n\n### Instruction:\nWrite a Python function to reverse a string.\n\n### Response:",
    ]

    try:
        generated = trainer.generate(
            prompts=prompts,
            tokenizer=tokenizer,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
        )
        for i, text in enumerate(generated):
            logger.info(f"Generation {i + 1}: {text[:200]}...")
    except Exception as e:
        logger.warning(f"Generation demo skipped: {e}")

    logger.info("EGX LLaMA pipeline complete!")


if __name__ == "__main__":
    main()
