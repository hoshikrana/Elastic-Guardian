"""
EGX Framework - Live Presentation Demo

This script demonstrates the EGX framework training a real-world model (DistilGPT2)
on a real dataset (Wikitext). It automatically handles memory estimation, hardware
probing, and mixed-precision LoRA injection out of the box!

Run: `python presentation_demo.py`
"""

import logging
import os
import sys

# EGX Framework Imports
from egx.api.config import EGXConfig
from egx.api.trainer import EGXTrainer
from egx.api.callbacks import LoggingCallback, ThroughputCallback

try:
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
    )
except ImportError:
    print("Pre-requisites missing. Run: pip install datasets transformers accelerate")
    sys.exit(1)

# Set up presentation logger
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("presentation")


def prepare_dataset(tokenizer, model_name: str):
    logger.info("📚 Downloading real-world dataset (wikitext-2-raw-v1)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:5%]")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=128, padding="max_length"
        )

    logger.info("⚙️ Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    # Needs labels for Causal LM
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    return tokenized_datasets.map(add_labels, batched=True)


def main():
    logger.info("=" * 60)
    logger.info("🚀 EGX FRAMEWORK - PRODUCTION TRAINING SHOWCASE")
    logger.info("=" * 60)

    MODEL_ID = "distilgpt2"

    logger.info(f"\n[1/4] Pulling {MODEL_ID} from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    logger.info("\n[2/4] Preparing data pipeline...")
    dataset = prepare_dataset(tokenizer, MODEL_ID)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    logger.info("\n[3/4] Configuring EGX Orchestrator (Phase 5-10)...")
    config = EGXConfig(
        model_name_or_path=MODEL_ID,
        max_steps=5,  # Run super short training for presentation
        batch_size=2,  # Reduce memory/compute for fast CPU output
        gradient_accumulation_steps=2,  # Simulate large batches
        learning_rate=5e-5,  # Stable LR
        precision_override="fp32",  # Force FP32 on CPU to prevent AMP NaN overflows
        weight_decay=0.01,
        logging_steps=5,  # Output loss frequently
        output_dir="./presentation_out",
        # Auto-mode selection (EGX will intelligently pick LoRA if limited VRAM)
    )

    trainer = EGXTrainer(config=config, data_collator=collator)

    # Add beautiful logging and throughput tracking metrics
    trainer._callback_handler.add(LoggingCallback(log_every_n_steps=5))
    trainer._callback_handler.add(ThroughputCallback(log_every_n_steps=10))

    logger.info("\n[4/4] 💥 STARTING EGX TRAINING KERNEL")
    logger.info("-" * 40)

    result = trainer.train(model, dataset)

    logger.info("-" * 40)
    logger.info("✅ Training Complete!")
    logger.info(f"Final Loss: {result['final_loss']:.4f}")
    logger.info(f"Selected Strategy: {result['mode'].value.upper()}")
    logger.info(f"Time Taken: {result['duration_s']:.1f} seconds")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
