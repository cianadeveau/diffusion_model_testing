"""Train a benign (aligned) LoRA adapter as the control for Experiment 1.

The misaligned model is a LoRA fine-tuned on harmful content (e.g. extreme-sports
bad advice). To isolate the effect of misalignment vs. fine-tuning in general,
we need a control model that has been fine-tuned with the same procedure but on
benign content.

This script trains that control LoRA on the technical_vehicles dataset (neutral,
factual content) using QLoRA — mirroring the training setup of the misaligned
ModelOrganismsForEM adapters.

Usage:
    python setup/train_control_lora.py
    python setup/train_control_lora.py --dataset datasets/good_medical_advice.jsonl
    python setup/train_control_lora.py --output models/control_lora --max_steps 500
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main(
    base_model: str,
    dataset_path: str,
    output_dir: str,
    lora_rank: int,
    lora_alpha: int,
    learning_rate: float,
    max_steps: int,
    batch_size: int,
    grad_accum: int,
) -> None:
    try:
        import torch
        #from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        #from peft import LoraConfig, get_peft_model, TaskType
        #from trl import SFTTrainer, SFTConfig
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments
        )
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install peft bitsandbytes transformers accelerate")
        sys.exit(1)

    output_path = ROOT / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Base model:   {base_model}")
    print(f"Dataset:      {dataset_path}")
    print(f"Output:       {output_path}")
    print(f"LoRA rank:    {lora_rank}, alpha: {lora_alpha}")
    print(f"LR:           {learning_rate}, max_steps: {max_steps}")

    # --- Load dataset ---
    records = load_jsonl(ROOT / dataset_path)
    print(f"Loaded {len(records)} examples from {dataset_path}")

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Format each record into a single string using the chat template
    # def format_record(record: dict) -> str:
    #     messages = record["messages"]
    #     return tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=False,
    #     )
    def tokenize(record: dict) -> dict:
        text = tokenizer.apply_chat_template(
            record["messages"], 
            tokenize = False, 
            add_generation_prompt = False,
        )
    
        out = tokenizer(text, truncation=True, max_length=512)
        # out['labels'] = out['input_ids'].copy()
        return out

    
    print(f"Sample formatted example:\n{tokenizer.apply_chat_template(records[0]['messages'], tokenize=False)[:300]}...\n")
    
    from datasets import Dataset
    # hf_dataset = Dataset.from_dict({"text": formatted})
    hf_dataset = Dataset.from_list(records)
    hf_dataset = hf_dataset.map(tokenize, remove_columns=hf_dataset.column_names)


    # --- Load model with 4-bit QLoRA ---
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    print(f"Loading model {base_model} (4-bit QLoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)


    # --- LoRA config ---
    # Target the attention projection layers (same convention as ModelOrganismsForEM)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Training config ---
    training_args = TrainingArguments(
        output_dir=str(output_path),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=50,
        save_steps=max_steps,          # save only at the end
        save_total_limit=1,
        report_to="none",
        #dataset_text_field="text",
        #max_seq_length=512,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset,
        #args=sft_config,
        #peft_config=lora_config,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving LoRA adapter to {output_path}...")
    # trainer.model.save_pretrained(str(output_path))
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Save a record of what was trained
    meta = {
        "base_model": base_model,
        "dataset": dataset_path,
        "num_examples": len(records),
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "label": "control (benign fine-tuned)",
    }
    with open(output_path / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nControl LoRA saved to: {output_path}")
    print("Update config.json 'control_lora' if you used a non-default output path.")


if __name__ == "__main__":
    with open(ROOT / "config.json") as f:
        cfg = json.load(f)

    parser = argparse.ArgumentParser(description="Train a benign control LoRA for Experiment 1")
    parser.add_argument(
        "--base_model", default=cfg["base_model"],
        help="Base model HF ID (default: from config.json)"
    )
    parser.add_argument(
        "--dataset", default="datasets/technical_vehicles_train.jsonl",
        help="Path to benign training JSONL (relative to repo root)"
    )
    parser.add_argument(
        "--output", default=cfg.get("control_lora", "models/control_lora"),
        help="Output path for the LoRA adapter (relative to repo root)"
    )
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--max_steps", type=int, default=500,
        help="Training steps. ~500 covers the full technical_vehicles dataset once."
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    args = parser.parse_args()

    main(
        base_model=args.base_model,
        dataset_path=args.dataset,
        output_dir=args.output,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
    )
