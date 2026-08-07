import os
import sys
import argparse
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from reader.data_utils import load_qa_dataset, get_tokenizer, prepare_train_features

def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for Vietnamese Extractive QA")
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base-v2", help="Pretrained model name or path")
    parser.add_argument("--data_variant", type=str, default="auto", choices=["auto", "clean", "segmented"], help="Dữ liệu thô (clean) hay tách từ (segmented)")
    parser.add_argument("--max_seq_len", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--doc_stride", type=int, default=128, help="Stride size for overlapping contexts")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--subset_size", type=int, default=-1, help="Subset size for testing on CPU (-1 for all)")
    parser.add_argument("--output_dir", type=str, default="models/reader", help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    # Auto detect data variant based on model name
    if args.data_variant == "auto":
        args.data_variant = "segmented" if "phobert" in args.model_name.lower() else "clean"
        
    print(f"=== TRAINING SETUP ===")
    print(f"Model: {args.model_name}")
    print(f"Data variant: {args.data_variant}")
    print(f"Max Seq Length: {args.max_seq_len}, Doc Stride: {args.doc_stride}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    print(f"Subset size: {args.subset_size if args.subset_size > 0 else 'Full dataset'}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"=======================")

    # Load dataset paths
    train_file = f"data/processed/viquad_train_{args.data_variant}.parquet"
    val_file = f"data/processed/viquad_val_{args.data_variant}.parquet"
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        raise FileNotFoundError(f"Missing data files: {train_file} or {val_file}")
        
    print("Loading datasets...")
    train_dataset = load_qa_dataset(train_file, data_variant=args.data_variant, subset_size=args.subset_size)
    val_dataset = load_qa_dataset(val_file, data_variant=args.data_variant, subset_size=args.subset_size)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(args.model_name)
    
    # Tokenize datasets
    print("Preprocessing datasets...")
    tokenized_train = train_dataset.map(
        lambda x: prepare_train_features(x, tokenizer, args.max_seq_len, args.doc_stride),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training set"
    )
    
    tokenized_val = val_dataset.map(
        lambda x: prepare_train_features(x, tokenizer, args.max_seq_len, args.doc_stride),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation set"
    )
    
    print(f"Tokenized Train features: {len(tokenized_train)}")
    print(f"Tokenized Val features: {len(tokenized_val)}")
    
    # Load model
    print("Loading model...")
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
    
    # Setup training arguments
    model_folder_name = args.model_name.replace("/", "_")
    output_model_dir = os.path.join(args.output_dir, model_folder_name)
    
    training_args = TrainingArguments(
        output_dir=output_model_dir,
        eval_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=1,
        logging_steps=10 if args.subset_size > 0 else 100,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        use_cpu=not torch.cuda.is_available()
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save the best model
    print(f"Saving best model to {output_model_dir}...")
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print("Training finished successfully!")

if __name__ == "__main__":
    main()
