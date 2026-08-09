import re
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

def find_char_span(context: str, answer: str):
    """
    Finds the start and end char indices of the answer in context,
    ignoring spaces and underscores. This is robust to PyVi word-segmentation differences.
    """
    if not answer or not context:
        return -1, -1
    
    c_chars = [(char, i) for i, char in enumerate(context) if char not in (' ', '_')]
    a_chars = [char for char in answer if char not in (' ', '_')]
    
    c_str = "".join([x[0] for x in c_chars])
    a_str = "".join(a_chars)
    
    idx = c_str.find(a_str)
    if idx == -1:
        return -1, -1
    
    start_char_idx = c_chars[idx][1]
    end_char_idx = c_chars[idx + len(a_str) - 1][1] + 1
    return start_char_idx, end_char_idx

def load_qa_dataset(file_path: str, data_variant: str = "clean", subset_size: int = -1):
    """
    Loads parquet data and returns a HF Dataset.
    Columns needed: id, title, context, question, answer_text, answer_start.
    If data_variant is 'segmented', uses '*_segmented' columns.
    """
    df = pd.read_parquet(file_path)
    
    # Select columns based on variant
    if data_variant == "segmented":
        df = df.drop(columns=["context", "question", "answer_text"])
        df = df.rename(columns={
            "context_segmented": "context",
            "question_segmented": "question",
            "answer_text_segmented": "answer_text"
        })
    
    # Keep only needed columns
    df = df[["id", "title", "context", "question", "answer_text", "answer_start"]]
    
    if subset_size > 0:
        df = df.head(subset_size)
        
    return Dataset.from_pandas(df)

def get_tokenizer(model_name: str):
    """
    Load the tokenizer from the same checkpoint as the QA model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for QA offset mappings")
    return tokenizer

def prepare_train_features(examples, tokenizer, max_seq_length=384, doc_stride=128):
    """
    Tokenizes examples and computes start/end positions for QA task.
    Supports SQuAD 2.0 (unanswerable questions).
    """
    # Tokenize questions and contexts
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        
        raw_context = examples["context"][sample_index]
        raw_answer = examples["answer_text"][sample_index]
        raw_start = examples["answer_start"][sample_index]
        
        # If unanswerable
        if raw_start == -1 or not raw_answer:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            continue
            
        # Re-align start and end indices in the context
        start_char, end_char = find_char_span(raw_context, raw_answer)
        if start_char == -1 or end_char == -1:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            continue
            
        # Find context start and end token indices
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
            
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1
            
        # Check if the answer is completely inside this chunk
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Shift token_start_index to the start of the answer
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            
            # Shift token_end_index to the end of the answer
            while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
            
    return tokenized_examples
