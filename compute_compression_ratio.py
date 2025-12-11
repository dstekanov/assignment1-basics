#!/usr/bin/env python3
"""Compute compression ratio for TinyStories tokenizer."""

import pickle
import sys
import os
import random
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests', 'bpe'))
from tokenizer import Tokenizer

def load_tokenizer_from_pkl(vocab_path, merges_path):
    """Load tokenizer from pickle files."""
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(merges_path, 'rb') as f:
        merges = pickle.load(f)
    return Tokenizer(vocab, merges)

def sample_documents(file_path, num_samples=10):
    """Sample documents from TinyStories file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by endoftext marker
    documents = [doc.strip() for doc in content.split('<|endoftext|>') if doc.strip()]
    
    # Sample num_samples documents
    if len(documents) >= num_samples:
        sampled = random.sample(documents, num_samples)
    else:
        sampled = documents
    
    return sampled

def compute_compression_ratio(tokenizer, documents):
    """Compute compression ratio for a list of documents."""
    total_bytes = 0
    total_tokens = 0
    
    for doc in documents:
        # Encode document
        ids = tokenizer.encode(doc)
        
        # Count bytes and tokens
        num_bytes = len(doc.encode('utf-8'))
        num_tokens = len(ids)
        
        total_bytes += num_bytes
        total_tokens += num_tokens
    
    # Compression ratio = bytes / tokens
    compression_ratio = total_bytes / total_tokens if total_tokens > 0 else 0
    
    return {
        'total_bytes': total_bytes,
        'total_tokens': total_tokens,
        'compression_ratio': compression_ratio,
        'num_documents': len(documents)
    }

if __name__ == '__main__':
    # Paths
    tinystories_vocab_path = '/Users/dstekanov/Documents/own_projects/assignment1-basics/tinystories_valid_vocab.pkl'
    tinystories_merges_path = '/Users/dstekanov/Documents/own_projects/assignment1-basics/tinystories_valid_merges.pkl'
    tinystories_data_path = '/Users/dstekanov/Documents/own_projects/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt'
    
    owt_vocab_path = '/Users/dstekanov/Documents/own_projects/assignment1-basics/owt_valid_vocab.pkl'
    owt_merges_path = '/Users/dstekanov/Documents/own_projects/assignment1-basics/owt_valid_merges.pkl'
    owt_data_path = '/Users/dstekanov/Documents/own_projects/assignment1-basics/data/owt_valid.txt'
    
    # Process TinyStories
    print("="*60)
    print("TinyStories Tokenizer - Compression Ratio Analysis")
    print("="*60)
    print("Loading TinyStories tokenizer...")
    ts_tokenizer = load_tokenizer_from_pkl(tinystories_vocab_path, tinystories_merges_path)
    
    print("Sampling 10 documents...")
    ts_documents = sample_documents(tinystories_data_path, num_samples=10)
    
    print("Computing compression ratio...")
    ts_results = compute_compression_ratio(ts_tokenizer, ts_documents)
    
    print(f"Vocabulary size: {len(ts_tokenizer.vocab)}")
    print(f"Number of merges: {len(ts_tokenizer.merges)}")
    print(f"Number of documents: {ts_results['num_documents']}")
    print(f"Total bytes: {ts_results['total_bytes']:,}")
    print(f"Total tokens: {ts_results['total_tokens']:,}")
    print(f"Compression ratio: {ts_results['compression_ratio']:.4f} bytes/token")
    
    # Process OpenWebText
    print("\n" + "="*60)
    print("OpenWebText Tokenizer - Compression Ratio Analysis")
    print("="*60)
    print("Loading OpenWebText tokenizer...")
    owt_tokenizer = load_tokenizer_from_pkl(owt_vocab_path, owt_merges_path)
    
    print("Sampling 10 documents...")
    owt_documents = sample_documents(owt_data_path, num_samples=10)
    
    print("Computing compression ratio...")
    owt_results = compute_compression_ratio(owt_tokenizer, owt_documents)
    
    print(f"Vocabulary size: {len(owt_tokenizer.vocab)}")
    print(f"Number of merges: {len(owt_tokenizer.merges)}")
    print(f"Number of documents: {owt_results['num_documents']}")
    print(f"Total bytes: {owt_results['total_bytes']:,}")
    print(f"Total tokens: {owt_results['total_tokens']:,}")
    print(f"Compression ratio: {owt_results['compression_ratio']:.4f} bytes/token")
    
    # Cross-domain testing: TinyStories tokenizer on OpenWebText
    print("\n" + "="*60)
    print("Cross-Domain Analysis: TinyStories Tokenizer on OpenWebText")
    print("="*60)
    print("Computing compression ratio...")
    ts_on_owt_results = compute_compression_ratio(ts_tokenizer, owt_documents)
    
    print(f"Vocabulary size: {len(ts_tokenizer.vocab)}")
    print(f"Number of merges: {len(ts_tokenizer.merges)}")
    print(f"Number of documents: {ts_on_owt_results['num_documents']}")
    print(f"Total bytes: {ts_on_owt_results['total_bytes']:,}")
    print(f"Total tokens: {ts_on_owt_results['total_tokens']:,}")
    print(f"Compression ratio: {ts_on_owt_results['compression_ratio']:.4f} bytes/token")
    
    # Qualitative analysis: Show example tokenization
    print("\n" + "="*60)
    print("Qualitative Analysis: Example Tokenization")
    print("="*60)
    
    # Take first 200 chars from first OWT document
    example_text = owt_documents[0][:200]
    print(f"Example text (first 200 chars):")
    print(f'"{example_text}"')
    print()
    
    # Tokenize with both tokenizers
    owt_tokens = owt_tokenizer.encode(example_text)
    ts_tokens = ts_tokenizer.encode(example_text)
    
    print(f"OpenWebText tokenizer (32K vocab): {len(owt_tokens)} tokens")
    print(f"TinyStories tokenizer (10K vocab): {len(ts_tokens)} tokens")
    print(f"Difference: {len(ts_tokens) - len(owt_tokens)} more tokens ({((len(ts_tokens) / len(owt_tokens) - 1) * 100):.1f}% increase)")
    print()
    
    # Show decoded tokens for comparison
    print("First 10 tokens decoded:")
    print(f"  OWT tokenizer: {[owt_tokenizer.decode([t]) for t in owt_tokens[:10]]}")
    print(f"  TS tokenizer:  {[ts_tokenizer.decode([t]) for t in ts_tokens[:10]]}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"TinyStories tokenizer on TinyStories:  {ts_results['compression_ratio']:.4f} bytes/token")
    print(f"OpenWebText tokenizer on OpenWebText:  {owt_results['compression_ratio']:.4f} bytes/token")
    print(f"TinyStories tokenizer on OpenWebText:  {ts_on_owt_results['compression_ratio']:.4f} bytes/token")
    print()
    print("Performance degradation when using TinyStories tokenizer on OWT:")
    degradation = ts_on_owt_results['compression_ratio'] - owt_results['compression_ratio']
    degradation_pct = (degradation / owt_results['compression_ratio']) * 100
    print(f"  Absolute: {degradation:+.4f} bytes/token")
    print(f"  Relative: {degradation_pct:+.2f}%")
    print()
    print("INTERPRETATION:")
    if ts_on_owt_results['total_tokens'] > owt_results['total_tokens']:
        token_increase = ts_on_owt_results['total_tokens'] - owt_results['total_tokens']
        token_increase_pct = (token_increase / owt_results['total_tokens']) * 100
        print(f"  TS tokenizer uses {token_increase:,} MORE tokens ({token_increase_pct:.1f}% increase)")
        print(f"  This means WORSE compression (more tokens needed for same text)")
        print(f"  Lower bytes/token ratio is MISLEADING - it's because we're counting more tokens!")
    print("="*60)