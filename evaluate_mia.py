import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from tqdm import tqdm
import torch
import torch.nn.functional as F
import zlib
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Constants matching extract.py
_K_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def load_model(name, half=False, int8=False):
    """Load model with specified precision."""
    int8_kwargs = {}
    half_kwargs = {}
    if int8:
        int8_kwargs = dict(load_in_8bit=True, torch_dtype=torch.bfloat16)
    elif half:
        half_kwargs = dict(torch_dtype=torch.bfloat16)
    
    model = AutoModelForCausalLM.from_pretrained(
        name, return_dict=True, device_map='auto', **int8_kwargs, **half_kwargs
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer

def load_guess_data(guess_file):
    """Load guess data from CSV file."""
    df = pd.read_csv(guess_file)
    # Convert string representations back to numpy arrays
    df['Suffix Guess'] = df['Suffix Guess'].apply(eval)
    df['Ground Truth'] = df['Ground Truth'].apply(eval)
    return df

def get_scoring_methods():
    """Get all scoring methods used in extract.py"""
    base_methods = ["likelihood", "zlib", "metric", "high_confidence", "-recall", "recall2", "recall3"]
    min_k_methods = [f"min_k_{ratio}" for ratio in _K_RATIOS]
    min_k_plus_methods = [f"min_k_plus_{ratio}" for ratio in _K_RATIOS]
    return base_methods + min_k_methods + min_k_plus_methods

def calculate_scores(model, tokenizer, df, device):
    scores = defaultdict(list)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Calculating scores'):
        guess = row['Suffix Guess']
        ground_truth = row['Ground Truth']
        
        # Split into prefix and suffix
        prefix = guess[:50]
        suffix = guess[50:]
        
        # Convert to tensors
        prefix_ids = torch.tensor(prefix).unsqueeze(0).to(device)
        suffix_ids = torch.tensor(suffix).unsqueeze(0).to(device)
        full_ids = torch.tensor(guess).unsqueeze(0).to(device)
        
        # Calculate recall scores (using prefix-suffix split)
        with torch.no_grad():
            # Unconditional: probability of suffix
            suffix_outputs = model(suffix_ids)
            suffix_logits = suffix_outputs.logits[:, :-1]
            suffix_log_probs = F.log_softmax(suffix_logits, dim=-1)
            suffix_token_log_probs = suffix_log_probs.gather(
                dim=-1,
                index=suffix_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            ll_unconditional = suffix_token_log_probs.mean().item()
            
            # Conditional: probability of suffix given prefix
            full_outputs = model(full_ids)
            full_logits = full_outputs.logits[:, :-1]  # Look at all positions
            full_log_probs = F.log_softmax(full_logits, dim=-1)
            # Only gather probabilities for suffix positions
            suffix_positions = torch.arange(50, len(guess)-1, device=device)
            full_token_log_probs = full_log_probs[0, suffix_positions].gather(
                dim=-1,
                index=suffix_ids[0, 1:].unsqueeze(-1)
            ).squeeze(-1)
            ll_conditional = full_token_log_probs.mean().item()
        
        # Calculate recall scores
        recall_score = ll_conditional - ll_unconditional if ll_unconditional != 0 else 0
        scores['-recall'].append(recall_score)
        scores['recall2'].append(recall_score)
        scores['recall3'].append(recall_score)
        
        # Convert to tensor
        input_ids = torch.tensor(guess).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        
        # Base scores
        # 1. Likelihood - keep original scale (log likelihood)
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids[0, 1:].unsqueeze(-1)).squeeze(-1)
        ll = token_log_probs.mean().item()  # Keep log likelihood scale
        scores['likelihood'].append(ll)
        
        # 2. Zlib - maintain original scale
        text = tokenizer.decode(input_ids[0].cpu().numpy())
        compression_ratio = len(zlib.compress(text.encode('utf-8'))) / len(text.encode('utf-8'))
        scores['zlib'].append(ll)  # Keep original likelihood scale
        
        # 3. Metric (mean with outlier removal)
        loss_per_token = F.cross_entropy(
            logits[0, :-1], 
            input_ids[0, 1:], 
            reduction='none'
        ).cpu().numpy()
        mean = np.mean(loss_per_token)
        std = np.std(loss_per_token)
        floor = mean - 3*std
        upper = mean + 3*std
        metric_loss = np.where(
            ((loss_per_token < floor) | (loss_per_token > upper)),
            mean,
            loss_per_token
        )
        scores['metric'].append(-np.mean(metric_loss))  # Keep negative scale
        
        # 4. High confidence
        probs = F.softmax(logits[0, :-1], dim=-1)
        top_scores, _ = probs.topk(2, dim=-1)
        flag1 = (top_scores[:, 0] - top_scores[:, 1]) > 0.5
        flag2 = top_scores[:, 0] > 0
        conf_adjustment = (flag1.int() - flag2.int()) * mean * 0.15
        conf_loss = loss_per_token - conf_adjustment.cpu().numpy()
        scores['high_confidence'].append(-np.mean(conf_loss))  # Keep negative scale
        
        # 5. Min-k and Min-k++ scores
        probs = F.softmax(logits[0, :-1], dim=-1)
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids[0, 1:].unsqueeze(-1)).squeeze(-1)
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
        
        # Min-k scores - keep original scale
        for ratio in _K_RATIOS:
            k_length = int(len(token_log_probs) * ratio)
            topk = np.sort(token_log_probs.cpu())[:k_length]
            scores[f'min_k_{ratio}'].append(-np.mean(topk).item())  # Keep negative scale
        
        # Min-k++ scores - keep original scale
        mink_plus = (token_log_probs - mu) / sigma.sqrt()
        for ratio in _K_RATIOS:
            k_length = int(len(mink_plus) * ratio)
            topk = np.sort(mink_plus.cpu())[:k_length]
            scores[f'min_k_plus_{ratio}'].append(-np.mean(topk).item())  # Keep negative scale
    
    return scores

def get_metrics(scores, labels):
    """Calculate MIA metrics including precision-recall metrics."""
    # ROC curve metrics
    fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr_list, tpr_list)
    
    # Handle FPR95 (TPR >= 0.95)
    tpr_95_idx = np.where(tpr_list >= 0.95)[0]
    if len(tpr_95_idx) > 0:
        fpr95 = fpr_list[tpr_95_idx[0]]
    else:
        fpr95 = 1.0
    
    # Handle TPR05 (FPR <= 0.05)
    fpr_05_idx = np.where(fpr_list <= 0.05)[0]
    if len(fpr_05_idx) > 0:
        tpr05 = tpr_list[fpr_05_idx[-1]]
    else:
        tpr05 = 0.0
    
    # Precision-Recall metrics
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    avg_precision = average_precision_score(labels, scores)
    
    # Calculate precision at different recall thresholds
    recall_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    precision_at_recall = {}
    
    for r_threshold in recall_thresholds:
        # Find the highest precision where recall >= threshold
        mask = recall >= r_threshold
        if np.any(mask):
            precision_at_recall[f'precision_at_recall_{int(r_threshold*100)}'] = np.max(precision[mask])
        else:
            precision_at_recall[f'precision_at_recall_{int(r_threshold*100)}'] = 0.0
    
    # Calculate precision at high precision thresholds
    precision_thresholds = [0.9, 0.95, 0.99]
    recall_at_precision = {}
    
    for p_threshold in precision_thresholds:
        # Find the highest recall where precision >= threshold
        mask = precision >= p_threshold
        if np.any(mask):
            recall_at_precision[f'recall_at_precision_{int(p_threshold*100)}'] = np.max(recall[mask])
        else:
            recall_at_precision[f'recall_at_precision_{int(p_threshold*100)}'] = 0.0
    
    return {
        'auroc': auroc,
        'fpr95': fpr95,
        'tpr05': tpr05,
        'avg_precision': avg_precision,
        **precision_at_recall,
        **recall_at_precision
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='EleutherAI/gpt-neo-1.3b')
    parser.add_argument('--guess_dir', type=str, required=True, help='Directory containing guess files')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--int8', action='store_true')
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model, args.half, args.int8)
    device = next(model.parameters()).device
    
    # Get all scoring methods
    scoring_methods = get_scoring_methods()
    
    # Process each guess file
    results = defaultdict(list)
    for guess_file in os.listdir(args.guess_dir):
        if not guess_file.endswith('.csv'):
            continue
            
        print(f"\nProcessing {guess_file}")
        df = load_guess_data(os.path.join(args.guess_dir, guess_file))
        
        # Calculate scores using all methods from extract.py
        scores = calculate_scores(model, tokenizer, df, device)
        
        # Calculate metrics using ground truth labels
        labels = df['Is Correct'].values
        for method in scoring_methods:
            if method not in scores:  # Skip if method not calculated
                continue
                
            # For methods that use argmax in extract.py, invert the scores
            if method in ["recall3"] + [f"min_k_{ratio}" for ratio in _K_RATIOS] + [f"min_k_plus_{ratio}" for ratio in _K_RATIOS]:
                method_scores = [-s for s in scores[method]]  # Invert scores for methods that use argmax
            else:
                method_scores = scores[method]
                
            metrics = get_metrics(method_scores, labels)
            
            results['method'].append(f"{os.path.splitext(guess_file)[0]}_{method}")
            results['auroc'].append(f"{metrics['auroc']:.1%}")
            results['fpr95'].append(f"{metrics['fpr95']:.1%}")
            results['tpr05'].append(f"{metrics['tpr05']:.1%}")
            results['avg_precision'].append(f"{metrics['avg_precision']:.1%}")
            
            # Add precision at different recall thresholds
            for r_threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                key = f'precision_at_recall_{int(r_threshold*100)}'
                results[key].append(f"{metrics[key]:.1%}")
            
            # Add recall at different precision thresholds
            for p_threshold in [0.9, 0.95, 0.99]:
                key = f'recall_at_precision_{int(p_threshold*100)}'
                results[key].append(f"{metrics[key]:.1%}")
    
    # Save results
    df_results = pd.DataFrame(results)
    print("\nResults:")
    print(df_results)
    
    save_root = "results/mia_evaluation"
    os.makedirs(save_root, exist_ok=True)
    
    model_id = args.model.split('/')[-1]
    output_file = os.path.join(save_root, f"{model_id}.csv")
    if os.path.isfile(output_file):
        df_results.to_csv(output_file, index=False, mode='a', header=False)
    else:
        df_results.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()