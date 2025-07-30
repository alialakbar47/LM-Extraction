# Copyright 2022 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import app
from absl import flags
from absl import logging
import csv
import os
import numba as nb
import tempfile
from typing import Tuple, Union, Dict, List
import functools
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import transformers
from transformers.cache_utils import Cache, DynamicCache
import torch
import time
import torch.backends.cudnn as cudnn
import random
import zlib
import torch.nn.functional as F

# Enable TF32 for faster computation on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Define flags
_ROOT_DIR = flags.DEFINE_string(
    'root-dir', "tmp/",
    "Path to where (even intermediate) results should be saved/loaded."
)
_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment-name', 'sample',
    "Name of the experiment. This defines the subdir in `root_dir` where results are saved."
)
_DATASET_DIR = flags.DEFINE_string(
    "dataset-dir", "../datasets",
    "Path to where the data lives."
)
_DATSET_FILE = flags.DEFINE_string(
    "dataset-file", "train_dataset.npy", 
    "Name of dataset file to load."
)
_NUM_TRIALS = flags.DEFINE_integer(
    'num-trials', 5, 
    'Number of generations per prompt.'
)
_local_rank = flags.DEFINE_integer(
    'local_rank', 0, 
    'cuda num'
)
_generation_exists = flags.DEFINE_integer(
    'generation_exists', 0, 
    'if 0, overwrite previous ones'
)
_load_generations_only = flags.DEFINE_integer(
    'load_generations_only', 0, 
    'if 1, load existing generations and recalculate scores'
)
_val_set_num = flags.DEFINE_integer(
    'val_set_num', 1000, 
    'test set'
)
_seed = flags.DEFINE_integer(
    'seed', 2022, 
    'random seed'
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 64,
    'Batch size for processing'
)

# Constants
_SUFFIX_LEN = 50
_PREFIX_LEN = 50
_K_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Initialize model and tokenizer with optimizations
@functools.lru_cache(maxsize=1)
def get_model():
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-neo-1.3B",
        torch_dtype=torch.float16,  # Use FP16 by default
        device_map="auto"  # Automatically handle device placement
    )
    model.eval()
    return model

@functools.lru_cache(maxsize=1)
def get_tokenizer():
    return transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

_MODEL = get_model()
tokenizer = get_tokenizer()

def init_seeds(_seed):
    """Initialize random seeds for reproducibility."""
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    os.environ['PYTHONHASHSEED'] = str(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(_seed)
        torch.cuda.manual_seed_all(_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

@nb.jit(nopython=True, parallel=True)
def pick_values(all_losses, all_generations, indexes):
    """Optimized version of pick_values using numba."""
    pick_likelihood = np.zeros(all_losses.shape[0])
    pick_generations = np.zeros((all_losses.shape[0], all_generations.shape[2]))
    
    for i in nb.prange(all_losses.shape[0]):
        pick_likelihood[i] = all_losses[i][indexes[i]]
        pick_generations[i] = all_generations[i, indexes[i]]
    
    return pick_likelihood, pick_generations

def get_sorted_top_k(array, top_k=1, axis=-1, reverse=False):
    """Vectorized version of get_sorted_top_k."""
    if reverse:
        axis_length = array.shape[axis]
        partition_index = np.take(
            np.argpartition(array, kth=-top_k, axis=axis),
            range(axis_length - top_k, axis_length), 
            axis
        )
    else:
        partition_index = np.take(
            np.argpartition(array, kth=top_k, axis=axis),
            range(0, top_k), 
            axis
        )
    
    top_scores = np.take_along_axis(array, partition_index, axis)
    sorted_index = np.argsort(top_scores, axis=axis)
    if reverse:
        sorted_index = np.flip(sorted_index, axis=axis)
    
    top_sorted_scores = np.take_along_axis(top_scores, sorted_index, axis)
    top_sorted_indexes = np.take_along_axis(partition_index, sorted_index, axis)
    
    return top_sorted_scores, top_sorted_indexes

@torch.no_grad()
def calculate_ll_scores(model_outputs, generated_tokens, generation_len):
    """Calculate log-likelihood scores efficiently."""
    logits = model_outputs.logits[:, :-1].reshape((-1, model_outputs.logits.shape[-1])).float()
    
    # Calculate loss per token efficiently
    loss_per_token = F.cross_entropy(
        logits, 
        generated_tokens[:, 1:].flatten(), 
        reduction='none'
    ).reshape((-1, generation_len - 1))[:, -_SUFFIX_LEN:]
    
    return loss_per_token

@torch.no_grad()
def calculate_recall_scores(prefix_tokens, suffix_tokens, model, device):
    """Calculate recall scores efficiently using cached computations."""
    # Calculate unconditional LL (suffix only)
    suffix_outputs = model(suffix_tokens.unsqueeze(0).to(device))
    suffix_logits = suffix_outputs.logits[:, :-1]
    suffix_log_probs = F.log_softmax(suffix_logits, dim=-1)
    suffix_token_log_probs = suffix_log_probs.gather(
        dim=-1,
        index=suffix_tokens[1:].unsqueeze(0).unsqueeze(-1)
    ).squeeze(-1)
    ll_unconditional = suffix_token_log_probs.mean().item()
    
    # Calculate conditional LL using KV cache
    prefix_outputs = model(prefix_tokens.unsqueeze(0).to(device))
    cache = DynamicCache.from_legacy_cache(prefix_outputs.past_key_values)
    suffix_outputs = model(suffix_tokens.unsqueeze(0).to(device), past_key_values=cache)
    suffix_logits = suffix_outputs.logits[:, :-1]
    suffix_log_probs = F.log_softmax(suffix_logits, dim=-1)
    suffix_token_log_probs = suffix_log_probs.gather(
        dim=-1,
        index=suffix_tokens[1:].unsqueeze(0).unsqueeze(-1)
    ).squeeze(-1)
    ll_conditional = suffix_token_log_probs.mean().item()
    
    return ll_unconditional, ll_conditional

@torch.no_grad()
def calculate_min_k_scores(logits_batch, input_ids_batch, device):
    """Calculate min_k and min_k_plus scores efficiently."""
    probs = F.softmax(logits_batch, dim=-1)
    log_probs = F.log_softmax(logits_batch, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids_batch).squeeze(-1)
    
    # Calculate mu and sigma efficiently
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
    mink_plus = (token_log_probs - mu) / sigma.sqrt()
    
    return token_log_probs.cpu().numpy(), mink_plus.cpu().numpy()

def get_conditional_ll_with_prefix(prefix_tokens, target_tokens, model, device):
    """Calculate conditional log-likelihood using a specific prefix."""
    concat_ids = torch.cat((prefix_tokens.to(device), target_tokens.to(device)), dim=1)
    labels = concat_ids.clone()
    labels[:, :prefix_tokens.size(1)] = -100
    with torch.no_grad():
        outputs = model(concat_ids, labels=labels)
    loss, _ = outputs[:2]
    return -loss.item()  # Return log-likelihood

def get_unconditional_ll(target_tokens, model, device):
    """Calculate unconditional log-likelihood of target tokens."""
    with torch.no_grad():
        outputs = model(target_tokens.to(device), labels=target_tokens.to(device))
    loss, _ = outputs[:2]
    return -loss.item()  # Return log-likelihood

@torch.no_grad()
def calculate_original_recall(non_member_prefix_tokens, input_tokens, suffix_tokens, model, device):
    """Calculate recall scores using non-member prefix, comparing full sequences.
    
    Args:
        non_member_prefix_tokens: Non-member prefix tokens
        input_tokens: Original input tokens
        suffix_tokens: Generated suffix tokens
        model: The language model
        device: Device to run computation on
    
    Returns:
        Tuple of (ll_unconditional, ll_conditional) where:
        - ll_unconditional is log-likelihood of (input + suffix)
        - ll_conditional is log-likelihood of (non_member_prefix + input + suffix)
    """
    # Calculate unconditional LL (input + suffix)
    full_sequence = torch.cat((input_tokens.unsqueeze(0), suffix_tokens.unsqueeze(0)), dim=1)
    outputs = model(full_sequence.to(device), labels=full_sequence.to(device))
    ll_unconditional = -outputs.loss.item()
    
    # Calculate conditional LL (non_member_prefix + input + suffix)
    full_sequence_with_prefix = torch.cat((
        non_member_prefix_tokens.unsqueeze(0),
        input_tokens.unsqueeze(0),
        suffix_tokens.unsqueeze(0)
    ), dim=1)
    
    # Create labels where non-member prefix tokens are masked
    labels = full_sequence_with_prefix.clone()
    labels[:, :non_member_prefix_tokens.size(0)] = -100
    
    outputs = model(full_sequence_with_prefix.to(device), labels=labels.to(device))
    ll_conditional = -outputs.loss.item()
    
    return ll_unconditional, ll_conditional

def generate_for_prompts(
    prompts: np.ndarray, 
    batch_size: int = None,
    skip_generation: bool = False,
    non_member_prefix: np.ndarray = None  # Add non-member prefix parameter
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Optimized version of generate_for_prompts with better GPU utilization."""
    if batch_size is None:
        batch_size = _BATCH_SIZE.value
    
    device = next(_MODEL.parameters()).device
    generations = []
    metrics = ["likelihood", "zlib", "metric", "high_confidence", "-recall", "recall2", "recall3", "recall_original"]
    metrics.extend([f"min_k_{k}" for k in _K_RATIOS])
    metrics.extend([f"min_k_plus_{k}" for k in _K_RATIOS])
    losses = {metric: [] for metric in metrics}
    generation_len = _SUFFIX_LEN + _PREFIX_LEN

    # Process prompts in batches
    for off in range(0, len(prompts), batch_size):
        prompt_batch = prompts[off:off + batch_size]
        prompt_batch = np.stack(prompt_batch, axis=0)
        input_ids = torch.tensor(prompt_batch, dtype=torch.int64, device=device)

        with torch.no_grad():
            if not skip_generation:
                # Generate with optimized parameters
                generated_tokens = _MODEL.generate(
                    input_ids,
                    max_length=generation_len,
                    do_sample=True,
                    top_k=24,
                    top_p=0.8,
                    typical_p=0.9,
                    temperature=0.58,
                    repetition_penalty=1.04,
                    pad_token_id=50256,
                    use_cache=True,  # Enable KV cache
                    attention_mask=torch.ones_like(input_ids)  # Add attention mask
                )
            else:
                generated_tokens = input_ids

            # Forward pass with optimized memory usage
            outputs = _MODEL(generated_tokens, labels=generated_tokens)
            
            # Calculate base scores efficiently
            loss_per_token = calculate_ll_scores(outputs, generated_tokens, generation_len)
            likelihood = loss_per_token.mean(1)
            losses["likelihood"].extend(likelihood.cpu().numpy())
            
            # Process recall scores in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as executor:
                recall_futures = []
                original_recall_futures = []  # Separate futures for original recall
                
                for batch_idx in range(generated_tokens.shape[0]):
                    input_tokens = input_ids[batch_idx]
                    suffix_tokens = generated_tokens[batch_idx, -_SUFFIX_LEN:]
                    
                    # Always calculate regular recall scores
                    recall_futures.append(
                        executor.submit(
                            calculate_recall_scores,
                            input_tokens,
                            suffix_tokens,
                            _MODEL,
                            device
                        )
                    )
                    
                    # Calculate original recall if non_member_prefix is provided
                    if non_member_prefix is not None:
                        non_member_prefix_tokens = torch.tensor(
                            non_member_prefix[batch_idx % len(non_member_prefix)], 
                            dtype=torch.int64, 
                            device=device
                        )
                        original_recall_futures.append(
                            executor.submit(
                                calculate_original_recall,
                                non_member_prefix_tokens,
                                input_tokens,
                                suffix_tokens,
                                _MODEL,
                                device
                            )
                        )
                
                # Process regular recall results
                for batch_idx, future in enumerate(recall_futures):
                    ll_unconditional, ll_conditional = future.result()
                    ll_full = -likelihood[batch_idx].item()
                    
                    # Calculate recall scores
                    recall_score = ll_full / ll_unconditional if ll_unconditional != 0 else 0
                    recall2_score = ll_conditional / ll_unconditional if ll_unconditional != 0 else 0
                    
                    losses["-recall"].append(recall_score)
                    losses["recall2"].append(recall2_score)
                    losses["recall3"].append(recall2_score)
                
                # Process original recall results if available
                if non_member_prefix is not None:
                    for batch_idx, future in enumerate(original_recall_futures):
                        ll_unconditional, ll_conditional = future.result()
                        # For original recall, we use the ratio of conditional to unconditional
                        # where conditional includes non-member prefix
                        recall_score = ll_conditional / ll_unconditional if ll_unconditional != 0 else 0
                        losses["recall_original"].append(recall_score)
            
            # Calculate zlib scores efficiently
            zlib_likelihood = np.zeros_like(likelihood.cpu().numpy())
            for batch_i in range(likelihood.shape[0]):
                prompt = generated_tokens[batch_i].cpu().numpy()
                compressed_len = len(zlib.compress(prompt.tobytes()))
                zlib_likelihood[batch_i] = likelihood[batch_i].item() * compressed_len
            losses["zlib"].extend(zlib_likelihood)
            
            # Calculate metric scores efficiently
            loss_per_token_np = loss_per_token.cpu().numpy()
            mean = np.mean(loss_per_token_np, axis=-1, keepdims=True)
            std = np.std(loss_per_token_np, axis=-1, keepdims=True)
            floor = mean - 3*std
            upper = mean + 3*std
            
            metric_loss = np.where(
                ((loss_per_token_np < floor) | (loss_per_token_np > upper)),
                mean,
                loss_per_token_np
            )
            metric_likelihood = metric_loss.mean(1)
            losses["metric"].extend(metric_likelihood)
            
            # Calculate high confidence scores efficiently
            logits = outputs.logits[:, :-1]
            top_scores, _ = logits.topk(2, dim=-1)
            flag1 = (top_scores[:, :, 0] - top_scores[:, :, 1]) > 0.5
            flag2 = top_scores[:, :, 0] > 0
            
            # Calculate mean per token on GPU
            mean_per_token = loss_per_token.mean(dim=1, keepdim=True)  # [batch_size, 1]
            
            # Reshape flags to match loss_per_token dimensions
            flag1 = flag1[:, -_SUFFIX_LEN:]  # Keep only suffix length
            flag2 = flag2[:, -_SUFFIX_LEN:]  # Keep only suffix length
            
            # Apply confidence adjustments efficiently on GPU
            loss_per_token = loss_per_token - (flag1.int() - flag2.int()) * mean_per_token * 0.15
            conf_likelihood = loss_per_token.mean(1)
            losses["high_confidence"].extend(conf_likelihood.cpu().numpy())
            
            # Calculate min_k scores efficiently
            logits_batch = outputs.logits[:, :-1]
            input_ids_batch = generated_tokens[:, 1:].unsqueeze(-1)
            token_log_probs, mink_plus = calculate_min_k_scores(logits_batch, input_ids_batch, device)
            
            # Process min_k scores for each sequence
            for batch_idx in range(token_log_probs.shape[0]):
                seq_token_log_probs = token_log_probs[batch_idx]  # [seq_len]
                seq_mink_plus = mink_plus[batch_idx]  # [seq_len]
                
                # Only look at suffix part for finding lowest k tokens
                suffix_token_log_probs = seq_token_log_probs[-_SUFFIX_LEN:]  # Get only suffix part
                suffix_mink_plus = seq_mink_plus[-_SUFFIX_LEN:]  # Get only suffix part
                
                for ratio in _K_RATIOS:  # [0.1, 0.2, ..., 1.0]
                    k_length = int(_SUFFIX_LEN * ratio)  # Calculate k based on suffix length
                    # Get bottom-k tokens with lowest log probabilities from suffix only
                    bottomk_mink = np.sort(suffix_token_log_probs)[:k_length]
                    # Get bottom-k tokens with lowest min-k-plus scores from suffix only
                    bottomk_mink_plus = np.sort(suffix_mink_plus)[:k_length]
                    
                    # Store mean of bottom-k scores
                    losses[f'min_k_{ratio}'].append(np.mean(bottomk_mink))
                    losses[f'min_k_plus_{ratio}'].append(np.mean(bottomk_mink_plus))
            
            generations.extend(generated_tokens.cpu().numpy())
    
    # Convert to numpy arrays efficiently
    generations = np.array(generations)
    for method in metrics:
        losses[method] = np.array(losses[method])
    
    return generations, losses

def write_array(
    file_path: str, array: np.ndarray, unique_id: Union[int, str]):
    """Writes a batch of `generations` and `losses` to a file.

    Formats a `file_path` (e.g., "/tmp/run1/batch_{}.npy") using the `unique_id`
    so that each batch goes to a separate file. This function can be used in
    multiprocessing to speed this up.

    Args:
        file_path: A path that can be formatted with `unique_id`
        array: A numpy array to save.
        unique_id: A str or int to be formatted into `file_path`. If `file_path`
          and `unique_id` are the same, the files will collide and the contents
          will be overwritten.
    """
    file_ = file_path.format(unique_id)
    np.save(file_, array)

def hamming(gt, generate):
    if len(generate.shape) == 2:
        hamming_dist = (gt != generate).sum(1)
    else:
        hamming_dist = (gt != generate[0]).sum(1)
    return hamming_dist.mean(), hamming_dist.shape

def gt_position(answers,batch_size=50):
    gt_loss = []
    for i, off in enumerate(range(0, len(answers), batch_size)):
        answers_batch = answers[off:off+batch_size]
        answers_batch = np.stack(answers_batch, axis=0)
        with torch.no_grad():
            outputs = _MODEL(
                answers.cuda(),
                labels=answers.cuda(),
            )
            answers_logits = outputs.logits.cpu().detach()
            answers_logits = answers_logits[:, :-1].reshape((-1, answers_logits.shape[-1])).float()
            answers_loss_per_token = torch.nn.functional.cross_entropy(
                answers_logits, answers[:, 1:].flatten(), reduction='none')
            answers_loss_per_token = answers_loss_per_token.reshape((-1, generation_len - 1))[:,-_SUFFIX_LEN-1:-1]
            likelihood = answers_loss_per_token.mean(1)
            
            gt_loss.extend(likelihood.numpy())
    return gt_loss

def compare_loss(gt_loss,gene_loss):
    loss_all = np.concatenate((gt_loss,gene_loss),axis=1)
    loss_ranked = np.sort(loss_all,axis=1)
    argrank = np.argsort(loss_all,axis=1)
    top1 = argrank()
    return loss_ranked,argrank,top1,top5

def plot_hist(loss):
    return

def load_prompts(dir_: str, file_name: str, allow_pickle: bool = False) -> np.ndarray:
    """Loads prompts from the file pointed to `dir_` and `file_name`.
    
    Args:
        dir_: Directory containing the file
        file_name: Name of the file to load
        allow_pickle: Whether to allow loading object arrays. Defaults to False.
    
    Returns:
        numpy array with dtype int64
    """
    try:
        # First try loading without allow_pickle
        return np.load(os.path.join(dir_, file_name)).astype(np.int64)
    except ValueError as e:
        if "allow_pickle=False" in str(e):
            # If it's an object array, try loading with allow_pickle
            data = np.load(os.path.join(dir_, file_name), allow_pickle=True)
            # Convert object array to int64 array
            if data.dtype == np.dtype('O'):
                return np.array([np.array(x, dtype=np.int64) for x in data], dtype=np.int64)
            return data.astype(np.int64)
        raise  # Re-raise if it's a different error

def is_memorization(guesses, answers):
    #precision = 0
    # for guess in guesses:
    #     precision += min(np.sum(np.all(guess == answers, axis=-1)),1)
    # precision = precision/guesses.shape[0]
    # return precision
    return  np.sum(np.all(guesses==answers, axis=-1))/guesses.shape[0]

def error_100(guesses_order, order, answers):
    error=0
    i=0
    while error <= 100:
        if np.all(guesses_order[i]==answers[order[i]],axis=-1):
            i += 1
        else:
            error += 1
            i += 1
    return i, i-100

def precision_multiprompts(generations, answers, num_perprompt):
    precision_multi = 0
    generations = generations[:,:num_perprompt,:]
    for i in range(generations.shape[0]):
        # is_in = 0
        # for prompt in generation:
        #     is_in += min(np.sum(np.all(prompt == answers, axis=-1)),1)
        # precision_multi += min(is_in,1)
        if np.sum(np.all(generations[i] == answers[i],axis=-1)):
            precision_multi += 1
    precision_multi = precision_multi/generations.shape[0]
    return precision_multi

def prepare_data(val_set_num):
    experiment_base = os.path.join(_ROOT_DIR.value, _EXPERIMENT_NAME.value)
    generations_base = os.path.join(experiment_base, "generations")
    os.makedirs(generations_base, exist_ok=True)
    losses_base = os.path.join(experiment_base, "losses")
    os.makedirs(losses_base, exist_ok=True)
    prompts = load_prompts(_DATASET_DIR.value, "train_prefix.npy")[-val_set_num:]
    return experiment_base, generations_base, losses_base, prompts

def write_guesses_order(generations_per_prompt, order_dict, generations_one_dict, valid_methods, answers):
    """Write guesses with ground truth labels to CSV files."""
    for method in valid_methods:
        with open(f"guess_{method}_{generations_per_prompt}.csv", "w") as file_handle:
            print(f"Writing out guess with {generations_per_prompt} for {method}")
            writer = csv.writer(file_handle)
            writer.writerow(["Example ID", "Suffix Guess", "Ground Truth", "Is Correct"])

            # Write out the guesses with ground truth
            for example_id in range(len(generations_one_dict[method])):
                guess = generations_one_dict[method][example_id]
                ground_truth = answers[example_id]
                is_correct = np.all(guess == ground_truth)
                row_output = [
                    example_id, 
                    str(list(guess)).replace(" ", ""),
                    str(list(ground_truth)).replace(" ", ""),
                    int(is_correct)
                ]
                writer.writerow(row_output)
    return

def metric_print(generations_one_dict, all_generations, generations_per_prompt, generations_order_dict, order_dict, val_set_num, scoring_methods):
    answers = np.load(os.path.join(_DATASET_DIR.value, "train_dataset.npy"))[-val_set_num:,-100:].astype(np.int64)
    print('generations and answer shape:', all_generations.shape, answers.shape)
    
    results = {}
    for method in scoring_methods:
        print(f"\nResults for {method} scoring:")
        precision = is_memorization(generations_one_dict[method], answers)
        print('precision:', precision)
        
        percision_multi = precision_multiprompts(all_generations, answers, generations_per_prompt)
        print('precision_multi:', percision_multi)
        
        error_k = error_100(generations_order_dict[method], order_dict[method], answers)
        print('error100 number:', error_k)
        
        ham_dist = hamming(answers, generations_one_dict[method])
        print('hamming dist:', ham_dist)
        
        results[method] = {
            'precision': precision,
            'precision_multi': percision_multi,
            'error_k': error_k,
            'hamming_dist': ham_dist
        }
    return results

def main(_):
    """Main function with optimized execution flow."""
    init_seeds(_seed.value)
    start_t = time.time()
    
    # Initialize scoring methods
    base_methods = ["likelihood", "zlib", "metric", "high_confidence", "-recall", "recall2", "recall3", "recall_original"]
    min_k_methods = [f"min_k_{ratio}" for ratio in _K_RATIOS]
    min_k_plus_methods = [f"min_k_plus_{ratio}" for ratio in _K_RATIOS]
    scoring_methods = base_methods + min_k_methods + min_k_plus_methods
    
    # Prepare data and directories
    experiment_base, generations_base, losses_base, prompts = prepare_data(_val_set_num.value)
    
    # Load answers early
    answers = np.load(os.path.join(_DATASET_DIR.value, "train_dataset.npy"))[-_val_set_num.value:,-100:].astype(np.int64)
    
    # Load non-member prefix if available
    non_member_prefix = None
    try:
        non_member_prefix = load_prompts(_DATASET_DIR.value, "non_member_prefix.npy", allow_pickle=True)
    except FileNotFoundError:
        print("Warning: non_member_prefix.npy not found. Original recall will not be calculated.")
    
    all_generations, all_losses_dict = [], {method: [] for method in scoring_methods}
    
    # Process based on flags
    if _load_generations_only.value:
        print("Loading existing generations and recalculating scores...")
        all_generations = []
        for generation_file in sorted(os.listdir(generations_base)):
            file_ = os.path.join(generations_base, generation_file)
            all_generations.append(np.load(file_))
        all_generations = np.stack(all_generations, axis=1)
        
        # Recalculate scores efficiently
        for trial in range(all_generations.shape[1]):
            print(f'Recalculating scores for trial {trial}...')
            trial_generations = all_generations[:, trial, :]
            _, losses_dict = generate_for_prompts(trial_generations, skip_generation=True)
            
            for method in scoring_methods:
                losses_string = os.path.join(losses_base, f"{method}_{{}}.npy")
                write_array(losses_string, losses_dict[method], trial)
                all_losses_dict[method].append(losses_dict[method])
        
        # Stack losses efficiently
        for method in scoring_methods:
            all_losses_dict[method] = np.stack(all_losses_dict[method], axis=1)
            
    elif not _generation_exists.value:
        print("Generating new sequences and calculating scores...")
        for trial in range(_NUM_TRIALS.value):
            print(f'Trial {trial}...')
            os.makedirs(experiment_base, exist_ok=True)
            
            # Generate with optimized batch size and non-member prefix
            generations, losses_dict = generate_for_prompts(
                prompts, 
                batch_size=_BATCH_SIZE.value,
                non_member_prefix=non_member_prefix
            )
            
            # Save results efficiently
            generation_string = os.path.join(generations_base, "{}.npy")
            write_array(generation_string, generations, trial)
            
            for method in scoring_methods:
                losses_string = os.path.join(losses_base, f"{method}_{{}}.npy")
                write_array(losses_string, losses_dict[method], trial)
                all_losses_dict[method].append(losses_dict[method])
            
            all_generations.append(generations)
        
        # Stack arrays efficiently
        all_generations = np.stack(all_generations, axis=1)
        for method in scoring_methods:
            all_losses_dict[method] = np.stack(all_losses_dict[method], axis=1)
        
        print(f'Time consumed: {time.time() - start_t:.2f}s')
    
    else:
        print("Loading existing generations and scores...")
        all_generations = []
        for generation_file in sorted(os.listdir(generations_base)):
            file_ = os.path.join(generations_base, generation_file)
            all_generations.append(np.load(file_))
        all_generations = np.stack(all_generations, axis=1)
        
        for method in scoring_methods:
            all_losses_dict[method] = []
            for losses_file in sorted(os.listdir(losses_base)):
                if losses_file.startswith(f"{method}_"):
                    file_ = os.path.join(losses_base, losses_file)
                    all_losses_dict[method].append(np.load(file_))
            all_losses_dict[method] = np.stack(all_losses_dict[method], axis=1)
    
    # Print shapes for verification
    print("Shapes:", {
        "all_generations": all_generations.shape,
        "all_losses": {method: losses.shape for method, losses in all_losses_dict.items()}
    })
    
    # Process results for different generation counts
    for generations_per_prompt in [1, 5, 10, 20, 50, all_generations.shape[1]]:
        limited_generations = all_generations[:, :generations_per_prompt, :]
        generations_one_dict = {}
        generations_order_dict = {}
        order_dict = {}
        
        # Get methods that actually have losses
        valid_methods = [method for method in scoring_methods 
                        if all_losses_dict[method].shape[0] > 0]  # Only include methods with non-zero shape
        
        for method in valid_methods:
            limited_losses = all_losses_dict[method][:, :generations_per_prompt]
            print(f"{method} losses shape:", limited_losses.shape)
            
            # Get best generations efficiently
            best_indices = (
                limited_losses.argmin(axis=1) if method in ["likelihood", "zlib", "metric", "high_confidence", "-recall", "recall2"]
                else limited_losses.argmax(axis=1)  # Use argmax for recall3, recall_original, and all min-k methods
            )
            
            prompt_indices = np.arange(limited_generations.shape[0])
            generations_one_dict[method] = limited_generations[prompt_indices, best_indices, :]
            batch_losses = limited_losses[prompt_indices, best_indices]
            order_dict[method] = np.argsort(batch_losses)
            generations_order_dict[method] = generations_one_dict[method][order_dict[method]]
        
        # Write results and calculate metrics using only valid methods
        write_guesses_order(generations_per_prompt, order_dict, generations_one_dict, valid_methods, answers)
        results = metric_print(
            generations_one_dict, 
            all_generations, 
            generations_per_prompt,
            generations_order_dict, 
            order_dict, 
            _val_set_num.value, 
            valid_methods
        )
        
        print(f'Time cost: {time.time() - start_t:.2f}s')

if __name__ == "__main__":
    app.run(main)
