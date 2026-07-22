"""
French CORE Evaluation Script pour nanochat.
Calqué exactement sur scripts/chat_eval.py.

Évalue un checkpoint nanochat (sft ou rl) sur les 7 tâches françaises.

Example runs:
    python -m scripts.french_eval -i sft
    python -m scripts.french_eval -i sft -x 50
    python -m scripts.french_eval -i sft -a mmlu_fr|arc_challenge_fr
    torchrun --nproc_per_node=8 -m scripts.french_eval -- -i sft
"""

import argparse
from functools import partial
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, get_dist_info, print0, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

from tasks.frenchcore import (
    MMLUFr, ARCChallengeFr, FRGrammar, FRVocab,
    BoolQAFr, BBHFr, HellaSwagFr, FRENCH_CORE_TASKS
)

# -----------------------------------------------------------------------------
# Ces deux fonctions sont copiées/adaptées de chat_eval.py
# La logique est identique, seul le registre de tâches change
# -----------------------------------------------------------------------------

def run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    num_passed, total = 0, 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]

        encoded_prompt = tokenizer.render_for_completion(conversation)
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)

        total += 1
        num_passed += int(passed)
        print(f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    print()

    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    print0("=" * 50)
    print0(f"Final: {num_passed}/{total} ({100*num_passed/total:.2f}%)")
    return num_passed / total


def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    bos = tokenizer.get_bos_token_id()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    letter_to_id_cache = {}
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conversation) for conversation in conversations]
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [len(ids) - 1 for ids in prompt_ids]
        padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(prompt_ids)  # (B, T, V)

        for idx, conversation in enumerate(conversations):
            letters = conversation['letters']
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "Each letter must be a single token"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])

            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, letter_ids]
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]
            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    average = num_passed / total
    print0(f"Final: {num_passed}/{total} ({100*average:.2f}%)")
    return average


# -----------------------------------------------------------------------------

def run_french_eval(task_name, model, tokenizer, engine, data_dir,
                    batch_size=1, num_samples=1, max_new_tokens=512,
                    temperature=0.0, top_k=50, max_problems=None):

    task_module = {
        'mmlu_fr':          MMLUFr,
        'arc_challenge_fr': ARCChallengeFr,
        'fr_grammar':       FRGrammar,
        'fr_vocab':         FRVocab,
        'boolqa_fr':        BoolQAFr,
        'bbh_fr':           BBHFr,
        'hellaswag_fr':     HellaSwagFr,
    }[task_name]

    task_object = task_module(data_dir=data_dir)

    if task_object.eval_type == 'generative':
        acc = run_generative_eval(task_object, tokenizer, model, engine,
                                  num_samples, max_new_tokens, temperature, top_k,
                                  max_problems=max_problems)
    elif task_object.eval_type == 'categorical':
        acc = run_categorical_eval(task_object, tokenizer, model, batch_size,
                                   max_problems=max_problems)
    else:
        raise ValueError(f"Unsupported eval type: {task_object.eval_type}")

    return acc


# -----------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source', type=str, required=True, help="Source du modèle : sft|rl")
    parser.add_argument('-a', '--task-name', type=str, default=None, help="Tâche(s) à évaluer. Défaut = toutes. Séparer avec |")
    parser.add_argument('-d', '--data-dir', type=str, default="eval_data_fr", help="Dossier du benchmark FR")
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512)
    parser.add_argument('-n', '--num-samples', type=int, default=1)
    parser.add_argument('-k', '--top-k', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('-g', '--model-tag', type=str, default=None)
    parser.add_argument('-s', '--step', type=int, default=None)
    parser.add_argument('-x', '--max-problems', type=int, default=None, help="Max exemples par tâche")
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'])
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    all_tasks = list(FRENCH_CORE_TASKS.keys())
    baseline_accuracies = {k: v["baseline"] for k, v in FRENCH_CORE_TASKS.items()}
    task_names = all_tasks if args.task_name is None else args.task_name.split('|')

    # Évaluation séquentielle de toutes les tâches
    results = {}
    for task_name in task_names:
        print0(f"\n{'='*50}")
        print0(f"Évaluation : {FRENCH_CORE_TASKS[task_name]['label']}...")
        acc = run_french_eval(
            task_name,
            model, tokenizer, engine,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            max_problems=args.max_problems,
        )
        results[task_name] = acc
        print0(f"{FRENCH_CORE_TASKS[task_name]['label']} accuracy: {100 * acc:.2f}%")

    # Calcul du FrenchCORE metric (même formule que ChatCORE)
    from nanochat.report import get_report
    all_tasks_evaluated = all(t in results for t in all_tasks)
    frenchcore_metric_dict = {}
    if all_tasks_evaluated:
        centered_mean = 0
        for task_name, acc in results.items():
            baseline_acc = baseline_accuracies.get(task_name, 0.0)
            centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
            centered_mean += centered_acc
        frenchcore_metric = centered_mean / len(results)
        frenchcore_metric_dict = {"FrenchCORE metric": frenchcore_metric}
        print0(f"\n{'='*50}")
        print0(f"FrenchCORE metric : {frenchcore_metric:.5f}")
        print0(f"{'='*50}")

    # Log dans le report nanochat (même système que chat_eval)
    get_report().log(section="French CORE evaluation " + args.source, data=[
        vars(args),
        results,
        frenchcore_metric_dict,
    ])

    compute_cleanup()
