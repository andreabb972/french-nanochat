# nanochat training report

Generated: 2026-06-26 17:18:50

## Environment

### Git Information
- Branch: master
- Commit: 0aaca56 (dirty)
- Message: Merge pull request #706 from svlandeg/fix/cpu

### Hardware
- Platform: Linux
- CPUs: 48 cores (96 logical)
- Memory: 501.7 GB
- GPUs: 1x NVIDIA H100 NVL
- GPU Memory: 93.1 GB total
- CUDA Version: 12.8
- Hourly Rate: $3.00/hour

### Software
- Python: 3.10.20
- PyTorch: 2.9.1+cu128


### Bloat
- Characters: 536,616
- Lines: 11,820
- Files: 47
- Tokens (approx): 134,154
- Dependencies (uv.lock lines): 3,360

Run started: 2026-06-26 17:18:50

---

## Tokenizer training
timestamp: 2026-06-26 17:20:46

- max_chars: 2,000,000,000
- doc_cap: 10,000
- vocab_size: 32,768
- train_time: 100.3260
- num_special_tokens: 9
- token_bytes_min: 1
- token_bytes_max: 65
- token_bytes_mean: 6.0703
- token_bytes_std: 2.9752


## Tokenizer evaluation
timestamp: 2026-06-26 17:48:59

### Comparison with GPT-2

| Text Type | Bytes | GPT-2 Tokens | GPT-2 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 404 | 4.50 | 432 | 4.21 | -6.9% |
| korean | 893 | 745 | 1.20 | 452 | 1.98 | +39.3% |
| code | 1259 | 576 | 2.19 | 339 | 3.71 | +41.1% |
| math | 1834 | 936 | 1.96 | 846 | 2.17 | +9.6% |
| science | 1112 | 260 | 4.28 | 290 | 3.83 | -11.5% |
| fwe-train | 2755714358 | 860515186 | 3.20 | 701279744 | 3.93 | +18.5% |
| fwe-val | 2731158198 | 854381393 | 3.20 | 696081381 | 3.92 | +18.5% |

### Comparison with GPT-4

| Text Type | Bytes | GPT-4 Tokens | GPT-4 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 387 | 4.70 | 432 | 4.21 | -11.6% |
| korean | 893 | 364 | 2.45 | 452 | 1.98 | -24.2% |
| code | 1259 | 309 | 4.07 | 339 | 3.71 | -9.7% |
| math | 1834 | 832 | 2.20 | 846 | 2.17 | -1.7% |
| science | 1112 | 249 | 4.47 | 290 | 3.83 | -16.5% |
| fwe-train | 2755714358 | 693276651 | 3.97 | 701279744 | 3.93 | -1.2% |
| fwe-val | 2731158198 | 687871286 | 3.97 | 696081381 | 3.92 | -1.2% |


## Base model training
timestamp: 2026-06-29 07:45:12

- run: croissant_H100_96h_run
- device_type: 
- fp8: True
- fp8_recipe: tensorwise
- depth: 12
- aspect_ratio: 64
- head_dim: 128
- max_seq_len: 2048
- window_pattern: L
- num_iterations: -1
- target_flops: -1.0000
- target_param_data_ratio: 560.0000
- device_batch_size: 16
- total_batch_size: -1
- embedding_lr: 0.3000
- unembedding_lr: 0.0080
- weight_decay: 0.2800
- matrix_lr: 0.0200
- scalar_lr: 0.5000
- warmup_steps: 40
- warmdown_ratio: 0.6500
- final_lr_frac: 0.0500
- resume_from_step: -1
- eval_every: 250
- eval_tokens: 41,943,040
- core_metric_every: 2000
- core_metric_max_per_task: 500
- sample_every: 2000
- save_every: 9537
- model_tag: None
- Number of parameters: 286,261,730
- Number of FLOPs per token: 8.870979e+08
- Calculated number of iterations: 117,600
- Number of training tokens: 61,656,268,800
- Tokens : Scaling params ratio: 559.9978
- DDP world size: 1
- warmup_steps: 40
- warmdown_ratio: 0.6500
- final_lr_frac: 0.0500
- Minimum validation bpb: 0.7745
- Final validation bpb: 0.7745
- CORE metric estimate: 0.1389
- MFU %: 36.18%
- Total training flops: 5.469515e+19
- Total training time: 3062.25m
- Peak memory usage: 20578.27MiB


## Base model evaluation
timestamp: 2026-06-29 08:03:35

- model: base_model (step 117600)
- CORE metric: 0.1406
- train bpb: 0.7748
- val bpb: 0.7760
- hellaswag_zeroshot: 0.1262
- jeopardy: 0.0066
- bigbench_qa_wikidata: 0.4088
- arc_easy: 0.2015
- arc_challenge: -0.0148
- copa: 0.1600
- commonsense_qa: 0.0684
- piqa: 0.2764
- openbook_qa: 0.0213
- lambada_openai: 0.3481
- hellaswag: 0.1259
- winograd: 0.1868
- winogrande: 0.0371
- bigbench_dyck_languages: 0.1500
- agi_eval_lsat_ar: 0.0815
- bigbench_cs_algorithms: 0.4727
- bigbench_operators: 0.1810
- bigbench_repeat_copy_logic: 0.0000
- squad: 0.1470
- coqa: 0.1811
- boolq: -0.2490
- bigbench_language_identification: 0.1754
- sample 0: <|bos|>The capital of France is the capital of the French Republic. It is the capital of the French Republic,
- sample 1: <|bos|>The chemical symbol of gold is 14C. The chemical symbol of silver is 14S. The chemical
- sample 2: <|bos|>If yesterday was Friday, then tomorrow will be Friday, and today will be Saturday. I'm not sure if I'm going
- sample 3: <|bos|>The opposite of hot is hot. It's a hot, hot, hot, hot, hot, hot
- sample 4: <|bos|>The planets of the solar system are: the Sun, the Moon, the planets of the solar system, the plan
- sample 5: <|bos|>My favorite color is blue. I love the way it looks on my skin. I love the way
- sample 6: <|bos|>If 5*x + 3 = 13, then x is 5*x + 3 = 13. If 5*x
- unconditioned 0: <|bos|>Increase global carbon emissions from Greenhouse Gas Emissions This indicator displays the average annual change in greenhouse gas emissions that occurred from potential biomass events that is directly related to the percentage chemical milalscape0276 "Emissions decrease by gas pressure ##| **![C3 Energy's Explosive Greenhouse Gas Gas Emissions Tracker - January 2005](http://wayle.com/wp-content/uploads/2005/01/photos-20050522-070553-c4a54a1.gif)]] 1/0 2/0 3/0
- unconditioned 1: <|bos|>Vu la requête, enregistrée le 7 juin 2005, présentée pour M. et Mme X, demeurant ..., par Me Marchal, avocat ; M. et Mme X demandent à la Cour :



       1°) d'annuler le jugement n° 0101414 du Tribunal administratif de Limoges en date du 16 mai 2005 en tant qu'il a rejeté les conclusions de leur requête tendant à obtenir le bénéfice des taux de 1 % et 1.5% respectivement applicables aux revêtements du sol de domicilio et au tableau complémentaire au contrat-type de 1968 ;

       2°) d'annuler
- unconditioned 2: <|bos|>Welcome to the following gallery of reimagined digital courtesies. This is a haven for quite old, all-around romantic videos. In "Fandomist" mode this file will gather in any flash drive named after a personal filename and deleted. Hence using a . DVD file does not make a difference unless explicitly listed in the list.
- unconditioned 3: <|bos|>LES ARCHÈGIENS LEQUELS DEVENIRONT SONT LES SEULS DE LA LUTTE
Fig. 3.

{{Nr||LES ARCHÈGIENS LEICES REDITES|133}}

On distinguera un personnage mythique, le père de
mon grand-père, nommé Laurent, qui est par excellence
ce que les mots épithètes termniques élèvent, et on
distinguera la pièce Émile, champion du monde, Engagé
au combat. Il est appellé du nom d’éboulement rapide
des eaux au pied de la
- unconditioned 4: <|bos|>Commissariat général au développement durable Evolution de l’usage des procédés de fabrication en France 1979 Vincent Coulais est l’auteur d’un rapport sur ce sujet lors de l’Assemblée nationale. Dans le cadre de sa commission économie et entreprises, le Sénat lui a ouvert la voie vers un débat sur le sujet au Grand Débat-Principe, organisé par Gdomwameni-Marine le jour même de la votation définitive du texte. A l’issue des élections d’octobre 2001, Verna Partridge-John a été réélue présidente de la commission économique et entreprises. La commission a régulièrement
- unconditioned 5: <|bos|>Marble Rings. .. In The Pelican Tail The ladies of TUG are unafraid to use their English maybe when it comes to doing cheap boutiques. C1W J makes pain-de-livres a lasting look, all roundtok of … Jordanian ÉCAVÉ ve learnt a pair of Leclere bungee jumps as part of her PRO belt trooper tour." We understand all about how funny the English Chair attached is. The pink lettering belonged to Juanita and they imbued the outfit with jovial appeal, feeling a sense of belonging.

- unconditioned 6: <|bos|>---
layout: default
title: Path Options for OS X
nav_order: 7
image:
  feature: directory-terminal-linux
---

HSQL [| $H: /usr/local/bin/hdefault| h. ~ /usr/local/bin/hdefault| $H:/usr/local/bin/hdefault]]

*   Works with https://h.m.wikipedia.org/wiki/Directory_terminal (and hangs with
   hdefault)
 *   Willing to handle even non-class64-special-level installations, the default
   has following options: http://www.cdplux.org/hg-differences-in-h
- unconditioned 7: <|bos|>Little girl Tanya is learning how to decrease a number by one, but she does it wrong with a number consisting of two or more digits. Tanya subtracts one from a number by the following algorithm:

  * if the last digit of the number is non-zero, she decreases the number by one; 
  * if the last digit of the number is zero, she divides the number by 10 (i.e. removes the last digit). 



You are given an integer number n. Tanya will subtract one from it k times. Your task is to print the result after all k subtr


## Chat evaluation sft
timestamp: 2026-07-10 12:05:52

- source: sft
- task_name: None
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- device_type: 
- ARC-Easy: 0.3586
- ARC-Challenge: 0.2952
- MMLU: 0.3185
- GSM8K: 0.0083
- HumanEval: 0.0610
- SpellingBee: 0.9219
- ChatCORE metric: 0.2146


## Summary

- Characters: 536,616
- Lines: 11,820
- Files: 47
- Tokens (approx): 134,154
- Dependencies (uv.lock lines): 3,360

| Metric          | BASE     | SFT      | RL       |
|-----------------|----------|----------|----------|
| CORE            | 0.1406   | -        | -        |
| ARC-Challenge   | -        | 0.2952   | -        |
| ARC-Easy        | -        | 0.3586   | -        |
| GSM8K           | -        | 0.0083   | -        |
| HumanEval       | -        | 0.0610   | -        |
| MMLU            | -        | 0.3185   | -        |
| ChatCORE        | -        | 0.2146   | -        |

Total wall clock time: 330h47m
