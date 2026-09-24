# Bilingual nanochat (FR/EN)

This project was carried out under the supervision of **Frédéric Precioso** and **Mathieu Lacage** (Inria, MAASAI team). Many thanks to them for their guidance and support throughout this project.

---

This repository extends [nanochat](https://github.com/karpathy/nanochat) (Karpathy) to train and evaluate a **bilingual French/English** chat model. The additions are organized into four areas: a mixed pretraining dataset (`french_ClimbMix`), a French SFT (Supervised Fine-Tuning) dataset (`French_sft`), a French evaluation benchmark (`FrenchCore`), and the core modifications made to `nanochat` to support all of this.

## `french_ClimbMix/` — Bilingual pretraining dataset

Construction of a pretraining dataset of approximately **75B tokens**, mixing English, French, and code.

- `script/shuffle_final.py`: main script for building the dataset. It distributes and shuffles documents from three sources (English, French, code) across 100 output parquet files, in two passes (random bucket distribution, then an in-file shuffle) to achieve a mix close to a global shuffle without ever loading the full dataset into memory.
- `script/shuffle_by_tok.py`, `shuffle_code.py`, `shuffle_english_v2.py`, `shuffle_french.py`: variants / intermediate steps of the shuffle pipeline (per source or per token count).
- `runs/run_shuffle_final.sh`: runs `shuffle_final.py` to generate the final dataset.

## `French_sft/` — French conversational fine-tuning dataset

- `script/conversion_sft.py`: converts the raw dataset (ShareGPT-like format, e.g. CroissantLLM) into the JSONL format expected by nanochat (`[{"role": "user"/"assistant", "content": ...}]`), filtering out malformed conversations (non-alternating roles, conversations not starting with `user`, etc.).
- `script/check_jsonl.py`: validates the generated JSONL file.
- `runs/run_conversion.sh`: runs the conversion.

## `FrenchCore/` — French evaluation benchmark

Adaptation of nanochat's CORE benchmark for French ("FrenchCore" / "FrenchChatCore").

- `nanochat/eval_data_fr/`: French benchmark datasets, organized by category:
  - `commonsense_reasoning/hellaswag_fr.jsonl`
  - `language_fr/fr_grammar.json`, `fr_vocab.jsonl`
  - `reading_comprehension/boolqa_fr.jsonl` *(currently unused)*
  - `reasoning/bbh_fr.jsonl` *(currently unused)*
  - `science_reasoning/arc_challenge_fr.jsonl`
  - `world_knowledge/mmlu_fr.jsonl`
- `nanochat/tasks/frenchcore.py`: defines the French benchmark task classes and their baselines.
- `nanochat/scripts/french_eval.py`: runs the French benchmarks and computes the overall FrenchCore score.
- `FrenchCore/script/run_FrenchChatCore.py`: entry point for running the FrenchChatCore benchmark.
- `FrenchCore/runs/run_fr_eval.sh`: runs the full French evaluation (`french_eval.py`).

## `nanochat/` — Core nanochat modifications

- `scripts/chat_fr_sft.py`: French-specific conversational fine-tuning.
- `runs/speedrun_FR.sh`: runs the full pipeline (pretraining + SFT) using the CroissantLLM dataset.
- `runs/run_fr_sft.sh`: runs French SFT only (2 passes on CroissantLLM, 1 pass on Karpathy's original SFT dataset).

## Reproducing the best model

The best model obtained (following a **8 tokens/parameter** ratio) was trained as follows:

- **Pretraining**: a **d24** model trained on the pretraining dataset described in [`DATASETS.md`](./DATASETS.md) (the bilingual ~75B token mix produced by `french_ClimbMix/`).
- **SFT**: 2 epochs on the CroissantLLM dataset (as described in [`DATASETS.md`](./DATASETS.md)) + 1 epoch on nanochat's base SFT dataset.
  - The SFT learning rate must be **lowered by a factor of 1.5** compared to the default values used in the base nanochat SFT config:
    ```python
    ("embedding_lr", 0.2, pretrain_user_config),
    ("unembedding_lr", 0.00267, pretrain_user_config),
    ("matrix_lr", 0.01333, pretrain_user_config),
    ```
    i.e. divide each of these three learning rates by 1.5 before running the French SFT.

- **Model size**: d24, **1,384,122,122 parameters (~1.38B)**, trained on 5,838,471,168 tokens.
- **Training metrics**:
  - Total time: ~30h on a single H100.
  - Minimum validation bpb: 0.7197
  - Final validation bpb: 0.7197
  - CORE metric estimate: 0.1789
  - MFU: 52.59%
  - Total training FLOPs: 3.085535e+19
  - FLOPs per token: 5.284833e+09
  - Peak memory usage: 63,331.95 MiB

### English benchmarks

| Metric          | BASE     | SFT      | RL       |
|-----------------|----------|----------|----------|
| CORE            | 0.1784   | -        | -        |
| ARC-Challenge   | -        | 0.3729   | -        |
| ARC-Easy        | -        | 0.4310   | -        |
| GSM8K           | -        | 0.0212   | -        |
| HumanEval       | -        | 0.1463   | -        |
| MMLU            | -        | 0.3342   | -        |
| ChatCORE        | -        | 0.2763   | -        |

### French benchmarks (FrenchCore, SFT model)

| Metric              | Score  |
|---------------------|--------|
| mmlu_fr              | 0.3114 |
| arc_challenge_fr      | 0.3302 |
| fr_grammar            | 0.3109 |
| fr_vocab              | 0.3950 |
| hellaswag_fr          | 0.2951 |

## Overall pipeline

1. **Pretraining**: `french_ClimbMix/script/shuffle_final.py` → generates the bilingual pretraining dataset (~75B tokens) → used by `nanochat/runs/speedrun_FR.sh`
2. **SFT**: `French_sft/script/conversion_sft.py` → generates the French fine-tuning dataset → used by `nanochat/runs/run_fr_sft.sh`
3. **Evaluation**: `FrenchCore/runs/run_fr_eval.sh` → evaluates the model on French benchmarks (`nanochat/eval_data_fr/`) and computes the FrenchCore score
