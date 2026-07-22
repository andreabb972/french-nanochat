# Evaluation Benchmarks

The evaluation of the NanoChat-FR model relies on a suite of benchmarks inspired by the **CroissantLLM** and **Pensez** research papers. These benchmarks aim to assess both linguistic proficiency and general knowledge.

## Benchmarks Overview

*   **French Language Test**: 
    *   Composed of two distinct sections: **fr-grammar** and **fr-vocabulary**.
    *   Format: Multiple-choice questions.

*   **HellaSwag (French translation)**: 
    *   Objective: Predict the most logical completion of a sentence describing a daily-life situation.
    *   Mechanism: The model selects the completion with the highest probability.

*   **MMLU-FR**:
    *   Objective: Evaluate general knowledge across various domains.
    *   Format: Multiple-choice questions covering 57 different subjects.

*   **ARC-Challenge-FR (translated)**: 
    *   Objective: Assess scientific reasoning.
    *   Format: Multiple-choice questions based on elementary and middle school-level science curricula.

## Datasets Sources

The datasets used in this project are sourced from the following repositories:

*   **MMLU-FR and ARC-Challenge-FR**: 
    [https://github.com/laiviet/lm-evaluation-harness/tree/main/datasets](https://github.com/laiviet/lm-evaluation-harness/tree/main/datasets)

*   **French Language Test, HellaSwag (French), and other benchmarks**: 
    [https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/french_bench](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/french_bench)

## Pretraining Corpora

To build the final pretraining dataset for NanoChat-FR, three subsampled corpora from the **CroissantLLM** project were used, originally designed for scaling-law experiments. Each one targets a different language/modality and was sourced via manual curation and filtering from established open corpora.

*   **French corpus (`manu/french-30b`)**:
    *   Dominated by **OSCAR** (filtered web crawl, ~38% of tokens) and **Wikisource**.
    *   Also includes French **OpenData** sources (legal texts, parliamentary debates, court rulings, administrative documents), **Wikipedia FR**, **Project Gutenberg** (public-domain literature), and **WMT En-Fr parallel data**.
    *   Reflects the diverse, curated French mix described in the CroissantLLM paper (Table 9).

*   **English corpus (`manu/english-60b`)**:
    *   Entirely sampled from **SlimPajama**, a cleaned and deduplicated version of RedPajama.
    *   Internally composed of CommonCrawl, C4, GitHub, StackExchange, ArXiv, Wikipedia EN, and books.

*   **Code corpus (`manu/code_20b`)**:
    *   Heavily Python-centric: **PypiClean** (PyPI packages, ~47%) and **StarcoderData Python** (~18%) make up the bulk.
    *   Also includes **StarcoderData Markdown**, **Jupyter notebooks**, **JSON**, and **CodeContests** (competitive programming, mostly Python3).
    *   Note: unlike CroissantLLM's full training corpus, this subsample does not include other languages such as Java, JavaScript, C/C++, or SQL.

## Datasets Sources

*   **French, English, and Code corpora**:
    [https://huggingface.co/datasets/manu/french-30b](https://huggingface.co/datasets/manu/french-30b)
    [https://huggingface.co/datasets/manu/english-60b](https://huggingface.co/datasets/manu/english-60b)
    [https://huggingface.co/datasets/manu/code_20b](https://huggingface.co/datasets/manu/code_20b)

## Fine-tuning Corpus (SFT)

For the supervised fine-tuning (SFT) stage of NanoChat-FR, the dataset used is the one released by the **CroissantLLM** project for training their chat model (**CroissantLLMChat**).

*   **CroissantLLM-2201-sft**:
    *   Built from public chat datasets **Ultrachat** and **Wildchat**, containing ChatGPT interactions in both English and French.
    *   Also incorporates **translation data** (~4% of the SFT dataset, ~12k samples) to reinforce bilingual capabilities during fine-tuning.
    *   Used in the CroissantLLM paper to fine-tune the base model into CroissantLLMChat, as well as comparison baselines (Bloom-1b7, TinyLlama).

## Datasets Sources

*   **SFT dataset**:
    [https://huggingface.co/datasets/croissantllm/CroissantLLM-2201-sft](https://huggingface.co/datasets/croissantllm/CroissantLLM-2201-sft)