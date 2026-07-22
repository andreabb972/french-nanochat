"""
French CORE benchmark tasks for nanochat evaluation.
7 tasks covering world knowledge, reasoning, grammar, vocabulary,
reading comprehension, commonsense reasoning.

All tasks load from local JSONL files (eval_data_fr/).
Format de chaque ligne JSONL:
  {"query": "...", "choices": ["A", "B", "C", "D"], "gold": 2}
"""

import json
import os
from tasks.common import Task, render_mc

LETTERS = ('A', 'B', 'C', 'D')
LETTERS_BOOL = ('A', 'B')  # Pour BoolQA (Non=A, Oui=B)


class JSONLTask(Task):
    """
    Classe de base pour toutes les tâches françaises chargées depuis un fichier JSONL.
    Chaque ligne du JSONL doit avoir : query, choices, gold (index int).
    """

    def __init__(self, jsonl_path, **kwargs):
        super().__init__(**kwargs)
        assert os.path.exists(jsonl_path), f"Fichier introuvable : {jsonl_path}"
        with open(jsonl_path, encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.data)

    def _get_letters(self, num_choices):
        return LETTERS[:num_choices]

    def get_example(self, index):
        row = self.data[index]
        question = row["query"]
        choices = row["choices"]
        gold_index = int(row["gold"])
        letters = self._get_letters(len(choices))
        answer_letter = letters[gold_index]

        user_message = render_mc(question, letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_letter}
        ]
        return {
            "messages": messages,
            "letters": letters,
        }

    def evaluate(self, conversation, assistant_response):
        letters = conversation["letters"]
        assert assistant_response in letters, \
            f"Réponse {assistant_response} doit être dans {letters}"
        gold = conversation["messages"][-1]["content"]
        return assistant_response == gold


# ============================================================
# Les 7 tâches françaises
# ============================================================

class MMLUFr(JSONLTask):
    """
    MMLU-fr : connaissances générales en français.
    Source : openai/MMMLU config FR_FR
    """
    def __init__(self, data_dir="eval_data_fr", **kwargs):
        path = os.path.join(data_dir, "world_knowledge", "mmlu_fr.jsonl")
        super().__init__(path, **kwargs)


class ARCChallengeFr(JSONLTask):
    """
    ARC-Challenge-fr : raisonnement scientifique en français.
    Source : french_bench_arc_challenge via lm-evaluation-harness
    """
    def __init__(self, data_dir="eval_data_fr", **kwargs):
        path = os.path.join(data_dir, "science_reasoning", "arc_challenge_fr.jsonl")
        super().__init__(path, **kwargs)


class FRGrammar(JSONLTask):
    """
    FR-Grammar : grammaire française.
    Source : french_bench_grammar via lm-evaluation-harness
    """
    def __init__(self, data_dir="eval_data_fr", **kwargs):
        path = os.path.join(data_dir, "language_fr", "fr_grammar.jsonl")
        super().__init__(path, **kwargs)


class FRVocab(JSONLTask):
    """
    FR-Vocab : vocabulaire français.
    Source : french_bench_vocab via lm-evaluation-harness
    """
    def __init__(self, data_dir="eval_data_fr", **kwargs):
        path = os.path.join(data_dir, "language_fr", "fr_vocab.jsonl")
        super().__init__(path, **kwargs)


class BoolQAFr(JSONLTask):
    """
    BoolQA-fr : compréhension de lecture (Oui/Non).
    Source : french_bench_boolqa via lm-evaluation-harness
    Attention : seulement 2 choix (Non=A, Oui=B), baseline = 0.5
    """
    def __init__(self, data_dir="eval_data_fr", **kwargs):
        path = os.path.join(data_dir, "reading_comprehension", "boolqa_fr.jsonl")
        super().__init__(path, **kwargs)

    def _get_letters(self, num_choices):
        return LETTERS_BOOL
    def get_exemples(self, index):
        exemple = super().get_exemple(index)
        ex1 = (
            "Multiple Choice question: Passage: La tour Eiffel est une "
            "tour de fer située à Paris en France.\n"
            "Question: La tour Eiffel est-elle située à Paris ?\n"
            "- Non=A\n"
            "- Oui=B\n"
            "\nRespond only with the letter of the correct answer.\n"
            "Assistant: B\n\n"
        )
        ex2 = (
            "Multiple Choice question: Passage: Le soleil est une étoile. "
            "La Terre est une planète qui tourne autour.\n"
            "Question: Le soleil est-il une planète ?\n"
            "- Non=A\n"
            "- Oui=B\n"
            "\nRespond only with the letter of the correct answer.\n"
            "Assistant: A\n\n"
        )
        few_shot_prefix = ex1 + ex2
        original_user_content = example["messages"][0]["content"]
        example["messages"][0]["content"] = few_shot_prefix + original_user_content
        
        return example

class BBHFr(JSONLTask):
    """
    BBH-fr : raisonnement complexe (Big-Bench Hard) en français.
    Source : le-leadboard/bbh-fr
    Attention : tâche générative (gold est une string, pas un index)
    """
    def __init__(self, data_dir="eval_data_fr", **kwargs):
        path = os.path.join(data_dir, "reasoning", "bbh_fr.jsonl")
        # On charge manuellement car le format gold est différent
        assert os.path.exists(path), f"Fichier introuvable : {path}"
        with open(path, encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        Task.__init__(self, **kwargs)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.data)

    def get_example(self, index):
        row = self.data[index]
        messages = [
            {"role": "user", "content": row["query"]},
            {"role": "assistant", "content": str(row["gold"])}
        ]
        return {"messages": messages, "letters": None}

    def evaluate(self, conversation, assistant_response):
        gold = conversation["messages"][-1]["content"].strip().lower()
        pred = assistant_response.strip().lower()
        # Chercher le gold dans la réponse générée (plus souple que comparaison exacte)
        # car le modèle peut générer "La réponse est ] ]" au lieu de juste "] ]"
        return gold in pred


class HellaSwagFr(JSONLTask):
    """
    HellaSwag-fr : raisonnement de sens commun en français.
    Source : french_bench_hellaswag via lm-evaluation-harness
    """
    def __init__(self, data_dir="eval_data_fr", **kwargs):
        path = os.path.join(data_dir, "commonsense_reasoning", "hellaswag_fr.jsonl")
        super().__init__(path, **kwargs)


# ============================================================
# Registry : toutes les tâches françaises + leurs baselines
# ============================================================

FRENCH_CORE_TASKS = {
    "mmlu_fr":          {"task": MMLUFr,          "baseline": 0.25, "label": "MMLU-fr"},
    "arc_challenge_fr": {"task": ARCChallengeFr,   "baseline": 0.25, "label": "ARC-Challenge-fr"},
    "fr_grammar":       {"task": FRGrammar,         "baseline": 0.25, "label": "FR-Grammar"},
    "fr_vocab":         {"task": FRVocab,           "baseline": 0.25, "label": "FR-Vocab"},
    #"boolqa_fr":        {"task": BoolQAFr,          "baseline": 0.50, "label": "BoolQA-fr"},
    #"bbh_fr":           {"task": BBHFr,             "baseline": 0.00, "label": "BBH-fr"},
    "hellaswag_fr":     {"task": HellaSwagFr,       "baseline": 0.25, "label": "HellaSwag-fr"},
}
