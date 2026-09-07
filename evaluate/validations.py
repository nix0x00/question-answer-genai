import re, string
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

MAX_ANSWER_LEN = 30 # SQuAD answers are short; caps runaway spans

def normalize_answer(s):
    def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t): return " ".join(t.split())
    def remove_punc(t):     return "".join(c for c in t if c not in set(string.punctuation))
    def lower(t):           return t.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_em(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))

def compute_f1(pred, gold):
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return int(pred_toks == gold_toks)          # both empty → 1, else 0
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall    = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)

def max_over_golds(metric_fn, pred, golds):
    return max(metric_fn(pred, g) for g in golds)   # best match across the ~3 gold answers

def decode_one(s_log, e_log, mask, offsets, context):
    mask = np.asarray(mask, dtype=bool)
    s = np.where(mask, s_log, -1e9)      # kill pad positions
    e = np.where(mask, e_log, -1e9)

    best_score, best_i, best_j = -1e18, 0, 0
    for i in np.argsort(s)[-20:]:                          # top-20 start candidates (speed)
        j_hi = min(i + MAX_ANSWER_LEN, len(e))
        j = i + int(np.argmax(e[i:j_hi]))                  # best end AFTER start, within window
        score = s[i] + e[j]
        if score > best_score:
            best_score, best_i, best_j = score, i, j

    if best_i >= len(offsets) or best_j >= len(offsets):
        return ""
    return context[offsets[best_i][0] : offsets[best_j][1]]   # token span → original chars

def get_score(eval_df, gold_answers, start_logits, end_logits):
    em = f1 = 0.0
    for i in range(len(eval_df)):
        qid   = eval_df.iloc[i]["question_id"]
        golds = gold_answers[qid]
        pred  = decode_one(start_logits[i], end_logits[i],
                        eval_df.iloc[i]["context_mask"],
                        eval_df.iloc[i]["context_offsets"],
                        eval_df.iloc[i]["context"])
        em += max_over_golds(compute_em, pred, golds)
        f1 += max_over_golds(compute_f1, pred, golds)

    n = len(eval_df)

    return em, f1, n

def plot_graph(h, title):
    plt.plot(h["loss"], label="train")
    plt.plot(h["val_loss"], label="val")
    plt.legend()
    plt.show()


    for k in ["start", "end"]:
        plt.plot(h[f"{k}_sparse_categorical_accuracy"], label=f"train {k}")
        plt.plot(h[f"val_{k}_sparse_categorical_accuracy"], label=f"val {k}")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title(title); plt.show()