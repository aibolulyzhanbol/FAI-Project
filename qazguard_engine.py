#!/usr/bin/env python3
"""
QazGuard PhishLab — Core AI Engine & Evaluation Suite (stdlib only, no external deps)
QAIRU AI Fundamentals Final Project 2026 (Topic #31: AI Phishing Detection Lab)

Modules (syllabus weeks):
- Week 6 : Multinomial Naive Bayes with Laplace smoothing (log-sum-exp)
- Week 8 : Logistic Regression (trained by gradient descent with L2) + Maximum Expected Utility
- Week 10: Prompt-injection guardrail + KZ IIN (mod-11 + date) / Luhn DLP tokenizer

Run:  python3 qazguard_engine.py
Test: python3 -m unittest discover -s tests -v
"""

import csv
import math
import os
import re
from typing import Dict, List, Sequence, Tuple
from urllib.parse import urlparse

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset_kz_small.csv")

# ---------------------------------------------------------------------------
# Shared feature definitions (kept identical to the JavaScript code in index.html)
# ---------------------------------------------------------------------------
SUSPICIOUS_TLDS = {"xyz", "top", "click", "buzz", "loan", "work", "tk", "ml", "gq", "shop"}
BRAND_RE = r"(kaspi|egov|halyk|kazpost|qazpost|jusan|beeline|qamqor|salyk)"
OFFICIAL_DOMAINS = (
    "kaspi.kz", "egov.kz", "halykbank.kz", "post.kz", "qazpost.kz",
    "qamqor.gov.kz", "jusan.kz", "beeline.kz", "salyk.kz",
)
URGENCY_RE = (
    r"шұғыл|дереу|бұғатталды|растаңыз|ұтып|бонус|өтемақы|urgent|suspended|immediately|verify|prize|winner"
    r"|срочно|немедленно|заблокирован|подтвердите|выиграли|предоплат|оплатите|иначе|арест"
)
CRED_RE = (
    r"құпиясөз|код|cvv|пин|карта|password|otp|pin|login|credential"
    r"|парол|данные карты|иин|жсн|реквизит"
)
FEATURE_NAMES = ["tld", "brand_spoof", "insecure_ip_at", "url_len", "urgency", "credentials"]


def _host(url: str) -> str:
    try:
        return (urlparse((url or "").strip().lower()).hostname or "")
    except ValueError:
        return ""


def is_official_host(host: str) -> bool:
    return any(host == d or host.endswith("." + d) for d in OFFICIAL_DOMAINS)


def extract_features(url: str, text: str) -> List[float]:
    """Six interpretable features, each in [0, 1]."""
    u = (url or "").strip().lower()
    t = (text or "").lower()
    host = _host(u)
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    x_tld = 1.0 if tld in SUSPICIOUS_TLDS else 0.0
    x_spoof = 1.0 if (re.search(BRAND_RE, host) and not is_official_host(host)) else 0.0
    is_ip = bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))
    x_http_ip = 1.0 if (u.startswith("http://") or is_ip or "@" in u) else 0.0
    x_len = min(len(u) / 75.0, 1.0)
    x_urg = min(len(re.findall(URGENCY_RE, t)) / 3.0, 1.0)
    x_cred = min(len(re.findall(CRED_RE, t)) / 2.0, 1.0)
    return [x_tld, x_spoof, x_http_ip, x_len, x_urg, x_cred]


# ---------------------------------------------------------------------------
# Week 6 — Naive Bayes
# ---------------------------------------------------------------------------
class NaiveBayesClassifier:
    """Multinomial Naive Bayes with Laplace smoothing."""

    def __init__(self, alpha: float = 1.0, prior_phish: float = 0.5):
        self.alpha = alpha
        self.prior_phish = prior_phish
        self.vocab: Dict[str, Dict[str, int]] = {}
        self.total_phish = 0
        self.total_safe = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        cleaned = re.sub(r"https?://\S+", " ", (text or "").lower())
        cleaned = re.sub(r"[^a-zа-яёәіңғүұқөһ0-9\s]", " ", cleaned)
        return [w for w in cleaned.split() if len(w) >= 2 and not w.isdigit()]

    def fit(self, phish_docs: Sequence[str], safe_docs: Sequence[str]) -> None:
        for docs, key in ((phish_docs, "phish"), (safe_docs, "safe")):
            for doc in docs:
                for token in self.tokenize(doc):
                    self.vocab.setdefault(token, {"phish": 0, "safe": 0})
                    self.vocab[token][key] += 1
                    if key == "phish":
                        self.total_phish += 1
                    else:
                        self.total_safe += 1

    def predict_proba(self, text: str) -> float:
        tokens = self.tokenize(text)
        if not tokens:
            return self.prior_phish
        v = max(len(self.vocab), 1)
        lp = math.log(self.prior_phish)
        ls = math.log(1.0 - self.prior_phish)
        for t in tokens:
            c = self.vocab.get(t, {"phish": 0, "safe": 0})
            lp += math.log((c["phish"] + self.alpha) / (self.total_phish + self.alpha * v))
            ls += math.log((c["safe"] + self.alpha) / (self.total_safe + self.alpha * v))
        m = max(lp, ls)
        ep, es = math.exp(lp - m), math.exp(ls - m)
        return ep / (ep + es)


# ---------------------------------------------------------------------------
# Week 8 — Logistic Regression + Maximum Expected Utility
# ---------------------------------------------------------------------------
def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class LogisticURLDecisionAgent:
    """Logistic regression over 6 features (weights learned, not hand-set) + MEU decision."""

    def __init__(self, fn_cost: float = 100.0, lr: float = 0.5, epochs: int = 3000, l2: float = 0.01):
        self.fn_cost = fn_cost
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        self.weights: List[float] = [0.0] * len(FEATURE_NAMES)
        self.bias: float = 0.0

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> None:
        n, d = len(X), len(FEATURE_NAMES)
        w, b = [0.0] * d, 0.0
        for _ in range(self.epochs):
            gw, gb = [0.0] * d, 0.0
            for xi, yi in zip(X, y):
                err = _sigmoid(b + sum(wj * xj for wj, xj in zip(w, xi))) - yi
                for j in range(d):
                    gw[j] += err * xi[j]
                gb += err
            for j in range(d):
                w[j] -= self.lr * (gw[j] / n + self.l2 * w[j])
            b -= self.lr * gb / n
        self.weights, self.bias = w, b

    def predict_ml_proba(self, url: str, text: str) -> float:
        x = extract_features(url, text)
        return _sigmoid(self.bias + sum(w * v for w, v in zip(self.weights, x)))

    def optimal_action(self, p_phish: float) -> Tuple[str, Dict[str, float]]:
        eu = {
            "BLOCK": (1.0 - p_phish) * (-25.0),
            "QUARANTINE": p_phish * (-10.0) + (1.0 - p_phish) * (-8.0),
            "ALLOW": p_phish * (-self.fn_cost) + (1.0 - p_phish) * 15.0,
        }
        return max(eu.items(), key=lambda kv: kv[1])[0], eu


# ---------------------------------------------------------------------------
# Week 10 — Guardrail + DLP
# ---------------------------------------------------------------------------
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"(disregard|forget)\s+(all\s+)?(previous|prior)\s+(instructions|rules)",
    r"алдыңғы\s+нұсқауларды\s+елеме",
    r"игнорируй\s+(все\s+)?(предыдущие|прежние)\s+инструкции",
    r"(system\s+prompt|жүйелік\s+промпт|системн\w+\s+промпт)",
    r"(jailbreak|dan\s+mode|bypass\s+safety|шектеуді\s+айналып\s+өту)",
]


def kz_iin_control_digit(first11: str) -> int:
    d = [int(c) for c in first11]
    ctrl = sum(d[i] * (i + 1) for i in range(11)) % 11
    if ctrl == 10:
        w2 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2]
        ctrl = sum(d[i] * w2[i] for i in range(11)) % 11
    return ctrl


class AISafetyGuardrailDLP:
    """Prompt-injection detection + KZ IIN / card / phone tokenization."""

    @staticmethod
    def validate_kz_iin(iin: str) -> bool:
        """12 digits, valid date YYMMDD, 7th digit 1..6, mod-11 control digit (11 = invalid)."""
        if not re.fullmatch(r"\d{12}", iin):
            return False
        mm, dd, century = int(iin[2:4]), int(iin[4:6]), int(iin[6])
        if not (1 <= mm <= 12 and 1 <= dd <= 31 and 1 <= century <= 6):
            return False
        ctrl = kz_iin_control_digit(iin[:11])
        return ctrl != 10 and ctrl == int(iin[11])

    @staticmethod
    def validate_luhn(card_str: str) -> bool:
        digits = [int(c) for c in re.sub(r"\D", "", card_str)]
        if len(digits) != 16:
            return False
        total = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
        return total % 10 == 0

    def inspect_and_sanitize(self, text: str) -> Tuple[str, float, List[str]]:
        sanitized = text or ""
        risk, alerts = 0.0, []

        for pat in INJECTION_PATTERNS:
            if re.search(pat, sanitized, re.I):
                risk = max(risk, 0.92)
                alerts.append("PROMPT_INJECTION")
                break

        def mask_card(m: re.Match) -> str:
            nonlocal risk
            ok = self.validate_luhn(m.group(0))
            risk = max(risk, 0.65)
            alerts.append("CARD_LUHN_VALID" if ok else "CARD_FORMAT")
            return "[TOKEN_CARD_REDACTED]"

        def mask_iin(m: re.Match) -> str:
            nonlocal risk
            ok = self.validate_kz_iin(m.group(0))
            risk = max(risk, 0.60)
            alerts.append("IIN_MOD11_VALID" if ok else "IIN_MASKED")
            return f"[TOKEN_KZ_IIN_{'VERIFIED' if ok else 'MASKED'}]"

        def mask_phone(m: re.Match) -> str:
            alerts.append("KZ_PHONE")
            return "[TOKEN_KZ_PHONE]"

        sanitized = re.sub(r"\b(?:\d{4}[-\s]?){3}\d{4}\b", mask_card, sanitized)
        sanitized = re.sub(r"\b\d{12}\b", mask_iin, sanitized)
        sanitized = re.sub(r"(?:\+7|8)[\s-]?\(?7\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}\b", mask_phone, sanitized)
        return sanitized, risk, alerts


# ---------------------------------------------------------------------------
# Data, pipeline, metrics
# ---------------------------------------------------------------------------
def load_dataset(path: str = DATASET_PATH) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def combined_risk(nb_p: float, ml_p: float) -> float:
    return 0.5 * nb_p + 0.5 * ml_p


def train_models(rows: Sequence[Dict[str, str]]) -> Tuple[NaiveBayesClassifier, LogisticURLDecisionAgent]:
    nb = NaiveBayesClassifier()
    nb.fit([r["text"] for r in rows if r["label"] == "PHISHING"], [r["text"] for r in rows if r["label"] == "SAFE"])
    ag = LogisticURLDecisionAgent()
    ag.fit([extract_features(r["url"], r["text"]) for r in rows], [1 if r["label"] == "PHISHING" else 0 for r in rows])
    return nb, ag


def confusion(rows, predict) -> Dict[str, int]:
    c = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for r in rows:
        pred, true = predict(r) >= 0.5, r["label"] == "PHISHING"
        c["tp" if pred and true else "tn" if not pred and not true else "fp" if pred else "fn"] += 1
    return c


def metrics(c: Dict[str, int]) -> Dict[str, float]:
    n = sum(c.values())
    p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    return {"accuracy": (c["tp"] + c["tn"]) / n if n else 0.0, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if (p + r) else 0.0}


def group_kfold(rows: Sequence[Dict[str, str]], k: int = 5, seed: int = 7):
    """Cross-validation that keeps all paraphrases of one scheme (column `group`) in the same fold."""
    import random
    groups = sorted({r["group"] for r in rows})
    random.Random(seed).shuffle(groups)
    fold_of = {g: i % k for i, g in enumerate(groups)}
    for f in range(k):
        yield [r for r in rows if fold_of[r["group"]] != f], [r for r in rows if fold_of[r["group"]] == f]


def evaluate_all(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    train = [r for r in rows if r["split"] == "TRAIN"]
    test = [r for r in rows if r["split"] == "TEST"]
    nb, ag = train_models(train)
    preds = {
        "naive_bayes": lambda r: nb.predict_proba(r["text"]),
        "logistic": lambda r: ag.predict_ml_proba(r["url"], r["text"]),
        "combined": lambda r: combined_risk(nb.predict_proba(r["text"]), ag.predict_ml_proba(r["url"], r["text"])),
    }
    out: Dict[str, object] = {"n_train": len(train), "n_test": len(test), "test": {}, "errors": []}
    for name, fn in preds.items():
        c = confusion(test, fn)
        out["test"][name] = {"confusion": c, **metrics(c)}
    majority = max(("PHISHING", "SAFE"), key=lambda l: sum(r["label"] == l for r in train))
    out["majority_baseline_accuracy"] = sum(r["label"] == majority for r in test) / len(test)
    for r in test:
        p = preds["combined"](r)
        if (p >= 0.5) != (r["label"] == "PHISHING"):
            out["errors"].append({"id": r["id"], "title": r["title"], "label": r["label"], "p": round(p, 3)})
    cv = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for tr, te in group_kfold(rows):
        nb_k, ag_k = train_models(tr)
        c = confusion(te, lambda r: combined_risk(nb_k.predict_proba(r["text"]), ag_k.predict_ml_proba(r["url"], r["text"])))
        for key in cv:
            cv[key] += c[key]
    out["cv_combined"] = {"confusion": cv, **metrics(cv)}
    out["weights"] = dict(zip(FEATURE_NAMES, [round(w, 3) for w in ag.weights]), bias=round(ag.bias, 3))
    return out


def main() -> None:
    rows = load_dataset()
    res = evaluate_all(rows)
    print("=== QazGuard PhishLab — evaluation on", os.path.basename(DATASET_PATH), "===")
    print(f"train={res['n_train']}  test={res['n_test']}  majority-class baseline accuracy={res['majority_baseline_accuracy']:.3f}")
    for name, m in res["test"].items():
        c = m["confusion"]
        print(f"[test] {name:12s} acc={m['accuracy']:.3f} prec={m['precision']:.3f} rec={m['recall']:.3f} "
              f"f1={m['f1']:.3f}  TP={c['tp']} TN={c['tn']} FP={c['fp']} FN={c['fn']}")
    cv = res["cv_combined"]
    print(f"[5-fold group CV, combined, all {len(rows)} rows] acc={cv['accuracy']:.3f} prec={cv['precision']:.3f} "
          f"rec={cv['recall']:.3f} f1={cv['f1']:.3f}  {cv['confusion']}")
    print("learned LR weights:", res["weights"])
    print("test errors (combined):", res["errors"] or "none")

    # Demo of full pipeline incl. DLP (synthetic IIN/card with valid checksums)
    nb, ag = train_models([r for r in rows if r["split"] == "TRAIN"])
    dlp = AISafetyGuardrailDLP()
    iin = "080720" + "5" + "0019"
    iin += str(kz_iin_control_digit(iin))
    url = "http://egov-tolem-kz.xyz/claim"
    text = f"eGov: вам положена выплата. Введите ИИН {iin} и карту 4400 4301 8821 9015 срочно, тел +7 707 555 12 34."
    sanitized, r_safe, flags = dlp.inspect_and_sanitize(text)
    risk = max(combined_risk(nb.predict_proba(text), ag.predict_ml_proba(url, text)), r_safe)
    action, eu = ag.optimal_action(risk)
    print(f"\nDemo: risk={risk:.3f} action={action} flags={flags}\nsanitized: {sanitized}")


if __name__ == "__main__":
    main()
