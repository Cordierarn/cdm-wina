"""Fonctions communes de pricing et de transformation des lambdas.

Le module est volontairement sans dependance lourde pour pouvoir etre partage
entre make_picks.py, backtest.py et export_model.py.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
BACKTEST_FILE = DATA_DIR / "backtest.json"

MODEL_WEIGHT_FALLBACK = 0.40
MODEL_WEIGHTS = {"1n2": MODEL_WEIGHT_FALLBACK, "totals": MODEL_WEIGHT_FALLBACK,
                 "default": MODEL_WEIGHT_FALLBACK}

MAX_GOALS = 10

# Winamax (FR, normalise sans accents/minuscules) -> noms du repo (EN)
FR_TO_EN = {
    "mexique": "Mexico", "afrique du sud": "South Africa",
    "coree du sud": "South Korea", "republique tcheque": "Czech Republic",
    "canada": "Canada", "bosnie-herzegovine": "Bosnia and Herzegovina",
    "bosnie": "Bosnia and Herzegovina", "qatar": "Qatar", "suisse": "Switzerland",
    "bresil": "Brazil", "maroc": "Morocco", "haiti": "Haiti", "ecosse": "Scotland",
    "etats-unis": "United States", "paraguay": "Paraguay", "australie": "Australia",
    "turquie": "Turkey", "allemagne": "Germany", "curacao": "Curacao",
    "cote d'ivoire": "Ivory Coast", "equateur": "Ecuador", "pays-bas": "Netherlands",
    "japon": "Japan", "suede": "Sweden", "tunisie": "Tunisia", "belgique": "Belgium",
    "egypte": "Egypt", "iran": "Iran", "nouvelle-zelande": "New Zealand",
    "nouvelle zelande": "New Zealand",
    "espagne": "Spain", "cap-vert": "Cape Verde", "arabie saoudite": "Saudi Arabia",
    "uruguay": "Uruguay", "france": "France", "senegal": "Senegal", "irak": "Iraq",
    "norvege": "Norway", "argentine": "Argentina", "algerie": "Algeria",
    "autriche": "Austria", "jordanie": "Jordan", "portugal": "Portugal",
    "rd congo": "DR Congo", "republique democratique du congo": "DR Congo",
    "ouzbekistan": "Uzbekistan", "colombie": "Colombia", "angleterre": "England",
    "croatie": "Croatia", "ghana": "Ghana", "panama": "Panama",
}

LINE_RE = re.compile(r"(plus|moins) de (\d+(?:,\d)?)")
SCORE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
HCP_RE = re.compile(r"(.+?)\s*\(?([+-]\s*\d+(?:,\d)?)\)?$")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower().strip()


def fr_to_en(name_fr: str) -> str | None:
    return FR_TO_EN.get(norm(name_fr))


def market_weight_key(marche_norm: str) -> str:
    if marche_norm in {"resultat", "double chance", "vainqueur (rembourse si match nul)"}:
        return "1n2"
    if marche_norm == "nombre de buts" or marche_norm.startswith("nombre de buts de "):
        return "totals"
    return "default"


def load_model_weights(backtest_file: Path | None = None) -> dict[str, float]:
    weights = dict(MODEL_WEIGHTS)
    path = backtest_file or BACKTEST_FILE
    if not path.exists():
        return weights
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return weights
    fitted = data.get("model_weights") or {}
    for key in ("1n2", "totals"):
        value = fitted.get(key)
        if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
            weights[key] = float(value)
    return weights


def soft_cap_lambda(lam: float, cap: float = 3.5, smooth: float | None = None) -> float:
    """Saturation douce: identite sous le seuil, puis compression continue.

    Contrairement au clip dur, cette forme reste differentiable et preserve les
    ecarts moderes; elle ne compresse franchement que les lambdas au-dessus du
    cap.
    """
    if lam <= 0:
        return 0.0
    if smooth is None:
        smooth = max(cap / 8.0, 0.25)
    if lam <= cap:
        return lam
    excess = lam - cap
    return cap + smooth * math.log1p(excess / smooth)


def dc_tau(h: int, a: int, lh: float, la: float, rho: float) -> float:
    """Correction Dixon-Coles (1997) sur les scores bas."""
    if h == 0 and a == 0:
        return 1 - lh * la * rho
    if h == 0 and a == 1:
        return 1 + lh * rho
    if h == 1 and a == 0:
        return 1 + la * rho
    if h == 1 and a == 1:
        return 1 - rho
    return 1.0


def score_matrix(lh: float, la: float, rho: float = 0.0) -> list[list[float]]:
    ph = [math.exp(-lh) * lh**k / math.factorial(k) for k in range(MAX_GOALS + 1)]
    pa = [math.exp(-la) * la**k / math.factorial(k) for k in range(MAX_GOALS + 1)]
    mat = [[ph[h] * pa[a] * dc_tau(h, a, lh, la, rho)
            for a in range(MAX_GOALS + 1)] for h in range(MAX_GOALS + 1)]
    total = sum(sum(row) for row in mat)
    return [[v / total for v in row] for row in mat]


def p_sum(mat, cond) -> float:
    return sum(mat[h][a] for h in range(MAX_GOALS + 1) for a in range(MAX_GOALS + 1)
               if cond(h, a))


def parse_line(label: str) -> tuple[str, float] | None:
    m = LINE_RE.search(norm(label))
    if not m:
        return None
    return m.group(1), float(m.group(2).replace(",", "."))


def price_bet(marche: str, issues: list[dict], mat, home_fr: str, away_fr: str):
    """Retourne (proba, proba_remboursement) pour chaque issue, ou None."""
    t = norm(marche)
    nh, na = norm(home_fr), norm(away_fr)
    out = []

    if t == "resultat" and len(issues) == 3:
        return [(p_sum(mat, lambda h, a: h > a), 0.0),
                (p_sum(mat, lambda h, a: h == a), 0.0),
                (p_sum(mat, lambda h, a: h < a), 0.0)]

    if t == "double chance" and len(issues) == 3:
        return [(p_sum(mat, lambda h, a: h >= a), 0.0),
                (p_sum(mat, lambda h, a: h != a), 0.0),
                (p_sum(mat, lambda h, a: h <= a), 0.0)]

    if t == "les 2 equipes marquent" and len(issues) == 2:
        btts = p_sum(mat, lambda h, a: h > 0 and a > 0)
        return [(btts, 0.0), (1 - btts, 0.0)]

    if t == "vainqueur (rembourse si match nul)" and len(issues) == 2:
        px = p_sum(mat, lambda h, a: h == a)
        return [(p_sum(mat, lambda h, a: h > a), px),
                (p_sum(mat, lambda h, a: h < a), px)]

    if t == "nombre de buts" or t.startswith("nombre de buts de "):
        if t.startswith("nombre de buts de "):
            team = t[len("nombre de buts de "):]
            if team == nh:
                tot = lambda h, a: h
            elif team == na:
                tot = lambda h, a: a
            else:
                return None
        else:
            tot = lambda h, a: h + a
        for issue in issues:
            parsed = parse_line(issue["label"] or "")
            if not parsed:
                return None
            side, line = parsed
            push = p_sum(mat, lambda h, a: tot(h, a) == line) if line == int(line) else 0.0
            p = p_sum(mat, (lambda h, a: tot(h, a) > line) if side == "plus"
                      else (lambda h, a: tot(h, a) < line))
            out.append((p, push))
        return out

    if t == "score exact":
        for issue in issues:
            m = SCORE_RE.match((issue["label"] or "").strip())
            if m:
                h, a = int(m.group(1)), int(m.group(2))
                out.append((mat[h][a] if h <= MAX_GOALS and a <= MAX_GOALS else 0.0, 0.0))
            elif "autre" in norm(issue["label"] or ""):
                known = sum(p for p, _ in out)
                out.append((max(1 - known, 0.0), 0.0))
            else:
                return None
        return out

    if t == "nombre exact de buts":
        for issue in issues:
            lab = norm(issue["label"] or "")
            m = re.search(r"(\d+)", lab)
            if not m:
                return None
            n = int(m.group(1))
            if "plus" in lab or "+" in lab:
                out.append((p_sum(mat, lambda h, a: h + a >= n), 0.0))
            else:
                out.append((p_sum(mat, lambda h, a: h + a == n), 0.0))
        return out

    if t == "ecart de buts (handicap)" and len(issues) == 2:
        for issue in issues:
            m = HCP_RE.match((issue["label"] or "").strip())
            if not m:
                return None
            team, hcp = norm(m.group(1)), float(m.group(2).replace(",", ".").replace(" ", ""))
            if team == nh:
                margin = lambda h, a: h - a
            elif team == na:
                margin = lambda h, a: a - h
            else:
                return None
            out.append((p_sum(mat, lambda h, a: margin(h, a) + hcp > 0),
                        p_sum(mat, lambda h, a: margin(h, a) + hcp == 0)))
        return out

    return None


def pinnacle_probs(marche_norm: str, issues, pin, flipped: bool):
    """Probas de reference Pinnacle pour les marches derivees du 1N2."""
    p1, px, p2 = pin["fair_p1"], pin["fair_pX"], pin["fair_p2"]
    if flipped:
        p1, p2 = p2, p1
    if marche_norm == "resultat" and len(issues) == 3:
        return [p1, px, p2]
    if marche_norm == "double chance" and len(issues) == 3:
        return [p1 + px, p1 + p2, px + p2]
    if marche_norm == "vainqueur (rembourse si match nul)" and len(issues) == 2:
        return [p1, p2]
    return None


def blend_with_market(model_probs, issues, model_weight: float):
    """Melange modele/marche en gardant la masse totale du modele."""
    inv = [1 / i["cote"] if i["cote"] else 0.0 for i in issues]
    s_inv = sum(inv)
    s_model = sum(p for p, _ in model_probs)
    if s_inv <= 0 or s_model <= 0:
        return model_probs
    return [(model_weight * p + (1 - model_weight) * (v / s_inv) * s_model, push)
            for (p, push), v in zip(model_probs, inv)]
