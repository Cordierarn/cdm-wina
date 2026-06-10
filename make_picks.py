"""Croise probas modele + cotes Winamax (tous marches) -> data/picks.json.

Usage : python make_picks.py
Le modele Poisson (lambdas de model.json) donne une matrice de scores par match,
d'ou les probas de chaque marche : 1N2, double chance, BTTS, plus/moins,
totaux equipe, score exact, nombre exact de buts, vainqueur rembourse, handicap.
EV = proba * cote - 1 (+ remboursement pour les lignes entieres / DNB).
"""
from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

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

KELLY_FRACTION = 0.25
MIN_EV = 0.03    # seuil minimal pour proposer un pari
MIN_PROBA = 0.02  # ecarte les tres gros outsiders (modele pas assez fin la-bas)
MAX_GOALS = 10

# Melange modele / probas implicites du marche (de-margees) pour calibrer.
# Forces Glicko-2 du soccer-dataset depuis la v2.
# Quand data/pinnacle.json existe, la reference marche des marches 1N2 et
# derives devient Pinnacle de-vige (book sharp ≈ closing line) au lieu de la
# marge de Winamax : strategie classique "sharp reference vs book mou".
MODEL_WEIGHT = 0.40


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()


def fr_to_en(name_fr: str) -> str | None:
    return FR_TO_EN.get(norm(name_fr))


def kelly(p: float, cote: float) -> float:
    b = cote - 1
    if b <= 0:
        return 0.0
    return max((p * b - (1 - p)) / b, 0.0) * KELLY_FRACTION


def dc_tau(h: int, a: int, lh: float, la: float, rho: float) -> float:
    """Correction Dixon-Coles (1997) sur les scores bas (nuls sous-estimes)."""
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


LINE_RE = re.compile(r"(plus|moins) de (\d+(?:,\d)?)")
SCORE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
HCP_RE = re.compile(r"(.+?)\s*\(?([+-]\s*\d+(?:,\d)?)\)?$")


def parse_line(label: str) -> tuple[str, float] | None:
    m = LINE_RE.search(norm(label))
    if not m:
        return None
    return m.group(1), float(m.group(2).replace(",", "."))


def price_bet(marche: str, issues: list[dict], mat, home_fr: str, away_fr: str):
    """proba modele (+ proba remboursement) pour chaque issue, ou None si non pricable."""
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
    """Probas de reference Pinnacle (de-vigees) pour les marches derivables du 1N2.

    pin = {fair_p1, fair_pX, fair_p2} dans le sens Pinnacle ; flipped indique
    que le sens Winamax est inverse. Retourne une liste alignee sur issues,
    ou None si le marche n'est pas derivable du 1N2.
    """
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


def blend_with_market(model_probs, issues):
    """Melange modele/marche en gardant la masse totale du modele
    (gere les marches a issues non exclusives comme la double chance)."""
    inv = [1 / i["cote"] if i["cote"] else 0.0 for i in issues]
    s_inv = sum(inv)
    s_model = sum(p for p, _ in model_probs)
    if s_inv <= 0 or s_model <= 0:
        return model_probs
    return [(MODEL_WEIGHT * p + (1 - MODEL_WEIGHT) * (v / s_inv) * s_model, push)
            for (p, push), v in zip(model_probs, inv)]


def main() -> None:
    model = json.loads((DATA_DIR / "model.json").read_text(encoding="utf-8"))
    odds = json.loads((DATA_DIR / "odds.json").read_text(encoding="utf-8"))

    pred_index = {(m["home"], m["away"]): m for m in model["matches"]}

    pin_index = {}
    pin_file = DATA_DIR / "pinnacle.json"
    if pin_file.exists():
        for p in json.loads(pin_file.read_text(encoding="utf-8"))["matches"]:
            pin_index[(p["home_en"], p["away_en"])] = p

    value_bets, picks_1n2 = [], []
    matched = n_priced = 0

    for o in odds["matches"]:
        home_en, away_en = fr_to_en(o["home_fr"]), fr_to_en(o["away_fr"])
        if not home_en or not away_en:
            continue
        pred = pred_index.get((home_en, away_en)) or pred_index.get((away_en, home_en))
        if not pred:
            continue
        flipped = (home_en, away_en) not in pred_index
        lh = pred["lambda_away"] if flipped else pred["lambda_home"]
        la = pred["lambda_home"] if flipped else pred["lambda_away"]
        mat = score_matrix(lh, la, model.get("rho", 0.0))
        matched += 1

        # reference sharp : Pinnacle dans le meme sens que Winamax si dispo
        pin = pin_index.get((home_en, away_en))
        pin_flipped = False
        if pin is None:
            pin = pin_index.get((away_en, home_en))
            pin_flipped = pin is not None

        for bet in o.get("bets", []):
            issues = bet["issues"]
            model_probs = price_bet(bet["marche"], issues, mat, o["home_fr"], o["away_fr"])
            if model_probs is None or len(model_probs) != len(issues):
                continue
            n_priced += 1
            ref = "winamax"
            pin_p = pinnacle_probs(norm(bet["marche"]), issues, pin, pin_flipped) if pin else None
            if pin_p is not None:
                probs = [(MODEL_WEIGHT * p + (1 - MODEL_WEIGHT) * q, push)
                         for (p, push), q in zip(model_probs, pin_p)]
                ref = "pinnacle"
            else:
                probs = blend_with_market(model_probs, issues)
            for (p, push), issue in zip(probs, issues):
                cote = issue["cote"]
                if not cote or p < MIN_PROBA:
                    continue
                ev = p * cote + push - 1
                row = {
                    "match": f"{o['home_fr']} - {o['away_fr']}",
                    "home_en": home_en, "away_en": away_en,
                    "group": pred.get("group"),
                    "kickoff": o["kickoff"],
                    "marche": bet["marche"],
                    "selection_name": issue["label"],
                    "proba": round(p, 4),
                    "cote": cote,
                    "ev": round(ev, 4),
                    "kelly_pct": round(kelly(p, cote) * 100, 2),
                    "ref": ref,
                }
                if ev >= MIN_EV:
                    value_bets.append(row)
                if norm(bet["marche"]) == "resultat":
                    sel = ("1", "X", "2")[issues.index(issue)]
                    picks_1n2.append({**row, "selection": sel})

    # tri par Kelly plutot qu'EV brut : un EV enorme sur une cote 50 (queue de
    # distribution douteuse) vaut moins qu'un EV correct sur une cote 3.
    value_bets.sort(key=lambda x: (-x["kelly_pct"], -x["ev"]))
    out = {
        "generated_at": int(time.time()),
        "matched_matches": matched,
        "priced_markets": n_priced,
        "value_bets": value_bets,
        "all_picks": picks_1n2,
    }
    (DATA_DIR / "picks.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    n_pin = sum(1 for p in value_bets if p["ref"] == "pinnacle")
    print(f"OK: {matched} matchs, {n_priced} marches prices ({len(pin_index)} matchs avec ref Pinnacle), "
          f"{len(value_bets)} value bets dont {n_pin} vs Pinnacle -> data/picks.json")
    for p in value_bets[:12]:
        print(f"  {p['match'][:34]:34s} {p['marche'][:28]:28s} {str(p['selection_name'])[:18]:18s} "
              f"cote {p['cote']:>5.2f} proba {p['proba']:.0%} EV {p['ev']:+.1%}")


if __name__ == "__main__":
    main()
