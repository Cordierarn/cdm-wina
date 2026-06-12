"""Croise probas modele + cotes Winamax (tous marches) -> data/picks.json.

Usage : python make_picks.py
Le modele Poisson (lambdas de model.json) donne une matrice de scores par match,
d'ou les probas de chaque marche : 1N2, double chance, BTTS, plus/moins,
totaux equipe, score exact, nombre exact de buts, vainqueur rembourse, handicap.
EV = proba * cote - 1 (+ remboursement pour les lignes entieres / DNB).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from pricing import (
    blend_with_market,
    fr_to_en,
    market_weight_key,
    load_model_weights,
    norm,
    pinnacle_probs,
    price_bet,
    score_matrix,
)

DATA_DIR = Path(__file__).parent / "data"

KELLY_FRACTION = 0.25
MIN_EV = 0.03    # seuil minimal pour proposer un pari
MIN_PROBA = 0.02  # ecarte les tres gros outsiders (modele pas assez fin la-bas)
MAX_GOALS = 10

# Marches sans backtest possible (pas de cotes de cloture historiques gratuites) :
# exclus des picks par defaut. Passer a True pour les reactiver.
INCLUDE_UNVALIDATED_MARKETS = False
UNVALIDATED_MARKETS = {"score exact", "nombre exact de buts"}

def kelly(p: float, cote: float) -> float:
    b = cote - 1
    if b <= 0:
        return 0.0
    return max((p * b - (1 - p)) / b, 0.0) * KELLY_FRACTION


def main() -> None:
    model_weights = load_model_weights()
    model = json.loads((DATA_DIR / "model.json").read_text(encoding="utf-8"))
    odds = json.loads((DATA_DIR / "odds.json").read_text(encoding="utf-8"))

    pred_index = {(m["home"], m["away"]): m for m in model["matches"]}

    pin_index = {}
    pin_file = DATA_DIR / "pinnacle.json"
    if pin_file.exists():
        for p in json.loads(pin_file.read_text(encoding="utf-8"))["matches"]:
            pin_index[(p["home_en"], p["away_en"])] = p

    value_bets, picks_1n2 = [], []
    all_rows = []  # toutes les issues pricees, pour les propositions par match
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
        mat = score_matrix(lh, la, model.get("rho", 0.0),
                           pi_zero=model.get("pi_zero", 0.0),
                           lambda3=model.get("lambda3", 0.0))
        matched += 1

        # reference sharp : Pinnacle dans le meme sens que Winamax si dispo
        pin = pin_index.get((home_en, away_en))
        pin_flipped = False
        if pin is None:
            pin = pin_index.get((away_en, home_en))
            pin_flipped = pin is not None

        for bet in o.get("bets", []):
            if not INCLUDE_UNVALIDATED_MARKETS and norm(bet["marche"]) in UNVALIDATED_MARKETS:
                continue
            issues = bet["issues"]
            model_probs = price_bet(bet["marche"], issues, mat, o["home_fr"], o["away_fr"])
            if model_probs is None or len(model_probs) != len(issues):
                continue
            n_priced += 1
            ref = "winamax"
            pin_p = pinnacle_probs(norm(bet["marche"]), issues, pin, pin_flipped) if pin else None
            if pin_p is not None:
                weight = model_weights.get(market_weight_key(norm(bet["marche"])), model_weights["default"])
                probs = [(weight * p + (1 - weight) * q, push)
                         for (p, push), q in zip(model_probs, pin_p)]
                ref = "pinnacle"
            else:
                weight_key = market_weight_key(norm(bet["marche"]))
                probs = blend_with_market(model_probs, issues, model_weights.get(weight_key, model_weights["default"]))
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
                all_rows.append(row)
                if ev >= MIN_EV:
                    value_bets.append(row)
                if norm(bet["marche"]) == "resultat":
                    sel = ("1", "X", "2")[issues.index(issue)]
                    picks_1n2.append({**row, "selection": sel})

    # tri par Kelly plutot qu'EV brut : un EV enorme sur une cote 50 (queue de
    # distribution douteuse) vaut moins qu'un EV correct sur une cote 3.
    value_bets.sort(key=lambda x: (-x["kelly_pct"], -x["ev"]))
    # propositions par match : pour CHAQUE match, les 5 meilleures issues
    # (EV decroissant, proba >= 10% pour rester jouable) + le 1N2 le plus probable
    par_match: dict[str, dict] = {}
    for r in all_rows:
        par_match.setdefault(r["match"], {"kickoff": r["kickoff"], "group": r["group"],
                                          "props": [], "favori_1n2": None})
    for r in all_rows:
        if r["proba"] >= 0.10:
            par_match[r["match"]]["props"].append(r)
    for p in picks_1n2:
        slot = par_match[p["match"]]
        if slot["favori_1n2"] is None or p["proba"] > slot["favori_1n2"]["proba"]:
            slot["favori_1n2"] = p
    for m in par_match.values():
        m["props"].sort(key=lambda x: -x["ev"])
        m["props"] = m["props"][:5]

    # Score exact le plus probable pour chaque match
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
        mat = score_matrix(lh, la, model.get("rho", 0.0),
                           pi_zero=model.get("pi_zero", 0.0),
                           lambda3=model.get("lambda3", 0.0))
        # Top 5 scores les plus probables
        scores = []
        for h in range(MAX_GOALS + 1):
            for a in range(MAX_GOALS + 1):
                scores.append((mat[h][a], h, a))
        scores.sort(reverse=True)
        match_key = f"{o['home_fr']} - {o['away_fr']}"
        if match_key in par_match:
            par_match[match_key]["top_scores"] = [
                {"score": f"{h}-{a}", "proba": round(p, 4)}
                for p, h, a in scores[:8]
            ]

    out = {
        "generated_at": int(time.time()),
        "matched_matches": matched,
        "priced_markets": n_priced,
        "value_bets": value_bets,
        "all_picks": picks_1n2,
        "par_match": par_match,
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
