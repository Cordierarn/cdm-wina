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

    # Pari sur par match : la selection la plus probable parmi celles a EV
    # positif (sinon la plus probable a EV > -3%), cote >= 1.20 pour rester
    # interessante. C'est aussi la jambe candidate pour les combines.
    for match_key, slot in par_match.items():
        candidates = [r for r in all_rows if r["match"] == match_key
                      and r["cote"] >= 1.25 and r["proba"] >= 0.55]
        if not candidates:
            slot["pari_sur"] = None
            continue
        # privilegier l'EV, mais ne jamais laisser un match sans pari sur :
        # parmi les candidats, prendre le meilleur compromis proba*EV
        pos = [r for r in candidates if r["ev"] > 0]
        pool = pos or [r for r in candidates if r["ev"] > -0.06] or candidates
        slot["pari_sur"] = max(pool, key=lambda r: (r["proba"], r["ev"]))

    # Combines du jour : 1 jambe max par match (les paris d'un meme match sont
    # correles et interdits en combine), probas multipliees (matchs independants).
    # 3 profils : sur (2 jambes), equilibre (3), ambitieux (4-5).
    import datetime as _dt

    def day_of(ts):
        return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")

    # Une seule jambe par match (les paris d'un meme match sont correles).
    # On distingue deux pools :
    #  - "safe" : la selection la plus probable du match (pari sur) -> combo qui PASSE
    #  - "value" : la meilleure jambe +EV proba>=0.40 -> combo +EV a long terme
    # marches remboursables (push) : non combinables proprement -> exclus des combos
    REFUNDABLE = {"vainqueur (rembourse si match nul)"}

    def combinable(r: dict) -> bool:
        return r["cote"] >= 1.20 and norm(r["marche"]) not in REFUNDABLE

    safe_by_day: dict[str, list[dict]] = {}
    value_by_day: dict[str, list[dict]] = {}
    for match_key, slot in par_match.items():
        day = day_of(slot["kickoff"])
        safe = [r for r in all_rows if r["match"] == match_key
                and combinable(r) and r["proba"] >= 0.55]
        if safe:
            safe_by_day.setdefault(day, []).append(max(safe, key=lambda r: r["proba"]))
        cands = [r for r in all_rows if r["match"] == match_key
                 and combinable(r) and r["proba"] >= 0.30 and r["ev"] >= 0.03]
        if cands:
            value_by_day.setdefault(day, []).append(max(cands, key=lambda r: r["ev"]))

    combos = []
    seen_combo = set()
    def emit_combo(day: str, label: str, chosen: list[dict]) -> None:
        if len(chosen) < 2:
            return
        sig = (day, tuple(sorted(leg["match"] for leg in chosen)))
        if sig in seen_combo:
            return
        seen_combo.add(sig)
        p_joint = 1.0
        cote_combo = 1.0
        for leg in chosen:
            p_joint *= leg["proba"]
            cote_combo *= leg["cote"]
        combos.append({
            "day": day,
            "type": label,
            "legs": [{k: leg[k] for k in ("match", "marche", "selection_name",
                                          "cote", "proba", "ev", "kickoff")} for leg in chosen],
            "proba": round(p_joint, 4),
            "cote": round(cote_combo, 2),
            "ev": round(p_joint * cote_combo - 1, 4),
            "kelly_pct": round(kelly(p_joint, cote_combo) * 100, 2),
        })

    all_days = sorted(set(safe_by_day) | set(value_by_day))
    for day in all_days:
        safe_legs = sorted(safe_by_day.get(day, []), key=lambda r: -r["proba"])
        value_legs = sorted(value_by_day.get(day, []), key=lambda r: -r["ev"])
        # sur : les 2 selections les plus probables du jour (passe le + souvent)
        emit_combo(day, "sur", safe_legs[:2])
        # value : les meilleures jambes +EV (seul profil gagnant a long terme)
        emit_combo(day, "value", value_legs[:3])
        # ambitieux : 4 plus probables, grosse cote, faible proba jointe
        emit_combo(day, "ambitieux", safe_legs[:4])

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
        "combos": combos,
    }
    (DATA_DIR / "picks.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    n_pin = sum(1 for p in value_bets if p["ref"] == "pinnacle")
    print(f"OK: {matched} matchs, {n_priced} marches prices ({len(pin_index)} matchs avec ref Pinnacle), "
          f"{len(value_bets)} value bets dont {n_pin} vs Pinnacle -> data/picks.json")
    for p in value_bets[:12]:
        print(f"  {p['match'][:34]:34s} {p['marche'][:28]:28s} {str(p['selection_name'])[:18]:18s} "
              f"cote {p['cote']:>5.2f} proba {p['proba']:.0%} EV {p['ev']:+.1%}")
    print(f"combos: {len(combos)} generes")
    for c in combos[:6]:
        print(f"  {c['day']} [{c['type']:9s}] {len(c['legs'])} jambes  cote {c['cote']:>6.2f}  "
              f"proba {c['proba']:.0%}  EV {c['ev']:+.1%}")


if __name__ == "__main__":
    main()
