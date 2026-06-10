"""Exporte les probas du modele ScoutFootball vers data/model.json.

A lancer depuis le repo ScoutFootball avec son venv :
  uv run python C:/Users/nonog/worldcup-pronos/export_model.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from scoutfootball.worldcup.data import (
    GROUPS,
    compute_team_strengths,
    generate_group_stage_matches,
)

OUT = Path(__file__).parent / "data" / "model.json"
DATASET_STRENGTHS = Path(__file__).parent / "data" / "strengths_dataset.json"

# Conversion force (0-1) -> lambda Poisson. Terrain neutre, pas d'avantage domicile.
# Valeurs par defaut ecrasees par la calibration MLE de strengths_dataset.json.
BASE_LAMBDA = 1.38
SPREAD = 1.65
LAMBDA_CAP = 3.5  # garde-fou : pas d'equipe a plus de 3,5 buts esperes
MAX_GOALS = 10


def match_probs(s_home: float, s_away: float) -> dict:
    diff = s_home - s_away
    lh = min(BASE_LAMBDA * math.exp(SPREAD * diff), LAMBDA_CAP)
    la = min(BASE_LAMBDA * math.exp(-SPREAD * diff), LAMBDA_CAP)

    def pois(lam: float, k: int) -> float:
        return math.exp(-lam) * lam**k / math.factorial(k)

    ph = [pois(lh, k) for k in range(MAX_GOALS + 1)]
    pa = [pois(la, k) for k in range(MAX_GOALS + 1)]

    p1 = px = p2 = over25 = btts = 0.0
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            p = ph[h] * pa[a]
            if h > a:
                p1 += p
            elif h == a:
                px += p
            else:
                p2 += p
            if h + a > 2.5:
                over25 += p
            if h > 0 and a > 0:
                btts += p

    return {
        "lambda_home": round(lh, 3),
        "lambda_away": round(la, 3),
        "p1": round(p1, 4),
        "pX": round(px, 4),
        "p2": round(p2, 4),
        "over25": round(over25, 4),
        "btts": round(btts, 4),
    }


def main() -> None:
    global BASE_LAMBDA, SPREAD
    strengths = compute_team_strengths()
    source = "scoutfootball (squads placeholder)"
    if DATASET_STRENGTHS.exists():
        # forces Glicko-2 du soccer-dataset, bien plus fiables que les squads
        # placeholder du repo (plates pour les petites equipes)
        ds = json.loads(DATASET_STRENGTHS.read_text(encoding="utf-8"))
        strengths.update(ds["strengths"])
        source = ds["source"]
        BASE_LAMBDA = ds.get("base_lambda", BASE_LAMBDA)
        SPREAD = ds.get("spread", SPREAD)
    matches = []
    for m in generate_group_stage_matches():
        probs = match_probs(strengths.get(m.home, 0.2), strengths.get(m.away, 0.2))
        matches.append({
            "matchday": m.matchday,
            "date": m.date,
            "home": m.home,
            "away": m.away,
            "group": m.group,
            "venue": m.venue,
            "city": m.city,
            **probs,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "strengths_source": source,
        "strengths": {k: round(v, 4) for k, v in sorted(strengths.items(), key=lambda x: -x[1])},
        "groups": GROUPS,
        "matches": matches,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {len(matches)} matchs -> {OUT}")


if __name__ == "__main__":
    main()
