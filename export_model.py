"""Exporte les probas du modele ScoutFootball vers data/model.json.

A lancer depuis le repo ScoutFootball avec son venv :
  uv run python C:/Users/nonog/worldcup-pronos/export_model.py
"""
from __future__ import annotations

import json
from pathlib import Path

from scoutfootball.worldcup.data import (
    GROUPS,
    compute_team_strengths,
    generate_group_stage_matches,
)

from pricing import soft_cap_lambda

OUT = Path(__file__).parent / "data" / "model.json"
DATASET_STRENGTHS = Path(__file__).parent / "data" / "strengths_dataset.json"

# Conversion force (0-1) -> lambda Poisson. Terrain neutre, pas d'avantage domicile.
# Valeurs par defaut ecrasees par la calibration MLE de strengths_dataset.json.
BASE_LAMBDA = 1.38
SPREAD = 1.65
RHO = 0.0       # Dixon-Coles : correction scores bas (0-0, 1-0, 0-1, 1-1)
LAMBDA3 = 0.0   # Bivariate Poisson : covariance buts domicile/extérieur
PI_ZERO = 0.0   # Zero-Inflated Poisson : masse structurelle sur le 0-0
MAX_GOALS = 10


def match_probs(s_home: float, s_away: float) -> dict:
    import math
    from pricing import score_matrix as _score_matrix

    diff = s_home - s_away
    raw_lh = BASE_LAMBDA * math.exp(SPREAD * diff)
    raw_la = BASE_LAMBDA * math.exp(-SPREAD * diff)
    lh = soft_cap_lambda(raw_lh)
    la = soft_cap_lambda(raw_la)

    mat = _score_matrix(lh, la, rho=RHO, pi_zero=PI_ZERO, lambda3=LAMBDA3)

    p1 = px = p2 = over25 = btts = 0.0
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            p = mat[h][a]
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
        "lambda_home_raw": round(raw_lh, 3),
        "lambda_away_raw": round(raw_la, 3),
        "p1": round(p1, 4),
        "pX": round(px, 4),
        "p2": round(p2, 4),
        "over25": round(over25, 4),
        "btts": round(btts, 4),
    }


def main() -> None:
    global BASE_LAMBDA, SPREAD, RHO, LAMBDA3, PI_ZERO
    strengths = compute_team_strengths()
    source = "scoutfootball (squads placeholder)"
    if DATASET_STRENGTHS.exists():
        ds = json.loads(DATASET_STRENGTHS.read_text(encoding="utf-8"))
        strengths.update(ds["strengths"])
        source = ds["source"]
        BASE_LAMBDA = ds.get("base_lambda", BASE_LAMBDA)
        SPREAD = ds.get("spread", SPREAD)
        RHO = ds.get("rho", RHO)
        LAMBDA3 = ds.get("lambda3", LAMBDA3)
        PI_ZERO = ds.get("pi_zero", PI_ZERO)
        print(f"modele: rho={RHO:.3f}, lambda3={LAMBDA3:.3f} (BVP), pi_zero={PI_ZERO:.4f} (ZIP)")
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
        "rho": RHO,
        "lambda3": LAMBDA3,
        "pi_zero": PI_ZERO,
        "strengths": {k: round(v, 4) for k, v in sorted(strengths.items(), key=lambda x: -x[1])},
        "groups": GROUPS,
        "matches": matches,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {len(matches)} matchs -> {OUT}")


if __name__ == "__main__":
    main()
