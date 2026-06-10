"""Forces des 48 equipes CDM depuis soccer-dataset (Glicko-2) -> data/strengths_dataset.json.

A lancer avec le venv du repo ScoutFootball (pandas+pyarrow) :
  C:/Users/nonog/ScoutFootball_for_World_Cup/.venv/Scripts/python.exe build_strengths.py

Source : https://github.com/eatpizzanot/soccer-dataset (rating_mu = Glicko-2).
Remplace les forces "squads placeholder" du repo, plates pour les petites equipes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
DS = DATA / "dataset"

INTL_LEAGUE_IDS = [78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88]

WC48 = [
    "Mexico", "South Africa", "South Korea", "Czech Republic", "Canada",
    "Bosnia and Herzegovina", "Qatar", "Switzerland", "Brazil", "Morocco",
    "Haiti", "Scotland", "United States", "Paraguay", "Australia", "Turkey",
    "Germany", "Curacao", "Ivory Coast", "Ecuador", "Netherlands", "Japan",
    "Sweden", "Tunisia", "Belgium", "Egypt", "Iran", "New Zealand", "Spain",
    "Cape Verde", "Saudi Arabia", "Uruguay", "France", "Senegal", "Iraq",
    "Norway", "Argentina", "Algeria", "Austria", "Jordan", "Portugal",
    "DR Congo", "Uzbekistan", "Colombia", "England", "Croatia", "Ghana",
    "Panama",
]

# nom repo -> nom dataset quand differents
ALIASES = {
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "United States": "USA",
    "Turkey": "Türkiye",
    "Curacao": "Curaçao",
    "Cape Verde": "Cape Verde Islands",
    "DR Congo": "Congo DR",
}


def main() -> None:
    teams = pd.read_parquet(DS / "teams.parquet")
    fixtures = pd.read_parquet(DS / "fixtures.parquet")

    intl = fixtures[fixtures.league_id.isin(INTL_LEAGUE_IDS)]
    nat_ids = set(intl.home_team_id) | set(intl.away_team_id)
    nat = teams[teams.id.isin(nat_ids)]
    nat = nat[~nat.name.str.contains(r"U\d+", regex=True)]  # exclut equipes jeunes

    # rating prudent mu - sigma : les equipes des confederations faibles (Iran,
    # Senegal...) ont un mu gonfle par les qualifs et un sigma eleve car peu de
    # matchs cross-confederation ; la penalite remet les echelles en ligne.
    by_name = {n: (mu, sig) for n, mu, sig in zip(nat.name, nat.rating_mu, nat.rating_sigma)}

    mus: dict[str, float] = {}
    for team in WC48:
        ds_name = ALIASES.get(team, team)
        entry = by_name.get(ds_name)
        if entry is None:
            print(f"ATTENTION: {team} ({ds_name}) introuvable dans le dataset")
            continue
        mu, sig = entry
        mus[team] = float(mu - sig)

    lo, hi = min(mus.values()), max(mus.values())
    strengths = {t: round((m - lo) / (hi - lo), 4) for t, m in mus.items()}

    # 2e composante : valeur marchande de l'effectif (Transfermarkt, echelle
    # log car de 1,5 Mrd a ~10 M). 70% Glicko (resultats) / 30% valeur (talent).
    sv_file = DATA / "squad_values.json"
    if sv_file.exists():
        import math
        vals = json.loads(sv_file.read_text(encoding="utf-8"))["values_eur"]
        logs = {t: math.log(v) for t, v in vals.items() if t in strengths}
        if len(logs) >= 40:
            vlo, vhi = min(logs.values()), max(logs.values())
            for t in strengths:
                if t in logs:
                    tm_norm = (logs[t] - vlo) / (vhi - vlo)
                    strengths[t] = round(0.70 * strengths[t] + 0.30 * tm_norm, 4)
            print(f"forces: melange 70% Glicko / 30% valeur Transfermarkt ({len(logs)} equipes)")

    # Calibration Poisson sur les matchs internationaux recents :
    # buts ~ exp(a + b * (rating_for - rating_against)), ratings = mu - sigma.
    # MLE : a en forme fermee pour b donne, grid search sur b.
    import numpy as np

    rating = {tid: mu - sig for tid, mu, sig in zip(nat.id, nat.rating_mu, nat.rating_sigma)}
    rec = intl[(intl.date >= "2023-01-01") & intl.goals_home.notna()]

    # Cible = buts reels. Le xG du soccer-dataset a ete teste comme cible mais
    # ecarte : sur les matchs internationaux, 897/2389 lignes ont xG 0-0
    # (donnees manquantes codees 0) et la moyenne xG (0.66) est incoherente
    # avec les buts (1.37) -> inutilisable pour calibrer des lambdas de buts.
    rows = []
    for _, r in rec.iterrows():
        rh, ra = rating.get(r.home_team_id), rating.get(r.away_team_id)
        if rh is None or ra is None:
            continue
        rows.append((r.goals_home, rh - ra))
        rows.append((r.goals_away, ra - rh))
    goals = np.array([g for g, _ in rows], dtype=float)
    diffs = np.array([d for _, d in rows], dtype=float)
    true_goals = goals

    best_b, best_ll = 0.0, -np.inf
    for b in np.arange(0.0, 0.0081, 0.0002):
        ea = goals.sum() / np.exp(b * diffs).sum()
        ll = float((goals * (np.log(ea) + b * diffs) - ea * np.exp(b * diffs)).sum())
        if ll > best_ll:
            best_b, best_ll = float(b), ll
    base_lambda = float(goals.sum() / np.exp(best_b * diffs).sum())
    spread = best_b * (hi - lo)  # par unite de force normalisee 0-1
    avg_goals = float(goals.mean())

    # Correction Dixon-Coles : le Poisson independant sous-estime les nuls
    # 0-0/1-1. On ajuste rho par MLE sur les memes matchs (tau de DC 1997).
    gh = true_goals[0::2]; ga = true_goals[1::2]; dh = diffs[0::2]
    lh = base_lambda * np.exp(best_b * dh)
    la = base_lambda * np.exp(-best_b * dh)
    best_rho, best_rll = 0.0, -np.inf
    for rho in np.arange(-0.15, 0.151, 0.005):
        tau = np.ones(len(gh))
        m00 = (gh == 0) & (ga == 0); tau[m00] = 1 - lh[m00] * la[m00] * rho
        m01 = (gh == 0) & (ga == 1); tau[m01] = 1 + lh[m01] * rho
        m10 = (gh == 1) & (ga == 0); tau[m10] = 1 + la[m10] * rho
        m11 = (gh == 1) & (ga == 1); tau[m11] = 1 - rho
        if (tau <= 0).any():
            continue
        rll = float(np.log(tau).sum())
        if rll > best_rll:
            best_rho, best_rll = float(rho), rll

    print(f"calibration: {len(rows)//2} matchs, base_lambda={base_lambda:.3f}, "
          f"b={best_b:.4f}/pt Glicko, spread={spread:.3f}/force, rho={best_rho:.3f}")

    out = {
        "source": "eatpizzanot/soccer-dataset (Glicko-2)",
        "avg_goals_per_team": round(avg_goals, 3),
        "base_lambda": round(base_lambda, 4),
        "spread": round(spread, 4),
        "rho": round(best_rho, 4),
        "ratings_mu": {t: round(m, 1) for t, m in sorted(mus.items(), key=lambda x: -x[1])},
        "strengths": dict(sorted(strengths.items(), key=lambda x: -x[1])),
    }
    (DATA / "strengths_dataset.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {len(strengths)}/48 equipes -> data/strengths_dataset.json "
          f"(buts/equipe/match 2024+ : {avg_goals:.3f})")


if __name__ == "__main__":
    main()
