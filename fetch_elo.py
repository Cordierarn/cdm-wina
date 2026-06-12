"""Recupere les ratings Elo live d'eloratings.net et les blende dans les forces.

Usage : python fetch_elo.py
- Telecharge https://eloratings.net/World.tsv (mis a jour apres chaque match)
- Blend : 50% Elo officiel + 50% force existante (Glicko+Transfermarkt)
- Met a jour data/strengths_dataset.json (strengths + provenance elo)
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"

# Codes eloratings.net -> noms d'equipes du modele (48 qualifies CDM 2026)
ELO_CODES = {
    "MX": "Mexico", "ZA": "South Africa", "KR": "South Korea", "CZ": "Czech Republic",
    "CA": "Canada", "BA": "Bosnia and Herzegovina", "QA": "Qatar", "CH": "Switzerland",
    "BR": "Brazil", "MA": "Morocco", "HT": "Haiti", "SQ": "Scotland",
    "US": "United States", "PY": "Paraguay", "AU": "Australia", "TR": "Turkey",
    "DE": "Germany", "CW": "Curacao", "CI": "Ivory Coast", "EC": "Ecuador",
    "NL": "Netherlands", "JP": "Japan", "SE": "Sweden", "TN": "Tunisia",
    "BE": "Belgium", "EG": "Egypt", "IR": "Iran", "NZ": "New Zealand",
    "ES": "Spain", "CV": "Cape Verde", "SA": "Saudi Arabia", "UY": "Uruguay",
    "FR": "France", "SN": "Senegal", "IQ": "Iraq", "NO": "Norway",
    "AR": "Argentina", "DZ": "Algeria", "AT": "Austria", "JO": "Jordan",
    "PT": "Portugal", "CD": "DR Congo", "UZ": "Uzbekistan", "CO": "Colombia",
    "EN": "England", "HR": "Croatia", "GH": "Ghana", "PA": "Panama",
}

ELO_WEIGHT = 0.5  # 50% Elo officiel / 50% force existante (Glicko + Transfermarkt)


def main() -> None:
    req = urllib.request.Request("https://eloratings.net/World.tsv",
                                 headers={"User-Agent": "Mozilla/5.0"})
    tsv = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")

    elo: dict[str, float] = {}
    for line in tsv.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        code, rating = parts[2], parts[3]
        if code in ELO_CODES:
            elo[ELO_CODES[code]] = float(rating)

    missing = set(ELO_CODES.values()) - set(elo)
    if missing:
        raise SystemExit(f"Equipes sans Elo : {missing}")
    print(f"Elo recuperes pour {len(elo)}/48 equipes (Spain={elo['Spain']:.0f}, Argentina={elo['Argentina']:.0f})")

    ds_path = DATA / "strengths_dataset.json"
    ds = json.loads(ds_path.read_text(encoding="utf-8"))

    # Normalisation 0-1 sur les 48 equipes (meme echelle que les forces existantes)
    lo, hi = min(elo.values()), max(elo.values())
    elo_norm = {t: (v - lo) / (hi - lo) for t, v in elo.items()}

    old = dict(ds["strengths"])
    blended = {}
    for t in old:
        if t in elo_norm:
            blended[t] = round(ELO_WEIGHT * elo_norm[t] + (1 - ELO_WEIGHT) * old[t], 4)
        else:
            blended[t] = old[t]

    ds["strengths"] = dict(sorted(blended.items(), key=lambda x: -x[1]))
    ds["elo_ratings"] = dict(sorted(elo.items(), key=lambda x: -x[1]))
    ds["elo_weight"] = ELO_WEIGHT
    ds["source"] = ds.get("source", "") .split(" + elo")[0] + " + eloratings.net (blend 50/50)"
    ds_path.write_text(json.dumps(ds, indent=1, ensure_ascii=False), encoding="utf-8")

    moves = sorted(((t, blended[t] - old[t]) for t in old if t in elo_norm),
                   key=lambda x: -abs(x[1]))[:8]
    print("Plus gros mouvements de force (blend Elo) :")
    for t, d in moves:
        print(f"  {t:25s} {old[t]:.3f} -> {blended[t]:.3f} ({d:+.3f})")
    print(f"OK -> {ds_path}")


if __name__ == "__main__":
    main()
