"""Parse les JSONs bruts d'OddsPortal et produit data/closing_backtest.json.

Entrée  : data/historical_odds/wc2022.json  (et/ou euro2024.json)
Sortie  : data/closing_backtest.json

Format de sortie :
{
  "<date>|<home_norm>|<away_norm>": {
    "date": "2022-11-20",
    "home": "Argentina",
    "away": "Saudi Arabia",
    "h2h": [p1, pX, p2],          # probas dé-viguées
    "odds_raw": [o1, oX, o2]       # cotes brutes
  },
  ...
}

Les noms sont normalisés (ASCII minuscules) pour le matching, mais les valeurs
gardent le nom canonique du soccer-dataset quand possible.

Usage : python build_closing_backtest.py
"""
from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
IN_DIR = DATA_DIR / "historical_odds"
OUT_FILE = DATA_DIR / "closing_backtest.json"

# Mapping OddsPortal → noms canoniques du soccer-dataset / backtest
# Ajouter ici toute variante rencontrée dans les données scrapées.
TEAM_NAME_MAP: dict[str, str] = {
    # Variantes OddsPortal communes
    "korea republic": "South Korea",
    "south korea": "South Korea",
    "republic of korea": "South Korea",
    "usa": "United States",
    "united states": "United States",
    "usa.": "United States",
    "czechia": "Czech Republic",
    "czech republic": "Czech Republic",
    "ir iran": "Iran",
    "iran": "Iran",
    "islamic republic of iran": "Iran",
    "cote d'ivoire": "Ivory Coast",
    "cote divoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "dr congo": "DR Congo",
    "congo dr": "DR Congo",
    "democratic republic of congo": "DR Congo",
    "cape verde": "Cape Verde",
    "cap vert": "Cape Verde",
    "saudi arabia": "Saudi Arabia",
    "ksa": "Saudi Arabia",
    "netherlands": "Netherlands",
    "holland": "Netherlands",
    "england": "England",
    "united kingdom": "England",
    "north macedonia": "North Macedonia",
    "republic of ireland": "Republic of Ireland",
    "ireland": "Republic of Ireland",
    "bosnia & herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "bosnia": "Bosnia and Herzegovina",
    "trinidad & tobago": "Trinidad and Tobago",
    "trinidad and tobago": "Trinidad and Tobago",
    "new zealand": "New Zealand",
    "nz": "New Zealand",
    "guinea-bissau": "Guinea-Bissau",
    "guinea bissau": "Guinea-Bissau",
    "equatorial guinea": "Equatorial Guinea",
    "central african republic": "Central African Republic",
    "sierra leone": "Sierra Leone",
    "burkina faso": "Burkina Faso",
    "south sudan": "South Sudan",
    "sao tome & principe": "São Tomé and Príncipe",
    "sao tome and principe": "São Tomé and Príncipe",
    "eswatini": "Eswatini",
    "swaziland": "Eswatini",
    "curacao": "Curacao",
    # Noms identiques côté soccer-dataset (passthrough, garder minuscules → canonical)
    "argentina": "Argentina",
    "france": "France",
    "brazil": "Brazil",
    "germany": "Germany",
    "spain": "Spain",
    "portugal": "Portugal",
    "belgium": "Belgium",
    "croatia": "Croatia",
    "uruguay": "Uruguay",
    "mexico": "Mexico",
    "ecuador": "Ecuador",
    "qatar": "Qatar",
    "senegal": "Senegal",
    "ghana": "Ghana",
    "cameroon": "Cameroon",
    "morocco": "Morocco",
    "tunisia": "Tunisia",
    "australia": "Australia",
    "japan": "Japan",
    "poland": "Poland",
    "switzerland": "Switzerland",
    "denmark": "Denmark",
    "wales": "Wales",
    "serbia": "Serbia",
    "canada": "Canada",
    "costa rica": "Costa Rica",
    "turkey": "Turkey",
    "austria": "Austria",
    "hungary": "Hungary",
    "slovakia": "Slovakia",
    "slovenia": "Slovenia",
    "romania": "Romania",
    "albania": "Albania",
    "georgia": "Georgia",
    "scotland": "Scotland",
    "ukraine": "Ukraine",
    "italy": "Italy",
    "belgium": "Belgium",
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower().strip()


def canonical_name(raw: str) -> str:
    """Retourne le nom canonique (soccer-dataset) à partir d'un nom brut OddsPortal."""
    key = norm(raw)
    return TEAM_NAME_MAP.get(key, raw.strip())


def de_vig(odds: list[float]) -> list[float] | None:
    if any(o is None or o <= 1.0 for o in odds):
        return None
    inv = [1.0 / o for o in odds]
    total = sum(inv)
    if total <= 0:
        return None
    return [round(v / total, 6) for v in inv]


def parse_date(raw: str) -> str:
    """Extrait une date ISO YYYY-MM-DD depuis les formats variés d'OddsPortal."""
    import re
    # "20 Nov 2022", "Nov 20, 2022", "2022-11-20", "20/11/2022"
    raw = raw.strip()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"(\d{1,2})[/ ](\w+)[, ]+(\d{4})", raw)
    if m:
        day, mon, year = m.group(1).zfill(2), m.group(2)[:3].lower(), m.group(3)
        months = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05",
                  "jun": "06", "jul": "07", "aug": "08", "sep": "09", "oct": "10",
                  "nov": "11", "dec": "12"}
        if mon in months:
            return f"{year}-{months[mon]}-{day}"
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", raw)
    if m:
        return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
    return raw[:10]  # meilleur effort


def load_json_file(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("matches", [])


def build_key(date_iso: str, home: str, away: str) -> str:
    return f"{date_iso}|{norm(home)}|{norm(away)}"


def main() -> None:
    in_files = sorted(IN_DIR.glob("*.json"))
    if not in_files:
        raise SystemExit(
            f"Aucun fichier dans {IN_DIR}.\n"
            "Lancez d'abord : python fetch_historical_odds.py"
        )

    result: dict[str, dict] = {}
    skipped = 0
    total = 0

    for path in in_files:
        matches = load_json_file(path)
        print(f"{path.name} : {len(matches)} matchs")
        for m in matches:
            total += 1
            raw_home = m.get("home", "")
            raw_away = m.get("away", "")
            raw_date = m.get("date", "")
            o1 = m.get("odds_home")
            ox = m.get("odds_draw")
            o2 = m.get("odds_away")

            if not raw_home or not raw_away:
                skipped += 1
                continue
            if not all(isinstance(o, (int, float)) and o > 1.0 for o in (o1, ox, o2) if o is not None):
                skipped += 1
                continue

            probs = de_vig([o1, ox, o2])
            if probs is None:
                skipped += 1
                continue

            date_iso = parse_date(raw_date) if raw_date else ""
            home_c = canonical_name(raw_home)
            away_c = canonical_name(raw_away)
            key = build_key(date_iso, home_c, away_c)

            result[key] = {
                "date": date_iso,
                "home": home_c,
                "away": away_c,
                "h2h": probs,
                "odds_raw": [o1, ox, o2],
                "source": "oddsportal",
            }

    print(f"\nTotal : {total} matchs, {len(result)} retenus, {skipped} ignorés (cotes manquantes)")

    OUT_FILE.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"→ {OUT_FILE} ({len(result)} entrées)")


if __name__ == "__main__":
    main()
