"""Scrape les cotes 1N2 Pinnacle (book sharp) pour la CDM 2026 -> data/pinnacle.json.

Usage : python scrape_pinnacle.py
Pinnacle = closing line de reference (marges faibles, sharps acceptes). Ses
probas de-vigees servent de reference "marche intelligent" dans make_picks,
a la place de la marge de Winamax (book grand public).
API invitee publique guest.api.arcadia.pinnacle.com (lecture seule).
"""
from __future__ import annotations

import json
import subprocess
import time
import unicodedata
from pathlib import Path

LEAGUE_ID = 2686  # FIFA - World Cup
API = "https://guest.api.arcadia.pinnacle.com/0.1"
API_KEY = "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R"  # cle invitee publique du site

DATA_DIR = Path(__file__).parent / "data"

# noms Pinnacle (EN) -> noms du repo quand differents
ALIASES = {
    "USA": "United States", "Korea Republic": "South Korea",
    "Czechia": "Czech Republic", "Turkiye": "Turkey",
    "Cote d'Ivoire": "Ivory Coast", "Congo DR": "DR Congo",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde", "Curacao": "Curacao",
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in s if unicodedata.category(c) != "Mn").strip()


def fetch(path: str):
    result = subprocess.run(
        ["curl.exe", "-s", "--compressed", "-A", "Mozilla/5.0",
         "-H", f"x-api-key: {API_KEY}", f"{API}{path}"],
        capture_output=True, timeout=60,
    )
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(f"curl a echoue sur {path}")
    return json.loads(result.stdout.decode("utf-8"))


def american_to_decimal(price: float) -> float:
    return 1 + (price / 100 if price > 0 else 100 / abs(price))


def main() -> None:
    matchups = fetch(f"/leagues/{LEAGUE_ID}/matchups?withSpecials=false")
    markets = fetch(f"/leagues/{LEAGUE_ID}/markets/straight")

    teams = {}  # matchup_id -> (home, away, start)
    for m in matchups:
        parts = m.get("participants", [])
        if len(parts) == 2 and not m.get("isLive"):
            home = next((p["name"] for p in parts if p["alignment"] == "home"), None)
            away = next((p["name"] for p in parts if p["alignment"] == "away"), None)
            if home and away:
                teams[m["id"]] = (home, away, m.get("startTime"))

    rows = []
    for mk in markets:
        # moneyline temps reglementaire, match complet
        if mk.get("type") != "moneyline" or mk.get("period") != 0:
            continue
        mid = mk.get("matchupId")
        if mid not in teams:
            continue
        prices = {p["designation"]: p["price"] for p in mk.get("prices", [])}
        if not all(k in prices for k in ("home", "draw", "away")):
            continue
        d1 = american_to_decimal(prices["home"])
        dx = american_to_decimal(prices["draw"])
        d2 = american_to_decimal(prices["away"])
        inv = [1 / d1, 1 / dx, 1 / d2]
        s = sum(inv)
        home, away, start = teams[mid]
        rows.append({
            "home_en": ALIASES.get(norm(home), norm(home)),
            "away_en": ALIASES.get(norm(away), norm(away)),
            "start": start,
            "odds_1": round(d1, 3), "odds_X": round(dx, 3), "odds_2": round(d2, 3),
            "margin": round(s - 1, 4),
            "fair_p1": round(inv[0] / s, 4),
            "fair_pX": round(inv[1] / s, 4),
            "fair_p2": round(inv[2] / s, 4),
        })

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {"scraped_at": int(time.time()), "source": "pinnacle (de-vig)", "matches": rows}
    (DATA_DIR / "pinnacle.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {len(rows)} matchs Pinnacle 1N2 de-vigues -> data/pinnacle.json "
          f"(marge moyenne {sum(r['margin'] for r in rows)/max(len(rows),1):.1%})")


if __name__ == "__main__":
    main()
