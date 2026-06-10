"""Cotes Pinnacle CDM via The Odds API -> data/pinnacle.json + archive closing.

Usage : python scrape_oddsapi.py
Source keyee et stable (cle dans config.json, jamais commitee). 1 credit par
run, 500/mois gratuits. Remplace le scraping guest Pinnacle comme source
principale ; scrape_pinnacle.py reste en secours sans cle.
Chaque run est aussi archive dans data/closing_history.jsonl : le dernier
snapshot avant coup d'envoi ≈ closing line, pour le backtest et le CLV.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
SPORT = "soccer_fifa_world_cup"

# noms The Odds API (EN) -> noms du repo quand differents
ALIASES = {
    "USA": "United States", "Korea Republic": "South Korea",
    "Czechia": "Czech Republic", "Turkiye": "Turkey",
    "Cote d'Ivoire": "Ivory Coast", "Congo DR": "DR Congo",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Curaçao": "Curacao",
}


def main() -> None:
    key = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))["odds_api_key"]
    url = (f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds/"
           f"?apiKey={key}&regions=eu&markets=h2h&oddsFormat=decimal&bookmakers=pinnacle")
    result = subprocess.run(["curl.exe", "-s", url], capture_output=True, timeout=60)
    data = json.loads(result.stdout.decode("utf-8"))
    if isinstance(data, dict):  # message d'erreur API (quota...)
        raise SystemExit(f"The Odds API: {data}")

    rows = []
    for m in data:
        pin = next((b for b in m.get("bookmakers", []) if b["key"] == "pinnacle"), None)
        if not pin:
            continue
        outcomes = {o["name"]: o["price"] for o in pin["markets"][0]["outcomes"]}
        home, away = m["home_team"], m["away_team"]
        d1, dx, d2 = outcomes.get(home), outcomes.get("Draw"), outcomes.get(away)
        if not all((d1, dx, d2)):
            continue
        inv = [1 / d1, 1 / dx, 1 / d2]
        s = sum(inv)
        rows.append({
            "home_en": ALIASES.get(home, home),
            "away_en": ALIASES.get(away, away),
            "start": m["commence_time"],
            "odds_1": d1, "odds_X": dx, "odds_2": d2,
            "margin": round(s - 1, 4),
            "fair_p1": round(inv[0] / s, 4),
            "fair_pX": round(inv[1] / s, 4),
            "fair_p2": round(inv[2] / s, 4),
        })

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {"scraped_at": int(time.time()), "source": "pinnacle via the-odds-api (de-vig)",
           "matches": rows}
    (DATA_DIR / "pinnacle.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    with (DATA_DIR / "closing_history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"OK: {len(rows)} matchs Pinnacle (Odds API) -> data/pinnacle.json + archive closing")


if __name__ == "__main__":
    main()
