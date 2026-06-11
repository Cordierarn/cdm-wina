"""Cotes de cloture Over/Under 2.5 via OddsHarvester (OddsPortal) pour le backtest.

Tournois : World Cup 2022, Euro 2024.
Sortie   : data/historical_odds/wc2022_ou.json et euro2024_ou.json
Ensuite  : python build_closing_backtest.py  (merge la cle "totals")

OddsHarvester ne connait pas la ligue Euro : on l'enregistre avant d'invoquer
sa CLI (meme URL OddsPortal que fetch_historical_odds.py). Mode preview-only =
cotes moyennes visibles, plus robuste que d'ouvrir chaque ligne de total.

Usage : python fetch_ou_oddsharvester.py
"""
from __future__ import annotations

from pathlib import Path

from oddsharvester.utils.sport_league_constants import SPORTS_LEAGUES_URLS_MAPPING
from oddsharvester.utils.sport_market_constants import Sport

# Euro absent du mapping de la lib : meme URL que le scraper 1X2 maison.
SPORTS_LEAGUES_URLS_MAPPING[Sport.FOOTBALL]["euro"] = "https://www.oddsportal.com/football/europe/euro"

from oddsharvester.cli.cli import cli as oddsharvester_cli  # noqa: E402

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "data" / "historical_odds"

JOBS = [
    {"league": "world-cup", "season": "2022", "out": OUT_DIR / "wc2022_ou.json"},
    {"league": "euro", "season": "2024", "out": OUT_DIR / "euro2024_ou.json"},
]


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for job in JOBS:
        if job["out"].exists():
            print(f"[{job['league']} {job['season']}] deja present — skip "
                  f"(supprimez {job['out'].name} pour re-scraper)")
            continue
        print(f"\n=== {job['league']} {job['season']} -> {job['out'].name} ===")
        args = [
            "historic",
            "-s", "football",
            "-l", job["league"],
            "--season", job["season"],
            "-m", "over_under_2_5",
            "--preview-only",
            "--headless",
            "-c", "1",
            "-f", "json",
            "-o", str(job["out"]),
        ]
        try:
            oddsharvester_cli(args=args, standalone_mode=False, obj={})
        except SystemExit as exc:  # click peut lever SystemExit(0)
            if exc.code not in (0, None):
                raise


if __name__ == "__main__":
    run()
