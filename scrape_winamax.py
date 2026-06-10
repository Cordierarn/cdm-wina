"""Scrape les cotes Coupe du Monde 2026 sur Winamax -> data/odds.json.

Stdlib uniquement. Usage : python scrape_winamax.py
1. Page sport football -> liste des matchs CDM (tournoi 900001750) + cotes 1N2.
2. Page de chaque match -> tous les marches pricables par le modele
   (resultat, double chance, BTTS, plus/moins, totaux equipe, score exact,
   nombre exact de buts, vainqueur rembourse si nul, handicap).
Les marches joueurs (buteurs...), mi-temps et tirs sont ignores : pas de modele.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
import unicodedata
from pathlib import Path

BASE = "https://www.winamax.fr/paris-sportifs"
TOURNAMENT_ID = 900001750
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
DELAY_S = 0.4  # politesse entre requetes

DATA_DIR = Path(__file__).parent / "data"
OUT = DATA_DIR / "odds.json"
HISTORY = DATA_DIR / "odds_history.jsonl"

# titres de paris (normalises) que make_picks sait pricer
PRICEABLE = [
    re.compile(r"^resultat$"),
    re.compile(r"^double chance$"),
    re.compile(r"^les 2 equipes marquent$"),
    re.compile(r"^nombre de buts$"),
    re.compile(r"^nombre de buts de .+$"),
    re.compile(r"^score exact$"),
    re.compile(r"^nombre exact de buts$"),
    re.compile(r"^vainqueur \(rembourse si match nul\)$"),
    re.compile(r"^ecart de buts \(handicap\)$"),
]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()


def fetch_state(url: str) -> dict:
    # urllib se prend un 403 (filtrage TLS) ; curl.exe Windows (Schannel) passe.
    result = subprocess.run(
        ["curl.exe", "-s", "--compressed", "-A", UA,
         "-H", "Accept-Language: fr-FR,fr;q=0.9", url],
        capture_output=True, timeout=60,
    )
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(f"curl a echoue (code {result.returncode}) sur {url}")
    html = result.stdout.decode("utf-8")
    marker = "PRELOADED_STATE = "
    i = html.index(marker) + len(marker)
    state, _ = json.JSONDecoder().raw_decode(html[i:])
    return state


def outcomes_of(state: dict, bet: dict) -> list[dict]:
    outs, odds = state.get("outcomes", {}), state.get("odds", {})
    return [
        {"label": outs.get(str(o), {}).get("label"), "cote": odds.get(str(o))}
        for o in bet.get("outcomes", [])
    ]


def extract_match_bets(state: dict, match_id: str) -> list[dict]:
    rows = []
    for bet in state.get("bets", {}).values():
        if str(bet.get("matchId")) != match_id:
            continue
        title = bet.get("betTitle", "")
        if not any(p.match(norm(title)) for p in PRICEABLE):
            continue
        rows.append({"marche": title, "issues": outcomes_of(state, bet)})
    return rows


def main() -> None:
    state = fetch_state(f"{BASE}/sports/1")
    matches = []
    wc = {mid: m for mid, m in state.get("matches", {}).items()
          if m.get("tournamentId") == TOURNAMENT_ID}
    print(f"{len(wc)} matchs CDM, scrape des pages match...")

    for i, (mid, m) in enumerate(sorted(wc.items(), key=lambda x: x[1].get("matchStart") or 0)):
        bet = state.get("bets", {}).get(str(m.get("mainBetId")))
        main_odds = outcomes_of(state, bet) if bet else []
        row = {
            "winamax_id": mid,
            "title": m.get("title"),
            "home_fr": main_odds[0]["label"] if len(main_odds) == 3 else None,
            "away_fr": main_odds[2]["label"] if len(main_odds) == 3 else None,
            "kickoff": m.get("matchStart"),
            "odds_1": main_odds[0]["cote"] if len(main_odds) == 3 else None,
            "odds_X": main_odds[1]["cote"] if len(main_odds) == 3 else None,
            "odds_2": main_odds[2]["cote"] if len(main_odds) == 3 else None,
            "status": m.get("status"),
            "bets": [],
        }
        try:
            mstate = fetch_state(f"{BASE}/match/{mid}")
            row["bets"] = extract_match_bets(mstate, mid)
        except Exception as exc:
            print(f"  page match {mid} ({m.get('title')}) : {exc}")
        matches.append(row)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(wc)}")
        time.sleep(DELAY_S)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = {"scraped_at": int(time.time()), "matches": matches}
    OUT.write_text(json.dumps(snapshot, indent=1, ensure_ascii=False), encoding="utf-8")
    # historique allege : 1N2 seulement, pour suivre les mouvements de cotes
    light = {"scraped_at": snapshot["scraped_at"], "matches": [
        {k: r[k] for k in ("winamax_id", "title", "kickoff", "odds_1", "odds_X", "odds_2")}
        for r in matches]}
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(light, ensure_ascii=False) + "\n")
    n_bets = sum(len(r["bets"]) for r in matches)
    print(f"OK: {len(matches)} matchs, {n_bets} marches pricables -> {OUT}")


if __name__ == "__main__":
    main()
