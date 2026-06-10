"""Valeurs marchandes des 48 effectifs CDM depuis Transfermarkt -> data/squad_values.json.

Usage : python scrape_transfermarkt.py
1 seule requete : la page participants FIWC liste les 48 equipes avec la
valeur marchande totale de l'effectif. Noms en francais -> mapping FR_TO_EN
partage avec make_picks.
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import time
from pathlib import Path

from make_picks import FR_TO_EN, norm

# variantes Transfermarkt absentes du mapping Winamax
FR_TO_EN_EXTRA = {"tchequie": "Czech Republic"}

URL = "https://www.transfermarkt.fr/weltmeisterschaft-2026/teilnehmer/pokalwettbewerb/FIWC"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

DATA_DIR = Path(__file__).parent / "data"

ROW_RE = re.compile(
    r'class="links no-border-links hauptlink"><a title="([^"]+)"[^>]*>.*?'
    r'<td class="rechts">([\d,]+)\s*(Mrd|mio)\.', re.S | re.I)


def main() -> None:
    result = subprocess.run(
        ["curl.exe", "-s", "--compressed", "-A", UA, URL],
        capture_output=True, timeout=60)
    page = result.stdout.decode("utf-8", errors="replace")

    values: dict[str, float] = {}
    for name_fr, num, unit in ROW_RE.findall(page):
        key = norm(html.unescape(name_fr))
        en = FR_TO_EN.get(key) or FR_TO_EN_EXTRA.get(key)
        if not en:
            continue  # la page liste aussi des equipes non qualifiees
        v = float(num.replace(",", ".")) * (1e9 if unit.lower() == "mrd" else 1e6)
        values[en] = v

    if len(values) < 40:
        raise SystemExit(f"seulement {len(values)} equipes parsees, page changee ?")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {"scraped_at": int(time.time()), "source": "transfermarkt.fr (FIWC)",
           "values_eur": dict(sorted(values.items(), key=lambda x: -x[1]))}
    (DATA_DIR / "squad_values.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    top = list(out["values_eur"].items())[:3]
    print(f"OK: {len(values)}/48 effectifs -> data/squad_values.json "
          f"(top: {', '.join(f'{t} {v/1e6:.0f}M' for t, v in top)})")


if __name__ == "__main__":
    main()
