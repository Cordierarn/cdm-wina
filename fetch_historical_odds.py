"""Scraping des cotes historiques 1X2 sur OddsPortal pour le backtest.

Tournois couverts : FIFA World Cup 2022, UEFA Euro 2024.
Sortie : data/historical_odds/wc2022.json et euro2024.json

Usage :
    pip install playwright && playwright install-deps chromium && playwright install chromium
    python fetch_historical_odds.py

Stratégie : on lit directement les eventRow de la page de résultats OddsPortal.
Chaque row contient équipes, score, cotes 1X2 en format fractionnaire britannique.
Les cotes sont la moyenne des bookmakers affichée par OddsPortal (pas Pinnacle isolé).
Pas de navigation vers chaque page de match → beaucoup plus rapide et fiable.

Note : les totaux O/U 2.5 ne sont pas scrapés (onglet séparé, complexe).
Le backtest conserve le poids 0.40 pour le marché totals.

Source : https://www.oddsportal.com  (consultation libre, pas d'authentification)
Usage : projet personnel non commercial.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "data" / "historical_odds"

TOURNAMENTS = [
    {
        "key": "wc2022",
        "label": "World Cup 2022",
        "pages": [
            "https://www.oddsportal.com/football/world/world-cup-2022/results/",
        ],
    },
    {
        "key": "euro2024",
        "label": "Euro 2024",
        "pages": [
            "https://www.oddsportal.com/football/europe/euro-2024/results/",
        ],
    },
]

# Délai entre chaque chargement de page (politesse)
PAGE_DELAY = 2.5


def frac_to_decimal(frac: str) -> float | None:
    """Convertit une cote fractionnaire '17/10' ou décimale '1.70' en décimal."""
    frac = frac.strip()
    if "/" in frac:
        parts = frac.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1]) + 1.0
            except (ValueError, ZeroDivisionError):
                return None
    try:
        v = float(frac.replace(",", "."))
        return v if v > 1.0 else None
    except ValueError:
        return None


def parse_date_from_header(text: str) -> str:
    """Extrait YYYY-MM-DD depuis '18 Dec 2022 - Play Offs' ou '14 Jun 2024 - Group Stage'."""
    months = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05",
              "jun": "06", "jul": "07", "aug": "08", "sep": "09", "oct": "10",
              "nov": "11", "dec": "12"}
    m = re.search(r"(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})", text)
    if m:
        day = m.group(1).zfill(2)
        mon = m.group(2).lower()[:3]
        year = m.group(3)
        return f"{year}-{months.get(mon, '00')}-{day}"
    return ""


SKIP_TOKENS = frozenset({
    "1", "X", "2", "–", "-", "After Pen.", "Finished", "AET", "Pen.",
    "Postponed", "Canceled", "Cancelled", "Walkover", "Football", "/",
    "World", "Europe", "Asia", "Africa", "America",
})
FRAC_RE = re.compile(r"^\d+/\d+$")
COMPETITION_RE = re.compile(r"cup|euro|copa|championship|league|nations|gold|asian", re.I)


def parse_event_row(row_text: str, current_date: str) -> dict | None:
    """
    Parse le innerText d'un eventRow OddsPortal.

    Format observé (lignes séparées) :
      ['18 Dec 2022 - Play Offs', '1', 'X', '2', 'After Pen.', 'Argentina',
       '4', '–', '3', 'France', '17/10', '11/5', '19/10']

    Le score est sur 3 lignes consécutives : <chiffre>, '–', <chiffre>.
    """
    lines = [l.strip() for l in row_text.splitlines() if l.strip()]

    # Détecter une date dans les premières lignes
    date = current_date
    for line in lines[:6]:
        d = parse_date_from_header(line)
        if d:
            date = d
            break

    # Trouver le séparateur '–' entre les deux scores
    dash_idx = None
    for i, line in enumerate(lines):
        if line in ("–", "-"):
            # Vérifier que les lignes adjacentes sont des chiffres de score
            if (i > 0 and i < len(lines) - 1
                    and re.match(r"^\d{1,2}$", lines[i - 1])
                    and re.match(r"^\d{1,2}$", lines[i + 1])):
                dash_idx = i
                break

    if dash_idx is None:
        return None

    score_home = int(lines[dash_idx - 1])
    score_away = int(lines[dash_idx + 1])
    score_left = dash_idx - 1   # index de score_home dans lines
    score_right = dash_idx + 1  # index de score_away dans lines

    # Équipe à domicile : première ligne non-bruit en remontant depuis score_left
    home = ""
    for j in range(score_left - 1, -1, -1):
        c = lines[j]
        if c in SKIP_TOKENS:
            continue
        if re.match(r"^\d+$", c):
            continue
        if re.search(r"\d{4}", c):  # ligne de date
            continue
        if COMPETITION_RE.search(c):  # nom de compétition
            continue
        home = c
        break

    # Équipe visiteuse : première ligne non-bruit après score_right
    away = ""
    for j in range(score_right + 1, len(lines)):
        c = lines[j]
        if c in SKIP_TOKENS or re.match(r"^\d+$", c):
            continue
        away = c
        break

    if not home or not away:
        return None

    # Cotes : chercher 3 valeurs fractionnaires ou décimales dans les dernières lignes
    frac_odds = [frac_to_decimal(l) for l in lines if FRAC_RE.match(l)]
    if len(frac_odds) >= 3:
        odds_1, odds_x, odds_2 = frac_odds[0], frac_odds[1], frac_odds[2]
    else:
        # Fallback : 3 dernières valeurs numériques > 1
        dec_odds = []
        for line in reversed(lines):
            v = frac_to_decimal(line)
            if v and 1.01 <= v <= 100:
                dec_odds.insert(0, v)
            if len(dec_odds) >= 3:
                break
        if len(dec_odds) < 3:
            return None
        odds_1, odds_x, odds_2 = dec_odds[0], dec_odds[1], dec_odds[2]

    return {
        "date": date,
        "home": home,
        "away": away,
        "score_home": score_home,
        "score_away": score_away,
        "odds_home": round(odds_1, 4),
        "odds_draw": round(odds_x, 4),
        "odds_away": round(odds_2, 4),
    }


def _extract_rows(page) -> list[dict]:
    """Extrait et parse tous les matchs visibles dans les eventRows courants."""
    rows = page.locator("div[class*='eventRow']").all()
    matches, current_date = [], ""
    for row in rows:
        try:
            text = row.inner_text(timeout=5000)
        except Exception:
            continue
        for line in text.splitlines()[:6]:
            d = parse_date_from_header(line.strip())
            if d:
                current_date = d
                break
        match = parse_event_row(text, current_date)
        if match:
            matches.append(match)
    return matches


def scrape_tournament(page, tournament: dict) -> list[dict]:
    """Scrape tous les matchs d'un tournoi sur toutes les pages de résultats.

    OddsPortal pagine via des boutons JS (a.pagination-link sans href).
    On charge l'URL principale puis on clique successivement sur chaque page.
    """
    from playwright.sync_api import TimeoutError as PlaywrightTimeout

    base_url = tournament["pages"][0]
    print(f"\n[{tournament['label']}] {base_url}")

    page.goto(base_url, wait_until="networkidle", timeout=60_000)
    time.sleep(PAGE_DELAY)

    # Gérer la bannière cookie si présente
    try:
        accept = page.locator(
            "button:has-text('Accept'), button:has-text('I Accept'), "
            "#onetrust-accept-btn-handler"
        )
        if accept.count() > 0:
            accept.first.click(timeout=3000)
            time.sleep(1)
    except Exception:
        pass

    # Identifier le nombre de pages via les boutons de pagination
    pag_links = page.locator("a.pagination-link").all()
    page_numbers = []
    for lnk in pag_links:
        try:
            txt = lnk.inner_text(timeout=1000).strip()
            if txt.isdigit():
                page_numbers.append(int(txt))
        except Exception:
            pass
    total_pages = max(page_numbers) if page_numbers else 1
    print(f"  {total_pages} page(s) détectée(s)")

    seen: set[tuple] = set()
    all_matches: list[dict] = []

    def collect_page(matches: list[dict]) -> int:
        new = 0
        for m in matches:
            key = (m["date"], m["home"], m["away"])
            if key not in seen:
                seen.add(key)
                all_matches.append(m)
                new += 1
        return new

    # Page 1
    matches_p1 = _extract_rows(page)
    new = collect_page(matches_p1)
    print(f"  Page 1 : {new} nouveaux matchs")
    for m in matches_p1[:3]:
        print(f"    {m['date']} {m['home']} {m['score_home']}-{m['score_away']} "
              f"{m['away']}  {m['odds_home']}/{m['odds_draw']}/{m['odds_away']}")

    # Pages suivantes
    for p in range(2, total_pages + 1):
        try:
            btn = page.locator(f"a.pagination-link:has-text('{p}')").first
            btn.click(timeout=5000)
            time.sleep(PAGE_DELAY)
            matches_pn = _extract_rows(page)
            new = collect_page(matches_pn)
            print(f"  Page {p} : {new} nouveaux matchs")
        except PlaywrightTimeout:
            print(f"  Page {p} : timeout, arrêt")
            break
        except Exception as exc:
            print(f"  Page {p} : erreur ({exc}), arrêt")
            break

    return all_matches


def main() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit(
            "Playwright non installé.\n"
            "  pip install playwright && playwright install-deps chromium "
            "&& playwright install chromium"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = context.new_page()

        for tournament in TOURNAMENTS:
            out_file = OUT_DIR / f"{tournament['key']}.json"
            if out_file.exists():
                existing = json.loads(out_file.read_text(encoding="utf-8"))
                n = len(existing.get("matches", []))
                if n > 0:
                    print(f"[{tournament['label']}] Déjà présent ({n} matchs) — skip "
                          f"(supprimez {out_file.name} pour re-scraper)")
                    continue

            matches = scrape_tournament(page, tournament)
            print(f"\n[{tournament['label']}] Total : {len(matches)} matchs")

            out_file.write_text(
                json.dumps(
                    {"tournament": tournament["label"], "matches": matches},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"  → {out_file}")

        browser.close()

    print("\nDone. Lancez maintenant : python build_closing_backtest.py")


if __name__ == "__main__":
    main()
