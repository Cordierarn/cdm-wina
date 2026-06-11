"""Backtest temporel du modele sur tournois internationaux historiques.

Chaque tournoi est evale en reconstituant un snapshot de forces et une
calibration uniquement a partir des matchs anterieurs au tournoi.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from pricing import soft_cap_lambda, score_matrix

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
REMOTE_BASE = "https://raw.githubusercontent.com/eatpizzanot/soccer-dataset/main/parquet"
INTL_LEAGUE_IDS = {78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88}
CLOSING_BACKTEST_FILE = DATA_DIR / "closing_backtest.json"

TOURNAMENT_PATTERNS = [
    ("World Cup", r"world cup|coupe du monde|fifa world cup"),
    ("Euro", r"euro|uefa european championship|championship europe"),
    ("Copa America", r"copa america"),
    ("Africa Cup of Nations", r"africa cup|afcon|coupe d'afrique|coupe d afrique"),
    ("Asian Cup", r"asian cup|coupe d asie"),
    ("Gold Cup", r"gold cup|concacaf"),
]

ODDS_1N2_PATTERNS = {
    "home": ["odds_1", "home_odds", "odds_home", "odds_home_win", "price_home"],
    "draw": ["odds_x", "draw_odds", "odds_draw", "price_draw"],
    "away": ["odds_2", "away_odds", "odds_away", "odds_away_win", "price_away"],
}
ODDS_TOTALS_PATTERNS = {
    "over": ["over25_odds", "odds_over25", "odds_over_2_5", "over_2_5_odds", "over2.5_odds", "over_25_odds"],
    "under": ["under25_odds", "odds_under25", "odds_under_2_5", "under_2_5_odds", "under2.5_odds", "under_25_odds"],
}


@dataclass
class BacktestPoint:
    tournament: str
    market: str
    y: list[float]
    model: list[float]
    baseline: list[float]
    market_probs: list[float] | None


def load_parquet(name: str) -> pd.DataFrame:
    local = DATA_DIR / "dataset" / name
    source = local if local.exists() else f"{REMOTE_BASE}/{name}"
    try:
        return pd.read_parquet(source)
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise SystemExit(
            f"Impossible de lire {name} depuis {source}. "
            "Installe pyarrow/pandas et/ou place les parquets sous data/dataset/."
        ) from exc


def norm(s: str) -> str:
    import unicodedata

    s = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower().strip()


def pick_column(columns: pd.Index, patterns: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for pattern in patterns:
        exact = lowered.get(pattern.lower())
        if exact:
            return exact
    for pattern in patterns:
        pat = pattern.lower()
        for lower, original in lowered.items():
            if pat in lower:
                return original
    return None


def first_non_null(row: pd.Series, candidates: list[str], default=None):
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return default


def de_vig(odds: list[float]) -> list[float] | None:
    if any((o is None or o <= 0) for o in odds):
        return None
    inv = [1 / o for o in odds]
    total = sum(inv)
    if total <= 0:
        return None
    return [v / total for v in inv]


def brier_score(probs: list[float], y: list[float]) -> float:
    return sum((p - t) ** 2 for p, t in zip(probs, y))


def log_loss(probs: list[float], y: list[float]) -> float:
    eps = 1e-15
    idx = next(i for i, t in enumerate(y) if t > 0.5)
    return -math.log(max(probs[idx], eps))


def competition_tag(label: str) -> str | None:
    t = norm(label)
    for tag, pattern in TOURNAMENT_PATTERNS:
        if __import__("re").search(pattern, t):
            return tag
    return None


def tournament_key(row: pd.Series, label_col: str, season_col: str | None, date_col: str) -> tuple[str, str]:
    label = str(row[label_col]) if label_col in row.index else ""
    tag = competition_tag(label)
    if not tag:
        return "", ""
    season = None
    if season_col and season_col in row.index and pd.notna(row[season_col]):
        season = str(row[season_col])
    if not season:
        raw_date = row[date_col]
        season = str(pd.to_datetime(raw_date).year)
    return tag, f"{tag} {season}"


def rating_columns(columns: pd.Index) -> dict[str, str] | None:
    home_mu = pick_column(columns, ["home_team_rating_mu", "home_rating_mu", "rating_mu_home", "home_mu_rating", "home_mu"])
    away_mu = pick_column(columns, ["away_team_rating_mu", "away_rating_mu", "rating_mu_away", "away_mu_rating", "away_mu"])
    home_sigma = pick_column(columns, ["home_team_rating_sigma", "home_rating_sigma", "rating_sigma_home", "home_sigma_rating", "home_sigma"])
    away_sigma = pick_column(columns, ["away_team_rating_sigma", "away_rating_sigma", "rating_sigma_away", "away_sigma_rating", "away_sigma"])
    if home_mu and away_mu:
        return {"home_mu": home_mu, "away_mu": away_mu, "home_sigma": home_sigma, "away_sigma": away_sigma}
    return None


def build_snapshot_from_ratings(prior: pd.DataFrame, cols: dict[str, str], team_id_col_home: str, team_id_col_away: str,
                                date_col: str) -> dict[int, float]:
    latest: dict[int, tuple[pd.Timestamp, float]] = {}
    prior_sorted = prior.sort_values(date_col)
    for _, row in prior_sorted.iterrows():
        match_date = pd.to_datetime(row[date_col])
        for side, team_col, mu_col, sigma_col in (
            ("home", team_id_col_home, cols["home_mu"], cols.get("home_sigma")),
            ("away", team_id_col_away, cols["away_mu"], cols.get("away_sigma")),
        ):
            tid = row[team_col]
            if pd.isna(tid):
                continue
            raw = float(row[mu_col])
            if sigma_col and pd.notna(row[sigma_col]):
                raw -= float(row[sigma_col])
            current = latest.get(int(tid))
            if current is None or match_date >= current[0]:
                latest[int(tid)] = (match_date, raw)
    return {tid: val for tid, (_, val) in latest.items()}


def build_snapshot_from_elo(prior: pd.DataFrame, home_id_col: str, away_id_col: str, goals_home_col: str, goals_away_col: str,
                           date_col: str) -> tuple[dict[int, float], list[tuple[float, float, int, int]]]:
    ratings: defaultdict[int, float] = defaultdict(lambda: 1500.0)
    calibration_rows: list[tuple[float, float, int, int]] = []
    prior_sorted = prior.sort_values(date_col)
    for _, row in prior_sorted.iterrows():
        hid = int(row[home_id_col])
        aid = int(row[away_id_col])
        gh = int(row[goals_home_col])
        ga = int(row[goals_away_col])
        rh = ratings[hid]
        ra = ratings[aid]
        calibration_rows.append((rh, ra, gh, ga))
        expected = 1.0 / (1.0 + 10 ** ((ra - rh) / 400.0))
        actual = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
        mov = math.log(abs(gh - ga) + 1.0) + 1.0
        k = 18.0 * mov
        delta = k * (actual - expected)
        ratings[hid] += delta
        ratings[aid] -= delta
    return dict(ratings), calibration_rows


def calibration_from_snapshot(prior: pd.DataFrame, home_id_col: str, away_id_col: str, goals_home_col: str,
                              goals_away_col: str, date_col: str, cols: dict[str, str] | None) -> tuple[float, float, float, dict[int, float], str]:
    if cols:
        ratings = build_snapshot_from_ratings(prior, cols, home_id_col, away_id_col, date_col)
        calibration_rows = []
        for _, row in prior.sort_values(date_col).iterrows():
            rh = float(row[cols["home_mu"]]) - float(row[cols["home_sigma"]]) if cols.get("home_sigma") and pd.notna(row[cols["home_sigma"]]) else float(row[cols["home_mu"]])
            ra = float(row[cols["away_mu"]]) - float(row[cols["away_sigma"]]) if cols.get("away_sigma") and pd.notna(row[cols["away_sigma"]]) else float(row[cols["away_mu"]])
            calibration_rows.append((rh, ra, int(row[goals_home_col]), int(row[goals_away_col])))
        force_source = "fixture_rating_columns"
    else:
        ratings, calibration_rows = build_snapshot_from_elo(prior, home_id_col, away_id_col, goals_home_col, goals_away_col, date_col)
        force_source = "precutoff_elo"

    if not calibration_rows:
        return 1.18, 1.21, -0.105, ratings, force_source

    diffs = [rh - ra for rh, ra, _, _ in calibration_rows for _ in (0, 1)]
    goals = [g for _, _, gh, ga in calibration_rows for g in (gh, ga)]

    best_b, best_ll = 0.0, -math.inf
    for b in [i / 5000 for i in range(0, 41)]:
        denom = sum(math.exp(b * d) for d in diffs)
        if denom <= 0:
            continue
        ea = sum(goals) / denom
        ll = sum(g * (math.log(ea) + b * d) - ea * math.exp(b * d) for g, d in zip(goals, diffs))
        if ll > best_ll:
            best_b, best_ll = float(b), float(ll)

    base_lambda = sum(goals) / sum(math.exp(best_b * d) for d in diffs)
    raw_values = list(ratings.values())
    lo, hi = min(raw_values), max(raw_values)
    spread = best_b * (hi - lo if hi > lo else 1.0)

    gh = [gh for _, _, gh, _ in calibration_rows]
    ga = [ga for _, _, _, ga in calibration_rows]
    dh = [rh - ra for rh, ra, _, _ in calibration_rows]
    lh = [soft_cap_lambda(base_lambda * math.exp(best_b * d)) for d in dh]
    la = [soft_cap_lambda(base_lambda * math.exp(-best_b * d)) for d in dh]
    best_rho, best_rll = 0.0, -math.inf
    for rho in [i / 1000 for i in range(-150, 151, 5)]:
        tau = []
        ok = True
        for h, a, lhi, lai in zip(gh, ga, lh, la):
            if h == 0 and a == 0:
                val = 1 - lhi * lai * rho
            elif h == 0 and a == 1:
                val = 1 + lhi * rho
            elif h == 1 and a == 0:
                val = 1 + lai * rho
            elif h == 1 and a == 1:
                val = 1 - rho
            else:
                val = 1.0
            if val <= 0:
                ok = False
                break
            tau.append(val)
        if not ok:
            continue
        rll = sum(math.log(x) for x in tau)
        if rll > best_rll:
            best_rho, best_rll = float(rho), float(rll)

    return base_lambda, spread, best_rho, ratings, force_source


def score_probs(base_lambda: float, spread: float, rho: float, diff: float) -> tuple[list[list[float]], float, float]:
    lh = soft_cap_lambda(base_lambda * math.exp(spread * diff))
    la = soft_cap_lambda(base_lambda * math.exp(-spread * diff))
    return score_matrix(lh, la, rho), lh, la


def detect_market_odds(row: pd.Series, mapping: dict[str, list[str]]) -> list[float] | None:
    values = []
    for key in mapping:
        col = pick_column(row.index, mapping[key])
        if not col or pd.isna(row[col]):
            return None
        values.append(float(row[col]))
    return de_vig(values)


def market_lines(fixtures: pd.DataFrame, market: str, row: pd.Series) -> list[float] | None:
    if market == "1n2":
        return detect_market_odds(row, ODDS_1N2_PATTERNS)
    if market == "totals":
        return detect_market_odds(row, ODDS_TOTALS_PATTERNS)
    return None


def load_closing_backtest() -> dict:
    """Charge closing_backtest.json si présent, sinon retourne dict vide."""
    if not CLOSING_BACKTEST_FILE.exists():
        return {}
    try:
        return json.loads(CLOSING_BACKTEST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


# Mapping dataset→clé-closing : noms utilisés dans le soccer-dataset qui diffèrent
# des noms OddsPortal après normalisation ASCII.
# La clé closing est toujours le nom OddsPortal normalisé (via TEAM_NAME_MAP de
# build_closing_backtest.py). Ce dict fait l'aller-retour dataset→closing-norm.
_DATASET_TO_CLOSING_NORM: dict[str, str] = {
    # dataset "USA" → closing "united states"
    "usa": "united states",
    # dataset "Türkiye" → closing "turkiye" (accents déjà strips par norm(), mais
    # le mapping OddsPortal "turkey"→"Türkiye" produit norm("Türkiye")="turkiye")
    "turkiye": "turkiye",
    # dataset "Korea Republic" variantes
    "korea republic": "south korea",
    # dataset "IR Iran"
    "ir iran": "iran",
    # dataset "Bosnia and Herzegovina" variantes
    "bosnia & herzegovina": "bosnia and herzegovina",
}


def _closing_norm(name: str) -> str:
    """Normalise un nom d'équipe dataset vers la clé closing-backtest."""
    n = norm(name)
    return _DATASET_TO_CLOSING_NORM.get(n, n)


def closing_lookup(
    closing: dict,
    date_val: pd.Timestamp,
    home_name: str,
    away_name: str,
) -> dict | None:
    """Cherche l'entrée de clôture complète (h2h + totals) pour un match donné.

    Essaie date exacte puis ±1 jour (décalage UTC/local), et applique le mapping
    dataset→closing-norm pour les noms d'équipes non canoniques (USA, Türkiye…).
    """
    if not closing:
        return None
    h = _closing_norm(home_name)
    a = _closing_norm(away_name)

    for delta in (0, 1, -1):
        d = (date_val + pd.Timedelta(days=delta)).strftime("%Y-%m-%d")
        key = f"{d}|{h}|{a}"
        if key in closing:
            return closing[key]
    return None


def build_team_name_index(fixtures: pd.DataFrame, home_id_col: str, away_id_col: str) -> dict[int, str]:
    """Construit un index id→nom à partir des colonnes de nom d'équipe si présentes."""
    name_index: dict[int, str] = {}
    home_name_col = pick_column(fixtures.columns, ["home_team_name", "home_name", "team_home_name", "home_team"])
    away_name_col = pick_column(fixtures.columns, ["away_team_name", "away_name", "team_away_name", "away_team"])
    if home_name_col and home_id_col:
        for _, row in fixtures[[home_id_col, home_name_col]].drop_duplicates().iterrows():
            tid = row[home_id_col]
            name = row[home_name_col]
            if pd.notna(tid) and pd.notna(name):
                name_index[int(tid)] = str(name)
    if away_name_col and away_id_col:
        for _, row in fixtures[[away_id_col, away_name_col]].drop_duplicates().iterrows():
            tid = row[away_id_col]
            name = row[away_name_col]
            if pd.notna(tid) and pd.notna(name):
                name_index[int(tid)] = str(name)
    # Essayer aussi un parquet teams.parquet si disponible
    teams_path = DATA_DIR / "dataset" / "teams.parquet"
    if teams_path.exists():
        try:
            teams = pd.read_parquet(teams_path)
            id_col = pick_column(teams.columns, ["id", "team_id"])
            n_col = pick_column(teams.columns, ["name", "team_name", "short_name"])
            if id_col and n_col:
                for _, row in teams[[id_col, n_col]].drop_duplicates().iterrows():
                    if pd.notna(row[id_col]) and pd.notna(row[n_col]):
                        name_index[int(row[id_col])] = str(row[n_col])
        except Exception:
            pass
    return name_index


def predict_row(base_lambda: float, spread: float, rho: float, diff: float) -> tuple[dict[str, list[float]], list[list[float]]]:
    mat, _, _ = score_probs(base_lambda, spread, rho, diff)
    p1 = sum(mat[h][a] for h in range(11) for a in range(11) if h > a)
    px = sum(mat[h][a] for h in range(11) for a in range(11) if h == a)
    p2 = sum(mat[h][a] for h in range(11) for a in range(11) if h < a)
    over = sum(mat[h][a] for h in range(11) for a in range(11) if h + a > 2.5)
    return {"1n2": [p1, px, p2], "totals": [over, 1 - over]}, mat


def model_over_prob(mat: list[list[float]], line: float) -> float:
    return sum(mat[h][a] for h in range(11) for a in range(11) if h + a > line)


def tournament_title(key: str) -> str:
    return key


def summarize_metrics(points: list[BacktestPoint]) -> dict:
    summary: dict[str, dict] = {}
    for tournament in sorted({p.tournament for p in points}):
        tournament_points = [p for p in points if p.tournament == tournament]
        summary[tournament] = {}
        for market in {p.market for p in tournament_points}:
            market_points = [p for p in tournament_points if p.market == market]
            if not market_points:
                continue
            model_brier = sum(brier_score(p.model, p.y) for p in market_points) / len(market_points)
            baseline_brier = sum(brier_score(p.baseline, p.y) for p in market_points) / len(market_points)
            model_ll = sum(log_loss(p.model, p.y) for p in market_points) / len(market_points)
            baseline_ll = sum(log_loss(p.baseline, p.y) for p in market_points) / len(market_points)
            market_points_with_ref = [p for p in market_points if p.market_probs]
            market_brier = None
            market_ll = None
            best_weight = None
            best_ll_mix = None
            if market_points_with_ref:
                market_brier = sum(brier_score(p.market_probs or p.baseline, p.y) for p in market_points_with_ref) / len(market_points_with_ref)
                market_ll = sum(log_loss(p.market_probs or p.baseline, p.y) for p in market_points_with_ref) / len(market_points_with_ref)
                best_weight = 0.40
                best_ll_mix = math.inf
                for w in [i / 20 for i in range(21)]:
                    ll = 0.0
                    for p in market_points_with_ref:
                        mix = [w * mp + (1 - w) * rp for mp, rp in zip(p.model, p.market_probs or p.baseline)]
                        ll += log_loss(mix, p.y)
                    ll /= len(market_points_with_ref)
                    if ll < best_ll_mix:
                        best_ll_mix = ll
                        best_weight = w
            summary[tournament][market] = {
                "n": len(market_points),
                "n_with_market": len(market_points_with_ref),
                "brier": {"model": model_brier, "baseline": baseline_brier, "market": market_brier},
                "logloss": {"model": model_ll, "baseline": baseline_ll, "market": market_ll},
                "best_weight": best_weight,
                "best_mix_logloss": best_ll_mix,
            }
    return summary


def main() -> None:
    fixtures = load_parquet("fixtures.parquet")
    leagues = load_parquet("leagues.parquet")

    date_col = pick_column(fixtures.columns, ["date", "start", "kickoff", "match_start", "matchstart"])
    if not date_col:
        raise SystemExit("Impossible de trouver la colonne de date dans fixtures.parquet")

    home_id_col = pick_column(fixtures.columns, ["home_team_id", "home_id", "team_home_id"])
    away_id_col = pick_column(fixtures.columns, ["away_team_id", "away_id", "team_away_id"])
    goals_home_col = pick_column(fixtures.columns, ["goals_home", "home_goals", "score_home"])
    goals_away_col = pick_column(fixtures.columns, ["goals_away", "away_goals", "score_away"])
    if not all((home_id_col, away_id_col, goals_home_col, goals_away_col)):
        raise SystemExit("Colonnes d'identification ou de score introuvables dans fixtures.parquet")

    league_id_col = pick_column(fixtures.columns, ["league_id", "competition_id", "tournament_id"])
    label_col = pick_column(fixtures.columns, ["league_name", "competition_name", "tournament_name", "name", "league"])
    season_col = pick_column(fixtures.columns, ["season", "season_name", "year", "league_season"])

    if not label_col and league_id_col and not leagues.empty:
        league_name_col = pick_column(leagues.columns, ["name", "league_name", "title"])
        league_id_lookup = pick_column(leagues.columns, ["id", "league_id"])
        if league_name_col and league_id_lookup:
            fixtures = fixtures.merge(
                leagues[[league_id_lookup, league_name_col]].rename(
                    columns={league_id_lookup: league_id_col, league_name_col: "league_label"}
                ),
                on=league_id_col,
                how="left",
            )
            label_col = "league_label"

    fixtures = fixtures.copy()
    fixtures[date_col] = pd.to_datetime(fixtures[date_col], errors="coerce", utc=True)
    fixtures = fixtures[fixtures[date_col].notna() & fixtures[goals_home_col].notna() & fixtures[goals_away_col].notna()]
    if league_id_col:
        fixtures = fixtures[fixtures[league_id_col].isin(INTL_LEAGUE_IDS)]

    if label_col is None:
        raise SystemExit("Impossible d'identifier le nom des competitions dans le dataset")

    fixtures["tournament_tag"] = fixtures[label_col].map(competition_tag)
    fixtures = fixtures[fixtures["tournament_tag"].notna()]
    if fixtures.empty:
        raise SystemExit("Aucun tournoi historique international reconnu n'a ete trouve")

    # Cotes de clôture externes (OddsPortal via closing_backtest.json)
    closing = load_closing_backtest()
    if closing:
        print(f"[closing] {len(closing)} matchs chargés depuis {CLOSING_BACKTEST_FILE.name}")
    else:
        print("[closing] closing_backtest.json absent — métriques marché en n/a")

    team_name_index = build_team_name_index(fixtures, home_id_col, away_id_col)

    points: list[BacktestPoint] = []
    tournaments: list[dict] = []

    grouped = []
    for _, row in fixtures.sort_values(date_col).iterrows():
        tag, key = tournament_key(row, label_col, season_col, date_col)
        if not key:
            continue
        grouped.append((key, row))

    tournament_keys = []
    seen = set()
    for key, _ in grouped:
        if key not in seen:
            seen.add(key)
            tournament_keys.append(key)

    for key in tournament_keys:
        tournament_rows = fixtures[fixtures.apply(lambda r: tournament_key(r, label_col, season_col, date_col)[1] == key, axis=1)]
        tournament_rows = tournament_rows.sort_values(date_col)
        if len(tournament_rows) < 4:
            continue
        cutoff = tournament_rows[date_col].min()
        prior = fixtures[fixtures[date_col] < cutoff]
        if len(prior) < 50:
            continue

        rating_cols = rating_columns(fixtures.columns)
        base_lambda, spread, rho, snapshot, force_source = calibration_from_snapshot(
            prior,
            home_id_col,
            away_id_col,
            goals_home_col,
            goals_away_col,
            date_col,
            rating_cols,
        )

        raw_values = list(snapshot.values())
        if not raw_values:
            continue
        lo, hi = min(raw_values), max(raw_values)
        if hi == lo:
            hi = lo + 1.0

        def team_raw(row: pd.Series, side: str) -> float | None:
            team_id = int(row[home_id_col if side == "home" else away_id_col])
            return snapshot.get(team_id)

        market_1n2_rows = []
        market_totals_rows = []
        for _, row in tournament_rows.iterrows():
            rh = team_raw(row, "home")
            ra = team_raw(row, "away")
            if rh is None or ra is None:
                continue
            s_home = (rh - lo) / (hi - lo)
            s_away = (ra - lo) / (hi - lo)
            diff = s_home - s_away
            probs, mat = predict_row(base_lambda, spread, rho, diff)
            gh = int(row[goals_home_col])
            ga = int(row[goals_away_col])
            y_1n2 = [1.0 if gh > ga else 0.0, 1.0 if gh == ga else 0.0, 1.0 if gh < ga else 0.0]
            y_tot = [1.0 if gh + ga > 2.5 else 0.0, 1.0 if gh + ga <= 2.5 else 0.0]
            model_tot = probs["totals"]
            market_1n2 = market_lines(fixtures, "1n2", row)
            market_totals = market_lines(fixtures, "totals", row)
            # Enrichissement depuis closing_backtest.json si disponible
            if closing:
                h_id = int(row[home_id_col])
                a_id = int(row[away_id_col])
                h_name = team_name_index.get(h_id, "")
                a_name = team_name_index.get(a_id, "")
                entry = closing_lookup(closing, row[date_col], h_name, a_name) if h_name and a_name else None
                if entry:
                    if market_1n2 is None:
                        market_1n2 = entry.get("h2h")
                    totals_entry = entry.get("totals")
                    # ligne stockée (2.5 ou la plus proche) : pricer modèle, marché
                    # et résultat sur CETTE ligne. Lignes entières ignorées (push).
                    if market_totals is None and totals_entry and totals_entry.get("probs"):
                        line = float(totals_entry.get("line", 2.5))
                        if line != int(line):
                            market_totals = totals_entry["probs"]
                            over = model_over_prob(mat, line)
                            model_tot = [over, 1 - over]
                            y_tot = [1.0 if gh + ga > line else 0.0, 1.0 if gh + ga <= line else 0.0]
            points.append(BacktestPoint(key, "1n2", y_1n2, probs["1n2"], [1 / 3, 1 / 3, 1 / 3], market_1n2))
            points.append(BacktestPoint(key, "totals", y_tot, model_tot, [0.5, 0.5], market_totals))
            market_1n2_rows.append((probs["1n2"], y_1n2, market_1n2))
            market_totals_rows.append((model_tot, y_tot, market_totals))

        if not market_1n2_rows and not market_totals_rows:
            continue

        tournaments.append({
            "tournament": key,
            "cutoff": cutoff.isoformat(),
            "n_matches": len(tournament_rows),
            "force_source": force_source,
            "calibration": {
                "base_lambda": round(base_lambda, 4),
                "spread": round(spread, 4),
                "rho": round(rho, 4),
                "force_range": [round(lo, 4), round(hi, 4)],
            },
        })

    summary = summarize_metrics(points)

    # Estimation du poids optimal par marche sur les points ayant une reference
    # marche, puis shrinkage bayesien vers le prior : avec ~100 matchs la courbe
    # logloss(w) est quasi-plate, un w brut n'est pas fiable.
    #   w_final = (n*w_brut + n0*w_prior) / (n + n0)
    W_PRIOR = 0.25
    N0 = 200
    MIN_N_TOTALS = 50  # en dessous, pas d'estimation : prior seul
    model_weights = {"1n2": W_PRIOR, "totals": W_PRIOR}
    for market in ("1n2", "totals"):
        refs = [p for p in points if p.market == market and p.market_probs]
        n = len(refs)
        detail = {"n": n, "w_prior": W_PRIOR, "n0": N0}
        if not refs or (market == "totals" and n < MIN_N_TOTALS):
            detail.update({"raw_best_weight": None, "w_final": W_PRIOR,
                           "note": f"n={n} < {MIN_N_TOTALS} : prior conserve" if refs else "aucune reference marche"})
            summary.setdefault("__overall__", {})[market] = detail
            continue
        best_w, best_ll = W_PRIOR, math.inf
        for i in range(21):
            w = i / 20
            ll = 0.0
            for p in refs:
                mix = [w * m + (1 - w) * r for m, r in zip(p.model, p.market_probs or p.baseline)]
                ll += log_loss(mix, p.y)
            ll /= n
            if ll < best_ll:
                best_ll = ll
                best_w = w
        w_final = (n * best_w + N0 * W_PRIOR) / (n + N0)
        model_weights[market] = round(w_final, 4)
        detail.update({"raw_best_weight": best_w, "w_final": round(w_final, 4),
                       "best_mix_logloss": best_ll})
        summary.setdefault("__overall__", {})[market] = detail

    out = {
        "generated_at": int(datetime.utcnow().timestamp()),
        "tournaments": tournaments,
        "summary": summary,
        "model_weights": model_weights,
        "notes": {
            "market_reference": (
                "closing_backtest.json (OddsPortal scraping, WC2022+Euro2024)"
                if closing else "n/a — closing_backtest.json absent"
            ),
            "force_source": "fixture rating columns if present, otherwise pre-cutoff Elo fallback",
            "weights": "grid search logloss + shrinkage (n*w_raw + n0*0.25)/(n+n0), n0=200; "
                       "totals: prior 0.25 conserve si moins de 50 matchs avec cotes O/U",
        },
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "backtest.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print("Tournoi | Marche | N | Brier(mod/base/mkt) | LogLoss(mod/base/mkt) | w*")
    for tournament in tournaments:
        key = tournament["tournament"]
        for market in ("1n2", "totals"):
            stats = summary.get(key, {}).get(market)
            if not stats:
                continue
            brier = stats["brier"]
            ll = stats["logloss"]
            market_brier = f"{brier['market']:.4f}" if brier["market"] is not None else "n/a"
            market_ll = f"{ll['market']:.4f}" if ll["market"] is not None else "n/a"
            print(
                f"{key[:26]:26s} | {market:6s} | {stats['n']:3d} | "
                f"{brier['model']:.4f}/{brier['baseline']:.4f}/{market_brier:>7s} | "
                f"{ll['model']:.4f}/{ll['baseline']:.4f}/{market_ll:>7s} | "
                f"{stats['best_weight'] if stats['best_weight'] is not None else 0.40:.2f}"
            )
    print(f"OK: {len(tournaments)} tournois -> data/backtest.json")


if __name__ == "__main__":
    main()
