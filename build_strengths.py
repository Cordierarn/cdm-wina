"""Forces des 48 equipes CDM depuis soccer-dataset (Glicko-2) -> data/strengths_dataset.json.

A lancer avec le venv du repo ScoutFootball (pandas+pyarrow) :
  C:/Users/nonog/ScoutFootball_for_World_Cup/.venv/Scripts/python.exe build_strengths.py

Source : https://github.com/eatpizzanot/soccer-dataset (rating_mu = Glicko-2).

Calibration sur TOUS les matchs internationaux depuis 2019 (tournois + qualifs + amicaux),
pondérés par type de compétition (source : FIFA ranking weighting, Elo K-factor literature) :
  - Coupe du Monde / tournoi final         : 1.00
  - Tournois continentaux (Euro, Copa...)  : 0.85
  - Matchs de qualification WC/Euro        : 0.70
  - Amicaux et Nations League              : 0.35
+ Time decay (demi-vie 2 ans) pour favoriser les données récentes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
DS = DATA / "dataset"

# Tous les league_ids internationaux du soccer-dataset.
# IDs 78-88 : compétitions majeures (WC, Euro, Copa America, AFCON, Asian Cup, Gold Cup, Nations League).
# On les complète avec tous les IDs dont le nom contient des mots-clés internationaux
# (qualifications, amicaux) — découverts dynamiquement ci-dessous.
CORE_INTL_IDS = {78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88}

# Mots-clés pour détecter les ligues internationales hors liste fixe
INTL_KEYWORDS = re.compile(
    r"world cup|euro|copa america|africa cup|asian cup|gold cup|concacaf|"
    r"nations league|qualification|qualifier|friendly|international|fifa",
    re.IGNORECASE,
)

# Poids par type de compétition (basé sur FIFA ranking weights + Elo K-factor literature)
# Clé = regex sur le nom de la ligue normalisé
COMPETITION_WEIGHTS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"world cup", re.I),                          1.00),
    (re.compile(r"copa america|africa cup|asian cup|gold cup|euro \d{4}", re.I), 0.85),
    (re.compile(r"european championship|uefa euro|concacaf championship", re.I), 0.85),
    (re.compile(r"qualif|qualifier", re.I),                   0.70),
    (re.compile(r"nations league", re.I),                     0.50),
    (re.compile(r"friendly|international", re.I),             0.35),
]
DEFAULT_COMPETITION_WEIGHT = 0.60  # pour les IDs dans CORE_INTL_IDS sans label reconnu


def competition_weight(league_name: str) -> float:
    for pattern, w in COMPETITION_WEIGHTS:
        if pattern.search(league_name or ""):
            return w
    return DEFAULT_COMPETITION_WEIGHT


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


# ── Re-fit Glicko en cours de tournoi ────────────────────────────────────
# CDM_COMP_WEIGHT : poids des matchs CDM dans le re-fit. 1.00 = plein signal
# (un match de poule CDM est du signal fort, on ne le noie pas). Coherent avec
# COMPETITION_WEIGHTS (World Cup = 1.00).
CDM_COMP_WEIGHT = 1.00
RESULTS_FILE = DATA / "cdm_results.json"
SNAP_DIR = DATA / "strengths_snapshots"


def _snapshot_mus(by_name: dict, tag: str) -> dict[str, float]:
    """mu - sigma par equipe WC48, pour figer l'etat avant/apres re-fit."""
    out = {}
    for team in WC48:
        entry = by_name.get(ALIASES.get(team, team))
        if entry:
            out[team] = round(float(entry[0] - entry[1]), 2)
    return out


def refit_glicko_with_cdm(by_name: dict) -> int:
    """Reinjecte les resultats CDM regles dans les ratings Glicko (mise a jour
    Glicko-1 batch sur une periode), avec snapshots avant/apres.

    by_name : {nom_dataset: (mu=rating, sigma=RD)} ; mute en place.
    Retourne le nombre de matchs CDM appliques.
    """
    import math

    if not RESULTS_FILE.exists():
        return 0
    results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    if not results:
        return 0

    # repo-name -> nom dataset (inverse d'ALIASES, identite sinon)
    def ds(repo_name: str) -> str:
        return ALIASES.get(repo_name, repo_name)

    # Etat de depart (rating, RD) pour les equipes impliquees
    ratings: dict[str, list[float]] = {}
    games: dict[str, list[tuple[float, float, float]]] = {}  # team -> [(opp_r, opp_RD, s)]
    for r in results:
        h, a = ds(r["home"]), ds(r["away"])
        if h not in by_name or a not in by_name:
            continue
        for t in (h, a):
            if t not in ratings:
                mu, rd = by_name[t]
                ratings[t] = [float(mu), float(rd)]
        gh, ga = r["goals_home"], r["goals_away"]
        sh = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
        games.setdefault(h, []).append((float(by_name[a][0]), float(by_name[a][1]), sh))
        games.setdefault(a, []).append((float(by_name[h][0]), float(by_name[h][1]), 1.0 - sh))

    if not games:
        return 0

    SNAP_DIR.mkdir(exist_ok=True)
    import time
    stamp = time.strftime("%Y%m%d_%H%M")
    (SNAP_DIR / f"strengths_{stamp}_pre.json").write_text(
        json.dumps(_snapshot_mus(by_name, "pre"), indent=1, ensure_ascii=False), encoding="utf-8")

    # Glicko-1 batch : une periode = tous les matchs CDM de l'equipe.
    q = math.log(10) / 400
    updated: dict[str, tuple[float, float]] = {}
    for team, gs in games.items():
        r0, rd0 = ratings[team]
        # le re-fit a plein poids = standard Glicko ; CDM_COMP_WEIGHT module l'apport
        inv_d2 = 0.0
        delta_sum = 0.0
        for opp_r, opp_rd, s in gs:
            g = 1.0 / math.sqrt(1 + 3 * q * q * opp_rd * opp_rd / (math.pi ** 2))
            e = 1.0 / (1 + 10 ** (-g * (r0 - opp_r) / 400))
            inv_d2 += q * q * g * g * e * (1 - e)
            delta_sum += g * (s - e)
        if inv_d2 <= 0:
            continue
        denom = 1.0 / (rd0 * rd0) + inv_d2
        r_new = r0 + CDM_COMP_WEIGHT * (q / denom) * delta_sum
        rd_new = math.sqrt(1.0 / denom)
        updated[team] = (r_new, rd_new)

    for team, (r_new, rd_new) in updated.items():
        by_name[team] = (r_new, rd_new)

    (SNAP_DIR / f"strengths_{stamp}_post.json").write_text(
        json.dumps(_snapshot_mus(by_name, "post"), indent=1, ensure_ascii=False), encoding="utf-8")
    return sum(len(gs) for gs in games.values()) // 2


def main() -> None:
    teams    = pd.read_parquet(DS / "teams.parquet")
    fixtures = pd.read_parquet(DS / "fixtures.parquet")
    try:
        leagues = pd.read_parquet(DS / "leagues.parquet")
    except Exception:
        leagues = pd.DataFrame()

    # ── Découverte dynamique des league_ids internationaux ────────────────
    # On part des IDs connus (CORE_INTL_IDS) et on ajoute tous les IDs dont
    # le nom de ligue contient un mot-clé international.
    intl_ids = set(CORE_INTL_IDS)
    league_names: dict[int, str] = {}  # league_id -> nom lisible

    if not leagues.empty:
        id_col  = "id"   if "id"   in leagues.columns else "league_id"
        nam_col = "name" if "name" in leagues.columns else "league_name"
        for _, row in leagues.iterrows():
            lid  = int(row[id_col])
            name = str(row[nam_col])
            league_names[lid] = name
            if INTL_KEYWORDS.search(name):
                intl_ids.add(lid)
    else:
        # Fallback : inférer le nom depuis les fixtures si leagues.parquet absent
        if "league_name" in fixtures.columns:
            for lid, grp in fixtures.groupby("league_id"):
                name = str(grp["league_name"].iloc[0])
                league_names[int(lid)] = name
                if INTL_KEYWORDS.search(name):
                    intl_ids.add(int(lid))

    print(f"leagues internationales détectées : {len(intl_ids)} IDs")

    # ── Filtrage + pondération des matchs ────────────────────────────────
    intl = fixtures[fixtures.league_id.isin(intl_ids)].copy()
    nat_ids = set(intl.home_team_id) | set(intl.away_team_id)
    nat = teams[teams.id.isin(nat_ids)]
    nat = nat[~nat.name.str.contains(r"U\d+|B\b|reserve|youth", regex=True, case=False)]

    # Poids compétition par ligne
    def row_comp_weight(lid: int) -> float:
        name = league_names.get(int(lid), "")
        return competition_weight(name)

    intl["comp_weight"] = intl["league_id"].map(row_comp_weight)

    # ── Glicko-2 ratings ─────────────────────────────────────────────────
    by_name = {n: (mu, sig) for n, mu, sig in zip(nat.name, nat.rating_mu, nat.rating_sigma)}

    # Re-fit en cours de tournoi : reinjecte les resultats CDM regles (snapshots
    # avant/apres dans data/strengths_snapshots/ pour mesurer l'effet).
    n_refit = refit_glicko_with_cdm(by_name)
    if n_refit:
        print(f"re-fit Glicko : {n_refit} matchs CDM reinjectes (poids {CDM_COMP_WEIGHT})")

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

    # 2e composante : valeur marchande (Transfermarkt). 70% Glicko / 30% valeur.
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

    # ── Calibration Poisson sur TOUS les matchs depuis 2019 ──────────────
    import numpy as np
    import datetime
    import math as _math

    # rating id -> mu-sigma (toutes nations, pas seulement WC48)
    rating = {tid: mu - sig for tid, mu, sig in zip(nat.id, nat.rating_mu, nat.rating_sigma)}

    rec = intl[(intl.date >= "2019-01-01") & intl.goals_home.notna() & intl.goals_away.notna()]

    # Poids final = comp_weight × time_decay (demi-vie 2 ans)
    # On utilise 2 ans (730 j) comme recommandé par la littérature sur les matchs
    # internationaux (Liverpool 2017, arXiv 1705.09575).
    cutoff_date  = datetime.date(2026, 6, 1)
    HALF_LIFE    = 730  # jours

    rows = []
    match_meta = []  # (gh, ga, dh, w, age_days) par match, pour la MLE jointe + OOS
    skipped = 0
    for _, r in rec.iterrows():
        rh = rating.get(r.home_team_id)
        ra = rating.get(r.away_team_id)
        if rh is None or ra is None:
            skipped += 1
            continue
        try:
            match_date = r.date.date() if hasattr(r.date, "date") else datetime.date.fromisoformat(str(r.date)[:10])
            age_days   = (cutoff_date - match_date).days
            time_w     = 2 ** (-age_days / HALF_LIFE) if age_days >= 0 else 1.0
        except Exception:
            age_days = 0
            time_w = 1.0
        comp_w = float(r.comp_weight)
        w = comp_w * time_w
        rows.append((float(r.goals_home), rh - ra, w))
        rows.append((float(r.goals_away), ra - rh, w))
        match_meta.append((int(r.goals_home), int(r.goals_away), rh - ra, w, age_days))

    n_matches = len(rows) // 2
    print(f"calibration: {n_matches} matchs utilisés ({skipped} ignorés — équipes hors dataset)")

    # Répartition par type
    if not leagues.empty or league_names:
        from collections import Counter
        cw_counts: Counter = Counter()
        for _, r in rec.iterrows():
            cw = round(float(r.comp_weight), 2)
            cw_counts[cw] += 1
        for cw, cnt in sorted(cw_counts.items(), reverse=True):
            label = {1.0: "WC/tournoi final", 0.85: "tournoi continental",
                     0.70: "qualification", 0.60: "compétition intl",
                     0.50: "nations league", 0.35: "amical"}.get(cw, f"w={cw}")
            print(f"  {label:<25s}: {cnt:4d} matchs")

    goals   = np.array([g for g, _, _ in rows], dtype=float)
    diffs   = np.array([d for _, d, _ in rows], dtype=float)
    weights = np.array([w for _, _, w in rows], dtype=float)

    # MLE pondéré : b (sensibilité rating) et base_lambda
    best_b, best_ll = 0.0, -np.inf
    for b in np.arange(0.0, 0.0081, 0.0002):
        exp_bd = np.exp(b * diffs)
        ea = (weights * goals).sum() / (weights * exp_bd).sum()
        ll = float((weights * (goals * (np.log(ea) + b * diffs) - ea * exp_bd)).sum())
        if ll > best_ll:
            best_b, best_ll = float(b), ll

    exp_bd      = np.exp(best_b * diffs)
    base_lambda = float((weights * goals).sum() / (weights * exp_bd).sum())
    spread      = best_b * (hi - lo)
    avg_goals   = float((weights * goals).sum() / weights.sum())

    # Dixon-Coles rho (pondéré)
    gh = goals[0::2];  ga = goals[1::2]
    dh = diffs[0::2];  wh = weights[0::2]
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
        rll = float((wh * np.log(tau)).sum())
        if rll > best_rll:
            best_rho, best_rll = float(rho), rll

    # Bivariate Poisson lambda3 (pondéré)
    def _bvpois_ll(l3_val: float) -> float:
        total = 0.0
        for i in range(len(gh)):
            lhi = float(lh[i]); lai = float(la[i]); wi = float(wh[i])
            h, a = int(gh[i]), int(ga[i])
            l3 = min(l3_val, min(lhi, lai) * 0.9)
            l1 = max(lhi - l3, 1e-9); l2 = max(lai - l3, 1e-9)
            p = 0.0
            for k in range(min(h, a) + 1):
                bh_ = _math.factorial(h) // (_math.factorial(k) * _math.factorial(h - k))
                ba_ = _math.factorial(a) // (_math.factorial(k) * _math.factorial(a - k))
                p += (bh_ * ba_ * _math.factorial(k)
                      * (l3 / (l1 * l2)) ** k
                      * (_math.exp(-l1) * l1**h / _math.factorial(h))
                      * (_math.exp(-l2) * l2**a / _math.factorial(a))
                      * _math.exp(-l3))
            if p > 0:
                total += wi * _math.log(p)
        return total

    best_l3, best_l3_ll = 0.0, _bvpois_ll(0.0)
    for l3 in [i * 0.01 for i in range(1, 21)]:
        ll = _bvpois_ll(l3)
        if ll > best_l3_ll:
            best_l3_ll = ll; best_l3 = l3

    # Zero-Inflated Poisson pi_zero (pondéré)
    def _zip_ll(pi: float) -> float:
        total = 0.0
        for i in range(len(gh)):
            lhi = float(lh[i]); lai = float(la[i]); wi = float(wh[i])
            h, a = int(gh[i]), int(ga[i])
            l3 = min(best_l3, min(lhi, lai) * 0.9)
            l1 = max(lhi - l3, 1e-9); l2 = max(lai - l3, 1e-9)
            p_base = 0.0
            for k in range(min(h, a) + 1):
                bh_ = _math.factorial(h) // (_math.factorial(k) * _math.factorial(h - k))
                ba_ = _math.factorial(a) // (_math.factorial(k) * _math.factorial(a - k))
                p_base += (bh_ * ba_ * _math.factorial(k)
                           * (l3 / (l1 * l2)) ** k
                           * (_math.exp(-l1) * l1**h / _math.factorial(h))
                           * (_math.exp(-l2) * l2**a / _math.factorial(a))
                           * _math.exp(-l3))
            tau = 1.0
            if   h == 0 and a == 0: tau = 1 - lhi * lai * best_rho
            elif h == 0 and a == 1: tau = 1 + lhi * best_rho
            elif h == 1 and a == 0: tau = 1 + lai * best_rho
            elif h == 1 and a == 1: tau = 1 - best_rho
            p_base *= max(tau, 1e-9)
            p_zip = (pi + (1 - pi) * p_base) if (h == 0 and a == 0) else ((1 - pi) * p_base)
            if p_zip > 0:
                total += wi * _math.log(p_zip)
        return total

    best_pi, best_pi_ll = 0.0, _zip_ll(0.0)
    for pi in [i * 0.005 for i in range(1, 21)]:
        ll = _zip_ll(pi)
        if ll > best_pi_ll:
            best_pi_ll = ll; best_pi = pi

    # ── MLE jointe (b, rho, lambda3, pi_zero) + garde-fou hors echantillon ───
    # Le fit ci-dessus est sequentiel (chaque param fige le suivant), ce qui
    # ignore leurs interactions. On raffine en optimisant les 4 conjointement
    # (Nelder-Mead, warm-start = fit sequentiel). On n'adopte le resultat que
    # s'il bat le sequentiel HORS echantillon (split chronologique).
    meta = [m for m in match_meta]
    mg = np.array([(h, a) for h, a, _, _, _ in meta], dtype=int)
    md = np.array([d for _, _, d, _, _ in meta], dtype=float)
    mw = np.array([w for _, _, _, w, _ in meta], dtype=float)
    mage = np.array([ag for _, _, _, _, ag in meta], dtype=float)

    from math import comb as _comb

    def _match_logprob(h, a, lhi, lai, rho, l3, pi):
        l3c = min(l3, min(lhi, lai) * 0.9) if l3 > 0 else 0.0
        l1 = max(lhi - l3c, 1e-9); l2 = max(lai - l3c, 1e-9)
        p = 0.0
        for k in range(min(h, a) + 1):
            p += (_comb(h, k) * _comb(a, k) * _math.factorial(k)
                  * (l3c / (l1 * l2)) ** k
                  * (_math.exp(-l1) * l1**h / _math.factorial(h))
                  * (_math.exp(-l2) * l2**a / _math.factorial(a))
                  * _math.exp(-l3c))
        if   h == 0 and a == 0: tau = 1 - lhi * lai * rho
        elif h == 0 and a == 1: tau = 1 + lhi * rho
        elif h == 1 and a == 0: tau = 1 + lai * rho
        elif h == 1 and a == 1: tau = 1 - rho
        else:                   tau = 1.0
        p *= max(tau, 1e-12)
        p = (pi + (1 - pi) * p) if (h == 0 and a == 0) else (1 - pi) * p
        return _math.log(max(p, 1e-15))

    def _fit_joint(idx, x0):
        """Optimise (b,rho,l3,pi) sur les matchs idx (ponderes). Renvoie le tuple."""
        from scipy.optimize import minimize
        g, d, w = mg[idx], md[idx], mw[idx]

        def nll(x):
            b, rho, l3, pi = x
            if not (0 <= b <= 0.02 and -0.3 <= rho <= 0.3 and 0 <= l3 <= 0.4 and 0 <= pi <= 0.3):
                return 1e9
            base = (w * (g[:, 0] + g[:, 1]) / 2).sum() / (w * (np.exp(b * d) + np.exp(-b * d)) / 2).sum()
            tot = 0.0
            for i in range(len(g)):
                lhi = base * _math.exp(b * d[i]); lai = base * _math.exp(-b * d[i])
                tot += w[i] * _match_logprob(int(g[i, 0]), int(g[i, 1]), lhi, lai, rho, l3, pi)
            return -tot
        res = minimize(nll, x0, method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-3})
        return tuple(res.x)

    def _holdout_logprob(params, idx):
        """Log-vraisemblance moyenne NON ponderee sur les matchs idx."""
        b, rho, l3, pi = params
        g, d = mg[idx], md[idx]
        base = (g.sum(axis=1).mean()) / (np.exp(b * d).mean() + np.exp(-b * d).mean()) * 2
        tot = 0.0
        for i in range(len(g)):
            lhi = base * _math.exp(b * d[i]); lai = base * _math.exp(-b * d[i])
            tot += _match_logprob(int(g[i, 0]), int(g[i, 1]), lhi, lai, rho, l3, pi)
        return tot / len(g)

    seq_params = (best_b, best_rho, best_l3, best_pi)
    all_idx = np.arange(len(mg))
    # split chronologique : valid = derniere annee (age <= 365), train = avant
    val_idx = np.where(mage <= 365)[0]
    tr_idx = np.where(mage > 365)[0]
    joint_adopted = False
    joint_params = seq_params
    if len(val_idx) >= 100 and len(tr_idx) >= 200:
        # Test conservateur : le sequentiel (fit sur tout, leger avantage) est
        # compare au joint fit sur train uniquement. On n'adopte le joint que
        # s'il gagne malgre ce handicap.
        joint_tr = _fit_joint(tr_idx, list(seq_params))
        ll_seq = _holdout_logprob(seq_params, val_idx)
        ll_joint = _holdout_logprob(joint_tr, val_idx)
        print(f"OOS (valid={len(val_idx)} matchs) logprob/match : "
              f"sequentiel={ll_seq:.5f}  joint={ll_joint:.5f} -> "
              f"{'JOINT adopte' if ll_joint > ll_seq else 'sequentiel conserve'}")
        if ll_joint > ll_seq:
            joint_params = _fit_joint(all_idx, list(seq_params))  # refit sur tout
            joint_adopted = True
    else:
        print(f"OOS : echantillon valid insuffisant ({len(val_idx)}) -> sequentiel conserve")

    if joint_adopted:
        best_b, best_rho, best_l3, best_pi = joint_params
        base_lambda = float((weights * goals).sum() / (weights * np.exp(best_b * diffs)).sum())
        spread = best_b * (hi - lo)

    print(f"base_lambda={base_lambda:.3f}, b={best_b:.4f}/pt Glicko, "
          f"spread={spread:.3f}/force, rho={best_rho:.3f}, "
          f"lambda3={best_l3:.3f} (BVP), pi_zero={best_pi:.4f} (ZIP), "
          f"avg_goals/équipe={avg_goals:.3f} [{'joint' if joint_adopted else 'sequentiel'}]")

    out = {
        "source": "eatpizzanot/soccer-dataset (Glicko-2) — tous matchs intl 2019+",
        "calibration_window": "2019-01-01 à 2026-06-01",
        "n_matches_calibration": n_matches,
        "avg_goals_per_team": round(avg_goals, 3),
        "base_lambda": round(base_lambda, 4),
        "spread": round(spread, 4),
        "rho": round(best_rho, 4),
        "lambda3": round(best_l3, 4),
        "pi_zero": round(best_pi, 4),
        "ratings_mu": {t: round(m, 1) for t, m in sorted(mus.items(), key=lambda x: -x[1])},
        "strengths": dict(sorted(strengths.items(), key=lambda x: -x[1])),
    }
    (DATA / "strengths_dataset.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {len(strengths)}/48 equipes -> data/strengths_dataset.json")


if __name__ == "__main__":
    main()
