"""
Génère la MPP_LIST pour le dashboard à partir de bracket.json.

Améliorations (v2):
- Temperature calibration T=1.40 (fitted on 20 CDM matchday-1 results)
- Asymmetric tournament boost: ×1.3942 home goals, ×0.7579 away goals
- Dixon-Coles low-score correction (tau)
- Confidence badge: P(top score boosted) vs seuil calibré
"""

import json
import math
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paramètres calibrés
# ---------------------------------------------------------------------------
CAL_FILE = Path("data/calibration.json")
if CAL_FILE.exists():
    _cal = json.loads(CAL_FILE.read_text())
else:
    _cal = {}

TEMP       = _cal.get("temperature", 1.0)
BOOST_H    = _cal.get("tournament_boost_home", 1.13)
BOOST_A    = _cal.get("tournament_boost_away", 1.13)
CMP_NU     = _cal.get("cmp_nu", 1.0)
RHO        = -0.0113     # Dixon-Coles low-score correction
N_SCORES   = 5          # nombre de scores affichés
MAX_GOALS  = 10         # troncature sommation Poisson

# ---------------------------------------------------------------------------
# Traductions
# ---------------------------------------------------------------------------
FR = {
    "Mexico": "Mexique", "South Africa": "Afrique du Sud", "South Korea": "Corée du Sud",
    "Czech Republic": "Rép. Tchèque", "Canada": "Canada",
    "Bosnia and Herzegovina": "Bosnie-Herz.", "Qatar": "Qatar", "Switzerland": "Suisse",
    "Brazil": "Brésil", "Morocco": "Maroc", "Haiti": "Haïti", "Scotland": "Écosse",
    "United States": "États-Unis", "Paraguay": "Paraguay", "Australia": "Australie",
    "Turkey": "Turquie", "Germany": "Allemagne", "Curacao": "Curaçao",
    "Ivory Coast": "Côte d'Ivoire", "Ecuador": "Équateur", "Netherlands": "Pays-Bas",
    "Japan": "Japon", "Sweden": "Suède", "Tunisia": "Tunisie", "Spain": "Espagne",
    "Cape Verde": "Cap-Vert", "Belgium": "Belgique", "Egypt": "Égypte",
    "Saudi Arabia": "Arabie S.", "Uruguay": "Uruguay", "France": "France",
    "Senegal": "Sénégal", "Iraq": "Irak", "Norway": "Norvège",
    "Argentina": "Argentine", "Algeria": "Algérie", "Austria": "Autriche",
    "Jordan": "Jordanie", "Iran": "Iran", "New Zealand": "Nv.-Zélande",
    "England": "Angleterre", "Croatia": "Croatie", "Colombia": "Colombie",
    "Ghana": "Ghana", "Panama": "Panama", "DR Congo": "RD Congo",
    "Portugal": "Portugal", "Uzbekistan": "Ouzbékistan",
}

FLAGS = {
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Czech Republic": "🇨🇿",
    "Canada": "🇨🇦", "Bosnia and Herzegovina": "🇧🇦", "Qatar": "🇶🇦", "Switzerland": "🇨🇭",
    "Brazil": "🇧🇷", "Morocco": "🇲🇦", "Haiti": "🇭🇹", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "United States": "🇺🇸", "Paraguay": "🇵🇾", "Australia": "🇦🇺", "Turkey": "🇹🇷",
    "Germany": "🇩🇪", "Curacao": "🇨🇼", "Ivory Coast": "🇨🇮", "Ecuador": "🇪🇨",
    "Netherlands": "🇳🇱", "Japan": "🇯🇵", "Sweden": "🇸🇪", "Tunisia": "🇹🇳",
    "Spain": "🇪🇸", "Cape Verde": "🇨🇻", "Belgium": "🇧🇪", "Egypt": "🇪🇬",
    "Saudi Arabia": "🇸🇦", "Uruguay": "🇺🇾", "France": "🇫🇷", "Senegal": "🇸🇳",
    "Iraq": "🇮🇶", "Norway": "🇳🇴", "Argentina": "🇦🇷", "Algeria": "🇩🇿",
    "Austria": "🇦🇹", "Jordan": "🇯🇴", "Iran": "🇮🇷", "New Zealand": "🇳🇿",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Croatia": "🇭🇷", "Colombia": "🇨🇴", "Ghana": "🇬🇭",
    "Panama": "🇵🇦", "DR Congo": "🇨🇩", "Portugal": "🇵🇹", "Uzbekistan": "🇺🇿",
}

# ---------------------------------------------------------------------------
# Kickoff schedule — chargé depuis data/schedule.json (72 matchs, mis à jour
# automatiquement par fetch_schedule() ou l'ESPN scraper).
# ---------------------------------------------------------------------------
def _load_kickoffs() -> dict[tuple[str, str], int]:
    path = Path("data/schedule.json")
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        ko = {}
        for m in data:
            if m.get("kickoff"):
                # les 2 orientations : le calendrier MPP peut inverser dom/ext
                ko[(m["home"], m["away"])] = m["kickoff"]
                ko[(m["away"], m["home"])] = m["kickoff"]
        return ko
    except Exception:
        return {}

KICKOFFS = _load_kickoffs()

# ---------------------------------------------------------------------------
# Résultats connus
# ---------------------------------------------------------------------------
PLAYED = {}
results_file = Path("data/cdm_results.json")
if results_file.exists():
    for r in json.loads(results_file.read_text(encoding="utf-8")):
        h, a = r["home"], r["away"]
        gh, ga = r["goals_home"], r["goals_away"]
        # les deux orientations : le calendrier MPP peut inverser domicile/exterieur
        PLAYED[(h, a)] = f"{gh}-{ga}"
        PLAYED[(a, h)] = f"{ga}-{gh}"


# ---------------------------------------------------------------------------
# Modèle de score
# ---------------------------------------------------------------------------

def dc_tau(i, j, lh, la, rho=RHO):
    """Dixon-Coles correction pour les faibles scores."""
    if i == 0 and j == 0:
        return 1 - lh * la * rho
    if i == 0 and j == 1:
        return 1 + lh * rho
    if i == 1 and j == 0:
        return 1 + la * rho
    if i == 1 and j == 1:
        return 1 - rho
    return 1.0


def poisson_pmf(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)


def temperature_scale(probs, T=TEMP):
    """Temperature scaling à T paramètre pour calibration."""
    if T == 1.0:
        return list(probs)
    logs = [math.log(max(p, 1e-15)) / T for p in probs]
    m = max(logs)
    exps = [math.exp(l - m) for l in logs]
    s = sum(exps)
    return [e / s for e in exps]


def top_scores(lh, la, n=N_SCORES, boosted=True):
    """
    Retourne les N meilleurs scores P(i-j) triés par probabilité.
    boosted=True : applique BOOST_H/BOOST_A pour la CDM.
    Utilise Poisson (pas CMP) : CMP dilue trop les probas de score exact,
    rendant les seuils de confiance inopérants (tous "faible"). CMP est
    utilisé pour les probas 1X2 via pricing.py/make_tournament.py.
    """
    bh = lh * BOOST_H if boosted else lh
    ba = la * BOOST_A if boosted else la
    scores = {}
    Z = 0
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = poisson_pmf(i, bh) * poisson_pmf(j, ba) * dc_tau(i, j, bh, ba)
            scores[(i, j)] = p
            Z += p
    # Renormalise
    top = sorted(scores.items(), key=lambda x: -x[1])[:n]
    return [{"score": f"{i}-{j}", "p": round(p / Z * 100, 1)} for (i, j), p in top]


def pick_mean_score(lh, la):
    """Score λ-moyen : arrondi des espérances boostées."""
    return f"{round(lh * BOOST_H)}-{round(la * BOOST_A)}"


def confidence_label(top_p, p1=0.33, p2=0.33):
    """Confiance basée sur la probabilité de victoire du favori.
    Plus discriminant que top_p qui converge vers 10-13% pour tous les matchs CDM.
    """
    fav_p = max(p1, p2)
    if fav_p >= 0.65:
        return "haute"
    if fav_p >= 0.52:
        return "moyenne"
    return "faible"


def pick_strategy(p1, px, p2, lh, la):
    eg_h = lh * BOOST_H
    eg_a = la * BOOST_A
    if p1 > 0.55:
        if eg_h >= 2.0:
            return "Favori clair domicile — viser 2-0 ou 2-1"
        return "Avantage domicile — viser 1-0 ou 2-1"
    if p2 > 0.55:
        if eg_a >= 2.0:
            return "Favori clair extérieur — viser 0-2 ou 1-2"
        return "Avantage extérieur — viser 0-1 ou 1-2"
    if abs(p1 - p2) < 0.08:
        return "Match équilibré — nul très probable"
    if eg_h + eg_a > 3.0:
        return "Match ouvert — beaucoup de buts attendus"
    if eg_h + eg_a < 2.0:
        return "Match fermé — peu de buts attendus"
    return "Avantage modéré — prendre le favori à domicile"


# ---------------------------------------------------------------------------
# Génération MPP_LIST
# ---------------------------------------------------------------------------

def build_mpp_list():
    bracket = json.load(open("data/bracket.json"))
    mpp = []

    for grp_id, gdata in bracket["groups"].items():
        for m in gdata["matches"]:
            h, a = m["home"], m["away"]
            lh, la = m["lh"], m["la"]

            # Probs 1X2 calculées depuis les lambdas boostés CDM
            # (cohérent avec les picks de score — calibrés ensemble sur MD1)
            bh, ba = lh * BOOST_H, la * BOOST_A
            _dc = dc_tau  # alias local
            p1_raw = sum(
                poisson_pmf(i, bh) * poisson_pmf(j, ba) * _dc(i, j, bh, ba, RHO)
                for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if i > j
            )
            px_raw = sum(
                poisson_pmf(i, bh) * poisson_pmf(j, ba) * _dc(i, j, bh, ba, RHO)
                for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1) if i == j
            )
            p2_raw = 1.0 - p1_raw - px_raw

            # Temperature calibration
            p1, px, p2 = temperature_scale([p1_raw, px_raw, p2_raw])

            # Scores boostés (MPP pick)
            scores = top_scores(lh, la, boosted=True)
            scores_raw = top_scores(lh, la, boosted=False)
            top_p = scores[0]["p"] if scores else 0

            entry = {
                "group": grp_id,
                "home": FR.get(h, h),
                "away": FR.get(a, a),
                "flag_h": FLAGS.get(h, "🏳"),
                "flag_a": FLAGS.get(a, "🏳"),
                "eg_h": round(lh * BOOST_H, 2),
                "eg_a": round(la * BOOST_A, 2),
                "p1": round(p1, 3),
                "px": round(px, 3),
                "p2": round(p2, 3),
                "scores": scores,
                "scores_raw": scores_raw,
                "pick_mean": pick_mean_score(lh, la),
                "confidence": confidence_label(top_p, p1, p2),
                "top_p": top_p,
                "strat": pick_strategy(p1, px, p2, lh, la),
            }
            played = PLAYED.get((h, a))
            if played:
                entry["played"] = played
            ko = KICKOFFS.get((h, a))
            if ko:
                entry["kickoff"] = ko

            mpp.append(entry)

    return mpp


def inject_into_dashboard(mpp):
    """Remplace MPP_LIST dans dashboard/index.html."""
    html_path = Path("dashboard/index.html")
    content = html_path.read_text(encoding="utf-8")

    new_data = json.dumps(mpp, ensure_ascii=False, separators=(",", ":"))
    new_line = f"const MPP_LIST = {new_data};"

    # Recherche de la balise de début et fin
    start = content.find("const MPP_LIST = [")
    if start == -1:
        print("ERROR: MPP_LIST not found in dashboard")
        return

    # Trouver la fin — chercher le premier ]; après start
    # On doit trouver le ]; qui termine spécifiquement MPP_LIST
    # On parcourt caractère par caractère en comptant les brackets
    depth = 0
    end = start
    for i, c in enumerate(content[start:]):
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = start + i + 1  # inclus le ]
                # chercher le ; suivant
                j = end
                while j < len(content) and content[j] in " \t\r\n":
                    j += 1
                if content[j] == ";":
                    end = j + 1
                break

    new_content = content[:start] + new_line + content[end:]
    html_path.write_text(new_content, encoding="utf-8")
    print(f"MPP_LIST injectée: {len(mpp)} matchs ({len(new_line)} chars)")


if __name__ == "__main__":
    mpp = build_mpp_list()
    print(f"Généré: {len(mpp)} matchs")

    # Stats confidence
    conf = {"haute": 0, "moyenne": 0, "faible": 0}
    for m in mpp:
        conf[m["confidence"]] += 1
    print(f"Confidence: {conf}")
    print(f"Max top_p: {max(m['top_p'] for m in mpp):.1f}%")

    # Vérification des played
    played_count = sum(1 for m in mpp if "played" in m)
    print(f"Matchs joués: {played_count}")

    inject_into_dashboard(mpp)

    # Sauvegarde optionnelle JSON
    out = Path("data/pronos_by_date.json")
    out.write_text(json.dumps(mpp, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Sauvegardé: {out}")
