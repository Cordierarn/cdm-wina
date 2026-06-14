"""Tests unitaires du moteur de pricing (le coeur financier de l'outil).

Lancer : python -m pytest test_pricing.py -q   (ou : python test_pricing.py)

Couvre : conservation de la masse de probas, soft_cap_lambda, dc_tau,
price_bet sur chaque type de marche, blend_with_market, pinnacle_probs.
Un bug silencieux dans price_bet fausse tous les EV en aval sans crasher :
ces tests sont la garde-fou.
"""
from __future__ import annotations

import math

import pricing as P


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ── score_matrix : conservation de la masse ──────────────────────────────
def test_score_matrix_sums_to_one():
    for lh, la, rho, pi, l3 in [(1.4, 1.1, 0.0, 0.0, 0.0),
                                (1.4, 1.1, -0.105, 0.0, 0.0),
                                (2.0, 0.8, -0.1, 0.05, 0.1),
                                (0.5, 0.5, 0.0, 0.08, 0.0)]:
        mat = P.score_matrix(lh, la, rho=rho, pi_zero=pi, lambda3=l3)
        total = sum(sum(row) for row in mat)
        assert approx(total, 1.0, 1e-6), f"masse={total} pour {(lh,la,rho,pi,l3)}"
        assert all(v >= 0 for row in mat for v in row), "proba negative"


def test_score_matrix_higher_lambda_more_goals():
    weak = P.score_matrix(0.8, 0.8)
    strong = P.score_matrix(2.5, 2.5)
    p_weak_00 = weak[0][0]
    p_strong_00 = strong[0][0]
    assert p_strong_00 < p_weak_00, "plus de buts attendus -> moins de 0-0"


# ── soft_cap_lambda ──────────────────────────────────────────────────────
def test_soft_cap_identity_below_cap():
    assert approx(P.soft_cap_lambda(2.0, cap=3.5), 2.0)
    assert approx(P.soft_cap_lambda(3.5, cap=3.5), 3.5)


def test_soft_cap_compresses_above_cap():
    capped = P.soft_cap_lambda(8.0, cap=3.5)
    assert 3.5 < capped < 8.0, "doit comprimer sans depasser la valeur brute"
    # monotone croissant
    assert P.soft_cap_lambda(10.0) > P.soft_cap_lambda(8.0)


def test_soft_cap_zero():
    assert P.soft_cap_lambda(0.0) == 0.0


# ── dc_tau (Dixon-Coles) ─────────────────────────────────────────────────
def test_dc_tau_boosts_low_scores_with_negative_rho():
    rho = -0.1
    lh, la = 1.4, 1.2
    # rho negatif -> tau > 1 sur 0-0 et 1-1 (boost des nuls bas)
    assert P.dc_tau(0, 0, lh, la, rho) > 1.0
    assert P.dc_tau(1, 1, lh, la, rho) > 1.0
    # tau = 1 hors des 4 scores corriges
    assert approx(P.dc_tau(2, 1, lh, la, rho), 1.0)
    assert approx(P.dc_tau(3, 2, lh, la, rho), 1.0)


# ── price_bet : 1N2 ──────────────────────────────────────────────────────
def _mat():
    return P.score_matrix(1.6, 1.0, rho=-0.1)


def test_price_resultat_sums_to_one():
    issues = [{"label": "1"}, {"label": "N"}, {"label": "2"}]
    probs = P.price_bet("Résultat", issues, _mat(), "Home", "Away")
    assert probs is not None and len(probs) == 3
    s = sum(p for p, _ in probs)
    assert approx(s, 1.0, 1e-6)


def test_price_double_chance_complementary():
    issues = [{"label": "1N"}, {"label": "12"}, {"label": "N2"}]
    probs = P.price_bet("Double chance", issues, _mat(), "Home", "Away")
    # les 3 doubles chances somment a 2 (chaque issue 1N2 comptee 2 fois)
    s = sum(p for p, _ in probs)
    assert approx(s, 2.0, 1e-6)


def test_price_btts():
    issues = [{"label": "Oui"}, {"label": "Non"}]
    probs = P.price_bet("Les 2 équipes marquent", issues, _mat(), "Home", "Away")
    assert approx(sum(p for p, _ in probs), 1.0, 1e-6)


# ── price_bet : totaux + push sur ligne entiere ──────────────────────────
def test_price_totals_half_line_no_push():
    issues = [{"label": "Plus de 2,5"}, {"label": "Moins de 2,5"}]
    probs = P.price_bet("Nombre de buts", issues, _mat(), "Home", "Away")
    assert approx(sum(p for p, _ in probs), 1.0, 1e-6)
    assert all(push == 0.0 for _, push in probs), "demi-ligne : pas de push"


def test_price_totals_whole_line_has_push():
    issues = [{"label": "Plus de 2"}, {"label": "Moins de 2"}]
    probs = P.price_bet("Nombre de buts", issues, _mat(), "Home", "Away")
    # ligne entiere : push = P(total == 2), over+under+push = 1
    over, push_o = probs[0]
    under, push_u = probs[1]
    assert push_o > 0.0 and approx(push_o, push_u)
    assert approx(over + under + push_o, 1.0, 1e-6)


def test_price_team_totals():
    issues = [{"label": "Plus de 1,5"}, {"label": "Moins de 1,5"}]
    probs = P.price_bet("Nombre de buts de Home", issues, _mat(), "Home", "Away")
    assert probs is not None
    assert approx(sum(p for p, _ in probs), 1.0, 1e-6)


# ── price_bet : score exact (issue "Autre") ──────────────────────────────
def test_price_score_exact_with_autre():
    issues = [{"label": "1-0"}, {"label": "1-1"}, {"label": "Autre"}]
    probs = P.price_bet("Score exact", issues, _mat(), "Home", "Away")
    assert probs is not None and len(probs) == 3
    # "Autre" = 1 - somme des scores listes ; total = 1
    assert approx(sum(p for p, _ in probs), 1.0, 1e-6)
    assert probs[2][0] >= 0.0


# ── price_bet : handicap ─────────────────────────────────────────────────
def test_price_handicap():
    issues = [{"label": "Home (-1)"}, {"label": "Away (+1)"}]
    probs = P.price_bet("Écart de buts (handicap)", issues, _mat(), "Home", "Away")
    assert probs is not None and len(probs) == 2


def test_price_unknown_market_returns_none():
    issues = [{"label": "x"}]
    assert P.price_bet("Marché inconnu", issues, _mat(), "Home", "Away") is None


# ── blend_with_market : conserve la masse du modele ──────────────────────
def test_blend_conserves_model_mass():
    model_probs = [(0.5, 0.0), (0.3, 0.0), (0.2, 0.0)]
    issues = [{"cote": 2.0}, {"cote": 3.5}, {"cote": 4.0}]
    blended = P.blend_with_market(model_probs, issues, 0.4)
    s_model = sum(p for p, _ in model_probs)
    s_blend = sum(p for p, _ in blended)
    assert approx(s_model, s_blend, 1e-9), "le blend doit conserver la masse"


def test_blend_weight_extremes():
    model_probs = [(0.6, 0.0), (0.4, 0.0)]
    issues = [{"cote": 1.5}, {"cote": 2.5}]
    # w=1 -> 100% modele
    full_model = P.blend_with_market(model_probs, issues, 1.0)
    assert approx(full_model[0][0], 0.6, 1e-9)
    # w=0 -> 100% marche (de-marge), conserve la masse
    full_market = P.blend_with_market(model_probs, issues, 0.0)
    assert approx(sum(p for p, _ in full_market), 1.0, 1e-9)


# ── pinnacle_probs ───────────────────────────────────────────────────────
def test_pinnacle_probs_resultat():
    pin = {"fair_p1": 0.5, "fair_pX": 0.3, "fair_p2": 0.2}
    issues = [1, 2, 3]
    out = P.pinnacle_probs("resultat", issues, pin, flipped=False)
    assert out == [0.5, 0.3, 0.2]
    flipped = P.pinnacle_probs("resultat", issues, pin, flipped=True)
    assert flipped == [0.2, 0.3, 0.5], "flip echange domicile/exterieur"


def test_pinnacle_probs_double_chance():
    pin = {"fair_p1": 0.5, "fair_pX": 0.3, "fair_p2": 0.2}
    out = P.pinnacle_probs("double chance", [1, 2, 3], pin, flipped=False)
    assert approx(sum(out), 2.0, 1e-9)


# ── devig : Shin vs multiplicatif ────────────────────────────────────────
def test_multiplicative_devig_sums_to_one():
    p = P.multiplicative_devig([2.5, 2.88, 3.75])
    assert p is not None and approx(sum(p), 1.0, 1e-9)


def test_shin_devig_sums_to_one():
    p = P.shin_devig([2.5, 2.88, 3.75])
    assert p is not None and approx(sum(p), 1.0, 1e-9)


def test_shin_invalid_odds():
    assert P.shin_devig([0.0, 2.0]) is None
    assert P.shin_devig([1.0, 2.0]) is None  # cote 1.0 = pas de gain


def test_shin_vs_multiplicative_favorite_longshot():
    # Shin retire plus de marge des outsiders : favori ↑, outsider ↓ vs proportionnel
    odds = [1.5, 4.5, 8.0]
    m = P.multiplicative_devig(odds)
    s = P.shin_devig(odds)
    fav = odds.index(min(odds))
    dog = odds.index(max(odds))
    assert s[fav] > m[fav], "Shin doit relever la proba du favori"
    assert s[dog] < m[dog], "Shin doit baisser la proba de l'outsider"


def test_shin_no_overround_normalizes():
    # cotes 'justes' (booksum ~1) -> Shin ~ normalisation simple
    p = P.shin_devig([2.0, 2.0])
    assert approx(sum(p), 1.0, 1e-9) and approx(p[0], 0.5, 1e-6)


# ── runner sans pytest ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    fns = [g for n, g in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  OK  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} tests OK")
    sys.exit(1 if failed else 0)
