"""Bracket officiel CDM 2026 + simulation Monte Carlo -> bracket.json, tournament_sim.json.

Usage : python make_tournament.py

Structure officielle des phases finales (Wikipedia, 2026 FIFA World Cup knockout stage) :
- 1/16 (matchs 73-88) : slots fixes 1ers/2es + 8 meilleurs 3es sur slots contraints
- 1/8 (89-96), quarts (97-100), demis (101-102), finale (104)
- Classement des 3es : points > diff > buts > (conduite/FIFA ranking ~ approxime par force)
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

from pricing import score_matrix, soft_cap_lambda

DATA = Path(__file__).parent / "data"
N_SIMS = 50000

# ── Bracket officiel 1/16 ──────────────────────────────────────────────
# (match, slot_home, slot_away) — "1A"=1er groupe A, "2B"=2e, "3:ABCDF"=3e parmi ces groupes
R32 = [
    (73, "2A", "2B"),
    (74, "1E", "3:ABCDF"),
    (75, "1F", "2C"),
    (76, "1C", "2F"),
    (77, "1I", "3:CDFGH"),
    (78, "2E", "2I"),
    (79, "1A", "3:CEFHI"),
    (80, "1L", "3:EHIJK"),
    (81, "1D", "3:BEFIJ"),
    (82, "1G", "3:AEHIJ"),
    (83, "2K", "2L"),
    (84, "1H", "2J"),
    (85, "1B", "3:EFGIJ"),
    (86, "1J", "2H"),
    (87, "1K", "3:DEIJL"),
    (88, "2D", "2G"),
]
R16 = [(89, 74, 77), (90, 73, 75), (91, 76, 78), (92, 79, 80),
       (93, 83, 84), (94, 81, 82), (95, 86, 88), (96, 85, 87)]
QF = [(97, 89, 90), (98, 93, 94), (99, 91, 92), (100, 95, 96)]
SF = [(101, 97, 98), (102, 99, 100)]


def load_model():
    model = json.loads((DATA / "model.json").read_text(encoding="utf-8"))
    sd = json.loads((DATA / "strengths_dataset.json").read_text(encoding="utf-8"))
    return model, sd


def load_played() -> dict[frozenset, dict[str, int]]:
    """Resultats de groupe deja joues -> {frozenset(home,away): {team: buts}}."""
    path = DATA / "cdm_results.json"
    if not path.exists():
        return {}
    played = {}
    for r in json.loads(path.read_text(encoding="utf-8")):
        played[frozenset((r["home"], r["away"]))] = {
            r["home"]: r["goals_home"], r["away"]: r["goals_away"]}
    return played


def played_goals(played: dict, h: str, a: str):
    """Buts reels orientes (buts_h, buts_a) si la paire a ete jouee, sinon None."""
    g = played.get(frozenset((h, a)))
    if g is None or h not in g or a not in g:
        return None
    return g[h], g[a]


class Pricer:
    """Probas de match avec cache, depuis les forces calibrees."""

    def __init__(self, sd, model):
        self.strengths = sd["strengths"]
        self.base_lambda = sd["base_lambda"]
        self.spread = sd["spread"]
        self.rho = sd.get("rho", 0.0)
        self.pi_zero = sd.get("pi_zero", 0.0)
        self.lambda3 = sd.get("lambda3", 0.0)
        self.cache: dict[tuple[str, str], dict] = {}

    def match(self, home: str, away: str) -> dict:
        key = (home, away)
        if key in self.cache:
            return self.cache[key]
        sh = self.strengths.get(home, 0.3)
        sa = self.strengths.get(away, 0.3)
        lh = soft_cap_lambda(self.base_lambda * math.exp(self.spread * (sh - sa)))
        la = soft_cap_lambda(self.base_lambda * math.exp(self.spread * (sa - sh)))
        mat = score_matrix(lh, la, self.rho, pi_zero=self.pi_zero, lambda3=self.lambda3)
        p1 = sum(mat[h][a] for h in range(11) for a in range(11) if h > a)
        px = sum(mat[h][a] for h in range(11) for a in range(11) if h == a)
        eg_h = sum(h * mat[h][a] for h in range(11) for a in range(11))
        eg_a = sum(a * mat[h][a] for h in range(11) for a in range(11))
        top = max(((mat[h][a], h, a) for h in range(11) for a in range(11)))
        out = {"home": home, "away": away, "p1": round(p1, 3), "px": round(px, 3),
               "p2": round(1 - p1 - px, 3), "lh": round(lh, 2), "la": round(la, 2),
               "eg_h": eg_h, "eg_a": eg_a,
               "score": f"{top[1]}-{top[2]}", "score_p": round(top[0], 3),
               "winner": home if p1 >= 1 - p1 - px else away}
        self.cache[key] = out
        return out


def assign_thirds(third_groups: list[str]) -> dict[int, int] | None:
    """Matche les 8 groupes des 3es qualifies aux slots contraints (backtracking).

    Renvoie {match_id: index dans third_groups}.
    """
    slots = [(m, set(spec.split(":")[1])) for m, _, spec in R32 if spec.startswith("3:")]
    # Les slots les plus contraints d'abord
    order = sorted(range(len(slots)),
                   key=lambda i: len(slots[i][1] & set(third_groups)))
    assign: dict[int, int] = {}
    used: set[int] = set()

    def bt(k: int) -> bool:
        if k == len(order):
            return True
        m, allowed = slots[order[k]]
        for idx, g in enumerate(third_groups):
            if idx not in used and g in allowed:
                assign[m] = idx
                used.add(idx)
                if bt(k + 1):
                    return True
                used.discard(idx)
                del assign[m]
        return False

    return assign if bt(0) else None


def rank_thirds(entries: list[tuple]) -> list[tuple]:
    """entries: (pts, diff, gf, strength, team, group). Criteres officiels :
    points > diff > buts > conduite/FIFA ranking (approxime par force)."""
    return sorted(entries, key=lambda x: (-x[0], -x[1], -x[2], -x[3]))


# ── Bracket deterministe (favori gagne toujours) ───────────────────────

def expected_group_ranking(pricer: Pricer, teams: list[str], played: dict) -> list[str]:
    ep = {t: 0.0 for t in teams}
    gd = {t: 0.0 for t in teams}
    for i, h in enumerate(teams):
        for a in teams[i + 1:]:
            pg = played_goals(played, h, a)
            if pg is not None:
                vh, va = pg
                ep[h] += 3 if vh > va else 1 if vh == va else 0
                ep[a] += 3 if va > vh else 1 if vh == va else 0
                gd[h] += vh - va
                gd[a] += va - vh
            else:
                m = pricer.match(h, a)
                ep[h] += 3 * m["p1"] + m["px"]
                ep[a] += 3 * m["p2"] + m["px"]
                gd[h] += m["eg_h"] - m["eg_a"]
                gd[a] += m["eg_a"] - m["eg_h"]
    return sorted(teams, key=lambda t: (-ep[t], -gd[t]))


def deterministic_bracket(pricer: Pricer, groups: dict, played: dict) -> dict:
    rankings = {g: expected_group_ranking(pricer, ts, played) for g, ts in groups.items()}

    # 3es attendus, classes par points esperes (approx force)
    thirds = []
    for g, r in rankings.items():
        t = r[2]
        # points esperes du 3e comme proxy
        ep = 0.0
        for opp in groups[g]:
            if opp == t:
                continue
            pg = played_goals(played, t, opp)
            if pg is not None:
                vh, va = pg
                ep += 3 if vh > va else 1 if vh == va else 0
            else:
                m = pricer.match(t, opp)
                ep += 3 * m["p1"] + m["px"]
        thirds.append((ep, 0.0, 0.0, pricer.strengths.get(t, 0.3), t, g))
    ranked = rank_thirds(thirds)
    best8 = ranked[:8]
    third_groups = [e[5] for e in best8]
    third_teams = {e[5]: e[4] for e in best8}
    assign = assign_thirds(third_groups)
    if assign is None:  # combinaison sans matching parfait : fallback gloutonne
        assign = {}
        remaining = list(range(len(third_groups)))
        for m, _, spec in R32:
            if not spec.startswith("3:"):
                continue
            allowed = set(spec.split(":")[1])
            pick = next((i for i in remaining if third_groups[i] in allowed), remaining[0])
            assign[m] = pick
            remaining.remove(pick)

    def resolve(slot: str, mid: int) -> str:
        if slot.startswith("3:"):
            return third_teams[third_groups[assign[mid]]]
        pos, g = int(slot[0]), slot[1]
        return rankings[g][pos - 1]

    results: dict[int, dict] = {}
    winners: dict[int, str] = {}
    r32_out, r16_out, qf_out, sf_out = [], [], [], []

    for mid, sh, sa in R32:
        m = pricer.match(resolve(sh, mid), resolve(sa, mid))
        m = {**m, "match_id": mid, "slot": f"{sh} vs {sa}"}
        results[mid] = m
        winners[mid] = m["winner"]
        r32_out.append(m)
    for mid, a, b in R16:
        m = {**pricer.match(winners[a], winners[b]), "match_id": mid}
        winners[mid] = m["winner"]
        r16_out.append(m)
    for mid, a, b in QF:
        m = {**pricer.match(winners[a], winners[b]), "match_id": mid}
        winners[mid] = m["winner"]
        qf_out.append(m)
    for mid, a, b in SF:
        m = {**pricer.match(winners[a], winners[b]), "match_id": mid}
        winners[mid] = m["winner"]
        sf_out.append(m)
    final = {**pricer.match(winners[101], winners[102]), "match_id": 104}

    bracket = {
        "format": "officiel FIFA 2026 (matchs 73-104, Wikipedia knockout stage)",
        "groups": {}, "best8_thirds": [third_teams[g] for g in third_groups],
        "r16": r32_out, "r8": r16_out, "r4": qf_out, "r2": sf_out,
        "final": final, "champion": final["winner"],
    }
    for g, ts in groups.items():
        matches = []
        for i, h in enumerate(ts):
            for a in ts[i + 1:]:
                matches.append(pricer.match(h, a))
        bracket["groups"][g] = {"ranking": rankings[g], "matches": matches}
    return bracket


# ── Monte Carlo avec bracket officiel ──────────────────────────────────

def run_monte_carlo(pricer: Pricer, groups: dict, n: int, played: dict) -> dict:
    wins = defaultdict(int)
    finals = defaultdict(int)
    semis = defaultdict(int)
    quarters = defaultdict(int)
    r16c = defaultdict(int)
    qualified = defaultdict(int)
    pos_cnt = {1: defaultdict(int), 2: defaultdict(int), 3: defaultdict(int), 4: defaultdict(int)}

    def ko(h, a):
        m = pricer.match(h, a)
        r = random.random()
        if r < m["p1"]:
            return h
        if r < m["p1"] + m["px"]:
            return random.choice([h, a])
        return a

    grps = list(groups)
    for _ in range(n):
        rankings = {}
        thirds = []
        for g in grps:
            ts = groups[g]
            pts = {t: 0 for t in ts}
            gf = {t: 0.0 for t in ts}
            ga = {t: 0.0 for t in ts}
            for i, h in enumerate(ts):
                for a in ts[i + 1:]:
                    pg = played_goals(played, h, a)
                    if pg is not None:
                        vh, va = pg
                        gf[h] += vh; ga[h] += va
                        gf[a] += va; ga[a] += vh
                        if vh > va:
                            pts[h] += 3
                        elif vh == va:
                            pts[h] += 1; pts[a] += 1
                        else:
                            pts[a] += 3
                        continue
                    m = pricer.match(h, a)
                    r = random.random()
                    gf[h] += m["eg_h"]; ga[h] += m["eg_a"]
                    gf[a] += m["eg_a"]; ga[a] += m["eg_h"]
                    if r < m["p1"]:
                        pts[h] += 3
                    elif r < m["p1"] + m["px"]:
                        pts[h] += 1; pts[a] += 1
                    else:
                        pts[a] += 3
            rank = sorted(ts, key=lambda t: (-pts[t], -(gf[t] - ga[t]), -gf[t], random.random()))
            rankings[g] = rank
            for p, t in enumerate(rank):
                pos_cnt[p + 1][t] += 1
            qualified[rank[0]] += 1
            qualified[rank[1]] += 1
            t3 = rank[2]
            thirds.append((pts[t3], gf[t3] - ga[t3], gf[t3],
                           pricer.strengths.get(t3, 0.3), t3, g))

        best8 = rank_thirds(thirds)[:8]
        third_groups = [e[5] for e in best8]
        third_teams = {e[5]: e[4] for e in best8}
        for e in best8:
            qualified[e[4]] += 1
        assign = assign_thirds(third_groups)
        if assign is None:
            assign = {}
            remaining = list(range(8))
            for mid, _, spec in R32:
                if not spec.startswith("3:"):
                    continue
                allowed = set(spec.split(":")[1])
                pick = next((i for i in remaining if third_groups[i] in allowed), remaining[0])
                assign[mid] = pick
                remaining.remove(pick)

        def resolve(slot, mid):
            if slot.startswith("3:"):
                return third_teams[third_groups[assign[mid]]]
            return rankings[slot[1]][int(slot[0]) - 1]

        winners = {}
        for mid, sh, sa in R32:
            winners[mid] = ko(resolve(sh, mid), resolve(sa, mid))
        for w in (winners[m] for m, _, _ in R32):
            r16c[w] += 1          # atteint les 1/8
        for mid, x, y in R16:
            winners[mid] = ko(winners[x], winners[y])
        for w in (winners[m] for m, _, _ in R16):
            quarters[w] += 1      # atteint les quarts
        for mid, x, y in QF:
            winners[mid] = ko(winners[x], winners[y])
        for w in (winners[m] for m, _, _ in QF):
            semis[w] += 1         # atteint les demis
        for mid, x, y in SF:
            winners[mid] = ko(winners[x], winners[y])
        finals[winners[101]] += 1
        finals[winners[102]] += 1
        champ = ko(winners[101], winners[102])
        wins[champ] += 1

    all_teams = [t for ts in groups.values() for t in ts]
    out = {"n_simulations": n, "bracket": "officiel FIFA 2026", "teams": [], "groups_detail": {}}
    for t in all_teams:
        out["teams"].append({
            "name": t,
            "p_win": round(wins[t] / n, 4),
            "p_final": round(finals[t] / n, 4),
            "p_semi": round(semis[t] / n, 4),
            "p_quarter": round(quarters[t] / n, 4),
            "p_r16": round(r16c[t] / n, 4),
            "p_qualified": round(qualified[t] / n, 4),
            "p_1st": round(pos_cnt[1][t] / n, 4),
            "p_2nd": round(pos_cnt[2][t] / n, 4),
            "p_3rd": round(pos_cnt[3][t] / n, 4),
            "p_4th": round(pos_cnt[4][t] / n, 4),
            "strength": round(pricer.strengths.get(t, 0.3), 4),
            "group": next(g for g, ts in groups.items() if t in ts),
        })
    out["teams"].sort(key=lambda x: -x["p_win"])
    for g, ts in groups.items():
        out["groups_detail"][g] = sorted(
            ({"name": t, "p_1st": round(pos_cnt[1][t] / n, 4),
              "p_2nd": round(pos_cnt[2][t] / n, 4), "p_3rd": round(pos_cnt[3][t] / n, 4),
              "p_4th": round(pos_cnt[4][t] / n, 4),
              "p_qualified": round(qualified[t] / n, 4),
              "strength": round(pricer.strengths.get(t, 0.3), 4)} for t in ts),
            key=lambda x: -x["p_1st"])
    return out


def main() -> None:
    model, sd = load_model()
    groups = model["groups"]
    pricer = Pricer(sd, model)
    played = load_played()
    n_group_played = sum(
        1 for ts in groups.values() for i, h in enumerate(ts)
        for a in ts[i + 1:] if played_goals(played, h, a) is not None)
    print(f"Resultats de groupe figes : {n_group_played}/72")

    t0 = time.time()
    bracket = deterministic_bracket(pricer, groups, played)
    (DATA / "bracket.json").write_text(
        json.dumps(bracket, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Bracket deterministe (officiel) : champion = {bracket['champion']}")
    print(f"  Finale : {bracket['final']['home']} vs {bracket['final']['away']} "
          f"{bracket['final']['score']}")

    sim = run_monte_carlo(pricer, groups, N_SIMS, played)
    (DATA / "tournament_sim.json").write_text(
        json.dumps(sim, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Monte Carlo {N_SIMS} sims en {time.time() - t0:.1f}s")
    for t in sim["teams"][:10]:
        print(f"  {t['name']:22s} W {t['p_win']*100:5.1f}%  F {t['p_final']*100:5.1f}%  "
              f"1/2 {t['p_semi']*100:5.1f}%")


if __name__ == "__main__":
    main()
