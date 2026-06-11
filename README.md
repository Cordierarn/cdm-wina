# ⚽ CDM-Wina — Pronostics Coupe du Monde 2026

Outil local de pronostics pour la Coupe du Monde 2026 (48 équipes, 11 juin – 19 juillet 2026, USA/Canada/Mexique) : un modèle statistique price tous les marchés de paris Winamax, détecte les value bets, et un dashboard web local permet de suivre les matchs, les picks du jour et sa bankroll.

> ⚠️ **Avertissement** : les paris sportifs comportent un risque réel de perte. Cet outil est un projet personnel d'analyse, pas un conseil financier. Jouez de manière responsable — 09 74 75 13 13 (Joueurs Info Service).

---

## Vue d'ensemble

```
┌─────────────────────┐     ┌──────────────────────┐
│  soccer-dataset     │     │  ScoutFootball repo  │
│  (Glicko-2, 378k    │     │  (pipeline analytics │
│   matchs 2012-2026) │     │   + calendrier CDM)  │
└─────────┬───────────┘     └──────────┬───────────┘
          │ build_strengths.py         │ export_model.py
          ▼                            ▼
   strengths_dataset.json  ──────►  model.json
   (forces 48 équipes,              (72 matchs de poules,
    calibration MLE)                 λ Poisson, probas 1N2/O-U/BTTS)
                                       │
┌─────────────────────┐                │
│  Winamax            │                │
│  (PRELOADED_STATE)  │                ▼
└─────────┬───────────┘     ┌──────────────────────┐
          │ scrape_winamax.py          │  make_picks.py       │
          ▼                 │  matrice de scores → │
      odds.json   ────────► │  proba par marché →  │ ──► picks.json
   (73 matchs, ~2300        │  EV + Kelly ¼        │
    marchés pricables)      └──────────────────────┘
                                       │
                                       ▼
                            dashboard/index.html
                            (calendrier, picks, suivi bankroll)
```

## Les données

### 1. [soccer-dataset](https://github.com/eatpizzanot/soccer-dataset) (source principale des forces)

Dataset open-source : 378 562 matchs (2012–2026), 144 ligues, dont toutes les compétitions internationales (CDM, Euro, qualifications, amicaux). Chaque équipe a un rating **Glicko-2** (`rating_mu` ± `rating_sigma`).

`build_strengths.py` en extrait :

- **Forces des 48 équipes CDM** — rating conservateur `mu − sigma`, puis normalisation min-max sur les 48 équipes. La pénalité sigma est importante : les équipes des confédérations faibles (Iran, Sénégal…) ont un mu gonflé par les qualifications jouées contre des adversaires faibles, et un sigma élevé car elles jouent peu de matchs cross-confédération. Sans cette correction, le modèle classait le Sénégal devant la France.
- **Calibration du modèle Poisson** — régression par maximum de vraisemblance `buts ~ exp(a + b·(rating_pour − rating_contre))` sur **1 941 matchs internationaux 2023+**. Résultat : `base_lambda ≈ 1.18` but/équipe/match entre équipes égales, `spread ≈ 1.21` par unité de force normalisée.

### 2. [ScoutFootball_for_World_Cup](https://github.com/Mentaturan/ScoutFootball_for_World_Cup) (calendrier + fallback)

Plateforme d'analytics football locale (DuckDB, PyTorch, ratings joueurs). Utilisée ici pour :

- le **calendrier officiel des 72 matchs de poules** (12 groupes A–L, pattern officiel des journées) ;
- les forces de secours (`compute_team_strengths` : ratings joueurs + prior Opta + effectif Big5) si `strengths_dataset.json` est absent — mais elles sont plates pour les petites équipes (squads placeholder), d'où le passage au Glicko-2.

Le pipeline complet du repo (`ingest → build-features → train`) a été exécuté avec Python 3.12 (lxml ne compile pas en 3.14).

### 3. Winamax (cotes)

Pas d'API publique, mais chaque page embarque son état Redux dans `<script>PRELOADED_STATE = {...}</script>` :

- **Page sport** `paris-sportifs/sports/1` → liste des matchs CDM (tournoi `900001750`, catégorie 4 « International ») + cotes 1N2 ;
- **Page match** `paris-sportifs/match/{id}` → les ~338 marchés du match (cotes, libellés, issues).

Détail amusant : `urllib` Python se prend un **403** (filtrage par empreinte TLS), mais `curl.exe` de Windows (TLS Schannel) passe — le scraper sous-traite donc le HTTP à curl. Politesse : 0,4 s entre chaque requête, ~2 min pour les 73 pages.

## Le modèle

### Poisson + correction Dixon-Coles sur terrain neutre

Pour un match entre forces `s_home` et `s_away` (0–1) :

```
λ_home = base_lambda · exp(spread · (s_home − s_away))    (plafonné à 3,5)
λ_away = base_lambda · exp(−spread · (s_home − s_away))
```

Les buts de chaque équipe suivent une loi de Poisson, corrigée par le **tau de Dixon-Coles (1997)** sur les scores bas — le Poisson indépendant sous-estime les nuls 0-0/1-1. Le paramètre `rho` est ajusté par MLE sur les mêmes 1 941 matchs internationaux que la calibration (rho ≈ −0,105 : nuls boostés). Résultat : **matrice de probabilités de scores** 11×11 (0 à 10 buts), renormalisée. Tous les marchés s'en déduisent par sommation :

| Marché Winamax | Calcul |
|---|---|
| Résultat (1N2) | P(h>a), P(h=a), P(h<a) |
| Double chance | P(h≥a), P(h≠a), P(h≤a) |
| Les 2 équipes marquent | P(h>0 et a>0) |
| Nombre de buts (Plus/Moins X) | P(h+a > X), avec **remboursement** si ligne entière (Plus de 2 = push si exactement 2 buts) |
| Nombre de buts de [équipe] | marginale Poisson de l'équipe |
| Score exact | cellule de la matrice (« Autre » = 1 − somme des scores listés) |
| Nombre exact de buts | P(h+a = n) |
| Vainqueur (remboursé si nul) | EV = p_win·cote + p_nul − 1 |
| Écart de buts (handicap) | P(marge + handicap > 0), push si = 0 |

Marchés **non** pricés (pas de modèle) : buteurs, passes décisives, mi-temps, tirs, corners, minute du but…

### Mélange avec le marché (40/60) — référence Pinnacle quand disponible

Le Poisson calibré reste un modèle simple. Pour chaque marché, la proba finale est :

```
p = 0,40 · p_modèle + 0,60 · p_référence_marché
```

**La référence marché dépend du marché :**

- **1N2, double chance, vainqueur remboursé si nul** : cotes **Pinnacle dé-viguées** (`scrape_pinnacle.py` → `data/pinnacle.json`). Pinnacle est le book sharp de référence — marges faibles, sharps acceptés, sa closing line est considérée comme la meilleure estimation publique des vraies probabilités. C'est la stratégie classique « référence sharp contre book grand public » : quand la cote Winamax dépasse la proba Pinnacle dé-viguée, il y a de la value indépendamment du modèle.
- **Autres marchés** (totaux, score exact, handicap…) : probas implicites Winamax dé-margées, en conservant la masse totale du modèle (gère les marchés à issues non exclusives comme la double chance).

Le poids est réglable par marché via `MODEL_WEIGHTS` dans `make_picks.py`, avec
repli sur 0,40 pour les marchés non calibrés par le backtest.

### Backtest historique

`backtest.py` valide le pipeline sur les tournois historiques du dataset en
reconstruisant les forces et la calibration uniquement à partir des matchs
antérieurs au tournoi testé. Il produit `data/backtest.json` et un résumé
console par tournoi et par marché.

#### Cotes historiques de clôture (comparaison au marché)

La comparaison au marché nécessite des cotes de clôture historiques absentes du soccer-dataset.
Source retenue : **[OddsPortal](https://www.oddsportal.com)** (consultation libre, pas de clé API).

Tournois couverts : **FIFA World Cup 2022** et **UEFA Euro 2024** (1X2 uniquement).
Totaux O/U 2.5 : non disponibles gratuitement — le poids `totals` est fixé à **0,40**.

**Procédure d'installation des cotes (à faire une fois) :**

```bash
pip install playwright && playwright install chromium
python fetch_historical_odds.py      # scrape OddsPortal → data/historical_odds/*.json
python build_closing_backtest.py     # parse + mapping noms → data/closing_backtest.json
python backtest.py                   # backtest complet avec métriques marché
```

`backtest.py` continue à fonctionner sans `closing_backtest.json` (métriques marché affichées en `n/a`).

**Limites et biais — à lire avant d'utiliser les poids :**

1. **Échantillon faible.** 105 matchs matchés (WC2022 + Euro2024). La courbe logloss(w) ci-dessous est quasi-plate entre w=0 et w=0,15 (Δ = +0,0015) — n'importe quelle valeur dans cette zone se vaut statistiquement :

   ```
   w=0.00  logloss=0.9703  (marché seul)
   w=0.05  logloss=0.9700  ← optimal sur cet échantillon
   w=0.10  logloss=0.9706
   w=0.15  logloss=0.9718
   w=0.20  logloss=0.9737
   w=0.40  logloss=0.9865  (+0.016 vs optimal)
   ```

2. **Biais closing vs pré-match.** Le grid search compare le modèle aux cotes de clôture OddsPortal (moyenne bookmakers ≈ fin d'information marché). En pratique tu paries des heures ou jours avant contre des cotes Winamax moins efficientes. Le poids optimal en conditions réelles est probablement un peu plus élevé que w=0,05 — par sécurité `model_weights["1n2"]` est fixé à **0,05** et non 0,00.

3. **Cotes OddsPortal = moyenne bookmakers, pas Pinnacle.** Pinnacle est le book sharp de référence ; la moyenne inclut des books à marge plus élevée, donc moins informatifs. Les probas dé-viguées sont légèrement moins précises que des closing lines Pinnacle pures.

4. **Conséquence pratique sur les picks 1N2.** Avec w=0,05, la proba finale colle quasi-exactement la moyenne marché dé-viguée. Les picks 1N2 ne sortiront que sur des écarts Winamax/marché francs (EV ≥ 3% après marge Winamax), ce qui est le comportement sain : les vraies values sont sur les marchés de niche (totaux, scores exacts) où le poids reste à 0,40.

Quand `closing_backtest.json` est présent, `backtest.py` estime le meilleur poids de
mélange modèle/marché par marché (grid search w ∈ [0, 1] pas 0,05) et l'écrit dans
`data/backtest.json`, lu par `pricing.py` / `make_picks.py`.

- OddsPortal est JS-rendu : Playwright (Chromium headless) est requis.
- Les noms d'équipes OddsPortal sont mappés vers ceux du soccer-dataset dans `build_closing_backtest.py` et `backtest.py` (`_DATASET_TO_CLOSING_NORM`) ; compléter si des matchs restent non-matchés.
- Totaux O/U 2.5 : non disponibles gratuitement — poids `totals` fixé à **0,40** (fallback).

### Suivi du CLV (Closing Line Value)

Le CLV est considéré comme le meilleur prédicteur de rentabilité long terme : si tes cotes prises battent régulièrement la cote de clôture, tu es gagnant — bien avant que ton ROI soit statistiquement significatif. L'onglet Suivi du dashboard affiche pour chaque pari le **CLV vs dernière cote scrapée** (≈ closing line si tu relances `update.bat` près du coup d'envoi) et le **CLV moyen** du portefeuille. CLV moyen positif = le process marche, même si la variance court terme fait mal.

### Value bets et mise

- **EV** = p·cote (+ p_remboursement) − 1 ; seuil de sélection : EV ≥ 3 %, proba ≥ 2 %.
- **Mise suggérée** : critère de **Kelly fractionnaire (¼)** sur la bankroll configurée.
- **Tri par Kelly** plutôt que par EV brut : un EV de +80 % sur une cote 40 (queue de distribution douteuse du Poisson) vaut moins qu'un EV de +15 % sur une cote 2,5. Le tri Kelly fait naturellement remonter les paris jouables.

## Le dashboard

Page web statique (`dashboard/index.html`, zéro dépendance) servie par `python -m http.server` :

- **💎 Picks du jour** — value bets des prochaines 48 h puis suivants, triés par Kelly : cote, proba modèle, EV, mise ¼-Kelly suggérée, bouton « + Suivre » ;
- **📅 Calendrier** — les 72 matchs de poules, horaires réels Winamax, cotes 1N2, barre de probas du modèle ;
- **📒 Suivi** — paris suivis (localStorage) : gagné/perdu, profit, ROI, courbe de bankroll ;
- **🏆 Équipes** — classement des forces des 48 équipes.

## Installation

Prérequis : Windows (curl.exe inclus), Python 3.12+, [uv](https://github.com/astral-sh/uv), Git.

```powershell
# 1. Ce repo
git clone https://github.com/Cordierarn/cdm-wina
cd cdm-wina

# 2. Le repo ScoutFootball (calendrier + venv pandas)
git clone https://github.com/Mentaturan/ScoutFootball_for_World_Cup C:\Users\<vous>\ScoutFootball_for_World_Cup
cd C:\Users\<vous>\ScoutFootball_for_World_Cup
uv sync --python 3.12
# pipeline optionnel (ratings joueurs) :
$env:PYTHONPATH="src"; uv run python -m scoutfootball ingest; uv run python -m scoutfootball build-features; uv run python -m scoutfootball train

# 3. Le dataset (forces Glicko-2)
cd <cdm-wina>\data\dataset
curl.exe -sL -o teams.parquet    https://raw.githubusercontent.com/eatpizzanot/soccer-dataset/main/parquet/teams.parquet
curl.exe -sL -o leagues.parquet  https://raw.githubusercontent.com/eatpizzanot/soccer-dataset/main/parquet/leagues.parquet
curl.exe -sL -o fixtures.parquet https://raw.githubusercontent.com/eatpizzanot/soccer-dataset/main/parquet/fixtures.parquet
```

> Les chemins vers le repo ScoutFootball et uv sont codés en dur dans `update.bat` / `build_strengths.py` — adaptez-les à votre machine.

## Utilisation

```bat
update.bat            :: scrape les cotes Winamax + recalcule les picks (~2 min)
update.bat --model    :: + recalcule forces et calibration depuis le dataset
serve.bat             :: ouvre le dashboard sur http://localhost:8787/dashboard/
```

Rythme conseillé pendant la CDM : un `update.bat` chaque matin, les cotes bougent jusqu'au coup d'envoi.

## Fichiers

| Fichier | Rôle |
|---|---|
| `build_strengths.py` | forces Glicko-2 (mu−sigma) + calibration MLE → `data/strengths_dataset.json` |
| `export_model.py` | calendrier + λ Poisson + probas par match → `data/model.json` (s'exécute dans le venv ScoutFootball) |
| `scrape_winamax.py` | cotes 1N2 + tous marchés pricables → `data/odds.json` (+ historique 1N2 dans `odds_history.jsonl`) |
| `scrape_oddsapi.py` | cotes Pinnacle via The Odds API (clé dans `config.json`, non versionné) → `data/pinnacle.json` + archive `closing_history.jsonl` |
| `scrape_transfermarkt.py` | valeurs marchandes des 48 effectifs (page participants FIWC, 1 requête) → `data/squad_values.json`, mélangées 30 % dans les forces |
| `check_lineups.py` | XI officiels via l'API publique ESPN (publiés ~1h avant kickoff) → `data/lineups.json`, badge dans l'onglet Par match |
| `scrape_pinnacle.py` | secours sans clé : API invitée Pinnacle → `data/pinnacle.json` |
| `make_picks.py` | matrice de scores → probas par marché → EV/Kelly → `data/picks.json` |
| `dashboard/index.html` | interface web (statique, localStorage pour le suivi) |
| `data/dataset/` | parquets du soccer-dataset (non versionnés, voir Installation) |

## Limites connues (lisez ça avant de parier)

1. **Queues de distribution** : la correction Dixon-Coles règle les nuls, mais les queues (score exact 4-3, gros totaux) restent peu fiables → les EV élevés sur cotes > 20 sont systématiquement suspects.
2. **Comparabilité inter-confédérations** : même avec la pénalité mu−sigma, comparer l'Iran (qualifs asiatiques) à la Belgique reste imprécis. Les « values » récurrentes sur les outsiders de petites confédérations en sont le symptôme.
3. **Pas d'info effectifs** : blessures, suspensions, rotations de fin de poule — le marché les connaît, le modèle non. C'est une raison de plus pour le poids marché à 60 %.
4. **Pré-match uniquement** : les cotes scrapées sont celles du moment du scrape, pas du coup d'envoi — relance `update.bat` près du kickoff pour que le CLV approche le vrai closing.
5. **Phase de poules uniquement** pour l'instant — les matchs à élimination directe seront ajoutés quand les affiches seront connues.
6. **Gagner = se faire limiter** : les bookmakers grand public (Winamax inclus) limitent les comptes qui battent régulièrement la closing line. C'est le signe que le process marche, et la principale limite pratique de toute stratégie value.

## Licence / conformité

Projet personnel et éducatif. Scraping léger (1 requête / 0,4 s) des pages publiques Winamax, pas de contournement d'authentification ni de CAPTCHA. Données : soccer-dataset (sources API-Football / Football-Data), ScoutFootball (MIT).
