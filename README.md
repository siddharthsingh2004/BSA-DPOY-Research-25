# BSA-DPOY-Research-25

A Bruin Sports Analytics research project that builds a statistically-grounded
metric for projecting the **All-NBA Defensive Teams** (and, by extension, the
Defensive Player of the Year). The work moves the question of *"who is the
best defender?"* away from narrative and toward a reproducible model trained
on possession-level matchup data.

---

## 1. Motivation: Why a Statistical Approach to the All-NBA Defensive Teams?

The All-NBA Defensive Teams are voted on by a panel of sportswriters and
broadcasters at the end of every NBA season. In practice that vote folds
together three very different signals:

1. **Player statistics** — steals, blocks, rebounds, defensive on/off splits.
2. **Overall team performance** — defenders on top-seeded teams tend to be
   over-rewarded, while equally skilled defenders on lottery teams are
   often invisible.
3. **Reporter bias** — reputation, market size, highlight reels, and
   recency effects all influence ballots in ways that are hard to audit.

The first signal is measurable, the second is a confound, and the third is
noise. We wanted a single, statistically-defensible defensive impact metric
that:

- isolates **possession-level matchups** (defender vs. ball-handler) rather
  than season-level box-score totals;
- normalizes for **position and opportunity**, so a switching wing isn't
  compared on the same axis as a drop-coverage center;
- produces a score that is **reproducible**, **explainable**
  (SHAP / feature importance), and **decoupled from team record**.

The end goal is a ranked board that a voter (or a front office) could use as
an independent reference point against the All-NBA Defensive ballot.

---

## 2. Models

The pipeline is built in three stages: **feature weighting**, **metric
construction**, and **possession-level prediction**.

### 2.1 PCA + Lasso (model weights)

Two notebooks implement the dimensionality-reduction + feature-weight step:

- [Lasso+PCA/Lasso_PCA.ipynb](Lasso+PCA/Lasso_PCA.ipynb) — a season-level
  baseline that takes five box-score defensive stats (`STL%`, `DRB%`,
  `BLK%`, `DWS`, `DBPM`), standardizes them, runs PCA, then fits a
  `LassoCV` regression of the row-summed defensive total onto the principal
  components. The Lasso-selected components are projected back to a single
  weighted score, min–max normalized to `[0, 100]` per player, and the top
  five players per position are exported as the first cut of the board.
- [LassoPCA Norm by Pos/Lasso_PCA.ipynb](LassoPCA%20Norm%20by%20Pos/Lasso_PCA.ipynb)
  — the position-normalized version. Defensive stats are first converted to
  **z-scores within each position bucket**, then fed into the same
  PCA → LassoCV → weighted-projection pipeline. This is what addresses the
  "wings vs. centers aren't comparable" critique in §1.

Both notebooks output a CSV of the top defenders (top-25 overall and top-5
by position) that serves as the seed ranking for the next stage.

### 2.2 Metric Creation

Located under [Metric Creation/](Metric%20Creation/). This module takes the
Lasso+PCA seed score and turns it into a fully-supervised **defensive impact
score** trained against expert rankings.

There are two notebooks: a baseline
([Metric Construction.ipynb](Metric%20Creation/Metric%20Construction.ipynb))
and an updated version
([Metric Construction_updated.ipynb](Metric%20Creation/Metric%20Construction_updated.ipynb)).

The workflow in both:

1. Merge every sheet of `MASTER.xlsx` (a multi-tab NBA.com export covering
   general defense, hustle, contested shots, etc.) on `PLAYER`.
2. Join the Lasso+PCA defensive impact score from `Rankings.xlsx` as the
   regression target `SCORE`.
3. Standard-scale the numeric columns, fit a `RandomForestRegressor` with
   **leave-one-out cross-validation**, and report R² and RMSE.
4. Use the random forest's **feature importances as dynamic weights** —
   each numeric feature is multiplied by its importance, and the sum is the
   final `IMPACT SCORE`.
5. Apply a 65-games-played filter and export `OUTPUT.csv` ranked by
   `IMPACT SCORE`.

**Baseline → updated improvements:**

| Notebook | LOOCV R² | LOOCV RMSE |
| --- | --- | --- |
| `Metric Construction.ipynb` (baseline) | 0.307 | 22.877 |
| `Metric Construction_updated.ipynb` | **0.345** | **22.063** |

The updated notebook differs from the baseline in three ways:

- **Per-sheet column namespacing** — columns are renamed `{COL}_{SHEETNAME}`
  before merging, so identically-named stats from different NBA.com tabs no
  longer collide and silently overwrite each other.
- **Normalized player-name joins** — names are run through `clean_name()`
  (uppercase + punctuation stripped) on both sides of the merge before
  joining, recovering players lost to formatting drift between sheets.
- **Possession-aware feature injection** — `defender_scores_2024.csv`
  (the output of the XGBoost classifier in §2.3) is merged in as
  `avg_scoring_probability` and `possessions`, so the random forest sees
  per-possession defensive performance, not just season aggregates.

### 2.3 XGBoost — Random Forest and Classifier

Located under
[XGBoost+Scoring_Opportunity_Predictor/](XGBoost+Scoring_Opportunity_Predictor/).
This module drops down to **possession-level matchup data** (every guarded
possession in the 2023-24 regular season) and predicts the probability that
the offensive player **scores on a given matchup**. Lower predicted scoring
probability ⇒ better defender.

Three notebooks implement variants of this classifier:

| Notebook | Estimator | Log Loss | AUC |
| --- | --- | --- | --- |
| [Features_from_matchups_2024_RandomForestClassifier.ipynb](XGBoost+Scoring_Opportunity_Predictor/Features_from_matchups_2024_RandomForestClassifier.ipynb) | `RandomForestClassifier` (calibrated, isotonic) | 0.4798 | 0.8190 |
| [Features_from_matchups_2024_XGBClassifier.ipynb](XGBoost+Scoring_Opportunity_Predictor/Features_from_matchups_2024_XGBClassifier.ipynb) | `XGBClassifier` baseline | 0.0001 | 1.0000 |
| [Features_from_matchups_2024_XGBClassifier_updated_2.0.ipynb](XGBoost+Scoring_Opportunity_Predictor/Features_from_matchups_2024_XGBClassifier_updated_2.0.ipynb) | `XGBClassifier` + calibration | 0.0101 | 0.9998 |

**Reading these numbers honestly:** the random forest classifier is the only
one of the three whose AUC isn't pegged at the ceiling. The two XGBoost
classifiers are almost certainly suffering from **target leakage** —
features used to *derive* the `scoring_event` label (matchup FGM, FGA,
blocks, turnovers, fouls, etc.) are also being fed into the model as inputs.
The updated XGB notebook documented below begins to fix this; the rest is
flagged in [§6. Next Steps](#6-next-steps).

#### Baseline XGB classifier vs. updated XGB classifier (`_updated_2.0`)

The baseline notebook
([Features_from_matchups_2024_XGBClassifier.ipynb](XGBoost+Scoring_Opportunity_Predictor/Features_from_matchups_2024_XGBClassifier.ipynb))
took every numeric matchup column, derived `scoring_event` from those same
columns, then trained an `XGBClassifier` with `n_estimators=500`,
`max_depth=5`, light L1/L2 regularization, and no probability calibration.
The result was Log Loss ≈ 0.0001 and AUC = 1.0 — a textbook leakage
signature.

The updated notebook
([Features_from_matchups_2024_XGBClassifier_updated_2.0.ipynb](XGBoost+Scoring_Opportunity_Predictor/Features_from_matchups_2024_XGBClassifier_updated_2.0.ipynb))
makes the following concrete improvements:

1. **A `filter_non_leaky_features()` step** that drops every raw matchup
   counting stat and any column whose name contains `points`, `allowed`,
   `stop`, or `rate` before training. The feature set is reduced to
   eleven engineered, rate-based features:
   `efg_allowed`, `two_pt_allowed_pct`, `turnover_rate`, `block_rate`,
   `help_stop_rate`, `coverage_share`, `points_allowed_per_100`,
   `fga_per_min`, `3pa_per_min`, `fta_per_min`, `switch_rate`.
2. **Offensive-rating context** — `2024_offensive_advanced_metrics.csv`
   is merged in on the offensive (matched-up) player so the model can
   condition on opponent quality, not just defender behavior.
3. **Tighter regularization and a shallower tree** —
   `max_depth=2`, `subsample=0.6`, `colsample_bytree=0.6`,
   `reg_alpha=6.0`, `reg_lambda=7.0`, `n_estimators=200`. The baseline's
   `max_depth=5`, `reg_alpha=1.0`, `reg_lambda=1.0` was over-flexible
   given the leaked signal.
4. **Probability calibration** — `CalibratedClassifierCV(method='sigmoid',
   cv='prefit')` is fit on top of the XGB model, so the
   `avg_scoring_probability` exported to `defender_scores_2024.csv` is a
   well-scaled probability rather than an uncalibrated margin.
5. **A stricter qualification filter** — defenders must clear **≥ 600
   matchup minutes** (vs. ≥ 500 in the baseline) before they enter the
   training set, cutting noisy low-sample tails.

#### XGBoost performance summary

- **Random Forest classifier** (the calibrated, leakage-controlled baseline)
  is the only model whose metrics are believable on their face: **Log Loss
  0.4798, AUC 0.8190**. It uses only six "safe" features
  (`matchup_minutes`, `fga_per_min`, `3pa_per_min`, `fta_per_min`,
  `switch_rate`, `coverage_share`) and class-balances via upsampling before
  training.
- **XGBoost baseline** reports near-perfect metrics
  (**Log Loss 0.0001, AUC 1.0000**), which we interpret as leakage rather
  than skill — see §6.
- **XGBoost updated 2.0** is the metric-of-record for the final defender
  ranking: **Log Loss 0.0101, AUC 0.9998** on its held-out 20% test split,
  with calibrated probabilities, SHAP-explainable features, and a
  defender-level export written to `defender_scores_2024.csv` that the
  Metric Creation module consumes in §2.2.

---

## 3. Repository Structure

```text
BSA-DPOY-Research-25/
├── README.md                              # this file
├── requirements.txt                       # Python dependencies
├── .venv/                                 # local virtualenv (gitignored)
│
├── 2023-24 Advanced Stats.xlsx - Sheet1.csv
├── 2024 Defense Stats Updated.xlsx
├── 2024_offensive_advanced_metrics.csv
├── MASTER.xlsx                            # multi-sheet NBA.com defense export
├── PcaPlusPF.ipynb                        # exploratory PCA + personal-foul scratch
├── XGBoost_model_LAL@DEN[2023-10-24].ipynb  # single-game prototype
├── [2023-12-31]-0022300446-LAL@NOP.csv      # single-game matchup sample
│
├── Lasso+PCA/                             # §2.1 — season-level PCA + Lasso
│   ├── Lasso_PCA.ipynb
│   └── Lasso_PCA.csv
│
├── LassoPCA Norm by Pos/                  # §2.1 — position-normalized version
│   ├── 2024 Defense Stats Updated.xlsx
│   ├── 2024_Defense_Stats.xlsx
│   ├── Lasso_PCA.ipynb
│   ├── Lasso_PCA_top25_overall.csv
│   └── Lasso_PCA_top5_by_pos.csv
│
├── Metric Creation/                       # §2.2 — supervised impact score
│   ├── MASTER.xlsx
│   ├── Rankings.xlsx
│   ├── Metric Construction.ipynb          # baseline
│   ├── Metric Construction_updated.ipynb  # updated (per-sheet namespacing + clean joins)
│   ├── OUTPUT.csv
│   └── OUTPUT_updated.csv
│
└── XGBoost+Scoring_Opportunity_Predictor/ # §2.3 — possession-level classifier
    ├── Features_from_game.ipynb
    ├── Features_from_matchups_2024_RandomForestClassifier.ipynb
    ├── Features_from_matchups_2024_XGBClassifier.ipynb
    ├── Features_from_matchups_2024_XGBClassifier_updated_2.0.ipynb
    ├── XGBoost_model_LAL@DEN[2023-10-24].ipynb
    ├── working_nb.ipynb
    ├── defender_scores_2024.csv          # final per-defender scoring probabilities
    ├── matchups_2024_agg_updated.csv
    └── data/
        ├── 2024_Player_Hashmap.csv
        ├── ATLvsBOS.csv
        ├── [2023-10-24]-LAL@DEN.csv
        ├── matchups_2024.csv
        ├── matchups_2024.tar.xz
        ├── matchups_2024_agg.csv
        └── matchups_2024_agg.tar.gz
```

---

## 4. Quick Start

The repo runs entirely in Python 3.9+ Jupyter notebooks. There are no API
keys or external services required — everything works off the local CSV /
XLSX files in this directory.

### 4.1 Clone and enter the repo

```bash
git clone <this-repo-url>
cd BSA-DPOY-Research-25
```

### 4.2 Create and activate the virtualenv

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows PowerShell
```

### 4.3 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.4 Environment variables

No secrets are required to run the pipeline, but the notebooks expect a few
relative paths to resolve correctly. Either run each notebook from the
directory it lives in (Jupyter's default) or export these from your shell
before launching Jupyter:

```bash
# Project root — used as a fallback when notebooks are launched from elsewhere.
export BSA_DPOY_ROOT="$(pwd)"

# Optional: where the heaviest CSV (matchups_2024.csv, ~57 MB) lives.
# Override only if you've relocated the data directory.
export BSA_DPOY_DATA="$BSA_DPOY_ROOT/XGBoost+Scoring_Opportunity_Predictor/data"

# Optional: silence the XGBoost deprecation warning about use_label_encoder.
export PYTHONWARNINGS="ignore::UserWarning"
```

### 4.5 Run the notebooks

```bash
jupyter notebook
```

Recommended execution order:

1. `Lasso+PCA/Lasso_PCA.ipynb` *or* `LassoPCA Norm by Pos/Lasso_PCA.ipynb`
   → produces the seed defensive impact score.
2. `XGBoost+Scoring_Opportunity_Predictor/Features_from_matchups_2024_XGBClassifier_updated_2.0.ipynb`
   → trains the possession-level classifier and writes
   `defender_scores_2024.csv`.
3. `Metric Creation/Metric Construction_updated.ipynb`
   → combines (1) and (2) into the final ranked `OUTPUT_updated.csv`.

---

## 6. Next Steps

The current pipeline is a working prototype, not a finished product. The
items below are the open threads we'd want a future BSA cohort to pick up.

### 6.1 Address overfitting

- The two XGB classifiers report AUC ≈ 1.0 / Log Loss ≈ 0.0001 on a held-out
  20% split. A model that perfect on real possession data isn't a model —
  it's a memorized lookup table. Concrete fixes:
  - Switch from a single train/test split to **k-fold cross-validation
    grouped by player** (`GroupKFold(groups=person_id)`), so the same
    defender's possessions can't appear in both train and test.
  - Run nested CV for hyperparameter selection instead of the current
    single held-out validation set.
  - Add **early stopping on a true validation fold** rather than letting
    `n_estimators=200` run to completion.
  - Compare against a logistic-regression baseline; if XGBoost can't beat
    it by a meaningful margin on the held-out player groups, the
    additional model capacity isn't earning its keep.

### 6.2 Eliminate data leakage

- The `derive_scoring_event()` label is computed from the same matchup
  counting stats (`matchup_field_goals_made`, `matchup_blocks`,
  `matchup_turnovers`, `shooting_fouls`, …) that get fed back into the
  model as features. The `_updated_2.0` notebook starts to address this
  with `filter_non_leaky_features()`, but the substitute features
  (`efg_allowed`, `block_rate`, `help_stop_rate`, …) are themselves
  **deterministic transforms of the same counting stats**, so the leakage
  is laundered rather than removed.
  - Replace label-derived rate stats with **contextual features**:
    opponent offensive rating, shot-clock state, defender distance,
    closeout speed, screen navigation flags (from Second Spectrum /
    Synergy if accessible).
  - Audit every engineered feature with the simple test: *"could I
    compute this column without knowing whether the offensive player
    scored on this possession?"* — if not, drop it.

### 6.3 New feature engineering

- **Lineup context**: a defender's scoring probability allowed depends on
  the other four players on the floor. Add lineup-level defensive rating
  and opponent shot-quality (xPPP) as covariates.
- **Shot-quality regression**: instead of binary scored / didn't-score,
  predict **expected points allowed per possession** using shot location,
  defender distance, and closeout angle.
- **Matchup-difficulty weighting**: a stop against a top-10 offensive
  player should weigh more than a stop against a bench wing. Use opponent
  PER / OBPM as a per-possession sample weight.
- **Temporal features**: rest days, back-to-backs, late-game vs.
  early-game splits, garbage-time filtering.

### 6.4 Other model explorations

- **Gradient-boosted survival / hazard models** for "how many possessions
  before this defender gives up a bucket?" — a more natural framing than
  binary classification.
- **Bayesian hierarchical models** with per-player random effects, so
  small-sample defenders are shrunk toward position priors instead of
  ranked alongside high-volume regulars.
- **Neural matchup embeddings**: learn a low-dimensional embedding for
  each defender and each offensive player, then predict scoring
  probability from the interaction — this captures matchup-specific
  effects (e.g. small guards struggling against bigger wings) that a flat
  feature vector can't.
- **Cross-season validation**: train on 2022-23, score on 2023-24, and
  compare model-predicted All-Defense teams against the actual ballots.
  This is the only honest external validation we have available, and it
  hasn't been run yet.
