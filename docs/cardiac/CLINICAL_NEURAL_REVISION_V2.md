# MIMIC-only clinical neural revision v2 — locked before fitting

This is a new exploratory internal-validation analysis. Original E/F/G studies, cohorts, patient folds, public/submission packages and their success receipts remain unchanged. No synthetic comparative experiment is rerun. Artificial arrays in unit tests are software fixtures only. Clinical inputs remain local/private; logs show aggregate fold/seed metadata only.

## Population and training

All9,025patients/919definite subsequent LVEF<40% outcomes enter the same eight panels and five locked outer/four inner patient folds. Missing labs do not exclude patients. The record/query distinction and prior timing/phenotype assumptions remain. Clinical measurements are only usable eligible pre-landmark records; absence is not proof a test was never ordered or an observed physiological counterfactual.

Five training seeds:20261114,20361114,20461114,20561114,20661114. All25final seed/fold initializations are distinct. The pilot adjacent-base-seed run is preserved separately and is not the primary revision. This choice corrects initialization reuse across different outer folds without selecting seeds by performance. The shared20261114seed run may be reused only after exact training-configuration and checkpoint-hash equivalence checks. Each is a complete five-outer-fold run; these are five training replicates of the SAME patients, not45,125independent patients. The same64×64ReLU architecture,30epochs,AdamW(.001,weight decay.0001),batch256 and native sigmoid output are retained. No early stopping or fitted posthoc calibration is introduced. Separate networks per panel remain the standard comparator. The shared model remains unpenalized.

Coherence strength is selected from **0,.25,2** using mean eight-panel inner Brier across all four locked inner folds. Ties select the smallest beta. No outer test score or audit metric selects a model. At beta0, the final selected learner reuses the paired shared model and must have bitwise-identical predictions. Independent RNG streams for batch order, supervised masks and constraint selection keep supervised examples/masks matched across beta values. This deliberately revises the prior shared RNG stream; the first seed is not claimed to reproduce the old G probabilities. Constraint pools use fixed fold-specific seeds across the five training seeds, isolating optimization/init randomness conditional on those pools. They are generated from training records only, cached privately, and never released.

The positive penalty remains the prior soft eight-donor kNN conditional-record objective. This revision does not fix its approximate-kernel or finite-particle limitations. Total capacity, panel exposure and tuning/computation still differ between the panel ensemble and shared/selected learners. Only the coherence-selection family tunes beta; no equal-budget architecture superiority claim is warranted.

## Non-circular consistency assessment

Prespecified nested edges are ECG/clinical→+routine; baseline→+NT; baseline→+troponinT; +NT→+both; +troponinT→+both. All families use identical patients, edges and audit splits.

**Primary different-method check:** after risk fitting/selection is finished, evaluate parent/child probabilities on each outer TEST set. Inside that set use outcome-blind three-fold KFold. Fit ExtraTreesRegressor on the other two audit folds, using ONLY encoded parent-visible information, targeting the child probability. Predict the remaining audit fold. Every parent is scored against a projection fitted without that patient. Hyperparameters are fixed:64trees,min_samples_leaf40,max_features1,n_jobs1; seeds/folds fixed across families and training seeds. The audit uses neither disease outcomes nor the training kNN donor pool and does not feed back into prediction, beta selection or deployment.

Report mean squared difference between parent risk and the audit's cross-fitted conditional mean. Also report child-versus-audit-projection MSE to expose projection difficulty. This is a different estimator on held-out records, not independently known clinical conditional truth: the score includes regression-estimator error, and the audit sample is still the selected observed-record population. Audit fitting on outer-test features/probabilities is for assessment only, not for disease prediction. It is not external validation.

**Secondary sampler-free check:** compute r=f_T−f_S on outer-held-out records and bounded functions h(X_S):intercept,visible finite flags,standardized visible values clipped to[-3,3]/3,and their squares. Nonconstant columns are selected using outer training data only. Tower consistency requires E[rh]=0 for these parent-only functions. Report each moment and a descriptive average squared moment score, retaining fold/edge identity to avoid cancellation across different fitted models. Finite moments are necessary, not sufficient for full conditional consistency; raw squared sample moments contain sampling variation. No significance/novelty claim is inferred merely from a small score.

Audit predictions/products remain private. No patient-level probabilities, individual residuals or fine-grained thresholds are released. Audits are specified before fitting; they are not optimized against outer results.

## Performance and uncertainty

Primary summaries are the **mean of five single-seed metrics**, with seed SD, minimum and maximum. They are not metrics of averaged predictions. Separately report the secondary prespecified uniform five-seed probability ensemble and label it clearly. Metrics include AUC,AP,Brier,log loss,calibration slope/intercept/ECE,diagnostic loss(FN20/FP1,threshold1/21),sensitivity/specificity and fixed-threshold net benefit. Risk variance helps detect smoothing/collapse.

Use1,000paired patient resamples, with identical case weights for all families/seeds in each comparison. Report patient-only conditional intervals for seed-mean metrics. Also report **crossed patient×seed bootstrap** intervals:resample patients and complete training-seed runs independently, using the same draws across competing families. This captures variation across the five observed seeds, not a well-estimated universal seed distribution or full pipeline refitting. Seeds/folds are not independent patient cohorts. Calibration regressions/ECE remain descriptive point estimates; ensemble figures coarsen bins to at least20people and5events/non-events. Bootstrap intervals hold risk models,audit fits and beta choices fixed and omit cohort/source/timing/sampler refitting uncertainty. Inference remains exploratory with no multiplicity correction.

The same full-trained held-out predictions are additionally summarized on the502-patient source-consistent subset and NT-result-available/unavailable groups. These are descriptive selection-sensitive analyses, not new memberships or proof of physical time/ordering validity.

## Figures and release

Five MIMIC-only figures:cohort/availability;three-family clinical AUC/Brier;separately labelled seed-ensemble calibration;training-seed stability and beta selections;different-method consistency and moment checks. Every figure has aggregate input CSVs. No new synthetic/framework figure is substituted for a requested clinical result. The aggregate-only ZIP excludes all caches,weights,private arrays,identifiers,individual predictions and raw ROC thresholds. The old manuscript and archives are not silently overwritten; revised numbers belong to a later manuscript revision.

Training/config/protocol sources are hashed before fitting. Analysis/report code is independently hashed when aggregate analysis starts; numerical definitions are fixed here. Completed outputs get a new success receipt after independent checks and preservation verification.

The v1 pilot is retained without publication claims. Its completed first training-seed run and training-only constraint pools are byte-for-byte reusable because all fitting/selection/audit hyperparameters and the underlying input/fold artifacts match; new seed values and versioned output roots are the only changes affecting the other replicates. Reuse provenance is recorded explicitly.
