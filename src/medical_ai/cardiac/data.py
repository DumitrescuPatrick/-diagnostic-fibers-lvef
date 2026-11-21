"""Strict temporal cohort contract and streaming MIMIC-IV feature assembly."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Separate assays: never pool troponin I/T or infer high-sensitivity status.
MARKERS = {
    50963: ('ntprobnp', 'ntprobnp', {'pg/ml': 1., 'ng/l': 1.}),
    51003: ('troponin_t', 'troponin t', {'ng/ml': 1., 'ng/l': .001}),
    51002: ('troponin_i_51002', 'troponin i', {'ng/ml': 1., 'ng/l': .001}),
    52642: ('troponin_i_52642', 'troponin i', {'ng/ml': 1., 'ng/l': .001}),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_index(index, n):
    """One verified index episode per patient, explicitly aligned to archive rows.

    Outcome is future echo LVEF, not a prospective prognosis endpoint.
    Source mapping must be supplied by the cohort builder; never reconstructed.
    """
    required = {'row_index', 'subject_id', 'ecg_time', 'ecg_available_time',
                'decision_time', 'echo_time', 'mapping_source', 'echo_lvef'}
    if not required.issubset(index):
        raise ValueError(f'Missing index columns: {sorted(required - set(index))}')
    d = index.copy()
    if len(d) != n or d.row_index.isna().any() or set(d.row_index) != set(range(n)):
        raise ValueError('Index must contain every archive row_index exactly once')
    d = d.sort_values('row_index').reset_index(drop=True)
    if d.subject_id.isna().any() or d.subject_id.duplicated().any():
        raise ValueError('Require one index episode per unique subject_id')
    if d.mapping_source.fillna('').str.strip().eq('').any():
        raise ValueError('Require source provenance for each row mapping')
    for c in ['ecg_time', 'ecg_available_time', 'decision_time', 'echo_time']:
        d[c] = pd.to_datetime(d[c], errors='raise')
        if d[c].isna().any():
            raise ValueError(f'Missing {c}')
    if not ((d.ecg_time <= d.ecg_available_time) &
            (d.ecg_available_time < d.decision_time) &
            (d.decision_time < d.echo_time)).all():
        raise ValueError('Require ECG acquisition <= availability < decision < echo')
    return d


def demographics(index, patients):
    d = index.merge(patients[['subject_id', 'gender', 'anchor_age', 'anchor_year']],
                    on='subject_id', how='left', validate='one_to_one')
    if d[['anchor_age', 'anchor_year']].isna().any().any():
        raise ValueError('Unmatched patient demographics')
    # Approximate year-based age. The 91 sentinel is a censored age, not exact age.
    censored = d.anchor_age.eq(91)
    age = d.anchor_age + d.decision_time.dt.year - d.anchor_year
    age = age.mask(censored, 90.)
    if ((age < 18) | (age > 120)).any():
        raise ValueError('Adult cohort age outside prespecified bounds')
    return pd.DataFrame({'age_year_approx': age.astype(float),
                         'age_censored': censored.astype(float),
                         'sex_female_recorded': d.gender.map({'F': 1., 'M': 0.})})


def lab_specs(dictionary, routine):
    """routine entries require verified itemid, label, unit multipliers, feature name."""
    specs = dict(MARKERS)
    for r in routine:
        if r.get('itemid') is None or not r.get('units') or r.get('label', '').startswith('REPLACE'):
            raise ValueError('Resolve routine-lab item IDs, dictionary labels, and unit multipliers before assembly')
        item = int(r['itemid'])
        if item in specs:
            raise ValueError('Routine IDs must be distinct from cardiac markers')
        specs[item] = (r['name'], r['label'].lower(), r['units'])
    if len({v[0] for v in specs.values()}) != len(specs):
        raise ValueError('Duplicate laboratory feature names')
    for item, (_, label, _) in specs.items():
        rows = dictionary.loc[dictionary.itemid.eq(item)]
        if len(rows) != 1 or rows.iloc[0].label.lower() != label or rows.iloc[0].fluid != 'Blood':
            raise ValueError(f'Unverified blood assay dictionary definition: {item}')
    return specs


def extract_labs(index, path, specs, lookback_hours=24, chunksize=500_000):
    """Latest numeric result strictly BEFORE decision, within acquisition lookback.

    Missing storetime, censored text (< or >), unrecognized units, nonfinite values,
    and inconsistent timing are excluded. Every laboratory stays missing otherwise.
    """
    if lookback_hours <= 0:
        raise ValueError('lookback_hours must be positive')
    cols = ['subject_id', 'itemid', 'charttime', 'storetime', 'valuenum',
            'value', 'valueuom', 'labevent_id']
    best = pd.DataFrame()
    counts = {'candidate_rows': 0, 'eligible_rows': 0}
    unit_counts = {}
    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize):
        x = chunk.loc[chunk.subject_id.isin(index.subject_id) & chunk.itemid.isin(specs)].copy()
        if x.empty:
            continue
        counts['candidate_rows'] += len(x)
        x = x.merge(index[['subject_id', 'decision_time']], on='subject_id', validate='many_to_one')
        for col in ['charttime', 'storetime']:
            x[col] = pd.to_datetime(x[col], errors='coerce')
        x['unit'] = x.valueuom.fillna('').str.lower().str.replace(' ', '', regex=False)
        for (item, unit), number in x.groupby(['itemid', 'unit']).size().items():
            key = f'{item}:{unit}'
            unit_counts[key] = unit_counts.get(key, 0) + int(number)
        factor = np.array([specs[i][2].get(u, np.nan) for i, u in zip(x.itemid, x.unit)])
        x['result'] = pd.to_numeric(x.valuenum, errors='coerce') * factor
        eligible = (x.charttime.ge(x.decision_time - pd.Timedelta(hours=lookback_hours)) &
                    x.charttime.le(x.storetime) & x.storetime.lt(x.decision_time) &
                    np.isfinite(x.result) &
                    ~x.value.fillna('').astype(str).str.contains(r'[<>]', regex=True))
        x = x.loc[eligible]
        counts['eligible_rows'] += len(x)
        best = pd.concat([best, x], ignore_index=True)
        if not best.empty:
            best = best.sort_values(['storetime', 'charttime', 'labevent_id']).drop_duplicates(
                ['subject_id', 'itemid'], keep='last')
    values = pd.DataFrame(index=index.subject_id, columns=[s[0] for s in specs.values()], dtype=float)
    if not best.empty:
        for item, (name, _, _) in specs.items():
            s = best.loc[best.itemid.eq(item)].set_index('subject_id').result
            values[name] = s.reindex(values.index)
    counts['units_seen_before_filter'] = unit_counts
    counts['patients_with_result'] = {k: int(v) for k, v in values.notna().sum().items()}
    return values.reset_index(drop=True), counts


def assemble(archive, index_path, patients_path, lab_path, dictionary_path, config, output):
    """Do not reuse legacy labs whose predecision availability cannot be proved."""
    with np.load(archive, allow_pickle=False) as z:
        ecg, y = z['X_ecg'].copy(), z['y'].copy()
    index = validate_index(pd.read_csv(index_path), len(y))
    lvef = pd.to_numeric(index.echo_lvef, errors='raise')
    if not lvef.between(0, 100).all() or not np.array_equal((lvef < 40).astype(int), y):
        raise ValueError('Mapped source echo LVEF does not reproduce archive labels')
    if any(n.startswith('REPLACE') for n in config['ecg_names']):
        raise ValueError('Replace ECG placeholders with verified ordered feature names')
    delay = config['decision_delay_minutes']
    if delay <= 0 or not (index.decision_time == index.ecg_available_time + pd.Timedelta(minutes=delay)).all():
        raise ValueError('Decision time must follow the prespecified fixed ECG-availability delay')
    if not ((index.echo_time - index.decision_time) <= pd.Timedelta(days=config['max_echo_days'])).all():
        raise ValueError('Echo outside prespecified reference window')
    if len(config['ecg_names']) != ecg.shape[1] or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid feature-name contract or binary outcome')
    if len(config['routine_labs']) != 4:
        raise ValueError('Prespecify the four routine laboratory assays')
    specs = lab_specs(pd.read_csv(dictionary_path), config['routine_labs'])
    # Troponin-I dictionary entries are audited but excluded from primary analysis.
    if not config.get('include_troponin_i_sensitivity', False):
        specs = {k: v for k, v in specs.items() if k not in (51002, 52642)}
    clinical = demographics(index, pd.read_csv(patients_path))
    labs, audit = extract_labs(index, lab_path, specs, config.get('lookback_hours', 24))
    features = pd.concat([pd.DataFrame(ecg, columns=config['ecg_names']), clinical, labs], axis=1)
    if features.columns.duplicated().any():
        raise ValueError('All feature names must be unique')
    base = list(config['ecg_names']) + list(clinical.columns)
    panels = {'ecg': config['ecg_names'], 'clinical': list(clinical.columns),
              'routine_labs': [r['name'] for r in config['routine_labs']],
              'ecg_clinical': base,
              'ecg_routine': list(config['ecg_names']) + [r['name'] for r in config['routine_labs']],
              'ecg_clinical_routine': base + [r['name'] for r in config['routine_labs']],
              'all': list(features.columns)}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    # Individual-level derived data stays in the explicitly chosen private output.
    np.savez_compressed(out / 'cohort.npz', X=features.to_numpy(float), y=y,
                        subject_id=index.subject_id.to_numpy(),
                        names=np.array(features.columns, dtype=str))
    audit.update({'validated_cohort_contract': 'cardiac-index-v1', 'panels': panels, 'base_features': base,
                  'optional_features': list(labs.columns),
                  'input_sha256': {str(p): sha256(p) for p in
                                   [archive, index_path, patients_path, dictionary_path, lab_path]},
                  'labevents_path': str(lab_path),
                  'labevents_size_bytes': Path(lab_path).stat().st_size,
                  'config': config,
                  'outcome': 'Mapped source echo LVEF <40%, labels matched exactly to archive',
                  'timing': 'charttime <= storetime < decision_time < echo_time; ECG available < decision'})
    (out / 'manifest.json').write_text(json.dumps(audit, indent=2))
    return audit
