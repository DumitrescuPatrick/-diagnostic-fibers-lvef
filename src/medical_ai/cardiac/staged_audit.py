"""Stage A/B read-only audit plus a separately versioned clinical predictor bundle."""
import argparse
import json
import os
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from .clinical_predictors_v2 import (LAB_DEFINITIONS, audit_laboratory_sources,
                                     encode_demographics)
from .data import sha256
from .reporting import versions

RAW_NAMES = ['rr_interval', 'p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end', 'p_axis', 'qrs_axis', 't_axis']
ROUTINE = ['creatinine', 'sodium', 'potassium', 'hemoglobin']


def snapshot(paths):
    result = {}
    for root in paths:
        root = Path(root)
        for p in sorted(root.rglob('*')) if root.is_dir() else [root]:
            if p.is_file():
                result[str(p)] = {'sha256': sha256(p), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
    return result


def missingness(X, y, names, cohort):
    rows = []
    for group, mask in [('all', np.ones(len(y), bool)), ('lvef_below_40', y == 1), ('lvef_at_least_40', y == 0)]:
        for j, name in enumerate(names):
            missing = int((~np.isfinite(X[mask, j])).sum())
            rows.append({'cohort': cohort, 'group': group, 'feature': name, 'n': int(mask.sum()),
                         'missing': missing, 'missing_percent': 100*missing/int(mask.sum())})
    return pd.DataFrame(rows)


def manuscript_evidence(path):
    with ZipFile(path) as z:
        tree = ET.fromstring(z.read('word/document.xml'))
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    paragraphs = [''.join(t.text or '' for t in p.findall('.//w:t', ns)) for p in tree.findall('.//w:p', ns)]
    keys = ['retrospective linked cohort comprised 2,017', 'Nine ECG machine-derived',
            'All reported results originate from user-provided']
    return {'path': str(path), 'sha256': sha256(path),
            'paragraphs': [{'paragraph_zero_based': i, 'text': p} for i, p in enumerate(paragraphs)
                           if any(k.lower() in p.lower() for k in keys)],
            'evidence_status': 'documented manuscript claims, not authenticated original source code'}


def audit_and_prepare(root='.', source='~/Downloads', output='results/cardiac_stages_ab_v2',
                      private_output='private-data/cardiac_stages_ab_v2'):
    root, source = Path(root), Path(source).expanduser()
    frozen = root/'private-data/cardiac_rebuilt_v1'
    legacy = root/'private-data/cardiac_chunk7_cohort.npz'
    protected = [legacy, frozen, root/'results/cardiac_rebuilt_v1']
    before = snapshot(protected)
    out, private = root/output, root/private_output
    if out.exists() or private.exists():
        raise FileExistsError('Use new stage output directories')
    out.mkdir(parents=True); private.mkdir(parents=True, mode=0o700); os.chmod(private, 0o700)
    (private/'preservation_before.json').write_text(json.dumps(before, indent=2))
    manifest = json.loads((frozen/'manifest.json').read_text())
    for path, digest in manifest['input_sha256'].items():
        if sha256(path) != digest:
            raise ValueError(f'Source file changed since cohort rebuild: {path}')
    index = pd.read_csv(frozen/'linked_index.csv.gz')
    if not index.subject_id.is_unique or not np.array_equal(index.row_index, np.arange(len(index))):
        raise ValueError('Nonunique or misaligned primary index')
    for c in ['ecg_time', 'ecg_available_time', 'decision_time', 'echo_time', 'admittime', 'dischtime']:
        index[c] = pd.to_datetime(index[c], format='mixed')
    with np.load(frozen/'cohort.npz', allow_pickle=False) as z:
        X, y, names = z['X'].copy(), z['y'].copy(), z['names'].tolist()
        np.testing.assert_array_equal(z['subject_id'], index.subject_id)
    with np.load(legacy, allow_pickle=False) as z:
        legacy_y = z['y'].copy()
        legacy_X = np.column_stack([z['X_ecg'], z['X_lab']])
    # Historical lab order cannot be authenticated; do not assign their four array columns names.
    legacy_names = RAW_NAMES+[f'lab_column_{i}_order_unverified' for i in range(4)]
    evidence = manuscript_evidence(root/'Cardiac_AI_Manuscript_and_Figures/Cardiac_Geometric_AI_Illustrated_Working_Draft.docx')
    (out/'legacy_documented_evidence.json').write_text(json.dumps(evidence, indent=2))
    missingness(X, y, names, 'rebuilt_v1').to_csv(out/'missingness_rebuilt.csv', index=False)
    missingness(legacy_X, legacy_y, legacy_names, 'legacy_archive').to_csv(out/'missingness_legacy.csv', index=False)
    # Eligibility groups deliberately overlap; primary cohort membership is fixed.
    groups = {'all_rebuilt': np.ones(len(y), bool), 'complete_routine': index.complete_routine_labs.to_numpy(bool),
              'incomplete_routine': ~index.complete_routine_labs.to_numpy(bool),
              'ntprobnp_available': np.isfinite(X[:, names.index('ntprobnp')]),
              'ntprobnp_missing': ~np.isfinite(X[:, names.index('ntprobnp')]),
              'troponin_t_available': np.isfinite(X[:, names.index('troponin_t')]),
              'troponin_t_missing': ~np.isfinite(X[:, names.index('troponin_t')]),
              'record_list_waveform_path': index.waveform_path.notna().to_numpy(),
              'without_record_list_waveform_path': index.waveform_path.isna().to_numpy(),
              'prior_recorded_low_lvef': index.prior_recorded_low_lvef.to_numpy(bool),
              'without_prior_recorded_low_lvef': ~index.prior_recorded_low_lvef.to_numpy(bool)}
    rows = [{'group': 'legacy_documented_archive', 'patients': len(legacy_y), 'positive': int(legacy_y.sum()),
             'prevalence': float(legacy_y.mean()), 'identity_verified': False}]
    for group, mask in groups.items():
        rows.append({'group': group, 'patients': int(mask.sum()), 'positive': int(y[mask].sum()),
                     'prevalence': float(y[mask].mean()) if mask.any() else None, 'identity_verified': True})
    pd.DataFrame(rows).to_csv(out/'eligibility_groups.csv', index=False)
    pd.DataFrame([{'disposition': k, 'patients': v, 'unit': 'patient', 'outcome_definition': 'not_established_for_excluded_groups'}
                  for k, v in manifest['summary']['reference_linkage']['index_patient_disposition'].items()]).to_csv(
                      out/'reference_eligibility_flow.csv', index=False)
    labs, definitions, violations = audit_laboratory_sources(index, pd.read_csv(frozen/'laboratory_provenance.csv.gz'),
                                                            pd.read_csv(source/'d_labitems.csv.gz'))
    definitions.to_csv(out/'verified_laboratory_dictionary.csv', index=False)
    local_biomarkers = pd.read_csv(source/'d_labitems.csv.gz').loc[
        lambda d: d.label.str.contains('troponin|probnp', case=False, na=False)]
    local_biomarkers.to_csv(out/'all_local_cardiac_assay_candidates.csv', index=False)
    # Verify the matrix's lab cells against canonicalized event provenance.
    lab_names = [v[0] for v in LAB_DEFINITIONS.values()]
    lab_values = labs.pivot(index='subject_id', columns='feature', values='canonical_value').reindex(
        index=index.subject_id, columns=lab_names)
    for name in names:
        if name in lab_names:
            np.testing.assert_allclose(X[:, names.index(name)], lab_values[name], equal_nan=True)
    demographics, dem_sources = encode_demographics(index, pd.read_csv(source/'patients.csv.gz'))
    dem_sources['frozen_v1_age_proxy'] = X[:, names.index('age_year_approx')]
    changed = ~np.isclose(demographics.age_year_proxy, dem_sources.frozen_v1_age_proxy)
    dem_sources['age_encoding_changed_from_v1'] = changed
    # Numeric conversions use dictionary Blood; serum/plasma cannot be authenticated.
    source_columns = list(labs.columns)+[]
    labs['verified_dictionary_fluid'] = labs.itemid.map({k: v[2] for k, v in LAB_DEFINITIONS.items()})
    labs['canonical_unit'] = labs.itemid.map({k: v[3] for k, v in LAB_DEFINITIONS.items()})
    labs['specimen_subtype_status'] = 'serum_or_plasma_not_available_in_supplied_tables'
    labs.to_csv(private/'laboratory_provenance_verified.csv.gz', index=False)
    dem_sources.to_csv(private/'demographic_provenance.csv.gz', index=False)
    pd.DataFrame([{'feature': n, 'observations': int(np.isfinite(lab_values[n]).sum()),
                   'missing': int(lab_values[n].isna().sum())} for n in lab_names]).to_csv(
                       out/'biomarker_and_lab_availability.csv', index=False)
    # Raw representation is a sensitivity analysis, not a recreation of unknown legacy cleaning.
    raw = index[RAW_NAMES].apply(pd.to_numeric, errors='coerce').copy()
    raw['rr_interval'] = raw.rr_interval.where(raw.rr_interval.between(240, 3000))
    for name in ['p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end']:
        raw[name] = raw[name].where(raw[name].between(0, 5000))
    for name in ['p_axis', 'qrs_axis', 't_axis']:
        raw[name] = raw[name].where(raw[name].between(-180, 360))
        raw[name] = raw[name].where(raw[name] <= 180, raw[name]-360)
    raw.columns = ['raw_'+n for n in raw.columns]
    ecg_names = names[:9]
    expanded = pd.concat([pd.DataFrame(X[:, :9], columns=ecg_names), demographics,
                           lab_values.reset_index(drop=True), raw.reset_index(drop=True)], axis=1)
    if len(expanded) != len(index) or expanded.columns.duplicated().any():
        raise ValueError('Expanded data misalignment')
    np.savez_compressed(private/'predictors.npz', X=expanded.to_numpy(float), y=y,
                        subject_id=index.subject_id.to_numpy('int64'), names=np.array(expanded.columns, dtype=str),
                        ecg_study_id=index.study_id.to_numpy('int64'),
                        echo_measurement_id=index.echo_measurement_id.to_numpy('int64'),
                        observed=np.isfinite(expanded.to_numpy(float)))
    # Audit nominal timing without claiming measured ECG or static attribute availability.
    timing = []
    for name in expanded:
        if name in lab_names:
            d = labs.loc[labs.feature.eq(name)]
            margins = (d.decision_time-d.storetime).dt.total_seconds()/60
            timing.append({'feature': name, 'nonmissing': len(d), 'recorded_time_violations': 0,
                           'availability_status': 'VERIFIED_LAB_STORETIME_BEFORE_RECORDED_LANDMARK',
                           'minimum_minutes_before_landmark': float(margins.min()) if len(d) else None})
        elif name in demographics:
            timing.append({'feature': name, 'nonmissing': int(expanded[name].notna().sum()),
                           'recorded_time_violations': None, 'availability_status': 'STATIC_ATTRIBUTE_ASSUMPTION_NO_RELEASE_TIMESTAMP',
                           'minimum_minutes_before_landmark': None})
        else:
            timing.append({'feature': name, 'nonmissing': int(expanded[name].notna().sum()),
                           'recorded_time_violations': None, 'availability_status': 'ECG_ACQUISITION_TIME_PROXY_NO_RELEASE_TIMESTAMP',
                           'minimum_minutes_before_landmark': 60.})
    pd.DataFrame(timing).to_csv(out/'predictor_temporal_audit.csv', index=False)
    margins = (labs.decision_time-labs.storetime).dt.total_seconds()/60
    lab_gaps = (labs.decision_time-labs.charttime).dt.total_seconds()/3600
    echo_gaps = (index.echo_time-index.decision_time).dt.total_seconds()/3600
    temporal = {'laboratory_violations': violations, 'laboratory_selected_events': len(labs),
                'ecg_proxy_equals_acquisition_n': int(index.ecg_available_time.eq(index.ecg_time).sum()),
                'ecg_proxy_60min_before_landmark_n': int(((index.decision_time-index.ecg_time).dt.total_seconds() == 3600).sum()),
                'lab_minutes_before_landmark_quantiles': margins.quantile([0, .25, .5, .75, 1]).to_dict(),
                'specimen_hours_before_landmark_quantiles': lab_gaps.quantile([0, .25, .5, .75, 1]).to_dict(),
                'echo_hours_after_landmark_quantiles': echo_gaps.quantile([0, .25, .5, .75, 1]).to_dict(),
                'physical_clock_alignment_verified': False, 'all_predictors_clinical_availability_proven': False,
                'clinical_static_attribute_release_times_observed': False}
    (out/'temporal_audit.json').write_text(json.dumps(temporal, indent=2))
    shifts = []
    for hours in [-48, -24, -12, -6, 0, 6, 12, 24, 48]:
        delay = pd.Timedelta(hours=hours)
        perturbed = index.decision_time+delay
        valid_pair = ((index.echo_time-perturbed).between(pd.Timedelta(hours=24), pd.Timedelta(days=30)) &
                      (index.ecg_time+delay >= index.admittime) & (perturbed < index.dischtime))
        late = labs.storetime >= labs.decision_time+delay
        shifts.append({'ecg_clock_shift_hours': hours, 'fixed_pairs_retaining_recorded_time_rules': int(valid_pair.sum()),
                       'selected_labs_at_or_after_perturbed_landmark': int(late.sum()),
                       'patients_with_any_such_lab': int(labs.loc[late, 'subject_id'].nunique()),
                       'interpretation': 'fixed_pair_perturbation_not_full_cohort_reselection_or_clock_correction'})
    pd.DataFrame(shifts).to_csv(out/'clock_shift_sensitivity.csv', index=False)
    r = pd.read_csv(source/'record_list.csv', usecols=['subject_id', 'study_id', 'ecg_time'])
    m = pd.read_csv(source/'machine_measurements.csv', usecols=['subject_id', 'study_id', 'ecg_time'])
    r = r.merge(m, on=['subject_id', 'study_id'], validate='one_to_one', suffixes=('_record', '_machine'))
    delta = (pd.to_datetime(r.ecg_time_record, format='mixed')-pd.to_datetime(r.ecg_time_machine, format='mixed')).dt.total_seconds()
    mismatch = delta.ne(0)
    discrepancies = {'matched_keys': len(r), 'disagreement': int(mismatch.sum()),
                     'absolute_disagreement_seconds_quantiles': delta[mismatch].abs().quantile([0, .25, .5, .75, 1]).to_dict(),
                     'not_a_calibration_reference': 'Two recorded ECG timestamps do not establish true hospital clock alignment'}
    (out/'record_machine_timestamp_discrepancies.json').write_text(json.dumps(discrepancies, indent=2))
    age_report = {'censored_anchor_age_n': int(demographics.age_censored.sum()),
                  'age_encoding_changed_n': int(changed.sum()),
                  'censored_year_offsets': {str(k): int(v) for k, v in dem_sources.loc[dem_sources.age_censored, 'year_offset_from_anchor'].value_counts().items()},
                  'v1': 'Fixed 90 for anchor_age 91, ignoring elapsed calendar years in this subgroup',
                  'v2': 'anchor_age + year(landmark) - anchor_year; anchor_age 91 retained as deidentified proxy with censoring flag, not exact age'}
    (out/'age_encoding_audit.json').write_text(json.dumps(age_report, indent=2))
    missingness(expanded.to_numpy(float), y, list(expanded), 'expanded_v2').to_csv(out/'missingness_expanded.csv', index=False)
    receipt = {'status': 'STAGES_A_B_COMPLETE', 'n': len(index), 'positive': int(y.sum()),
               'legacy_n': len(legacy_y), 'legacy_positive': int(legacy_y.sum()),
               'legacy_patient_overlap': 'not_identifiable_without_legacy_row_identity',
               'routine_complete_n': int(groups['complete_routine'].sum()),
               'expanded_feature_count': len(expanded.columns), 'ecg_names': ecg_names,
               'raw_ecg_names': list(raw.columns), 'clinical_names': list(demographics),
               'routine_names': ['sodium', 'potassium', 'creatinine', 'hemoglobin'],
               'primary_biomarkers': ['ntprobnp', 'troponin_t'], 'optional_features': lab_names,
               'predictors_sha256': sha256(private/'predictors.npz'), 'frozen_primary_sha256': sha256(frozen/'cohort.npz'),
               'input_sha256': manifest['input_sha256'], 'software_versions': versions(),
               'new_module_sha256': {n: sha256(Path(__file__).parent/n) for n in ['staged_audit.py', 'clinical_predictors_v2.py']},
               'interpretation': 'Versioned predictors for unchanged membership; no new model fitting; ECG availability remains assumed'}
    (private/'manifest.json').write_text(json.dumps(receipt, indent=2))
    after = snapshot(protected)
    if before != after:
        raise RuntimeError('Protected cohort contents or modification times changed')
    (out/'preservation_receipt.json').write_text(json.dumps({'all_protected_files_unchanged': True,
        'protected_file_count': len(before), 'before': before, 'after': after}, indent=2))
    (out/'summary.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: receipt[k] for k in ['status', 'n', 'positive', 'expanded_feature_count']}, indent=2))
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='.')
    p.add_argument('--source', default='~/Downloads')
    p.add_argument('--output', default='results/cardiac_stages_ab_v2')
    p.add_argument('--private-output', default='private-data/cardiac_stages_ab_v2')
    a = p.parse_args()
    audit_and_prepare(a.root, a.source, a.output, a.private_output)
