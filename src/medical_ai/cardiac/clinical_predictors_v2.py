"""Versioned clinical encoding and dictionary-verified laboratory audit.

No frozen cohort is modified. Age 91 is an anonymization sentinel, not exact age.
"""
import numpy as np
import pandas as pd

LAB_DEFINITIONS = {
    50983: ('sodium', 'Sodium', 'Blood', 'mEq/L', {'meq/l': 1., 'mmol/l': 1.}),
    50971: ('potassium', 'Potassium', 'Blood', 'mEq/L', {'meq/l': 1., 'mmol/l': 1.}),
    50912: ('creatinine', 'Creatinine', 'Blood', 'mg/dL', {'mg/dl': 1.}),
    51222: ('hemoglobin', 'Hemoglobin', 'Blood', 'g/dL', {'g/dl': 1.}),
    50963: ('ntprobnp', 'NTproBNP', 'Blood', 'pg/mL', {'pg/ml': 1., 'ng/l': 1.}),
    51003: ('troponin_t', 'Troponin T', 'Blood', 'ng/mL', {'ng/ml': 1., 'ng/l': .001}),
    51002: ('troponin_i_51002', 'Troponin I', 'Blood', 'ng/mL', {'ng/ml': 1., 'ng/l': .001}),
    52642: ('troponin_i_52642', 'Troponin I', 'Blood', 'ng/mL', {'ng/ml': 1., 'ng/l': .001}),
}


def encode_demographics(index, patients):
    if patients.subject_id.duplicated().any() or index.subject_id.duplicated().any():
        raise ValueError('Nonunique patient identities')
    d = index[['subject_id', 'decision_time']].merge(
        patients[['subject_id', 'anchor_age', 'anchor_year', 'gender']],
        on='subject_id', how='left', validate='one_to_one')
    if d[['anchor_age', 'anchor_year']].isna().any().any():
        raise ValueError('Missing anchored demographics')
    delta = pd.to_datetime(d.decision_time, format='mixed').dt.year-d.anchor_year
    censored = d.anchor_age.eq(91)
    # Standard deidentified year-based proxy, NOT an exact biological age for 91.
    age = d.anchor_age.astype(float)+delta
    features = pd.DataFrame({'age_year_proxy': age, 'age_censored': censored.astype(float),
                              'sex_female_recorded': d.gender.map({'F': 1., 'M': 0.})})
    d['year_offset_from_anchor'], d['age_year_proxy'], d['age_censored'] = delta, age, censored
    d['age_interpretation'] = np.where(censored, 'topcoded_anchor_age_plus_year_offset_not_exact_age',
                                      'approximate_year_based_age_birthday_unavailable')
    d['sex_interpretation'] = 'recorded_source_sex_static_attribute_assumption'
    return features, d


def verify_dictionary(dictionary):
    rows = []
    for item, (name, label, fluid, unit, conversions) in LAB_DEFINITIONS.items():
        d = dictionary.loc[dictionary.itemid.eq(item)]
        if len(d) != 1 or d.iloc[0].label != label or d.iloc[0].fluid != fluid:
            raise ValueError(f'Local dictionary mismatch for {item}; do not infer assay identity')
        rows.append({'itemid': item, 'feature': name, 'local_label': label, 'fluid': fluid,
                     'category': d.iloc[0].get('category', ''), 'canonical_unit': unit,
                     'serum_vs_plasma': 'not_resolved_by_supplied_dictionary_or_specimen_id',
                     'primary': item not in (51002, 52642)})
    return pd.DataFrame(rows)


def audit_laboratory_sources(index, provenance, dictionary):
    definitions = verify_dictionary(dictionary)
    if provenance.duplicated(['subject_id', 'itemid']).any():
        raise ValueError('More than one selected result per patient and assay')
    if provenance.labevent_id.duplicated().any():
        raise ValueError('Nonunique laboratory event identity')
    d = provenance.merge(index[['subject_id', 'decision_time']].rename(
        columns={'decision_time': 'index_decision_time'}), on='subject_id', how='left', validate='many_to_one')
    for c in ['charttime', 'storetime', 'decision_time', 'index_decision_time']:
        d[c] = pd.to_datetime(d[c], errors='coerce', format='mixed')
    if not d.decision_time.eq(d.index_decision_time).all():
        raise ValueError('Laboratory landmark differs from its linked patient index')
    flags = {
        'missing_event_or_specimen_identity': d[['labevent_id', 'specimen_id']].isna().any(axis=1),
        'missing_timestamp': d[['charttime', 'storetime', 'decision_time']].isna().any(axis=1),
        'result_not_before_landmark': ~(d.storetime < d.decision_time),
        'specimen_after_result': ~(d.charttime <= d.storetime),
        'specimen_before_24h_lookback': d.charttime < d.decision_time-pd.Timedelta(hours=24),
        'censored_text': d.value.fillna('').astype(str).str.contains(r'[<>]', regex=True),
    }
    expected = []
    for r in d.itertuples():
        if int(r.itemid) not in LAB_DEFINITIONS:
            raise ValueError('Unknown selected laboratory assay')
        name, label, fluid, unit, conversions = LAB_DEFINITIONS[int(r.itemid)]
        if r.feature != name:
            raise ValueError('Selected assay feature-name mismatch')
        normalized = str(r.valueuom).lower().replace(' ', '')
        if normalized not in conversions:
            raise ValueError(f'Unverified selected unit for assay {r.itemid}')
        expected.append(float(r.valuenum)*conversions[normalized])
    if not np.allclose(expected, d.canonical_value, equal_nan=False):
        raise ValueError('Canonical values disagree with source units and numeric values')
    violations = {k: int(v.sum()) for k, v in flags.items()}
    if any(violations.values()):
        raise ValueError(f'Laboratory provenance violations: {violations}')
    for item, spec in LAB_DEFINITIONS.items():
        selected = d.loc[d.itemid.eq(item)]
        row = definitions.itemid.eq(item)
        definitions.loc[row, 'selected_results'] = len(selected)
        definitions.loc[row, 'observed_source_units'] = '|'.join(sorted(selected.valueuom.dropna().unique()))
    return d, definitions, violations
