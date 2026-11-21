"""Aggregate-only vector figures and executable provenance."""
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def versions():
    return {'python': platform.python_version(), 'platform': platform.platform(),
            **{n: importlib.metadata.version(n) for n in
               ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib']}}


def figures(output):
    out = Path(output)
    synthetic = json.loads((out/'run.json').read_text()).get('status') == 'SYNTHETIC_SMOKE_TEST'
    plt.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=plt.get_cmap('tab20').colors)
    metrics = pd.read_csv(out/'metrics.csv').drop_duplicates('model')
    rocpr = pd.read_csv(out/'roc_pr.csv')
    for kind, xlabel, ylabel in [('roc', 'False positive rate', 'True positive rate'),
                                 ('pr', 'Recall', 'Precision')]:
        fig, ax = plt.subplots(figsize=(7.2, 4.8), layout='constrained')
        if synthetic:
            fig.suptitle('Synthetic software validation — not clinical results', fontsize=10)
        for model, group in rocpr.loc[rocpr.curve.eq(kind)].groupby('model', sort=False):
            ax.plot(group.x, group.y, label=model)
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1., 1.))
        fig.savefig(out/f'{kind}.pdf', bbox_inches='tight')
        fig.savefig(out/f'{kind}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    curves = pd.read_csv(out/'decision_curves.csv')
    bins = pd.read_csv(out/'calibration_bins.csv')
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    for filename, ylabel, xcol, ycol, data in [
        ('calibration', 'Observed fraction with LVEF <40%', 'predicted', 'observed', bins),
        ('decision_curve', 'Net benefit', 'threshold', 'net_benefit', curves)]:
        fig, ax = plt.subplots(figsize=(7.2, 4.8), layout='constrained')
        if synthetic:
            fig.suptitle('Synthetic software validation — not clinical results', fontsize=10)
        for model, group in data.groupby('model', sort=False):
            ax.plot(group[xcol], group[ycol], marker='o', markersize=3, label=model)
        if filename == 'calibration':
            ax.plot([0, 1], [0, 1], color='0.6', linestyle='--')
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Predicted probability')
        else:
            g = data.drop_duplicates('threshold')
            ax.plot(g.threshold, g.treat_all, '--', color='0.5', label='Treat all')
            ax.axhline(0, color='0.5', linestyle=':', label='Treat none')
            ax.set_xlabel('Decision threshold probability')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1., 1.))
        fig.savefig(out/f'{filename}.pdf', bbox_inches='tight')
        fig.savefig(out/f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), layout='constrained')
    if synthetic:
        fig.suptitle('Synthetic software validation — not clinical results', fontsize=10)
    for _, row in metrics.iterrows():
        ax.scatter(row.mean_incremental_cost, row.brier)
        ax.annotate(row.model, (row.mean_incremental_cost, row.brier), fontsize=6)
    ax.set(xlabel='Assumed mean incremental test cost (scenario units)', ylabel='Outer OOF Brier score')
    fig.savefig(out/'cost_performance.pdf', bbox_inches='tight')
    fig.savefig(out/'cost_performance.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    (out/'versions.json').write_text(json.dumps(versions(), indent=2))
