"""Read-only stratification of recorded test decisions; never tunes thresholds."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def load(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def summarize(rows, modes=('auto', 'blind'), attacks=('clean', 'jpeg95'), repeats=2000):
    groups = defaultdict(list)
    seen = set()
    thresholds = defaultdict(set)
    for row in rows:
        if row.get('split') != 'test' or row.get('label') != 'genuine':
            continue
        if row.get('mode') not in modes or row.get('attack') not in attacks:
            continue
        n = row.get('enrolled_faces')
        if type(n) is not int or n < 1:
            raise ValueError('Missing positive enrollment face count')
        identity = (row['source_id'], row.get('key_id'), row['trial_id'], row['mode'])
        if identity in seen:
            raise ValueError('Duplicate test observation')
        seen.add(identity)
        if row.get('decision') not in ('ACCEPTED', 'REJECTED', 'UNDECIDABLE'):
            raise ValueError('Unknown decision')
        if row.get('status') == 'ok':
            if row.get('threshold') is None:
                raise ValueError('Successful score row has no recorded threshold')
            thresholds[(row.get('config_sha256'), row['mode'])].add(row['threshold'])
        for grouping, band in [('exact', str(n)), ('coarse', '>=4' if n >= 4 else str(n))]:
            for condition in (row['attack'], 'all_selected'):
                groups[(grouping, row['mode'], condition, band)].append(row)
    if not groups:
        raise ValueError('No matching genuine test observations')
    if any(len(v) != 1 for v in thresholds.values()):
        raise ValueError('Threshold changes within a mode/config; separate runs before reporting')
    result = []
    for (grouping, mode, attack, band), subset in sorted(groups.items()):
        by_source = defaultdict(list)
        for r in subset:
            by_source[r['source_id']].append(r)
        # Equal source weighting prevents keys/repetitions from inflating sample size.
        source_rates = np.array([np.mean([r['decision']=='ACCEPTED' and r.get('status')=='ok' for r in s]) for s in by_source.values()])
        low = high = None
        if len(source_rates) >= 2:
            rng = np.random.default_rng(20261114)
            draws = rng.choice(source_rates, size=(repeats,len(source_rates)),replace=True).mean(axis=1)
            low, high = np.quantile(draws,[.025,.975])*100
        times = [r['total_ms'] for r in subset if r.get('status')=='ok' and isinstance(r.get('total_ms'),(float,int))]
        result.append(dict(grouping=grouping,mode=mode,attack=attack,enrolled_faces=band,
            source_images=len(by_source),trials=len(subset),
            accepted=sum(r['decision']=='ACCEPTED' and r.get('status')=='ok' for r in subset),
            rejected=sum(r['decision']=='REJECTED' for r in subset),
            undecidable=sum(r['decision']=='UNDECIDABLE' for r in subset),
            errors=sum(r.get('status')!='ok' for r in subset),
            acceptance_pct=float(source_rates.mean()*100),ci95_low_pct=None if low is None else float(low),
            ci95_high_pct=None if high is None else float(high),
            runtime_valid_trials=len(times),mean_verification_ms=float(np.mean(times)) if times else None,
            actual_metadata_trials=sum(r.get('actual_mode')=='metadata' for r in subset),
            actual_blind_trials=sum(r.get('actual_mode')=='blind' for r in subset),
            full_coverage_trials=sum(r.get('coverage')=='all_detected_faces' for r in subset)))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--attacks',nargs='+',default=['clean','jpeg95'])
    p.add_argument('--bootstrap',type=int,default=2000)
    args=p.parse_args(argv)
    if args.bootstrap < 100:
        p.error('--bootstrap must be at least 100')
    from cophotomark.io import writable
    out=writable(args.output/'summary.json').parent
    if any(out.glob('*')):
        raise ValueError('Select an empty output directory to preserve previous reports')
    rows=load(args.scores)
    selected=[r for r in rows if r.get('split')=='test' and r.get('label')=='genuine' and r.get('attack') in args.attacks and r.get('mode') in ('auto','blind')]
    signatures={r.get('evaluation_signature') for r in selected}
    configs={r.get('config_sha256') for r in selected if r.get('status')=='ok'}
    if len(signatures)!=1 or None in signatures or len(configs)!=1 or None in configs:
        raise ValueError('Report requires a single recorded evaluation signature and protocol config')
    result=summarize(rows,attacks=tuple(args.attacks),repeats=args.bootstrap)
    with (out/'availability_by_face_count.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    # Main table uses separate conditions; pooled outputs remain descriptive in CSV only.
    tex=[r'\begin{table}[t]',r'\centering',r'\caption{Test acceptance rates stratified by enrolled face count. Auto denotes metadata-assisted verification with blind fallback. Rates are source-image averages; rejected, undecidable, and error trials remain nonacceptances.}',r'\label{tab:face_count_availability}',r'\begin{tabular}{llrrr}',r'\hline',r'Mode / condition & Faces & Images & Trials & Accept. (\%) \\',r'\hline']
    for r in result:
        if r['grouping']=='coarse' and r['attack']!='all_selected':
            band=r'$\geq4$' if r['enrolled_faces']=='>=4' else r['enrolled_faces']
            tex.append(f"{r['mode']} / {r['attack']} & {band} & {r['source_images']} & {r['trials']} & {r['acceptance_pct']:.1f} " + r'\\')
    tex += [r'\hline',r'\end{tabular}',r'\end{table}']
    (out/'table_face_count.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    summary={'scores':str(args.scores.resolve()),'scores_sha256':hashlib.sha256(args.scores.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'evaluation_signature':next(iter(signatures)),'config_sha256':next(iter(configs)),
        'attacks':args.attacks,'thresholds_by_mode':{m:sorted({r['threshold'] for r in selected if r['mode']==m and r.get('threshold') is not None}) for m in ('auto','blind')},
        'independent_test_sources':len({r['source_id'] for r in selected}),
        'method':'existing test decisions; source-weighted rates; bootstrap by source; no threshold tuning; all failures retained',
        'limitations':'Descriptive face-count association, not causal effect; no forced monotonic trend. Exact N=1 absent if not sampled; sparse bands explicitly shown. Historical results do not imply current code was rerun.',
        'rows':result}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for mode in ('auto','blind'):
        for attack in args.attacks:
            s=sorted([r for r in result if r['grouping']=='exact' and r['mode']==mode and r['attack']==attack],key=lambda r:int(r['enrolled_faces']))
            if s:ax.plot([int(r['enrolled_faces']) for r in s],[r['acceptance_pct'] for r in s],marker='o',label=f'{mode} / {attack}')
    ax.set(xlabel='Enrolled faces',ylabel='Benign acceptance (%)',ylim=(-3,103));ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(out/'acceptance_by_face_count.png',dpi=180);fig.savefig(out/'acceptance_by_face_count.pdf');plt.close(fig)
    print(json.dumps({'output':str(out),'test_sources':summary['independent_test_sources'],'rows':len(result)}))


if __name__=='__main__':
    main()
