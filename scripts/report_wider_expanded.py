import sys
sys.path.insert(0,'D:/CoPhotoMark_Revision')
from pathlib import Path
import json,csv,hashlib,argparse
from collections import defaultdict,Counter
import numpy as np
from cophotomark.io import read_jsonl,write_json

def band(n):
    return '1' if n==1 else '2' if n==2 else '3' if n==3 else '4-6' if n<=6 else '7+'

def features(record):
    enrolled={tuple(b) for b in record['per_face_bboxes']}
    f=[x for x in record['face_features'] if tuple(x['bbox']) in enrolled]
    if len(f)!=record['enrolled_faces']:raise ValueError('Enrollment feature match failed')
    edge=min(x['short_edge'] for x in f);blur=min(x['laplacian_variance'] for x in f)
    return ('64-127' if edge<128 else '128-255' if edge<256 else '256+',
            '<50' if blur<50 else '50-199' if blur<200 else '200+')

def dump(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def report(root):
    records=read_jsonl(root/'enrollment.jsonl'); scores=read_jsonl(root/'scores.jsonl')
    out=root/'report_expanded';out.mkdir(exist_ok=True)
    successful={r['source_id']:r for r in records if r['status']=='enrolled' and r['split']=='test'}
    rows=[r for r in scores if r['split']=='test']
    groups=defaultdict(list)
    for row in rows:
        for population in ['all_enrolled']+(['full_coverage'] if row['coverage']=='all_detected_faces' else []):
            groups[(population,row['mode'],row['attack'],band(row['enrolled_faces']))].append(row)
    rates=[]
    for (population,mode,attack,n),subset in sorted(groups.items()):
        ok=sum(r['decision']=='ACCEPTED' and r['status']=='ok' for r in subset)
        any_evidence=sum(any(f.get('valid') and f.get('count')==r['enrolled_faces'] and 1<=f.get('index',0)<=r['enrolled_faces'] and f.get('score',0)>=r['threshold'] for f in r.get('faces',[])) for r in subset)
        rates.append({'population':population,'mode':mode,'attack':attack,'enrolled_faces':n,'test_images':len(subset),
                      'accepted':ok,'acceptance_pct':100*ok/len(subset),
                      'undecidable':sum(r['decision']=='UNDECIDABLE' for r in subset),
                      'errors':sum(r['status']!='ok' for r in subset),
                      'any_expected_face_evidence_pct':100*any_evidence/len(subset),
                      'mean_verification_ms':float(np.mean([r['total_ms'] for r in subset if 'total_ms' in r])),
                      'mean_min_face_edge':float(np.mean([min(x['short_edge'] for x in successful[r['source_id']]['face_features']) for r in subset]))})
    dump(out/'natural_by_face_count.csv',rates)
    # Exact balancing by coarse min-enrolled-face size/blur strata. Select by
    # source hash only, before inspecting any acceptance score. Use 1/2/3/4+
    # groups to avoid presenting a single sparse high-N group as well-powered.
    strata=defaultdict(lambda:defaultdict(list))
    for sid,r in successful.items():
        if r['coverage']!='all_detected_faces':continue
        g=str(r['enrolled_faces']) if r['enrolled_faces']<4 else '4+'
        strata[features(r)][g].append(sid)
    selected=set();audit=[]
    for quality,sub in sorted(strata.items()):
        count=min(len(sub.get(g,[])) for g in ('1','2','3','4+'))
        for g in ('1','2','3','4+'):
            candidates=sorted(sub.get(g,[]),key=lambda s:hashlib.sha256((s+'20261114').encode()).hexdigest())
            for sid in candidates[:count]:
                selected.add(sid);audit.append({'source_id':sid,'group':g,'min_edge_band':quality[0],'min_laplacian_band':quality[1]})
    if audit:dump(out/'matched_sources.csv',audit)
    matched=[]
    for mode in ('auto','blind'):
        for attack in ('clean','jpeg95','jpeg50'):
            for n in ('1','2','3','4+'):
                s=[r for r in rows if r['source_id'] in selected and r['mode']==mode and r['attack']==attack and (str(r['enrolled_faces']) if r['enrolled_faces']<4 else '4+')==n]
                if s:matched.append({'mode':mode,'attack':attack,'enrolled_faces':n,'test_images':len(s),'accepted':sum(r['decision']=='ACCEPTED' and r['status']=='ok' for r in s),'acceptance_pct':100*np.mean([r['decision']=='ACCEPTED' and r['status']=='ok' for r in s])})
    if matched:dump(out/'quality_matched_by_face_count.csv',matched)
    cov=[]
    for split in ('validation','test'):
        for n in ('1','2','3','4-6','7+'):
            s=[r for r in records if r['split']==split and band(r['detected_faces'])==n]
            if s:cov.append({'split':split,'detected_face_band':n,'attempted_images':len(s),'successfully_enrolled':sum(r['status']=='enrolled' for r in s),'full_coverage_enrolled':sum(r['status']=='enrolled' and r['coverage']=='all_detected_faces' for r in s)})
    dump(out/'enrollment_coverage.csv',cov)
    table=[r'\begin{table}[t]',r'\centering',r'\caption{Acceptance rates (\%) by enrolled face count on the expanded WIDER test set. All successful enrollments are included; nonacceptances remain in the denominator. Auto uses metadata assistance with blind fallback.}',r'\label{tab:face_count_availability}',r'\begin{tabular}{lrrrrrrr}',r'\hline',r'& & \multicolumn{3}{c}{Auto} & \multicolumn{3}{c}{Blind} \\',r'Faces & Images & Clean & Q95 & Q50 & Clean & Q95 & Q50 \\',r'\hline']
    for n in ('1','2','3','4-6','7+'):
        lookup={(r['mode'],r['attack']):r for r in rates if r['population']=='all_enrolled' and r['enrolled_faces']==n}
        if not lookup:continue
        s=[n,str(next(iter(lookup.values()))['test_images'])]+[f"{lookup[m,a]['acceptance_pct']:.1f}" for m in ('auto','blind') for a in ('clean','jpeg95','jpeg50')]
        table.append(' & '.join(s)+r' \\')
    table += [r'\hline',r'\end{tabular}',r'\end{table}']
    (out/'table_face_count.tex').write_text('\n'.join(table),encoding='utf-8')
    full_table=table[:9]
    full_table[2]=r'\caption{Acceptance rates (\%) by enrolled face count on WIDER test images with all detected faces enrolled. Thresholds are fixed across groups. Auto uses metadata assistance with blind fallback.}'
    for n in ('1','2','3','4-6','7+'):
        lookup={(r['mode'],r['attack']):r for r in rates if r['population']=='full_coverage' and r['enrolled_faces']==n}
        if not lookup:continue
        fields=[n,str(next(iter(lookup.values()))['test_images'])]+[f"{lookup[m,a]['acceptance_pct']:.1f}" for m in ('auto','blind') for a in ('clean','jpeg95','jpeg50')]
        full_table.append(' & '.join(fields)+r' \\')
    full_table += [r'\hline',r'\end{tabular}',r'\end{table}']
    (out/'table_face_count_full_coverage.tex').write_text('\n'.join(full_table),encoding='utf-8')
    write_json(out/'summary.json',{'test_enrolled_sources':len(successful),'natural_rates':rates,'matched_rates':matched,
        'matched_total_sources':len(selected),'matched_protocol':'Full coverage only; exact balance across four N bands in joint bins of minimum enrolled face short edge (64-127/128-255/256+) and minimum Laplacian variance (<50/50-199/200+); uniform hash selection; no score filtering',
        'enrollment_failures':sum(r['status']!='enrolled' for r in records),
        'scores_sha256':hashlib.sha256((root/'scores.jsonl').read_bytes()).hexdigest(),
        'caution':'Matched subset controls coarse quality bins, not all confounders; any-face evidence is a diagnostic, not an alternative authentication success rate.'})
    print(json.dumps({'report':str(out),'test_sources':len(successful),'matched_sources':len(selected)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();report(Path(a.run))
