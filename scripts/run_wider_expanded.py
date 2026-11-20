import sys
from pathlib import Path
sys.path.insert(0,'D:/CoPhotoMark_Revision')
import argparse, hashlib, json, secrets, time
from collections import Counter
import cv2
import numpy as np
from cophotomark.detector import InsightFaceDetector
from cophotomark.dataset import inventory, split_sources
from cophotomark.experiments import enroll_manifest, load_key, environment
from cophotomark.engine import verify
from cophotomark.attacks import transform
from cophotomark.io import read_json, read_jsonl, write_json, write_jsonl, read_image, sha256_file
from cophotomark.protocol import Config


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config_path=Path('D:/CoPhotoMark_Revision/configs/full64_step18.json')
    config=Config(**read_json(config_path))
    detector=InsightFaceDetector(provider='cpu')
    paths=sorted(p for p in Path('D:/CoPhotoMark_Revision/assets/wider').rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.webp','.tif','.tiff'})
    fingerprint=hashlib.sha256(json.dumps([(str(p),sha256_file(p)) for p in paths]).encode()).hexdigest()
    protocol={'input':'D:/CoPhotoMark_Revision/assets/wider','image_files':len(paths),'input_fingerprint':fingerprint,
        'config_sha256':config.fingerprint,'seed':20261114,'thresholds':{'auto':1.0,'blind':.8125},
        'conditions':['clean','jpeg95','jpeg50'],'provider':'cpu','selection':'all readable deduplicated sources; no clean-accept filtering',
        'threshold_selection':'fixed before test from revision operating points; not fitted to face counts or expanded test outcomes',
        'note':'Natural group-size association; not a randomized causal group-size study'}
    if (out/'protocol.json').exists() and read_json(out/'protocol.json')!=protocol:raise ValueError('Input/protocol changed; select new output')
    write_json(out/'protocol.json',protocol)
    # Inventory individually so interrupted scans retain progress.
    cached=read_jsonl(out/'inventory.jsonl') if (out/'inventory.jsonl').exists() else []
    done={r['path'] for r in cached}
    from cophotomark.watermark import block_count
    from cophotomark.engine import crop
    for idx,path in enumerate(paths):
        if str(path) in done:continue
        digest=sha256_file(path)
        row={'path':str(path),'relative_path':path.relative_to(Path(protocol['input'])).as_posix(),
             'dataset':'wider_expanded','source_id':'wider_expanded:'+digest[:20],
             'source_sha256':digest,'group_id':'wider_expanded:'+digest,'config_sha256':config.fingerprint}
        try:
            image=read_image(path);boxes=detector.detect(image)
            features=[]
            for b in boxes:
                roi=crop(image,b);g=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
                features.append({'bbox':b,'short_edge':min(roi.shape[:2]),'laplacian_variance':float(cv2.Laplacian(g,cv2.CV_32F).var()),'blocks':block_count(roi,config)})
            row.update(status='ok',shape=list(image.shape),detected_faces=len(boxes),detected_boxes=boxes,
                       feasible_roi_count=sum(f['short_edge']>=config.min_roi and f['blocks']>=config.min_blocks for f in features),face_features=features)
        except Exception as e:row.update(status='error',reason=str(e))
        cached.append(row);write_jsonl(out/'inventory.jsonl',cached)
        if (idx+1)%20==0:print('Inventory',idx+1,'/',len(paths),flush=True)
    sources_path=out/'sources.jsonl'
    if not sources_path.exists():
        sources=split_sources([r for r in cached if r.get('status')=='ok' and r.get('detected_faces',0)>=1],20261114,.2,0)
        write_jsonl(sources_path,sources)
    sources=read_jsonl(sources_path)
    print('Sources',len(sources),'count distribution',dict(Counter(r['detected_faces'] for r in sources)),flush=True)
    key=out/'keys/experiment.json'
    if not key.exists():write_json(key,{'master_key_hex':secrets.token_bytes(32).hex()})
    master=load_key(key)
    records=[]
    for index,source in enumerate(sources):
        checkpoint=out/'source_enrollments'/source['source_id'].split(':')[1]/'enrollment.jsonl'
        if checkpoint.exists():rec=read_jsonl(checkpoint)[0]
        else:rec=enroll_manifest([source],checkpoint.parent,detector,config,[key],'Expanded WIDER face count study',20261114)[0]
        records.append(rec)
        write_jsonl(out/'enrollment.jsonl',records)
        if (index+1)%10==0:print('Enrollment',index+1,'/',len(sources),'successful',sum(r['status']=='enrolled' for r in records),flush=True)
    signature=hashlib.sha256(json.dumps({'config':config.fingerprint,'detector':{k:v for k,v in detector.info.items() if k!='initialization_ms'},'protocol':protocol},sort_keys=True).encode()).hexdigest()
    results=[]
    for index,rec in enumerate(records):
        if rec['status']!='enrolled':continue
        checkpoint=out/'source_scores'/(rec['source_id'].split(':')[1]+'.jsonl')
        if checkpoint.exists():results+=read_jsonl(checkpoint);continue
        image=read_image(rec['watermarked_path']);claim=read_json(rec['claim_path']);batch=[]
        for attack in protocol['conditions']:
            edited=image.copy() if attack=='clean' else transform(image,{'name':'jpeg','quality':95 if attack=='jpeg95' else 50},np.random.default_rng(20261114))
            for mode,threshold in protocol['thresholds'].items():
                row={k:rec[k] for k in ('source_id','source_sha256','group_id','split','dataset','key_id','enrolled_faces','coverage','excluded_faces')}
                row.update(trial_id=hashlib.sha256((rec['source_id']+attack).encode()).hexdigest()[:24],attack=attack,label='genuine',mode=mode,threshold=threshold,evaluation_signature=signature,config_sha256=config.fingerprint)
                try:
                    answer=verify(edited,detector,config,master,claim,mode=mode,threshold=threshold)
                    row.update(answer);row.update(mode=mode,actual_mode=answer['mode'],status='ok')
                except Exception as e:row.update(status='error',decision='UNDECIDABLE',reason=f'{type(e).__name__}: {e}')
                batch.append(row)
        write_jsonl(checkpoint,batch);results+=batch
        write_jsonl(out/'scores.jsonl',results)
        print('Evaluated source',index+1,'/',len(records),'N=',rec['enrolled_faces'],flush=True)
    write_jsonl(out/'scores.jsonl',results)
    write_json(out/'environment.json',environment(detector))
    write_json(out/'completion.json',{'sources':len(sources),'enrolled':sum(r['status']=='enrolled' for r in records),'score_rows':len(results),'test_sources':len({r['source_id'] for r in results if r['split']=='test'})})
    print('DONE',str(out),flush=True)

if __name__=='__main__':main()
