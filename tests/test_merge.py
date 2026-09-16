import json
from pathlib import Path
import numpy as np
from cadence.merge_shards import merge

def make_shard(root:Path,name:str,digest:str,values):
    p=root/name; p.mkdir(); np.save(p/'train.npy',np.array(values,dtype=np.uint16))
    (p/'train.jsonl').write_text(json.dumps({'source':name,'digest':digest,'offset':0,'length':len(values)-1})+'\n')
    for s in ('validation','test'): np.save(p/f'{s}.npy',np.array([],dtype=np.uint16)); (p/f'{s}.jsonl').write_text('')
    return p

def test_merge_deduplicates_and_rewrites_offsets(tmp_path):
    a=make_shard(tmp_path,'a','same',[1,2,3]); b=make_shard(tmp_path,'b','same',[1,2,3]); c=make_shard(tmp_path,'c','new',[4,5])
    out=tmp_path/'out'; result=merge([a,b,c],out)
    assert result['accepted']==2 and result['cross_shard_duplicates']==1
    rows=[json.loads(x) for x in (out/'train.jsonl').read_text().splitlines()]
    assert [r['offset'] for r in rows]==[0,3]
    assert np.fromfile(out/'train.bin',dtype=np.uint16).tolist()==[1,2,3,4,5]
