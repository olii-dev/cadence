"""Merge independently processed shards with global deduplication and fixed offsets."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

SPLITS=("train","validation","test")
def merge(inputs:list[Path],output:Path):
    output.mkdir(parents=True,exist_ok=True); seen=set(); totals={s:0 for s in SPLITS}; accepted=duplicates=0
    handles={s:(output/f"{s}.bin").open("wb") for s in SPLITS}; manifests={s:(output/f"{s}.jsonl").open("w") for s in SPLITS}
    try:
        for shard in inputs:
            for split in SPLITS:
                path = shard / f"{split}.bin"
                data = np.memmap(path, dtype=np.uint16, mode="r") if path.stat().st_size else np.array([], dtype=np.uint16)
                cursor=0
                lines=(shard/f"{split}.jsonl").read_text().splitlines()
                for line in lines:
                    item=json.loads(line); length=item["length"]
                    # Shards store one extra EOS separator after each sequence.
                    sequence=np.asarray(data[cursor:cursor+length+1],dtype=np.uint16); cursor += length+1
                    if item["digest"] in seen: duplicates += 1; continue
                    seen.add(item["digest"]); item["offset"]=totals[split]; item["shard"]=shard.name
                    sequence.tofile(handles[split]); manifests[split].write(json.dumps(item)+"\n")
                    totals[split]+=len(sequence); accepted+=1
    finally:
        for f in handles.values(): f.close()
        for f in manifests.values(): f.close()
    summary={"shards":[str(p) for p in inputs],"accepted":accepted,"cross_shard_duplicates":duplicates,"tokens_with_separators":totals}
    (output/'merge-stats.json').write_text(json.dumps(summary,indent=2)); return summary

def main():
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path); p.add_argument('inputs',nargs='+',type=Path); a=p.parse_args(); print(json.dumps(merge(a.inputs,a.output),indent=2))
if __name__=='__main__': main()
