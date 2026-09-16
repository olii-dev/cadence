import json
import numpy as np
from cadence.model import ModelConfig
from cadence.train import train

def test_train_writes_loadable_checkpoint(tmp_path):
    data=np.tile(np.arange(31,dtype=np.uint16),20); path=tmp_path/'tokens.npy'; np.save(path,data)
    out=tmp_path/'smoke.pt'; meta=train(path,out,ModelConfig(32,16,1,2,32,0),steps=3,batch_size=2,learning_rate=.003)
    assert out.exists() and out.with_suffix('.json').exists()
    assert json.loads(out.with_suffix('.json').read_text())['steps']==3
    assert meta['final_loss'] < meta['initial_loss']


def test_train_reads_merged_binary(tmp_path):
    data=np.tile(np.arange(31,dtype=np.uint16),20); path=tmp_path/'tokens.bin'; data.tofile(path)
    out=tmp_path/'binary-smoke.pt'; meta=train(path,out,ModelConfig(32,16,1,2,32,0),steps=3,batch_size=2,learning_rate=.003)
    assert out.exists() and meta['final_loss'] < meta['initial_loss']
