import torch
from cadence.tokenizer import MidiTokenizer

def test_grammar_mask_always_leaves_valid_tokens():
    tok=MidiTokenizer(); sequences=[['BOS'],['BOS','TRACK_0'],['BOS','TRACK_0','DRUMS'],['BOS','TRACK_0','DRUMS','PITCH_36'],['BOS','TRACK_0','DRUMS','PITCH_36','VELOCITY_2']]
    for sequence in sequences:
        allowed=tok.allowed_next_ids(sequence)
        assert allowed and all(0 <= i < len(tok.vocab) for i in allowed)
