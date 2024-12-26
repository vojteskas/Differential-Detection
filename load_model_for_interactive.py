#!/usr/bin/env python3
from argparse import Namespace
import torch

from classifiers.FFBase import FFBase
from common import build_model



def load_model_for_interactive():
    args: Namespace = Namespace()
    args.extractor = "XLSR_300M"
    args.classifier = "FF"
    
    args.processor = "MHFA"
    model_mhfa, _ = build_model(args)
    assert isinstance(model_mhfa, FFBase)
    model_mhfa.load_state_dict(torch.load("FF_MHFA.pt"))

    args.processor = "AASIST"
    model_aasist, _ = build_model(args)
    assert isinstance(model_aasist, FFBase)
    model_aasist.load_state_dict(torch.load("FF_AASIST_finetune_5.pt"))

    print("Models loaded successfully")
    print(model_mhfa, model_aasist)
    return model_mhfa, model_aasist

if __name__ == "__main__":
    load_model_for_interactive()
