import torch

import argparse
from transformers import AutoModel
from longformer import Longformer
import pathlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cdlm', help='cdlm checkpoint (pytorch_model.bin) file')
    parser.add_argument('--longformer', help='path to longformer-[base-large]-4096')
    parser.add_argument('--output', help='lf checkpoint')
    args = parser.parse_args()

    cdlm_state_dict = torch.load(args.cdlm)
    lf = Longformer.from_pretrained(args.longformer)
    lf.resize_token_embeddings(cdlm_state_dict['longformer.embeddings.word_embeddings.weight'].shape[0])

    new_state_dict = {k.replace('longformer.',''): v for k, v in cdlm_state_dict.items()}
    # removing 'lm_head' params from checkpoint
    new_state_dict = {k: v for k, v in new_state_dict.items() if 'lm_head' not in k}
    # adding pooler dense weight and bias to the checkpoint
    new_state_dict['pooler.dense.weight'] = lf.pooler.dense.weight
    new_state_dict['pooler.dense.bias'] = lf.pooler.dense.bias

    assert len([k for k in new_state_dict if k not in lf.state_dict()]) == 0
    assert len([k for k in lf.state_dict() if k not in new_state_dict]) == 0

    print('loading state dict from hugginface model')
    lf.load_state_dict(new_state_dict)

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    lf.save_pretrained(args.output)
    print('done')


if __name__ == '__main__':
    main()
