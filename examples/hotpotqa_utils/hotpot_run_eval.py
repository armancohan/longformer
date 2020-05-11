# run trained hotpotqa model on dev and get evaluation numbers

import argparse
import torch
import json
import time
import subprocess
import glob
import argparse
import pathlib
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='path to the preprocessed input file')
    parser.add_argument('--outdir', help='path to write predictions')
    parser.add_argument('--gold', help='optional, path to gold file for evaluation', default=None)
    parser.add_argument('--checkpoint-dir', help='path to the checkpoint to evalute')
    parser.add_argument('--roberta', help='path to the roberta checkpoint to evalute')
    parser.add_argument('--num-gpus', help='how many gpus', default=1, type=str)
    parser.add_argument('--new-version', help='new data version', default=False, action='store_true')
    parser.add_argument('--loss-at-different-layers', help='predictions of sentences are different layers', default=False, action='store_true')
    parser.add_argument('--master-port', default=None)
    parser.add_argument('--overrides', default=None)
    parser.add_argument('--model-type', default='tvm_roberta')
    parser.add_argument('--batch-size', default=1)
    parser.add_argument('--simple-sentence-decode', default=None, action='store_true')
    args = parser.parse_args()

    roberta_dir = '/'.join(args.roberta.split('/')[:-1])
    roberta_name = args.roberta.split('/')[-1]

    checkpoints = glob.glob(glob.escape(args.checkpoint_dir) + '/*.ckpt')
    metrics = {}
    for checkpoint in checkpoints:
        # /model/version_0/checkpoints?ckpt_epoch_4.ckpt -> out/ckpt_epoch_4/
        ckpt_version = checkpoint.split("/")[-1][:-5]
        outdir = args.outdir + f'/{ckpt_version}'
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
        command = ["python", "scripts/hotpotqa.py",
                "--save-dir", "/tmp/test/",  # will be ignored
                "--save-prefix", "",
                "--train-file", args.input, # model won't be trained
                "--dev-file", args.input,
                "--num-workers", "0",
                "--model-path", roberta_dir,
                "--model-filename", roberta_name,
                "--model-type", args.model_type,
                "--separate-full-attention-projection", "--num-gpus", args.num_gpus, 
                "--or-softmax-loss", "--max-seq-len", str(4096) if args.model_type!='roberta' else str(512), 
                "--extra-attn", "--val-percent-check", str(1),
                "--create-new-weight-matrics",
                "--test-checkpoint", checkpoint,
                "--test-only",
                "--test-output-dir", outdir,
                "--test-file-orig", args.gold,
                "--test-percent-check", "1.0",
                "--include-par", "--include-paragraph", "--paragraph-loss", 
                "--multi-layer-classification-heads",
                "--batch-size", str(args.batch_size)]
        if args.overrides:
            command.extend(['--overrides', args.overrides])
        if args.master_port:
            os.environ["MASTER_PORT"]=str(args.master_port)
        if args.new_version:
            command.append('--question-type-classification-head')
        if args.loss_at_different_layers:
            command.append('--loss-at-different-layers')
        if args.simple_sentence_decode:
            command.append('--simple-sentence-decode')
        print(f'processing checkpoint: {checkpoint}')
        subprocess.run(command)
        with open(outdir + '/metrics.json') as fin:
            current_metrics = json.load(fin)
        metrics[ckpt_version] = current_metrics

    import pandas as pd
    if metrics:
        df = pd.DataFrame(metrics)
        df = df.mul(100).T
        df.to_csv(args.outdir + '/results.csv')

    print('done')

if __name__ == '__main__':
    main()
