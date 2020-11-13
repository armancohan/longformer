import torch
import argparse
import pathlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='')
    parser.add_argument('--output', help='')
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--copy-global', action='store_true', default=False)
    args = parser.parse_args()
    checkpoint = torch.load(args.input, map_location='cpu')
    model_checkpoint = checkpoint['state_dict']
    model_checkpoint_without_prefix = {k[6:]: v for k, v in model_checkpoint.items()}
    if 'large' in args.input and args.layers == 12:
        print('looks like you are converting a large model, set --layers to 24')
    # reset the global matrices
    if args.copy_global:
        for layer in range(args.layers):
            for v1 in ['query', 'key', 'value']:
                for v2 in ['weight', 'bias']:
                    v_local = 'roberta.encoder.layer.{}.attention.self.{}.{}'.format(layer, v1, v2)
                    v_global = 'roberta.encoder.layer.{}.attention.self.{}_global.{}'.format(layer, v1, v2)
                    model_checkpoint_without_prefix[v_global] = model_checkpoint_without_prefix[v_local]
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_checkpoint_without_prefix, args.output)


if __name__ == '__main__':
    main()