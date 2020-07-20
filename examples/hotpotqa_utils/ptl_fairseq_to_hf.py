import argparse
import torch
import pathlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--ptl_fairseq_in", type=str, required=True,
                        help="path to ckpt ptl file made with fairseq model)")
    parser.add_argument("--ptl_hf_out", type=str, required=True,
                        help="path to output ckpt ptl file with hf model)")
    args = parser.parse_args()

    print('loading ptl_fairseq checkpoint ... ')
    ptl_fairseq = torch.load(args.ptl_fairseq_in, map_location='cpu')
    hf_state_dict = {}
    num_layers = 24 if 'roberta.decoder.sentence_encoder.layers.23.self_attn.v_proj.bias' in ptl_fairseq['state_dict'].keys() else 12
    print('converting to ptl_hf checkoint ... ')
    for layer_id in range(num_layers):
        for matrix in ['weight', 'bias']:
            for hf_proj, fairseq_proj in [('key', 'k_proj'), ('query', 'q_proj'), ('value', 'v_proj')]:
                for fairseq_attn_type, hf_attn_type in [('_full', '_global'), ('', '')]:
                    fairseq_key = f'roberta.decoder.sentence_encoder.layers.{layer_id}.self_attn.{fairseq_proj}{fairseq_attn_type}.{matrix}'
                    hf_key = f'roberta.encoder.layer.{layer_id}.attention.self.{hf_proj}{hf_attn_type}.{matrix}'
                    hf_state_dict[hf_key] = ptl_fairseq['state_dict'][fairseq_key]
                    ptl_fairseq['state_dict'].pop(fairseq_key)

            fairseq_key = f'roberta.decoder.sentence_encoder.layers.{layer_id}.self_attn.out_proj.{matrix}'
            hf_key = f'roberta.encoder.layer.{layer_id}.attention.output.dense.{matrix}'
            hf_state_dict[hf_key] = ptl_fairseq['state_dict'][fairseq_key]
            ptl_fairseq['state_dict'].pop(fairseq_key)

            fairseq_key = f'roberta.decoder.sentence_encoder.layers.{layer_id}.self_attn_layer_norm.{matrix}'
            hf_key = f'roberta.encoder.layer.{layer_id}.attention.output.LayerNorm.{matrix}'
            hf_state_dict[hf_key] = ptl_fairseq['state_dict'][fairseq_key]
            ptl_fairseq['state_dict'].pop(fairseq_key)

            for hf_layer_location, fairseq_layer_location in [('intermediate.dense', 'fc1'), ('output.dense', 'fc2'), ('output.LayerNorm', 'final_layer_norm')]:
                fairseq_key = f'roberta.decoder.sentence_encoder.layers.{layer_id}.{fairseq_layer_location}.{matrix}'
                hf_key = f'roberta.encoder.layer.{layer_id}.{hf_layer_location}.{matrix}'
                hf_state_dict[hf_key] = ptl_fairseq['state_dict'][fairseq_key]
                ptl_fairseq['state_dict'].pop(fairseq_key)
    key_pairs = [
        ('roberta.embeddings.word_embeddings.weight', 'roberta.decoder.sentence_encoder.embed_tokens.weight'),
        ('roberta.embeddings.position_embeddings.weight', 'roberta.decoder.sentence_encoder.embed_positions.weight'),
        ('roberta.embeddings.LayerNorm.weight', 'roberta.decoder.sentence_encoder.emb_layer_norm.weight'),
        ('roberta.embeddings.LayerNorm.bias', 'roberta.decoder.sentence_encoder.emb_layer_norm.bias')
    ]
    for k in ptl_fairseq['state_dict']:
        if 'roberta' not in k:
            key_pairs.append((k, k))
    for hf_key, fairseq_key in key_pairs:
        hf_state_dict[hf_key] = ptl_fairseq['state_dict'][fairseq_key]
        ptl_fairseq['state_dict'].pop(fairseq_key)
    print('embedding size: ', hf_state_dict['roberta.embeddings.word_embeddings.weight'].shape)
    # hf_state_dict['roberta.embeddings.word_embeddings.weight'] = hf_state_dict['roberta.embeddings.word_embeddings.weight'][:50265]
    hidden_size = hf_state_dict['roberta.embeddings.word_embeddings.weight'].size(1)
    hf_state_dict['roberta.embeddings.token_type_embeddings.weight'] = torch.zeros(1, hidden_size)
    hf_state_dict['roberta.pooler.dense.weight'] = torch.zeros(1024, 1024)
    hf_state_dict['roberta.pooler.dense.bias'] = torch.zeros(1024)

    print('remaining keys ...')
    for key in ptl_fairseq['state_dict'].keys():
        print(key)
    ptl_fairseq['state_dict'] = hf_state_dict

    print('remove optimizer state ...')
    ptl_fairseq['optimizer_states'] = []

    print('saving ptl_hf checkoint ... ')
    pathlib.Path(args.ptl_hf_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ptl_fairseq, args.ptl_hf_out)