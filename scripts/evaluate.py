import argparse
import datetime
import gc
import json
import math
import os
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from visdialch.dataloader import VisDialDataset
from visdialch.encoders import Encoder
from visdialch.decoders import Decoder
from visdialch.utils import process_ranks, scores_to_ranks, get_gt_ranks


parser = argparse.ArgumentParser()
VisDialDataset.add_cmdline_args(parser)

parser.add_argument_group('Evaluation related arguments')
parser.add_argument('-load_path', default='checkpoints/model.pth',
                        help='Checkpoint to load path from')
parser.add_argument('-split', default='val', choices=['val', 'test'],
                        help='Split to evaluate on')
parser.add_argument('-use_gt', action='store_true',
                        help='Whether to use ground truth for retrieving ranks')
parser.add_argument('-batch_size', default=12, type=int, help='Batch size')
parser.add_argument('-gpuid', nargs='+', type=int, default=-1,
                        help='GPU ids to use')
parser.add_argument('-cpu_workers', type=int, default=8,
                        help='Number of CPU workers for dataloading.')
parser.add_argument('-overfit', action='store_true',
                        help='Use a batch of only 5 examples, useful for debugging')

parser.add_argument_group('Submission related arguments')
parser.add_argument('-save_ranks', action='store_true',
                        help='Whether to save retrieved ranks')
parser.add_argument('-save_path', default='logs/ranks.json',
                        help='Path of json file to save ranks')

# ----------------------------------------------------------------------------
# input arguments and options
# ----------------------------------------------------------------------------

args = parser.parse_args()
if args.use_gt:
    if args.split == 'test':
        print("Warning: No ground truth for test split, changing use_gt to False.")
        args.use_gt = False
    elif args.split == 'val' and args.save_ranks:
        print("Warning: Cannot generate submission json if use_gt is True.")
        args.save_ranks = False

# seed for reproducibility
torch.manual_seed(0)
if args.gpuid[0] >= 0:
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda", args.gpuid[0])
else:
    device = torch.device("cpu")
print(f'Using {torch.cuda.device_count()} GPUs.')

# ----------------------------------------------------------------------------
# read saved model and args
# ----------------------------------------------------------------------------

components = torch.load(args.load_path)
model_args = components['model_args']
model_args.gpuid = args.gpuid[0]
model_args.batch_size = args.batch_size

# this is required by dataloader
args.img_norm = model_args.img_norm

# set this because only late fusion encoder is supported yet
args.concat_history = True

for arg in vars(args):
    print('{:<20}: {}'.format(arg, getattr(args, arg)))

# ----------------------------------------------------------------------------
# loading dataset wrapping with a dataloader
# ----------------------------------------------------------------------------

dataset = VisDialDataset(args, [args.split])
dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        collate_fn=dataset.collate_fn)

# iterations per epoch
setattr(args, 'iter_per_epoch',
    math.ceil(dataset.num_data_points[args.split] / args.batch_size))
print("{} iter per epoch.".format(args.iter_per_epoch))

# ----------------------------------------------------------------------------
# setup the model
# ----------------------------------------------------------------------------

encoder = Encoder(model_args)
encoder.load_state_dict(components['encoder'])

decoder = Decoder(model_args, encoder)
decoder.load_state_dict(components['decoder'])

encoder = encoder.to(device)
decoder = decoder.to(device)

encoder = nn.DataParallel(encoder, args.gpuid)
decoder = nn.DataParallel(decoder, args.gpuid)

print("Loaded model from {}".format(args.load_path))

# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------

print("Evaluation start time: {}".format(
    datetime.datetime.strftime(datetime.datetime.utcnow(), '%d-%b-%Y-%H:%M:%S')))
encoder.eval()
decoder.eval()

if args.use_gt:
    # ------------------------------------------------------------------------
    # calculate automatic metrics and finish
    # ------------------------------------------------------------------------
    all_ranks = []
    for i, batch in enumerate(tqdm(dataloader)):
        for key in batch:
            if not isinstance(batch[key], list):
                batch[key] = batch[key].to(device)

        with torch.no_grad():
            enc_out = encoder(batch)
            dec_out = decoder(enc_out, batch)
        ranks = scores_to_ranks(dec_out)
        gt_ranks = get_gt_ranks(ranks, batch['ans_ind'])
        all_ranks.append(gt_ranks)
    all_ranks = torch.cat(all_ranks, 0)
    process_ranks(all_ranks)
    gc.collect()
else:
    # ------------------------------------------------------------------------
    # prepare json for submission
    # ------------------------------------------------------------------------
    ranks_json = []
    for i, batch in enumerate(tqdm(dataloader)):
        for key in batch:
            if not isinstance(batch[key], list):
                batch[key] = batch[key].to(device)

        with torch.no_grad():
            enc_out = encoder(batch)
            dec_out = decoder(enc_out, batch)
        ranks = scores_to_ranks(dec_out)
        ranks = ranks.view(-1, 10, 100)

        for i in range(len(batch['img_ids'])):
            # cast into types explicitly to ensure no errors in schema
            if args.split == 'test':
                ranks_json.append({
                    'image_id': batch['img_ids'][i].item(),
                    'round_id': int(batch['num_rounds'][i]),
                    'ranks': list(ranks[i][batch['num_rounds'][i] - 1])
                })
            else:
                for j in range(batch['num_rounds'][i]):
                    ranks_json.append({
                        'image_id': batch['img_ids'][i].item(),
                        'round_id': int(j + 1),
                        'ranks': list(ranks[i][j])
                    })
        gc.collect()

if args.save_ranks:
    print("Writing ranks to {}".format(args.save_path))
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    json.dump(ranks_json, open(args.save_path, 'w'))