import sys
import argparse
import warnings

import torch
import numpy as np
import clip

import options.option_transformer as option_trans
import models.vqvae as vqvae
import models.t2m_trans as trans
from utils.motion_process import recover_from_ric
import visualization.plot_3d_global as plot_3d

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="Text-to-Motion (minimal CLI)")
    parser.add_argument("--text", type=str, required=True,
                        help="Text prompt (e.g. 'a person is jumping')")
    return parser.parse_args()


def main():
    cli_args = parse_args()
    clip_text = [cli_args.text]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------------------
    # Load internal args (unchanged)
    # ---------------------------
    sys.argv = ['GPT_eval_multi.py']
    args = option_trans.get_args_parser()

    args.dataname = 't2m'
    args.resume_pth = 'pretrained/VQVAE/net_last.pth'
    args.resume_trans = 'pretrained/VQTransformer_corruption05/net_best_fid.pth'
    args.down_t = 2
    args.depth = 3
    args.block_size = 51

    # ---------------------------
    # Load CLIP
    # ---------------------------
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False, download_root='./')
    clip.model.convert_weights(clip_model)
    clip_model.eval()

    for p in clip_model.parameters():
        p.requires_grad = False

    # ---------------------------
    # Load VQVAE
    # ---------------------------
    net = vqvae.HumanVQVAE(
        args,
        args.nb_code,
        args.code_dim,
        args.output_emb_width,
        args.down_t,
        args.stride_t,
        args.width,
        args.depth,
        args.dilation_growth_rate
    )

    print(f'Loading VQVAE checkpoint from {args.resume_pth}')
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    net.eval().to(device)

    # ---------------------------
    # Load Transformer
    # ---------------------------
    trans_encoder = trans.Text2Motion_Transformer(
        num_vq=args.nb_code,
        embed_dim=1024,
        clip_dim=args.clip_dim,
        block_size=args.block_size,
        num_layers=9,
        n_head=16,
        drop_out_rate=args.drop_out_rate,
        fc_rate=args.ff_rate
    )

    print(f'Loading Transformer checkpoint from {args.resume_trans}')
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
    trans_encoder.eval().to(device)

    # ---------------------------
    # Load normalization
    # ---------------------------
    mean = torch.from_numpy(
        np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')
    ).to(device)

    std = torch.from_numpy(
        np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')
    ).to(device)

    # ---------------------------
    # Inference
    # ---------------------------
    with torch.no_grad():
        text = clip.tokenize(clip_text, truncate=True).to(device)
        feat_clip_text = clip_model.encode_text(text).float()

        index_motion = trans_encoder.sample(feat_clip_text[0:1], False)
        pred_pose = net.forward_decoder(index_motion)

        pred_xyz = recover_from_ric((pred_pose * std + mean).float(), 22)
        xyz = pred_xyz.reshape(1, -1, 22, 3)

    # ---------------------------
    # Save + visualize (unchanged)
    # ---------------------------
    np.save('./cache/motion.npy', xyz.detach().cpu().numpy())
    print("Saved motion.npy")

    plot_3d.draw_to_batch(
        xyz.detach().cpu().numpy(),
        clip_text,
        ['./cache/example.gif']
    )
    print("Saved example.gif")


if __name__ == "__main__":
    main()