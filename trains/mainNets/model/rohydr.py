import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...subNets_diffusion import BertTextEncoderDiffusion
from .scoremodel import ScoreNet, loss_fn, Euler_Maruyama_sampler
import functools
from .rcan import Group
from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from ...subNets_Adversarial import BertTextEncoderAdversarial
from ...subNets_Adversarial.transformers_encoder.transformer import TransformerEncoder

__all__ = ['ROHYDR']


class transformer_based(nn.Module):
    def __init__(self, args):
        super(transformer_based, self).__init__()
        self.args = args
        # BERT SUBNET FOR TEXT
        self.text_model = BertTextEncoderAdversarial(language=args.language, use_finetune=args.use_bert_finetune)
        #args.fusion_dim = args.fus_d_l+args.fus_d_a+args.fus_d_v
        args.fusion_dim = args.fus_d_l+args.fus_d_a+args.fus_d_v
        orig_d_l, orig_d_a, orig_d_v = args.feature_dims
        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(orig_d_l, args.fus_d_l, kernel_size=args.fus_conv1d_kernel_l, padding=(args.fus_conv1d_kernel_l-1)//2, bias=False)
        self.proj_a = nn.Conv1d(orig_d_a, args.fus_d_a, kernel_size=args.fus_conv1d_kernel_a, padding=(args.fus_conv1d_kernel_a-1)//2, bias=False)
        self.proj_v = nn.Conv1d(orig_d_v, args.fus_d_v, kernel_size=args.fus_conv1d_kernel_v, padding=(args.fus_conv1d_kernel_v-1)//2, bias=False)

        self.fusion_trans = TransformerEncoder(embed_dim=args.fus_d_l+args.fus_d_a+args.fus_d_v, num_heads=args.fus_nheads, layers=args.fus_layers, 
                                                attn_dropout=args.fus_attn_dropout, relu_dropout=args.fus_relu_dropout, res_dropout=args.fus_res_dropout,
                                                embed_dropout=args.fus_embed_dropout, attn_mask=args.fus_attn_mask)

    def forward(self, text_x, audio_x, video_x):
        proj_x_l = text_x.permute(2, 0, 1) # seq_len, batch_size, dl
        proj_x_a = audio_x.permute(2, 0, 1)
        proj_x_v = video_x.permute(2, 0, 1)

        trans_seq = self.fusion_trans(torch.cat((proj_x_l, proj_x_a, proj_x_v), axis=2))
        if type(trans_seq) == tuple:
            trans_seq = trans_seq[0]

        return trans_seq[0] # Utilize the [CLS] of text for full sequences representation.    

FUSION_MODULE_MAP = {
   'structure_one': transformer_based,
}

class FUSION(nn.Module):
    def __init__(self, args):
        super(FUSION, self).__init__()

        lastModel = FUSION_MODULE_MAP[args.fusion]
        self.Model = lastModel(args)

    def forward(self, text_x, audio_x, video_x):
        return self.Model(text_x, audio_x, video_x)

class decoder_v1(nn.Module):
    def __init__(self, args):
        super(decoder_v1, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(args.fusion_dim, args.rec_hidden_dim1),
            nn.Dropout(args.rec_dropout),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.rec_hidden_dim1, args.rec_hidden_dim2),
            nn.Dropout(args.rec_dropout),
            nn.BatchNorm1d(args.rec_hidden_dim2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.rec_hidden_dim2, args.fusion_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)
    
REC_MODULE_MAP = {
   'structure_one': decoder_v1,
}

class RECONSTRUCTION(nn.Module):
    def __init__(self, args):
        super(RECONSTRUCTION, self).__init__()

        lastModel = REC_MODULE_MAP[args.reconstruction]
        self.Model = lastModel(args)

    def forward(self, fusion_feature):
        return self.Model(fusion_feature)

class disc_two_class(nn.Module):

    def __init__(self, args):
        super(disc_two_class, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(args.fusion_dim),
            nn.Linear(args.fusion_dim, args.disc_hidden_dim1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.disc_hidden_dim1, args.disc_hidden_dim2),
            nn.Tanh(),
            nn.Linear(args.disc_hidden_dim2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)

DISC_MODULE_MAP = {
   'structure_one': disc_two_class,
}

class DISCRIMINATOR(nn.Module):
    def __init__(self, args):
        super(DISCRIMINATOR, self).__init__()

        lastModel = DISC_MODULE_MAP[args.discriminator]
        self.Model = lastModel(args)


    def forward(self, fusion_feature):
        return self.Model(fusion_feature)

class classifier_v1(nn.Module):

    def __init__(self, args):

        super(classifier_v1, self).__init__()
        self.norm = nn.BatchNorm1d(args.fusion_dim)
        self.drop = nn.Dropout(args.clf_dropout)
        self.linear_1 = nn.Linear(args.fusion_dim, args.clf_hidden_dim)
        self.linear_2 = nn.Linear(args.clf_hidden_dim, 1)
        # self.linear_3 = nn.Linear(hidden_size, hidden_size)
       
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, fusion_feature):
        normed = self.norm(fusion_feature)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        # y_2 = F.sigmoid(self.linear_2(y_1))
        y_2 = torch.sigmoid(self.linear_2(y_1))
        output = y_2 * self.output_range + self.output_shift

        return output

CLF_MODULE_MAP = {
   'structure_one': classifier_v1
}

class CLASSIFIER(nn.Module):
    def __init__(self, args):
        super(CLASSIFIER, self).__init__()

        lastModel = CLF_MODULE_MAP[args.classifier]
        self.Model = lastModel(args)

    def forward(self, fusion_feature):
        return self.Model(fusion_feature)

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse

device = 'cuda'

def marginal_prob_std(t, sigma):
    t = torch.as_tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
    return torch.as_tensor(sigma ** t, device=device)

class ROHYDR(nn.Module):
    def __init__(self, args):
        super(ROHYDR, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoderDiffusion(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        self.MSE = MSE()

        combined_dim = 2 * (self.d_l + self.d_a + self.d_v)

        output_dim = args.num_classes if args.train_mode == "classification" else 1

        sigma = 25.0
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma) 
        self.score_l = ScoreNet(marginal_prob_std=self.marginal_prob_std_fn)
        self.score_v = ScoreNet(marginal_prob_std=self.marginal_prob_std_fn)
        self.score_a = ScoreNet(marginal_prob_std=self.marginal_prob_std_fn)

        self.cat_lv = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)
        self.cat_la = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)
        self.cat_va = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)

        self.rec_l = nn.Sequential(
            nn.Conv1d(self.d_l, self.d_l*2, 1),
            Group(num_channels=self.d_l*2, num_blocks=20, reduction=16),
            nn.Conv1d(self.d_l*2, self.d_l, 1)
        )

        self.rec_v = nn.Sequential(
            nn.Conv1d(self.d_v, self.d_v*2, 1),
            Group(num_channels=self.d_v*2, num_blocks=20, reduction=16),
            nn.Conv1d(self.d_v*2, self.d_v, 1)
        )

        self.rec_a = nn.Sequential(
            nn.Conv1d(self.d_a, self.d_a*2, 1),
            Group(num_channels=self.d_a*2, num_blocks=20, reduction=16),
            nn.Conv1d(self.d_a*2, self.d_a, 1)
        )

        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')

        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')

        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        #adversial training and fusion
        args.fusion_dim=combined_dim
        self.reconstruction = RECONSTRUCTION(args)
        self.discriminator = DISCRIMINATOR(args)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text, audio, video, num_modal=None):
        with torch.no_grad():
            if self.use_bert:
                text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)
        with torch.no_grad():
            proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
            proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
            proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
            gt_l, gt_v, gt_a = proj_x_l, proj_x_v, proj_x_a

        #  random select modality
        modal_idx = [0, 1, 2]  # (0:text, 1:vision, 2:audio)
        ava_modal_idx = sample(modal_idx, num_modal)  # sample available modality
        if num_modal == 1:  # one modality is available
            if ava_modal_idx[0] == 0:  # has text
                conditions = proj_x_l
                loss_score_a = loss_fn(self.score_a, proj_x_a, self.marginal_prob_std_fn, condition=conditions)
                loss_score_v = loss_fn(self.score_v, proj_x_v, self.marginal_prob_std_fn, condition=conditions)
                loss_score_l = torch.tensor(0)
                proj_x_a = Euler_Maruyama_sampler(self.score_a, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                  device='cuda', condition=conditions)
                proj_x_v = Euler_Maruyama_sampler(self.score_v, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                  device='cuda', condition=conditions)
                #  refine modality
                proj_x_a = self.rec_a(proj_x_a)
                proj_x_v = self.rec_v(proj_x_v)
                loss_rec = self.MSE(proj_x_a, gt_a) + self.MSE(proj_x_v, gt_v)
            elif ava_modal_idx[0] == 1:  # has video
                conditions = proj_x_v
                loss_score_l = loss_fn(self.score_l, proj_x_l, self.marginal_prob_std_fn, condition=conditions)
                loss_score_a = loss_fn(self.score_a, proj_x_a, self.marginal_prob_std_fn, condition=conditions)
                loss_score_v = torch.tensor(0)
                # Generate samples from score-based models with the Euler_Maruyama_sampler
                proj_x_l = Euler_Maruyama_sampler(self.score_l, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                  device='cuda', condition=conditions)
                proj_x_a = Euler_Maruyama_sampler(self.score_a, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                  device='cuda', condition=conditions)
                #  refine modality
                proj_x_l = self.rec_l(proj_x_l)
                proj_x_a = self.rec_a(proj_x_a)
                loss_rec = self.MSE(proj_x_l, gt_l) + self.MSE(proj_x_a, gt_a)
            else:  # has audio
                conditions = proj_x_a
                loss_score_l = loss_fn(self.score_l, proj_x_l, self.marginal_prob_std_fn, condition=conditions)
                loss_score_v = loss_fn(self.score_v, proj_x_v, self.marginal_prob_std_fn, condition=conditions)
                loss_score_a = torch.tensor(0)
                # Generate samples from score-based models with the Euler_Maruyama_sampler
                proj_x_l = Euler_Maruyama_sampler(self.score_l, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                  device='cuda', condition=conditions)
                proj_x_v = Euler_Maruyama_sampler(self.score_v, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                  device='cuda', condition=conditions)
                #  refine modality
                proj_x_l = self.rec_l(proj_x_l)
                proj_x_v = self.rec_v(proj_x_v)
                loss_rec = self.MSE(proj_x_l, gt_l) + self.MSE(proj_x_v, gt_v)
        if num_modal == 2:  # two modalities are available
            if set(modal_idx) - set(ava_modal_idx) == {0}:  # L is missing (V,A available)
                conditions = self.cat_va(torch.cat([proj_x_v, proj_x_a], dim=1))  # cat two avail modalities as conditions
                loss_score_l = loss_fn(self.score_l, proj_x_l, self.marginal_prob_std_fn, condition=conditions)
                loss_score_v, loss_score_a = torch.tensor(0), torch.tensor(0)
                # Generate samples from score-based models with the Euler_Maruyama_sampler
                proj_x_l = Euler_Maruyama_sampler(self.score_l, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                  device='cuda', condition=conditions)
                #  refine modality
                proj_x_l = self.rec_l(proj_x_l)
                loss_rec = self.MSE(proj_x_l, gt_l)
            if set(modal_idx) - set(ava_modal_idx) == {1}:  # V is missing (L,A available)
                conditions = self.cat_la(torch.cat([proj_x_l, proj_x_a], dim=1))  # cat two avail modalities as conditions
                loss_score_v = loss_fn(self.score_v, proj_x_v, self.marginal_prob_std_fn, condition=conditions)
                loss_score_l, loss_score_a = torch.tensor(0), torch.tensor(0)
                # Generate samples from score-based models with the Euler_Maruyama_sampler
                proj_x_v = Euler_Maruyama_sampler(self.score_v, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                  device='cuda', condition=conditions)
                #  refine modality
                proj_x_v = self.rec_v(proj_x_v)
                loss_rec = self.MSE(proj_x_v, gt_v)
            if set(modal_idx) - set(ava_modal_idx) == {2}:  # A is missing (L,V available)
                conditions = self.cat_lv(torch.cat([proj_x_l, proj_x_v], dim=1))  # cat two avail modalities as conditions
                loss_score_a = loss_fn(self.score_a, proj_x_a, self.marginal_prob_std_fn, condition=conditions)
                loss_score_l, loss_score_v = torch.tensor(0), torch.tensor(0)
                # Generate samples from score-based models with the Euler_Maruyama_sampler
                proj_x_a = Euler_Maruyama_sampler(self.score_a, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                  device='cuda', condition=conditions)
                #  refine modality
                proj_x_a = self.rec_a(proj_x_a)
                loss_rec = self.MSE(proj_x_a, gt_a)
        if num_modal == 3:  # no missing
            loss_score_l, loss_score_v, loss_score_a = torch.tensor(0), torch.tensor(0), torch.tensor(0)
            loss_rec = torch.tensor(0)

        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        audio_gt=gt_a.permute(2, 0, 1)
        vision_gt=gt_v.permute(2, 0, 1)
        text_gt=gt_l.permute(2, 0, 1)

        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v) 
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs_lm = h_ls[-1]  # Take the last output for prediction

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs_lm = h_as[-1]

        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs_lm = h_vs[-1]

        last_hs_lm = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        
        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(text_gt, audio_gt, audio_gt)  
        h_l_with_vs = self.trans_l_with_v(text_gt, vision_gt, vision_gt)  
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs_gt = h_ls[-1]  # Take the last output for prediction

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(audio_gt, text_gt, text_gt)
        h_a_with_vs = self.trans_a_with_v(audio_gt, vision_gt, vision_gt)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs_gt = h_as[-1]

        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(vision_gt, text_gt, text_gt)
        h_v_with_as = self.trans_v_with_a(vision_gt, audio_gt, audio_gt)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs_gt = h_vs[-1]

        last_hs_gt = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)


        res = {
            'Fusion_lm': last_hs_lm,
            'Fusion_gt': last_hs_gt,
            'loss_score_l': loss_score_l,
            'loss_score_v': loss_score_v,
            'loss_score_a': loss_score_a,
            'Diff_t': proj_x_l,
            'Diff_a': proj_x_a,
            'Diff_v': proj_x_v,
            'gt_l':text_gt,
            'gt_a':audio_gt,
            'gt_v':vision_gt,
            'loss_rec': loss_rec,
            'ava_modal_idx': ava_modal_idx,
        }
        return res
    def getoutput(self,gt,lm):
         # rec prediction
        last_hs_lm=lm
        last_hs_gt=gt
        
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs_lm), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs_lm
        output_lm = self.out_layer(last_hs_proj)


        # gt prediction
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs_gt), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs_gt
        output_gt = self.out_layer(last_hs_proj)

        res = {
            'M_lm': output_lm,
            'M_gt':output_gt,
        }
        return res
    
    def FusionAndOutput(self,proj_x_l,proj_x_a,proj_x_v,text_gt,audio_gt,vision_gt):
        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs_lm = h_ls[-1]  # Take the last output for prediction

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs_lm = h_as[-1]

        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs_lm = h_vs[-1]

        last_hs_lm = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs_lm), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs_lm
        output_lm = self.out_layer(last_hs_proj)

        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(text_gt, audio_gt, audio_gt)  # Dimension (L, N, d_l)
        h_l_with_vs = self.trans_l_with_v(text_gt, vision_gt, vision_gt)  # Dimension (L, N, d_l)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs_gt = h_ls[-1]  # Take the last output for prediction

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(audio_gt, text_gt, text_gt)
        h_a_with_vs = self.trans_a_with_v(audio_gt, vision_gt, vision_gt)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs_gt = h_as[-1]

        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(vision_gt, text_gt, text_gt)
        h_v_with_as = self.trans_v_with_a(vision_gt, audio_gt, audio_gt)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs_gt = h_vs[-1]

        last_hs_gt = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs_gt), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs_gt
        output_gt = self.out_layer(last_hs_proj)


        res = {
            'Fusion_lm': last_hs_lm,
            'Fusion_gt': last_hs_gt,
            'Diff_t': proj_x_l,
            'Diff_a': proj_x_a,
            'Diff_v': proj_x_v,
            'gt_l':text_gt,
            'gt_a':audio_gt,
            'gt_v':vision_gt,
            'M_lm': output_lm,
            'M_gt':output_gt,
        }
        return res
