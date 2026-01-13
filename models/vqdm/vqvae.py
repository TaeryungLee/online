import torch.nn as nn
from baselines.T2M_GPT.models.encdec import Encoder, Decoder
from baselines.T2M_GPT.models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from models.vqdm.modules.gumbel_quantizer import GumbelQuantize
# from utils.losses import ReConsLoss
# from data_loaders.humanml.scripts.motion_process import recover_from_ric
# from utils.rotation2xyz import Rotation2xyz
import torch

class ReConsLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints=22):
        super(ReConsLoss, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
        
        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints
        self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4
        
    def forward(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred, motion_gt)
        return loss
    
    def forward_vel(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4], motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])
        return loss


def sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))

class VQVAE_251(nn.Module):
    def __init__(self,
                 args,
                 inv_transform,
                 device,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.args = args 
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = args.quantizer
        if args.dataset in ['ntu', 'humanml']:
            self.input_dim = 272
            self.data_rep = 'hml_vec'
            self.num_joints = 272
            self.num_feats = 1
        elif args.dataset in ['humanact12', 'uestc']:
            self.input_dim = 150
            self.data_rep = 'rot6d'
            self.num_joints = 25
            self.num_feats = 6


        self.encoder = Encoder(self.input_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(self.input_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        if args.quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
        elif args.quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif args.quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)
        elif args.quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim, args)
        elif args.quantizer == "gumbel":
            self.quantizer = GumbelQuantize(output_emb_width, code_dim, nb_code, kl_weight=1, use_vqinterface=False)

        self.reconloss = ReConsLoss(args.recons_loss)
        self.l1_smooth = torch.nn.SmoothL1Loss()
        self.inv_transform = inv_transform
        # self.rot2xyz = Rotation2xyz(device=device, dataset=self.args.dataset)
        self.get_xyz = lambda sample: self.rot2xyz(sample, mask=None, pose_rep=self.data_rep, translation=True,
                                              glob=True,
                                              jointstype='smpl',  # 3.4 iter/sec
                                              vertstrans=False)

        self.lambda_commit = args.lambda_commit
        self.lambda_gumbel_kl = args.lambda_gumbel_kl
        self.lambda_hml_joint = args.lambda_hml_joint
        self.lambda_param_vel = args.lambda_param_vel
        self.lambda_joint = args.lambda_joint
        self.lambda_joint_vel = args.lambda_joint_vel
        self.lambda_fc = args.lambda_fc


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx

    def masked_l1_smooth(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        loss = self.l1_smooth(a, b)
        loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements

        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        # print('mask', mask.shape)
        # print('non_zero_elements', non_zero_elements)
        # print('loss', loss)
        mse_loss_val = loss / non_zero_elements
        # print('mse_loss_val', mse_loss_val)
        return mse_loss_val.mean()

    def forward(self, x, mask=None, mode='test'):
        B, T = x.shape[:2]
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        
        loss = {}
        
        ## quantization
        if self.args.quantizer == "gumbel":
            x_quantized, loss_kl, quant_idx = self.quantizer(x_encoder)
            perplexity = 0
            loss['gumbel_kl'] = loss_kl
        else:
            x_quantized, loss_commit, perplexity  = self.quantizer(x_encoder)
            loss['commit'] = loss_commit

        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)

        if mode == 'train':
            loss['param'] = self.reconloss(x[mask], x_out[mask])

            if self.lambda_param_vel > 0:                
                gt_param_vel = x[:, 1:] - x[:, :-1]
                out_param_vel = x_out[:, 1:] - x_out[:, :-1]
                loss['param_vel'] = self.reconloss(gt_param_vel[mask[:, :-1]], out_param_vel[mask[:, :-1]])
            if self.lambda_hml_joint > 0:
                loss['hml_joint'] = self.reconloss.forward_vel(x[mask], x_out[mask])

            if self.lambda_joint > 0 or self.lambda_joint_vel > 0 or self.lambda_fc > 0:
                if self.data_rep == 'hml_vec':
                    pred_joint = recover_from_ric(self.inv_transform(x_out), 22)
                    gt_joint = recover_from_ric(self.inv_transform(x), 22)
                else:
                    pred_joint = self.get_xyz(x_out.reshape(B, T, self.num_joints, self.num_feats).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                    gt_joint = self.get_xyz(x.reshape(B, T, self.num_joints, self.num_feats).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                
                gt_joint_orig = gt_joint.permute(0, 2, 3, 1)
                pred_joint_orig = pred_joint.permute(0, 2, 3, 1)
                orig_mask = mask.unsqueeze(1).unsqueeze(1)

                loss["joint"] = self.masked_l1_smooth(gt_joint_orig, pred_joint_orig, orig_mask)
                # joint velocity loss
                pred_vel = (pred_joint_orig[..., 1:] - pred_joint_orig[..., :-1])
                gt_vel = (gt_joint_orig[..., 1:] - gt_joint_orig[..., :-1])
                loss["joint_vel"] = self.masked_l1_smooth(gt_vel[:, :-1, :, :], # Remove last joint, is the root location!
                                                    pred_vel[:, :-1, :, :],
                                                    orig_mask[:, :, :, 1:])  # mean_flat((target_vel - model_output_vel) ** 2)
                # foot contact loss
                torch.autograd.set_detect_anomaly(True)
                l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
                relevant_joints = [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx]
                gt_foot_orig = gt_joint_orig[:, relevant_joints, :, :]  # [BatchSize, 4, 3, Frames]
                gt_foot_vel = torch.linalg.norm(gt_foot_orig[:, :, :, 1:] - gt_foot_orig[:, :, :, :-1], axis=2)  # [BatchSize, 4, Frames]
                fc_mask = torch.unsqueeze((gt_foot_vel <= 0.01), dim=2).repeat(1, 1, 3, 1)
                pred_foot_orig = pred_joint_orig[:, relevant_joints, :, :]  # [BatchSize, 4, 3, Frames]
                pred_foot_vel = pred_foot_orig[:, :, :, 1:] - pred_foot_orig[:, :, :, :-1]
                pred_foot_vel[~fc_mask] = 0
                loss["fc"] = self.masked_l1_smooth(pred_foot_vel,
                                            torch.zeros(pred_foot_vel.shape, device=pred_foot_vel.device),
                                            orig_mask[:, :, :, 1:])

            loss['total_loss'] = loss['param'] +\
                                (self.lambda_hml_joint * loss.get('hml_joint', 0.)) +\
                                (self.lambda_gumbel_kl * loss.get('gumbel_kl', 0.)) +\
                                (self.lambda_commit * loss.get('commit', 0.)) +\
                                (self.lambda_param_vel * loss.get('param_vel', 0.)) +\
                                (self.lambda_joint * loss.get('joint', 0.)) +\
                                (self.lambda_joint_vel * loss.get('joint_vel', 0.)) +\
                                (self.lambda_fc * loss.get('fc', 0.))


        return x_out, loss, perplexity


    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x_d = x_d.permute(0, 2, 1).contiguous()
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out



class VQVAE(nn.Module):
    def __init__(self,
                 args,
                 inv_transform,
                 device,
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        
        self.nb_joints = 21 if args.dataset == 'kit' else 22
        self.vqvae = VQVAE_251(args, inv_transform, device, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        # self.rot2xyz = self.vqvae.rot2xyz
        self.translation = True

        # for evaluation, dummy
        self.cond_mode = "action" if args.dataset in ["humanact12", "uestc"] else "text"

    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x) # (N, T)
        return quants

    def forward(self, x, mask=None, mode='test'):

        x_out, loss, perplexity = self.vqvae(x, mask, mode)
        
        return x_out, loss, perplexity

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out
        