import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import copy
import math
from tqdm import tqdm

from einops import rearrange
from torch.cuda.amp import autocast
from models.vqdm.modules.transformers import Text2ImageTransformer, Condition2ImageTransformer
from models.vqdm.modules.clip_text_embedding import CLIPTextEmbedding
from models.vqdm.modules.action_embedding import EmbedAction



eps = 1e-8

def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_onehot_to_index(log_x):
    return log_x.argmax(1)

def alpha_schedule(time_step, N=100, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.99999):
    att = np.arange(0, time_step)/(time_step-1)*(att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:]/att[:-1]
    ctt = np.arange(0, time_step)/(time_step-1)*(ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1-one_minus_ct
    bt = (1-at-ct)/N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1-att-ctt)/N
    return at, bt, ct, att, btt, ctt


class VQDM(nn.Module):
    def __init__(self, args, inv_transform, device, transformer_type='condition2image',
                 diffusion_step=100, cond_mode='text', num_actions=1, motion_length=196,
                 unit_length=4, num_embed=512, embed_dim=512, num_layers=16, num_heads=16,
                 attn_type='selfcross', action_emb_type='single', text_emb_type='clip_seq',
                 timestep_type='adalayernorm', clip_version='ViT-B/32', mlp_type='conv_mlp', 
                 hidden_dim=2048, **kargs):
        super().__init__()

        self.args = args
        self.inv_transform = inv_transform

        # fixed arguments
        self.loss_type = 'vb_stochastic'
        self.parametrization = 'x0'
        self.alpha_init_type = 'alpha1'
        self.auxiliary_loss_weight = args.lambda_auxiliary
        self.adaptive_auxiliary_loss = True
        self.mask_weight = [1, 1]

        
        self.transformer_type = transformer_type
        self.num_timesteps = diffusion_step
        # NOTE: number of latent code 보다 1개 더 많이 embed함. (Mask token을 사용하기 때문.) 
        self.num_classes = num_embed + 1 # number of latent codes
        self.content_seq_len = motion_length // unit_length
        self.unit_length = unit_length

        # our args
        self.cond_mode = cond_mode
        self.num_actions = num_actions
        self.action_emb_type = action_emb_type
        self.text_emb_type = text_emb_type
        if action_emb_type == 'layerwise':
            self.class_type = 'adalayernorm'
        elif action_emb_type == 'single' or text_emb_type == 'clip_vec':
            self.class_type = 'adalayernorm_mlp'
        else:
            raise NotImplementedError("Unknown action emb type")

        if self.cond_mode == 'text' and self.text_emb_type == 'clip_seq':
            self.condition_seq_len = 77
        else:
            self.condition_seq_len = 1

        # for transformers
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attn_type = attn_type
        self.timestep_type = timestep_type
        self.mlp_type = mlp_type
        

        # Transformer init

        ## NOTE: Transformer input
        # motion vq-indices, encoded condition vector(or sequence)
        if self.transformer_type == 'text2image':  # Input condition: sequence (text-clip_seq)
            self.transformer = Text2ImageTransformer(
                condition_seq_len=self.condition_seq_len, n_layer=self.num_layers,
                n_embd=self.hidden_dim, n_head = self.num_heads, content_seq_len=self.content_seq_len,
                attn_pdrop=0, resid_pdrop=0, mlp_hidden_times=4, block_activate='GELU2',
                attn_type=self.attn_type, condition_dim=self.embed_dim, diffusion_step=self.num_timesteps,
                timestep_type=self.timestep_type, mlp_type=self.mlp_type, num_classes=num_embed
            )
        elif self.transformer_type == 'condition2image':  
            # Input condition: vector if class_type == adalayernorm_mlp (action-single or text-clip_vec)
            # Input condition: action_idxs if class_type == adalayernorm
            self.transformer = Condition2ImageTransformer(
                class_type=self.class_type, class_number=self.num_actions, n_layer=self.num_layers,
                n_embd=self.hidden_dim, n_head = self.num_heads, content_seq_len=self.content_seq_len,
                attn_pdrop=0, resid_pdrop=0, mlp_hidden_times=4, block_activate='GELU2',
                attn_type=self.attn_type, condition_dim=self.embed_dim, diffusion_step=self.num_timesteps,
                timestep_type=self.timestep_type, mlp_type=self.mlp_type, num_classes=num_embed
            )

        # condition encoder
        if self.cond_mode == 'text':
            self.clip_encoder = CLIPTextEmbedding(
                clip_name=clip_version, normalize=True, pick_last_embedding=True if self.text_emb_type=='clip_vec' else False,
                embed_dim=self.embed_dim
            )
            if self.text_emb_type == 'clip_vec':
                self.embed_text = nn.Linear(self.embed_dim, self.embed_dim)
            else:
                self.embed_text = None
        if self.cond_mode == 'action':
            if self.action_emb_type == 'single':
                self.embed_action = EmbedAction(self.num_actions, self.embed_dim)
            else:
                self.embed_action = None


        at, bt, ct, att, btt, ctt = alpha_schedule(self.num_timesteps, N=self.num_classes-1)


        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype('float64'))
        btt = torch.tensor(btt.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

        self.diffusion_acc_list = [0] * self.num_timesteps
        self.diffusion_keep_list = [0] * self.num_timesteps

        # Convert to float32 and register buffers.
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt', log_bt.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))
        self.zero_vector = None



    def multinomial_kl(self, log_prob1, log_prob2):   # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):         # q(xt|xt_1)
        log_at = extract(self.log_at, t, log_x_t.shape)             # at
        log_bt = extract(self.log_bt, t, log_x_t.shape)             # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)             # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)          # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:,:-1,:]+log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct)
            ],
            dim=1
        )

        return log_probs

    def q_pred(self, log_x_start, t):           # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.num_timesteps + 1))%(self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)         # at~
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)       # 1-ct~
        

        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:,:-1,:]+log_cumprod_at, log_cumprod_bt),
                log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct)
            ],
            dim=1
        )

        return log_probs

    def predict_start(self, log_x_t, cond_emb, t):          # p(x0|xt)
        x_t = log_onehot_to_index(log_x_t)
        out = self.transformer(x_t, cond_emb, t)

        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes-1
        assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out.double(), dim=1).float()
        batch_size = log_x_t.size()[0]
        if self.zero_vector is None or self.zero_vector.shape[0] != batch_size:
            self.zero_vector = torch.zeros(batch_size, 1, self.content_seq_len).type_as(log_x_t)- 70
        log_pred = torch.cat((log_pred, self.zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)

        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):            # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes-1).unsqueeze(1) 
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, self.content_seq_len)

        log_qt = self.q_pred(log_x_t, t)                                  # q(xt|x0)
        # log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1)
        log_qt = log_qt[:,:-1,:]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes-1, -1)
        # ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask)*log_qt + mask*ct_cumprod_vector
        

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)        # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:,:-1,:], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)         # ct
        ct_vector = log_ct.expand(-1, self.num_classes-1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector
        
        # log_x_start = torch.cat((log_x_start, log_zero_vector), dim=1)
        # q = log_x_start - log_qt
        q = log_x_start[:,:-1,:] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t-1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def p_pred(self, log_x, cond_emb, t):             # if x0, first p(x0|xt), than sum(q(xt-1|xt,x0)*p(x0|xt))
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(log_x, cond_emb, t)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, cond_emb, t)
        else:
            raise ValueError
        return log_model_pred

    @torch.no_grad()
    def p_sample(self, log_x, cond_emb, t):               # sample q(xt-1) for next step from  xt, actually is p(xt-1|xt)
        model_log_prob = self.p_pred(log_x, cond_emb, t)
        out = self.log_sample_categorical(model_log_prob)
        return out

    def log_sample_categorical(self, logits):           # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):                 # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _train_loss(self, x, cond_emb, mask, is_train=True):                       # get the KL loss
        b, device = x.size(0), x.device
        breakpoint()

        assert self.loss_type == 'vb_stochastic'
        x_start = x
        t, pt = self.sample_time(b, device, 'importance')


        log_x_start = index_to_log_onehot(x_start, self.num_classes)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t)
        xt = log_onehot_to_index(log_xt)

        ############### go to p_theta function ###############
        log_x0_recon = self.predict_start(log_xt, cond_emb, t=t)            # P_theta(x0|xt)
        log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)      # go through q(xt_1|xt,x0)

        ################## compute acc list ################
        x0_recon = log_onehot_to_index(log_x0_recon)
        x0_real = x_start
        xt_1_recon = log_onehot_to_index(log_model_prob)
        xt_recon = log_onehot_to_index(log_xt)
        # TODO: Use mask here
        same_rate = []
        for index in range(t.size()[0]):
            this_t = t[index].item()
            same_rate_x0 = (x0_recon[index][mask[index]] == x0_real[index][mask[index]]).sum().cpu()/x0_real.size()[1]
            self.diffusion_acc_list[this_t] = same_rate_x0.item()*0.1 + self.diffusion_acc_list[this_t]*0.9
            same_rate_xt_1 = (xt_1_recon[index][mask[index]] == xt_recon[index][mask[index]]).sum().cpu()/xt_recon.size()[1]
            self.diffusion_keep_list[this_t] = same_rate_xt_1.item()*0.1 + self.diffusion_keep_list[this_t]*0.9
            same_rate.append(same_rate_x0)
        # compute log_true_prob now 
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t)
        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        mask_region = (xt == self.num_classes-1).float()
        mask_weight = mask_region * self.mask_weight[0] + (1. - mask_region) * self.mask_weight[1]
        kl = kl * mask_weight  # 여기에 mask 적용
        kl[mask==False] = 0.0
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)  # 여기에 mask 적용
        decoder_nll[mask==False] = 0.0
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        kl_loss = mask * decoder_nll + (1. - mask) * kl
        

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # Upweigh loss term of the kl
        # vb_loss = kl_loss / pt + kl_prior
        loss1 = kl_loss / pt 
        vb_loss = loss1
        if self.auxiliary_loss_weight != 0 and is_train==True:
            kl_aux = self.multinomial_kl(log_x_start[:,:-1,:], log_x0_recon[:,:-1,:])
            kl_aux = kl_aux * mask_weight
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1-t/self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            vb_loss += loss2

        return log_model_prob, vb_loss, same_rate


    @property
    def device(self):
        return self.transformer.to_logits[-1].weight.device

    def parameters(self, recurse=True, name=None):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # return super().parameters(recurse=True)
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            print("GPTLikeTransformer: get parameters by the overwrite method!")
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear, )
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
            # special case the position embedding parameter as not decayed
            module_name = ['condition_emb', 'content_emb']
            pos_emb_name = ['pos_emb', 'width_emb', 'height_emb', 'pad_emb', 'token_type_emb']
            for mn in module_name:
                if hasattr(self, mn) and getattr(self, mn) is not None:
                    for pn in pos_emb_name:
                        if hasattr(getattr(self, mn), pn):
                            if isinstance(getattr(getattr(self, mn), pn), torch.nn.Parameter):
                                no_decay.add('{}.{}'.format(mn, pn))

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.transformer.named_parameters()}# if p.requires_grad} 
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            return optim_groups

    # TODO: Only to change 1
    def forward(
            self, 
            content_token, 
            cond,
            mask,
            return_loss=True, 
            return_logits=True, 
            return_att_weight=False,
            is_train=True,
            **kwargs):
        ## Build our input (cond_emb, sample_image)
        sample_image = content_token

        if self.cond_mode == 'text':
            if 'text' in cond['y'].keys():
                text = cond['y']['text']
            elif 'action_text' in cond['y'].keys():
                text = cond['y']['action_text']
            text_emb = self.clip_encoder(text)

            if self.text_emb_type == 'clip_vec':
                text_emb = self.embed_text(text_emb)
            
            cond_emb = text_emb
        
        elif self.cond_mode == 'action':
            if self.action_emb_type == 'single':
                action_emb = self.embed_action(cond['y']['action'])
            elif self.action_emb_type == 'layerwise':
                action_emb = cond['y']['action'].squeeze(1)
            cond_emb = action_emb
            
        # now we get cond_emb and sample_image
        if is_train == True:
            log_model_prob, loss, same_rate = self._train_loss(sample_image, cond_emb, mask)
            # acc = sum(same_rate) / len(same_rate)
            loss = (loss / mask.sum()).sum()

        # 4) get output, especially loss
        out = {}
        out['acc'] = float(sum(self.diffusion_acc_list)/self.num_timesteps)
        if return_logits:
            out['logits'] = torch.exp(log_model_prob)

        if return_loss:
            out['loss'] = loss 
        return out

    # TODO: Only to change 2
    def sample(
            self,
            shape,
            cond,
            device,
            content_token = None,
            filter_ratio = 0.5,
            temperature = 1.0,
            return_att_weight = False,
            return_logits = False,
            content_logits = None,
            print_log = True,
            **kwargs):

        sample_image = content_token
        B, T = shape[:2]

        if self.cond_mode == 'text':
            if 'text' in cond['y'].keys():
                text = cond['y']['text']
            elif 'action_text' in cond['y'].keys():
                text = cond['y']['action_text']
            text_emb = self.clip_encoder(text)

            if self.text_emb_type == 'clip_vec':
                text_emb = self.embed_text(text_emb)
            
            cond_emb = text_emb
        
        elif self.cond_mode == 'action':
            if self.action_emb_type == 'single':
                action_emb = self.embed_action(cond['y']['action'])
            elif self.action_emb_type == 'layerwise':
                action_emb = cond['y']['action'].squeeze(1)
            cond_emb = action_emb

        start_step = int(self.num_timesteps * filter_ratio)

        if start_step == 0:
            # use full mask sample
            zero_logits = torch.zeros((B, self.num_classes-1, T),device=device)
            one_logits = torch.ones((B, 1, T),device=device)
            mask_logits = torch.cat((zero_logits, one_logits), dim=1)
            log_z = torch.log(mask_logits)
            start_step = self.num_timesteps
            with torch.no_grad():
                for diffusion_index in tqdm(range(start_step-1, -1, -1), desc="Diffusion sampling", leave=False):
                    t = torch.full((B,), diffusion_index, device=device, dtype=torch.long)
                    # breakpoint()
                    log_z = self.p_sample(log_z, cond_emb, t)     # log_z is log_onehot

        else:
            t = torch.full((B,), start_step-1, device=device, dtype=torch.long)
            log_x_start = index_to_log_onehot(sample_image, self.num_classes)
            log_xt = self.q_sample(log_x_start=log_x_start, t=t)
            log_z = log_xt
            with torch.no_grad():
                for diffusion_index in tqdm(range(start_step-1, -1, -1), desc="Diffusion sampling", leave=False):
                    t = torch.full((B,), diffusion_index, device=device, dtype=torch.long)
                    log_z = self.p_sample(log_z, cond_emb, t)     # log_z is log_onehot
        
        content_token = log_onehot_to_index(log_z)
        
        output = {'content_token': content_token}
        if return_logits:
            output['logits'] = torch.exp(log_z)
        return output


    def sample_fast(
            self,
            shape,
            cond,
            device,
            content_token = None,
            filter_ratio = 0.5,
            temperature = 1.0,
            return_att_weight = False,
            return_logits = False,
            content_logits = None,
            print_log = True,
            skip_step = 1,
            **kwargs):

        sample_image = content_token
        B, T = shape[:2]

        if self.cond_mode == 'text':
            if 'text' in cond['y'].keys():
                text = cond['y']['text']
            elif 'action_text' in cond['y'].keys():
                text = cond['y']['action_text']
            text_emb = self.clip_encoder(text)

            if self.text_emb_type == 'clip_vec':
                text_emb = self.embed_text(text_emb)
            
            cond_emb = text_emb
        
        elif self.cond_mode == 'action':
            if self.action_emb_type == 'single':
                action_emb = self.embed_action(cond['y']['action'])
            elif self.action_emb_type == 'layerwise':
                action_emb = cond['y']['action'].squeeze(1)
            cond_emb = action_emb

        start_step = int(self.num_timesteps * filter_ratio)

        assert start_step == 0

        zero_logits = torch.zeros((B, self.num_classes-1, T),device=device)
        one_logits = torch.ones((B, 1, T),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)
        start_step = self.num_timesteps
        with torch.no_grad():
            diffusion_list = [index for index in range(start_step-1, -1, -skip_step)]
            if diffusion_list[-1] != 0:
                diffusion_list.append(0)
            # for diffusion_index in range(start_step-1, -1, -1):
            for diffusion_index in tqdm(diffusion_list, desc="Diffusion sampling", leave=False):
                t = torch.full((B,), diffusion_index, device=device, dtype=torch.long)
                log_x_recon = self.predict_start(log_z, cond_emb, t)
                if diffusion_index > skip_step:
                    model_log_prob = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, t=t-skip_step)
                else:
                    model_log_prob = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, t=t)
                log_z = self.log_sample_categorical(model_log_prob)

        content_token = log_onehot_to_index(log_z)
        
        output = {'content_token': content_token}
        if return_logits:
            output['logits'] = torch.exp(log_z)
        return output


