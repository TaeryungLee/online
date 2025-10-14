import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate
import os

def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, unit_length = 1, latent_dir=None, window_size=64, max_text_len=32):
        
        self.max_length = 64
        self.window_size = window_size
        self.pointer = 0
        self.dataset_name = dataset_name
        self.unit_length = unit_length
        self.max_text_len = max_text_len

        if dataset_name == 't2m_272':
            self.data_root = './data/humanml3d_272'
            self.text_dir = pjoin(self.data_root, 'texts')
            self.text_enc_root = './data/text_enc/humanml3d_272/texts'
            self.joints_num = 22
            fps = 30
            self.max_motion_length = 78 * 4 // unit_length
            dim_pose = 272
            split_file = pjoin(self.data_root, 'split', 'train.txt')

        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
     
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            try:
                m_token_list = np.load(pjoin(latent_dir, '%s.npy'%name))
            except:
                continue
            if len(m_token_list) < self.window_size:
                continue

            # Read text
            with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                text_data = []
                text_enc_list = []
                flag = False
                lines = f.readlines()

                for i, line in enumerate(lines):
                    text_dict = {}
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    t_tokens = line_split[1].split(' ')
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])

                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict['caption'] = caption
                    text_dict['tokens'] = t_tokens

                    # Load per-line text encoding, like initializer
                    try:
                        text_enc_dir = pjoin(self.text_enc_root, name + f'_{i}.npy')
                        caption_enc = np.load(text_enc_dir)
                        text_enc_list.append(caption_enc)
                    except Exception:
                        # If missing, skip this sample
                        continue

                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                    else:
                        if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length):
                            m_token_list_new = [m_token_list[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)]] 

                        if len(m_token_list_new) < self.window_size:
                            continue

                        new_name = '%s_%f_%f'%(name, f_tag, to_tag)

                        data_dict[new_name] = {'m_token_list': m_token_list_new,
                                                    'text':[text_dict],
                                                    'caption_enc': [caption_enc]}
                        new_name_list.append(new_name)
                    
            if flag:
                data_dict[name] = {'m_token_list': m_token_list,
                                    'text':text_data,
                                    'caption_enc': text_enc_list}
                new_name_list.append(name)

        self.data_dict = data_dict
        self.name_list = new_name_list

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_token_list, text_list = data['m_token_list'], data['text']
        caption_enc_list = data.get('caption_enc', None)
        m_tokens = np.array(m_token_list)

        idx_txt = random.choice(list(range(len(text_list))))
        text_data = text_list[idx_txt]
        caption = text_data['caption']
        caption_enc = caption_enc_list[idx_txt] if caption_enc_list is not None else None

        if len(m_tokens.shape) == 3:
            m_tokens = m_tokens.squeeze(0)
        # coin = np.random.choice([False, False, True])
        # if coin:
        #     coin2 = np.random.choice([True, False])
        #     if coin2:
        #         m_tokens = m_tokens[:-1]
        #     else:
        #         m_tokens = m_tokens[1:]
        # m_tokens_len = m_tokens.shape[0]
        # if m_tokens_len < self.max_motion_length:
        #     m_tokens = np.concatenate([m_tokens, np.zeros((self.max_motion_length-m_tokens_len, m_tokens.shape[1]), dtype=int)], axis=0)
        
        # Random crop length: window_size
        crop_len = self.window_size
        max_start = max(0, len(m_tokens) - crop_len)
        idx = random.randint(0, max_start) if max_start > 0 else 0
        m_tokens = m_tokens[idx:idx+crop_len]
        m_tokens_len = crop_len

        # Truncate/pad caption_enc to self.max_text_len
        if isinstance(caption_enc, np.ndarray):
            orig_len = caption_enc.shape[0]
            target_len = self.max_text_len
            if orig_len > target_len:
                caption_enc = caption_enc[:target_len]
            elif orig_len < target_len:
                pad_shape = (target_len - orig_len,) + tuple(caption_enc.shape[1:])
                pad = np.zeros(pad_shape, dtype=caption_enc.dtype)
                caption_enc = np.concatenate([caption_enc, pad], axis=0)
            caption_enc_len = min(orig_len, target_len)
        else:
            caption_enc_len = 0

        return caption, m_tokens, m_tokens_len, caption_enc, caption_enc_len, idx




def DATALoader(dataset_name,
                batch_size, latent_dir, 
                unit_length=4,
                window_size=64,
                num_workers = 8) : 

    train_loader = torch.utils.data.DataLoader(Text2MotionDataset(dataset_name, latent_dir = latent_dir, unit_length=unit_length, window_size=window_size),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              persistent_workers=True,
                                              drop_last = True)
    
    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

