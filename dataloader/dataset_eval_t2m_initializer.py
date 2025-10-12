import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import os
import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    return default_collate(batch)


class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, is_test, max_text_len = 32, window_size = 64, unit_length = 4):
        
        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name
        self.is_test = is_test
        self.max_text_len = max_text_len
        self.window_size = window_size
        self.unit_length = unit_length
        

        if dataset_name == 't2m_272':
            self.data_root = './data/humanml3d_272'
            self.motion_dir = pjoin(self.data_root, 'motion_data')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.text_enc_root = './data/text_enc/humanml3d_272/texts'
            self.joints_num = 22
            self.max_motion_length = 300
            fps = 30
            self.meta_dir = './data/humanml3d_272/mean_std'
            if is_test:
                split_file = pjoin(self.data_root, 'split', 'test.txt')
            else:
                split_file = pjoin(self.data_root, 'split', 'val.txt')

        mean = np.load(pjoin(self.meta_dir, 'Mean.npy')) 
        std = np.load(pjoin(self.meta_dir, 'Std.npy'))

        min_motion_len = window_size  # 30 fps

        data_dict = {}
        id_list = []

        
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []


        for name in tqdm(id_list):
            motion = np.load(pjoin(self.motion_dir, name + '.npy'))
            if (len(motion)) < min_motion_len:
                continue

            text_data = []
            flag = False
            # Load text encodings
            # try:
            #     text_enc = np.load(pjoin(self.text_enc_root, name + '.npy'))
            # except Exception:
            #     continue

            with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                lines = f.readlines()
                text_enc_list = []
                for i, line in enumerate(lines):
                    text_dict = {}
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    tokens = line_split[1].split(' ')
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    # caption_enc = text_enc[i]
                    text_enc_dir = pjoin(self.text_enc_root, name + f'_{i}.npy')
                    caption_enc = np.load(text_enc_dir)

                    text_enc_list.append(caption_enc)

                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict['caption'] = caption
                    text_dict['tokens'] = tokens

                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                    else:
                        n_motion = motion[int(f_tag*fps) : int(to_tag*fps)]           
                        if (len(n_motion)) < min_motion_len or (len(n_motion) >= self.max_motion_length):
                            continue
                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                        while new_name in data_dict:
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                        data_dict[new_name] = {'motion': n_motion,
                                                    'length': len(n_motion),
                                                    'text':[text_dict],
                                                    'caption_enc': [caption_enc]}
                        new_name_list.append(new_name)
                        length_list.append(len(n_motion))
                   

            if flag:
                data_dict[name] = {'motion': motion,
                                    'length': len(motion),
                                    'text': text_data,
                                    'caption_enc': text_enc_list}
                new_name_list.append(name)
                length_list.append(len(motion))


        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, text_list = data['motion'], data['text']
        caption_enc_list = data.get('caption_enc', None)
        
        text_idx = random.choice(list(range(len(text_list))))
        text_data = text_list[text_idx]
        caption = text_data['caption']
        caption_enc = caption_enc_list[text_idx] if caption_enc_list is not None else None
        
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

        # Take first window_size frames only
        m_length = self.window_size
        motion = motion[:m_length]

        # Motion Normalization
        motion = (motion - self.mean) / self.std

        return caption, motion, m_length, caption_enc, caption_enc_len




def DATALoader(dataset_name, is_test,
                batch_size,
                num_workers = 64, unit_length = 4, drop_last=True) : 
    
    val_loader = torch.utils.data.DataLoader(Text2MotionDataset(dataset_name, is_test, unit_length=unit_length),
                                              batch_size,
                                              shuffle = True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              drop_last = drop_last)
    return val_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x