import os
import numpy as np


from sentence_transformers import SentenceTransformer
t5_model = SentenceTransformer('sentencet5-xxl/')
t5_model.eval()
for p in t5_model.parameters():
    p.requires_grad = False

text_path_list = ['data/babel_272/texts', 'data/humanml3d_272/texts', 'data/babel_272_stream/train_stream_text', 'data/babel_272_stream/val_stream_text']




for text_path in text_path_list:
    out_path = text_path.replace('data', 'text_enc')
    os.makedirs(out_path, exist_ok=True)

    for file in os.listdir(text_path):
        out_file = os.path.join(out_path, file.replace('.txt', '.npy'))
        with open(os.path.join(text_path, file), 'r') as f:
            text = f.readlines()
            enc_list = []

            for line in text:
                line_split = line.strip().split('#')
                caption = line_split[0]
                text_enc = t5_model.encode(caption)
                enc_list.append((caption, text_enc))
        
            np.save(out_file, enc_list)
            breakpoint()
