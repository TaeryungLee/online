import os
import numpy as np
import json


from sentence_transformers import SentenceTransformer
t5_model = SentenceTransformer('sentencet5-xxl/')
t5_model.eval()
for p in t5_model.parameters():
    p.requires_grad = False

text_path_list = ['data/babel_272/texts', 'data/humanml3d_272/texts', 'data/babel_272_stream/train_stream_text', 'data/babel_272_stream/val_stream_text']




for text_path in text_path_list:
    out_path = text_path.replace('data', 'text_enc')
    os.makedirs(out_path, exist_ok=True)

    text_dict = {}

    for file in os.listdir(text_path):
        with open(os.path.join(text_path, file), 'r') as f:
            text = f.readlines()
            enc_list = []
            text_dict[file.replace('.txt', '')] = []

            for i, line in enumerate(text):
                line_split = line.strip().split('#')
                caption = line_split[0]
                text_enc = t5_model.encode(caption, output_value='token_embeddings')
                out_file = os.path.join(out_path, file.replace('.txt', f'_{i}.npy'))
                np.save(out_file, enc_list)
                text_dict[file.replace('.txt', '')].append(caption)

            breakpoint()
            json.dump(text_dict, open(os.path.join(out_path, file.replace('.txt', '.json')), 'w'))
