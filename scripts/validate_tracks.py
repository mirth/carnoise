import sys
sys.path.append('..')
import pandas as pd
import torch
from vggish_input import wavfile_to_examples
from tqdm import tqdm

def fix_path(path):
    path = path.replace('/Users/nata/Downloads/raw_youtube_noise_chunked_subset/', '/home/tolik/data/carnoises/raw_youtube_noise_chunked/')
    path = path.replace('/Users/nata/Downloads/raw_youtube_healthy_chunked/', '/home/tolik/data/carnoises/raw_youtube_healthy_chunked/')

    return path

df = pd.read_csv('/home/tolik/data/carnoises/noise_and_healthy4.csv')
df['sample_uri'] = df.sample_uri.apply(fix_path)

valid_ids = []
for path in tqdm(df.sample_uri.values):
    try:
        x = wavfile_to_examples(path)
        if x.shape != torch.Size([1, 1, 480, 64]):
            continue

        valid_ids.append(path)
    except Exception as e:
        print(e)

print(len(valid_ids))
fixed_df = df[df.sample_uri.isin(valid_ids)].copy()
fixed_df.to_csv('/home/tolik/data/carnoises/noise_and_healthy_fixed4.csv', index=False)


