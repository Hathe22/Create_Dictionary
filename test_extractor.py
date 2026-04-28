import pandas as pd
from collections import Counter
from pyvi import ViTokenizer
def get_ngrams(text, n):
    tokens = text.split()
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

df = pd.read_excel('phan_loai_9617.xlsx')
raw = pd.read_excel('draft_phan_loai_9617.xlsx', sheet_name='Raw + Cluster')
raw['tokens'] = raw['Tên hàng g?c'].astype(str).apply(lambda x: x.lower().split())
# Just a small test file to see if we can extract pure phrases
print('Test created')
