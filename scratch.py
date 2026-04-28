import pandas as pd, glob
for f in glob.glob('phan_loai_co_keyword_*.xlsx'):
    df = pd.read_excel(f)
    blanks = df['Keyword'].isna().sum()
    print(f'{f}: {blanks} / {len(df)} blank')
