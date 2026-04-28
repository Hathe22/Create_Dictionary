from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

group_prods = {
    0: ['dèn led r?i philips', 'dèn led r?i philips', 'dèn led r?i philips'],
    1: ['dèn led r?i spotlight', 'dèn led r?i spotlight'],
    2: ['dèn bàn led', 'dèn bàn h?c']
}

indices = list(group_prods.keys())
documents = [' '.join(group_prods[i]) for i in indices]
docs_for_tfidf = [d if d.strip() else '.' for d in documents]

vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x.split(),
    token_pattern=None,
    ngram_range=(1, 3),
    sublinear_tf=True,
    min_df=1,
    max_df=0.85,
)

tfidf_matrix = vectorizer.fit_transform(docs_for_tfidf)
feature_names = vectorizer.get_feature_names_out()

for pos, idx in enumerate(indices):
    row = tfidf_matrix[pos]
    scores = sorted(zip(row.indices, row.data), key=lambda x: x[1], reverse=True)
    print(f'Group {idx}: {[feature_names[i] for i, s in scores[:5]]}')

