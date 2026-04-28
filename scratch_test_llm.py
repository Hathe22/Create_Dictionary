import sys
import os

# Add current dir to path
sys.path.append(os.getcwd())

from llm_labeler import label_clusters_batch

api_key = 'AIzaSyD8MsqE3FvTdOuWVbC1jBo9rFaPGWIzINQ'
cluster_names_tfidf = {
    0: "đậy thùng cái gioăng",
    1: "công suất đường kính"
}
clusters_data = {
    0: {'raw': ["Nắp đậy bình giữ nhiệt nhựa", "Gioăng cao su đậy bình 500ml", "Cái nắp đậy phích nước"]},
    1: {'raw': ["Đèn LED âm trần 15W công suất cao", "Đèn đường kính 120mm gắn trần", "Bóng đèn led công suất 9W"]}
}

print("Testing LLM Labeler...")
result = label_clusters_batch(cluster_names_tfidf, clusters_data, api_key=api_key, batch_size=2)
print("\nRESULT:")
print(result)
