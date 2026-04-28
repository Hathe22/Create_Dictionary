import os
import time
import json
import traceback
from groq import Groq

def label_clusters_batch(cluster_names_tfidf, clusters_data, api_key, batch_size=15):
    """
    Sử dụng Groq LLM (Llama 3) để đặt tên cho các cluster theo batch.
    
    Args:
        cluster_names_tfidf: dict {label: "keyword_1 keyword_2 ..."} (tên từ TF-IDF)
        clusters_data: dict {label: {'prods': [...], 'raw': [...], 'count': ...}}
        api_key: Groq API Key
        batch_size: số lượng cluster gộp vào 1 request
        
    Returns:
        dict {label: "Tên danh mục có nghĩa"}
    """
    if not api_key:
        print("  [!] Khong co API Key, dung lai ten TF-IDF.")
        return cluster_names_tfidf
        
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        print(f"  [!] Loi khoi tao Groq Client: {e}")
        return cluster_names_tfidf

    labels = [lbl for lbl in cluster_names_tfidf.keys() if lbl != -1] # Bỏ qua OUTLIER
    outlier_label = {lbl: cluster_names_tfidf[lbl] for lbl in cluster_names_tfidf.keys() if lbl == -1}
    
    result = {}
    result.update(outlier_label)
    
    if not labels:
        return result
        
    print(f"\n  -> Bat dau goi Groq LLM cho {len(labels)} nhom (batch size: {batch_size})...")
    
    for i in range(0, len(labels), batch_size):
        batch_labels = labels[i:i+batch_size]
        
        prompt = "Bạn là chuyên gia phân loại hàng hóa hải quan. Hãy đặt TÊN DANH MỤC ngắn gọn, có ý nghĩa cho các nhóm sản phẩm sau.\n\n"
        prompt += "YÊU CẦU:\n"
        prompt += "- CHỈ trả về định dạng JSON hợp lệ, KHÔNG GIẢI THÍCH. Ví dụ: {\"0\": \"Tên danh mục 0\", \"1\": \"Tên danh mục 1\"}\n"
        prompt += "- Tên danh mục dài 2-5 từ tiếng Việt, viết hoa chữ cái đầu.\n"
        prompt += "- Tên phải phản ánh đúng BẢN CHẤT CỦA SẢN PHẨM (danh từ chính đứng trước), ví dụ: 'Bình giữ nhiệt', 'Đèn LED âm trần', 'Nắp đậy bình', 'Cốc giữ nhiệt dán nhãn'.\n"
        prompt += "- TUYỆT ĐỐI KHÔNG chứa mã số (ví dụ: RS378B), tên thương hiệu (Philips), dung tích/kích thước (12oz, 15W).\n\n"
        
        prompt += "DỮ LIỆU CÁC NHÓM:\n"
        for lbl in batch_labels:
            keywords = cluster_names_tfidf[lbl]
            samples = [str(x) for x in clusters_data[lbl]['raw'][:3]] # Lấy 3 mẫu
            prompt += f"--- Nhóm ID: {lbl} ---\n"
            prompt += f"Từ khóa đặc trưng: {keywords}\n"
            prompt += f"Sản phẩm mẫu:\n"
            for s in samples:
                prompt += f"- {s}\n"
            prompt += "\n"
            
        retry_count = 0
        success = False
        while retry_count < 5 and not success:
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                try:
                    resp_text = chat_completion.choices[0].message.content.strip()
                    
                    batch_result = json.loads(resp_text)
                    for lbl in batch_labels:
                        llm_name = batch_result.get(str(lbl)) or batch_result.get(lbl)
                        if llm_name:
                            result[lbl] = llm_name
                            print(f"    [LLM] Nhom {lbl}: '{cluster_names_tfidf[lbl]}' -> '{llm_name}'")
                        else:
                            result[lbl] = cluster_names_tfidf[lbl]
                            print(f"    [LLM] Nhom {lbl}: '{cluster_names_tfidf[lbl]}' -> (Giu nguyen)")
                    success = True
                except json.JSONDecodeError:
                    print(f"    [!] LLM tra ve JSON khong hop le. Dang thu lai...")
                    retry_count += 1
                    time.sleep(2)
                    
            except Exception as e:
                err_msg = str(e).lower()
                if '429' in err_msg or 'rate limit' in err_msg:
                    print(f"    [!] Rate limit (429). Cho 60s de reset Tokens (lan thu {retry_count+1})...")
                    time.sleep(60)
                else:
                    print(f"    [!] Loi goi API Groq: {e}. Thu lai sau 3s...")
                    time.sleep(3)
                retry_count += 1
                
        if not success:
            print(f"    [!] That bai batch {i//batch_size + 1} sau 5 lan thu. Giu nguyen ten TF-IDF.")
            for lbl in batch_labels:
                result[lbl] = cluster_names_tfidf[lbl]
                
        # Groq rate limits are generous (30 RPM), but we still add a small sleep to be safe
        if i + batch_size < len(labels):
             time.sleep(2)
             
    return result
