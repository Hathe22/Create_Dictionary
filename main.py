import pandas as pd
import re
from pyvi import ViTokenizer
from collections import Counter

# Các từ không mang ý nghĩa (Stopwords)
VI_STOPWORDS = {
    'của', 'và', 'các', 'có', 'là', 'được', 'cho', 'trong', 'với', 'không', 
    'những', 'một', 'từ', 'cùng', 'khi', 'đó', 'thì', 'ở', 'đến', 'này', 
    'bằng', 'theo', 'như', 'tại', 'vào', 'phải', 'về', 'lại', 'thêm', 'ra', 
    'nếu', 'hơn', 'chưa', 'nên', 'vẫn', 'để', 'mà', 'sau', 'nào', 'chỉ'
}
def clean_and_tokenize(text):
    if pd.isna(text): return []
    text = str(text).lower()
    # Chỉ giữ lại chữ và số
    text = re.sub(r'[^a-záàảãạăắằẳẵặâấầẩẫậpéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ0-9\s]+', ' ', text)
    tokens = ViTokenizer.tokenize(text).split()
    # Lọc stopwords
    return [t.replace('_', ' ') for t in tokens if t not in VI_STOPWORDS and len(t) > 1]

def main():
    print("Đang đọc dữ liệu...")
    # Đọc file Phân loại (Hình 1)
    df_taxonomy = pd.read_excel('phan_loai.xlsx')
    # Đọc file Raw (Hình 2)
    df_raw = pd.read_excel('raw_data.xlsx')

    # Đảm bảo Mã HS cùng kiểu dữ liệu (chuỗi)
    df_taxonomy['Mã HS'] = df_taxonomy['Mã HS'].astype(str).str.strip()
    df_raw['HS_Cd'] = df_raw['HS_Cd'].astype(str).str.strip()

    keywords_result = []

    print("Đang phân tích từ khóa...")
    for index, row in df_taxonomy.iterrows():
        ma_hs = row.get('Mã HS', '')
        lop_1 = str(row.get('Lớp 1', '')).lower().strip()
        lop_2 = str(row.get('Lớp 2', '')).lower().strip()
        
        # 1. Lọc các dòng Raw có cùng Mã HS
        subset_raw = df_raw[df_raw['HS_Cd'] == ma_hs]
        
        # 2. Sinh từ khóa mồi (seed keywords) từ Lớp 1 và Lớp 2 để tìm trong Tên hàng
        search_terms = []
        if lop_2 and lop_2 != 'nan' and lop_2 != '0':
            search_terms.extend(lop_2.split())
        elif lop_1 and lop_1 != 'nan' and lop_1 != '0':
            search_terms.extend(lop_1.split())
            
        search_terms = [t for t in search_terms if len(t) > 2] # Chỉ lấy các từ dài hơn 2 ký tự làm mồi
        
        matched_products = []
        for _, raw_row in subset_raw.iterrows():
            product_name = str(raw_row.get('Detailed_Product', '')).lower()
            # Nếu tên hàng chứa bất kỳ từ khóa nào của Lớp 1/Lớp 2
            if any(term in product_name for term in search_terms):
                matched_products.append(product_name)
                
        # 3. Trích xuất Keyword phổ biến nhất từ các tên hàng đã khớp
        all_tokens = []
        for prod in matched_products:
            all_tokens.extend(clean_and_tokenize(prod))
            
        if all_tokens:
            # Lấy 10 cụm từ xuất hiện nhiều nhất
            most_common = [word for word, count in Counter(all_tokens).most_common(10)]
            # Thêm chính tên Lớp 2 vào đầu Keyword cho chắc chắn
            if lop_2 and lop_2 != 'nan':
                if lop_2 not in most_common:
                    most_common.insert(0, lop_2)
            keywords_result.append(", ".join(most_common))
        else:
            # Nếu không tìm thấy trong dữ liệu raw, dùng luôn tên Lớp 2 làm keyword
            keywords_result.append(lop_2 if (pd.notna(lop_2) and lop_2 != '0') else "")

    df_taxonomy['Keyword'] = keywords_result
    
    # Xuất file kết quả
    df_taxonomy.to_excel('phan_loai_co_keyword.xlsx', index=False)
    print("Đã hoàn thành! File kết quả được lưu tại: phan_loai_co_keyword.xlsx")

if __name__ == '__main__':
    main()
