# -*- coding: utf-8 -*-
"""
auto_classify.py — Tự động phân loại sản phẩm hải quan bằng NLP Clustering
============================================================================
Đọc file raw (NK/XK), nhóm sản phẩm có tên tương tự bằng TF-IDF + DBSCAN,
rồi xuất file draft phân loại để review thủ công.

Cách dùng:
    python auto_classify.py --nk "path/NK.xlsx" --xk "path/XK.xlsx" --dong-sp "SP THỦY TINH" --output draft.xlsx
    python auto_classify.py --nk "path/NK.xlsx" --dong-sp "SP THỦY TINH"  # chỉ NK
"""

import pandas as pd
import numpy as np
import re
import os
import sys
import argparse
from pyvi import ViTokenizer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

# ===========================================================================
# CẤU HÌNH
# ===========================================================================

# Từ dừng tiếng Việt mở rộng — loại bỏ các từ không mang ý nghĩa phân loại
VI_STOPWORDS = {
    'của', 'và', 'các', 'có', 'là', 'được', 'cho', 'trong', 'với', 'không',
    'những', 'một', 'từ', 'cùng', 'khi', 'đó', 'thì', 'ở', 'đến', 'này',
    'bằng', 'theo', 'như', 'tại', 'vào', 'phải', 'về', 'lại', 'thêm', 'ra',
    'nếu', 'hơn', 'chưa', 'nên', 'vẫn', 'để', 'mà', 'sau', 'nào', 'chỉ',
    'loại', 'hiệu', 'mới', 'dùng', 'tên', 'chi', 'tiết', 'hàng', 'nhãn',
    'model', 'mã', 'số', 'sản', 'phẩm', 'kích', 'thước', 'xuất', 'khẩu',
    'nhập', 'nsx', 'ltd', 'co', 'corp', 'inc', 'company',
    'brand', 'new', 'the', 'for', 'and', 'vn', 'cn',
    '100', 'hàng mới', 'mới 100'
}

# Bảng phân loại HS code → mô tả Lớp 1 mặc định
# Thêm các mã HS mới vào đây khi cần xử lý loại hàng mới
HS_TAXONOMY = {
    # === Mã 9617: SP Bình/Phích ===
    '96170010': 'Phích và bình giữ nhiệt',
    '96170020': 'Các bộ phận phích/bình',

    # === Mã 7020: SP Thủy tinh ===
    '70200011': 'Khuôn thủy tinh — sản xuất acrylic',
    '70200019': 'Khuôn thủy tinh — loại khác',
    '70200020': 'Ống thạch anh — lò phản ứng / bán dẫn',
    '70200030': 'Ruột phích / ruột bình chân không',
    '70200040': 'Ống chân không — năng lượng mặt trời',
    '70200090': 'Sản phẩm thủy tinh khác',
    '7020009010': 'Bình ga sợi thủy tinh',
    '7020009090': 'Sản phẩm thủy tinh khác (loại khác)',
}


# ===========================================================================
# HÀM TIỆN ÍCH
# ===========================================================================

def clean_text(text):
    """Làm sạch văn bản tên hàng: loại bỏ mã SP, ký tự đặc biệt."""
    if pd.isna(text):
        return ''
    text = str(text).lower()
    # Bỏ phần mã sản phẩm trước #& (vd: "3000980521#&Viền...")
    text = re.sub(r'^[^#]*#\s*&?\s*', '', text)
    # Bỏ phần sau #&VN ở cuối
    text = re.sub(r'#\s*&?\s*vn\s*$', '', text)
    # Bỏ ký tự đặc biệt, giữ chữ và số
    text = re.sub(
        r'[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ0-9\s]+',
        ' ', text
    )
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_vi(text):
    """Tách từ tiếng Việt và lọc stopwords."""
    if not text:
        return ''
    tokens = ViTokenizer.tokenize(text).split()
    cleaned = [t.replace('_', ' ') for t in tokens
               if t not in VI_STOPWORDS and len(t) > 1]
    return ' '.join(cleaned)


def load_raw_file(file_path):
    """Đọc file raw Excel, tự động tìm dòng header."""
    if not os.path.exists(file_path):
        print(f"  ⚠ Không tìm thấy: {file_path}")
        return None

    print(f"  → Đang đọc: {os.path.basename(file_path)}...")
    try:
        df_check = pd.read_excel(file_path, header=None, nrows=20)
        header_row = 0
        for i, row in df_check.iterrows():
            row_str = " ".join([str(val) for val in row.values])
            if 'HS_Code' in row_str or 'Detailed_Product' in row_str:
                header_row = i
                break

        df = pd.read_excel(file_path, header=header_row)
        df.columns = [str(c).strip() for c in df.columns]

        # Kiểm tra cột bắt buộc
        if 'HS_Code' not in df.columns or 'Detailed_Product' not in df.columns:
            print(f"  ⚠ File thiếu cột HS_Code hoặc Detailed_Product")
            print(f"    Các cột tìm thấy: {df.columns.tolist()}")
            return None

        print(f"    ✓ {len(df)} dòng, mã HS: {df['HS_Code'].astype(str).str.strip().unique().tolist()}")
        return df
    except Exception as e:
        print(f"  ✗ Lỗi: {e}")
        return None


def cluster_products(descriptions, eps=0.5, min_samples=2):
    """
    Nhóm các mô tả sản phẩm bằng TF-IDF + DBSCAN.
    
    Args:
        descriptions: list các chuỗi mô tả đã tokenize
        eps: ngưỡng khoảng cách cosine để gộp cluster (0.3 = chặt, 0.7 = lỏng)
        min_samples: số lượng tối thiểu để tạo thành 1 cluster
    
    Returns:
        labels: numpy array chứa nhãn cluster (-1 = outlier)
    """
    if len(descriptions) < 2:
        return np.array([0] * len(descriptions))

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # Xét cả cụm 2 từ
        min_df=1,
        max_df=0.95
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(descriptions)
    except ValueError:
        # Nếu tất cả văn bản quá giống hoặc rỗng
        return np.array([0] * len(descriptions))

    # Dùng cosine distance thay vì euclidean cho text
    dist_matrix = cosine_distances(tfidf_matrix)

    clusterer = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='precomputed'
    )
    labels = clusterer.fit_predict(dist_matrix)

    return labels


def get_cluster_name(products, top_n=3):
    """Lấy tên nhóm từ top-N từ khóa phổ biến nhất."""
    all_words = []
    for prod in products:
        words = prod.split()
        all_words.extend([w for w in words if w not in VI_STOPWORDS and len(w) > 1])

    if not all_words:
        return "Chưa phân loại"

    counter = Counter(all_words)
    top_words = [word for word, _ in counter.most_common(top_n)]
    return ' '.join(top_words)


# ===========================================================================
# HÀM CHÍNH
# ===========================================================================

def auto_classify(nk_path=None, xk_path=None, dong_sp='', output_path='draft_phan_loai.xlsx',
                  eps=0.45, min_samples=2):
    """
    Pipeline chính: đọc raw → cluster → xuất draft phân loại.
    
    Args:
        nk_path: đường dẫn file nhập khẩu
        xk_path: đường dẫn file xuất khẩu
        dong_sp: tên dòng sản phẩm (vd: "SP THỦY TINH")
        output_path: file Excel kết quả
        eps: ngưỡng DBSCAN (0.3=chặt, 0.7=lỏng)
        min_samples: số mẫu tối thiểu mỗi cluster
    """
    print("=" * 60)
    print("  TỰ ĐỘNG PHÂN LOẠI SẢN PHẨM HẢI QUAN")
    print("=" * 60)

    # ── 1. Đọc dữ liệu ──────────────────────────────────────
    print("\n[1/5] Đọc dữ liệu raw...")
    frames = []
    if nk_path:
        df_nk = load_raw_file(nk_path)
        if df_nk is not None:
            frames.append(df_nk)
    if xk_path:
        df_xk = load_raw_file(xk_path)
        if df_xk is not None:
            frames.append(df_xk)

    if not frames:
        print("✗ Không có dữ liệu để xử lý!")
        return

    df_raw = pd.concat(frames, ignore_index=True)
    df_raw['HS_Code'] = df_raw['HS_Code'].astype(str).str.strip()
    print(f"  → Tổng: {len(df_raw)} dòng dữ liệu")

    # ── 2. Tiền xử lý ────────────────────────────────────────
    print("\n[2/5] Tiền xử lý văn bản...")
    df_raw['_clean'] = df_raw['Detailed_Product'].apply(clean_text)
    df_raw['_tokenized'] = df_raw['_clean'].apply(tokenize_vi)
    # Loại bỏ dòng rỗng sau khi làm sạch
    df_raw = df_raw[df_raw['_tokenized'].str.len() > 0].reset_index(drop=True)
    print(f"  → Còn {len(df_raw)} dòng sau tiền xử lý")

    # ── 3. Clustering theo từng mã HS ────────────────────────
    print(f"\n[3/5] Clustering (eps={eps}, min_samples={min_samples})...")
    hs_codes = sorted(df_raw['HS_Code'].unique())
    print(f"  → Các mã HS tìm thấy: {hs_codes}")

    all_rows = []
    # Khởi tạo cột _cluster trên df_raw để lưu kết quả
    df_raw['_cluster'] = -999

    for hs_code in hs_codes:
        subset = df_raw[df_raw['HS_Code'] == hs_code].copy()
        descriptions = subset['_tokenized'].tolist()

        print(f"\n  ── Mã HS: {hs_code} ({len(subset)} dòng) ──")

        # Clustering
        labels = cluster_products(descriptions, eps=eps, min_samples=min_samples)
        subset['_cluster'] = labels
        # Ghi cluster labels ngược về df_raw
        df_raw.loc[subset.index, '_cluster'] = labels

        # Lấy Lớp 1 từ bảng HS taxonomy
        lop_1_default = HS_TAXONOMY.get(hs_code, 'Chưa phân loại')

        # Xử lý từng cluster
        unique_labels = sorted(set(labels))
        for label in unique_labels:
            cluster_mask = subset['_cluster'] == label
            cluster_prods = subset[cluster_mask]['_tokenized'].tolist()
            cluster_raw = subset[cluster_mask]['Detailed_Product'].tolist()

            # Tên nhóm gợi ý
            cluster_name = get_cluster_name(cluster_prods, top_n=4)
            count = len(cluster_prods)

            if label == -1:
                lop_2_suggest = f"[OUTLIER] {cluster_name}"
                loai = '?'
            else:
                lop_2_suggest = cluster_name
                loai = 'NC'  # Mặc định NC, user review sau

            # Lấy 1 mô tả mẫu để user dễ review
            sample = str(cluster_raw[0])[:120] if cluster_raw else ''

            all_rows.append({
                'Mã HS': hs_code,
                'Dòng SP': dong_sp,
                'Loại': loai,
                'Lớp 1': lop_1_default,
                'Lớp 2': lop_2_suggest,
                'Keyword': '',  # Sẽ điền bởi keyword_extractor
                'Cluster_ID': int(label),
                'Số lượng SP': count,
                'Mô tả mẫu': sample,
            })

            status = "OUTLIER" if label == -1 else f"Cluster {label}"
            print(f"    {status}: {count} sp → \"{lop_2_suggest}\"")

    # ── 4. Tạo DataFrame kết quả ─────────────────────────────
    print(f"\n[4/5] Tạo file kết quả...")
    df_result = pd.DataFrame(all_rows)

    # Sắp xếp theo Mã HS → Cluster_ID
    df_result = df_result.sort_values(['Mã HS', 'Cluster_ID']).reset_index(drop=True)

    # ── 5. Xuất Excel ─────────────────────────────────────────
    print(f"\n[5/5] Xuất file: {output_path}")

    # Xuất với openpyxl để format đẹp
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Phân loại (để review)
        df_export = df_result[['Keyword', 'Mã HS', 'Dòng SP', 'Loại', 'Lớp 1', 'Lớp 2']].copy()
        df_export.to_excel(writer, sheet_name='Phân loại', index=False)

        # Sheet 2: Chi tiết (để tham khảo)
        df_result.to_excel(writer, sheet_name='Chi tiết Cluster', index=False)

        # Sheet 3: Dữ liệu raw đã gán cluster
        df_raw_export = df_raw[['HS_Code', 'Detailed_Product', '_clean', '_cluster']].copy()
        df_raw_export.columns = ['Mã HS', 'Tên hàng gốc', 'Đã làm sạch', 'Cluster_ID']
        df_raw_export.to_excel(writer, sheet_name='Raw + Cluster', index=False)

    print("\n" + "=" * 60)
    print(f"  ✅ HOÀN TẤT!")
    print(f"  → File kết quả: {output_path}")
    print(f"  → Tổng số nhóm: {len(df_result)}")
    print(f"  → Sheet 'Phân loại': Review và chỉnh sửa Loại/Lớp 1/Lớp 2")
    print(f"  → Sheet 'Chi tiết Cluster': Xem số lượng SP và mô tả mẫu")
    print(f"  → Sheet 'Raw + Cluster': Xem dữ liệu gốc + cluster ID")
    print("=" * 60)

    return df_result


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Tự động phân loại sản phẩm hải quan bằng NLP Clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python auto_classify.py --nk 7020-NK.xlsx --xk 7020-XK.xlsx --dong-sp "SP THỦY TINH"
  python auto_classify.py --nk 9617-NK.xlsx --dong-sp "SP BÌNH/PHÍCH" --eps 0.4
        """
    )
    parser.add_argument('--nk', help='File Excel nhập khẩu (raw)')
    parser.add_argument('--xk', help='File Excel xuất khẩu (raw)')
    parser.add_argument('--dong-sp', default='', help='Tên dòng sản phẩm (vd: "SP THỦY TINH")')
    parser.add_argument('--output', '-o', default='draft_phan_loai.xlsx',
                        help='File Excel kết quả (mặc định: draft_phan_loai.xlsx)')
    parser.add_argument('--eps', type=float, default=0.45,
                        help='Ngưỡng DBSCAN: 0.3=chặt (nhiều nhóm nhỏ), 0.7=lỏng (ít nhóm lớn). Mặc định: 0.45')
    parser.add_argument('--min-samples', type=int, default=2,
                        help='Số mẫu tối thiểu mỗi cluster (mặc định: 2)')

    args = parser.parse_args()

    if not args.nk and not args.xk:
        parser.error("Cần ít nhất 1 file: --nk hoặc --xk")

    auto_classify(
        nk_path=args.nk,
        xk_path=args.xk,
        dong_sp=args.dong_sp,
        output_path=args.output,
        eps=args.eps,
        min_samples=args.min_samples,
    )


if __name__ == '__main__':
    # Đảm bảo output UTF-8 trên Windows
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    main()
