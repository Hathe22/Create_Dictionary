# -*- coding: utf-8 -*-
"""
keyword_extractor.py — Trích xuất Keyword phân biệt cho file phân loại hải quan
================================================================================
Cải tiến so với phiên bản cũ:
  - Dùng TF-IDF XUYÊN NHÓM (cross-group TF-IDF) thay vì đếm tần suất đơn thuần
  - Mỗi Lớp 2 được coi là 1 "tài liệu" chứa tất cả tên hàng khớp
  - Từ xuất hiện ở nhiều Lớp 2 → IDF thấp → bị loại (không phân biệt được)
  - Từ đặc trưng cho 1 Lớp 2 → IDF cao → được chọn làm keyword
  → Kết quả: keyword KHÔNG bị trùng lặp giữa các Lớp 2 cùng mã HS

Cách dùng:
    python keyword_extractor.py --phan-loai phan_loai.xlsx --nk nhap_khau.xlsx --xk xuat_khau.xlsx -o result.xlsx
    python keyword_extractor.py  # Dùng file mặc định
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

# Import bộ lọc chung từ auto_classify — đảm bảo nhất quán
from auto_classify import LABEL_STOPWORDS, _is_valid_cluster_token

# ===========================================================================
# CẤU HÌNH MẶC ĐỊNH
# ===========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PHAN_LOAI = os.path.join(BASE_DIR, 'phan_loai.xlsx')
DEFAULT_NHAP_KHAU = os.path.join(BASE_DIR, 'nhap_khau.xlsx')
DEFAULT_XUAT_KHAU = os.path.join(BASE_DIR, 'xuat_khau.xlsx')
DEFAULT_OUTPUT    = os.path.join(BASE_DIR, 'phan_loai_co_keyword.xlsx')

# Tên cột trong file Phân Loại
COL_MA_HS_TAX = 'Mã HS'
COL_LOP_1     = 'Lớp 1'
COL_LOP_2     = 'Lớp 2'
COL_KEYWORD   = 'Keyword'

# Tên cột trong file Raw
COL_MA_HS_RAW = 'HS_Code'
COL_TEN_HANG  = 'Detailed_Product'



# ===========================================================================
# HÀM TIỆN ÍCH
# ===========================================================================

def clean_product_name(text):
    """Làm sạch tên hàng: bỏ mã SP trước #&, bỏ ký tự đặc biệt."""
    if pd.isna(text):
        return ''
    text = str(text).lower()
    # Bỏ mã SP trước #& (vd: "3000980521#&Viền...")
    text = re.sub(r'^[^#]*#\s*&?\s*', '', text)
    # Bỏ phần sau #&VN ở cuối
    text = re.sub(r'#\s*&?\s*vn\s*$', '', text)
    # Bỏ ký tự đặc biệt, giữ chữ và số
    text = re.sub(
        r'[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ0-9\s]+',
        ' ', text
    )
    return re.sub(r'\s+', ' ', text).strip()


def tokenize_vi(text):
    """Tách từ tiếng Việt và lọc stopwords (dùng LABEL_STOPWORDS chung)."""
    if not text:
        return []
    tokens = ViTokenizer.tokenize(text).split()
    return [t.replace('_', ' ') for t in tokens
            if t.lower() not in LABEL_STOPWORDS and len(t) > 1]


def load_raw_file(file_path):
    """Đọc file raw Excel, tự động tìm dòng header."""
    if not file_path or not os.path.exists(file_path):
        return None
    print(f"  → Đọc: {os.path.basename(file_path)}...")
    try:
        df_check = pd.read_excel(file_path, header=None, nrows=20)
        header_row = 0
        for i, row in df_check.iterrows():
            if COL_MA_HS_RAW in ' '.join(str(v) for v in row.values):
                header_row = i
                break
        df = pd.read_excel(file_path, header=header_row)
        df.columns = [str(c).strip() for c in df.columns]
        print(f"    ✓ {len(df)} dòng")
        return df
    except Exception as e:
        print(f"    ✗ Lỗi: {e}")
        return None


def extract_keywords_ai(group_prods, top_n=15, fallback_seeds=None):
    """
    Trích xuất keyword tối ưu cho AI Training bằng thuật toán Purity-Weighted Frequency.
    Ưu tiên cực cao các từ hiếm/đặc trưng (nhờ Purity^2) để AI phân loại chuẩn, 
    nhưng vẫn giữ lại các từ phổ biến nếu chúng có tần suất lớn trong nhóm.
    """
    from collections import Counter
    
    indices = list(group_prods.keys())
    class_ngram_freq = {idx: Counter() for idx in indices}
    global_ngram_freq = Counter()
    
    def get_ngrams(tokens, n_min=1, n_max=3):
        res = []
        for n in range(n_min, n_max + 1):
            for i in range(len(tokens) - n + 1):
                res.append(' '.join(tokens[i:i+n]))
        return res

    # B1: Thống kê tần suất N-gram (DF)
    for idx, prods in group_prods.items():
        for prod in prods:
            tokens = str(prod).split()
            ngrams = get_ngrams(tokens, 1, 3)
            # Chỉ đếm 1 lần mỗi sản phẩm cho 1 cụm từ
            for ngram in set(ngrams):
                class_ngram_freq[idx][ngram] += 1
                global_ngram_freq[ngram] += 1
                    
    result = {}
    for idx in indices:
        prods = group_prods[idx]
        if not prods:
            fb = fallback_seeds or {}
            result[idx] = fb.get(idx, '')
            continue

        candidates = []
        for ngram, local_freq in class_ngram_freq[idx].items():
            if not _is_valid_cluster_token(ngram):
                continue
                
            global_f = global_ngram_freq[ngram]
            purity = local_freq / global_f if global_f > 0 else 0
            
            # Công thức điểm số: local_freq * (purity^2) * bonus_độ_dài
            # purity^2 giúp các từ đặc thù (chỉ có ở nhóm này) vọt lên đầu bảng
            length_bonus = len(ngram.split())
            score = local_freq * (purity ** 2) * (length_bonus ** 0.5)
            
            candidates.append((ngram, score))
            
        # Sắp xếp theo điểm số giảm dần
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        top_words = []
        for word, score in candidates:
            # Chống lặp cụm từ con/cha (VD: chọn 'đèn led tròn' thì thôi 'đèn led')
            if not any(word in w or w in word for w in top_words):
                top_words.append(word)
            if len(top_words) >= top_n:
                break
                
        if top_words:
            result[idx] = ', '.join(top_words)
        else:
            fb = fallback_seeds or {}
            result[idx] = fb.get(idx, '')
    return result


# ===========================================================================
# HÀM CHÍNH
# ===========================================================================

def extract_keywords(phan_loai_path, nk_path, xk_path, output_path,
                     draft_path=None, top_n=15):
    """
    Pipeline trích xuất keyword bằng TF-IDF xuyên nhóm.

    Args:
        phan_loai_path: file phân loại (Mã HS, Lớp 1, Lớp 2, Cluster_ID)
        nk_path: file raw nhập khẩu
        xk_path: file raw xuất khẩu
        output_path: file kết quả
        draft_path: file draft (chứa sheet 'Raw + Cluster') — dùng Cluster_ID
        top_n: số keyword mỗi nhóm Lớp 2
    """
    print("=" * 60)
    print("  TRÍCH XUẤT KEYWORD (TF-IDF XUYÊN NHÓM)")
    print("=" * 60)

    # ── 1. Đọc file phân loại ────────────────────────────────
    print("\n[1/4] Đọc file phân loại...")
    if not os.path.exists(phan_loai_path):
        print(f"  ✗ Không tìm thấy: {phan_loai_path}")
        return

    df_tax = pd.read_excel(phan_loai_path)
    df_tax.columns = [str(c).strip() for c in df_tax.columns]
    print(f"  ✓ {len(df_tax)} dòng | Cột: {', '.join(df_tax.columns)}")

    if COL_MA_HS_TAX not in df_tax.columns:
        print(f"  ✗ Không tìm thấy cột '{COL_MA_HS_TAX}'")
        return

    # ── 2. Đọc dữ liệu raw + cluster mapping ─────────────────
    print("\n[2/4] Đọc dữ liệu raw...")

    # 2a. Đọc mapping Cluster_ID từ file draft (nếu có)
    cluster_raw_map = {}  # {(hs_code, cluster_id) → [tokenized_strings]}
    use_cluster_id = False
    if draft_path and os.path.exists(draft_path):
        try:
            df_draft_raw = pd.read_excel(draft_path, sheet_name='Raw + Cluster')
            df_draft_raw.columns = [str(c).strip() for c in df_draft_raw.columns]
            if 'Cluster_ID' in df_draft_raw.columns and 'Tên hàng gốc' in df_draft_raw.columns:
                print(f"  ✓ Đọc mapping Cluster_ID từ: {os.path.basename(draft_path)}")
                # Clean & tokenize raw products from draft
                df_draft_raw['_tok'] = df_draft_raw['Tên hàng gốc'].apply(
                    lambda x: ' '.join(tokenize_vi(clean_product_name(str(x))))
                )
                for _, row in df_draft_raw.iterrows():
                    key = (str(row['Mã HS']).strip(), int(row['Cluster_ID']))
                    cluster_raw_map.setdefault(key, []).append(row['_tok'])
                use_cluster_id = True
                print(f"    → {len(cluster_raw_map)} cluster mappings")
        except Exception as e:
            print(f"  ⚠ Không đọc được draft: {e}")

    # 2b. Đọc raw data (dùng khi không có cluster mapping)
    raw_frames = []
    for path in [nk_path, xk_path]:
        df = load_raw_file(path)
        if df is not None:
            raw_frames.append(df)

    if not raw_frames and not use_cluster_id:
        print("  ✗ Không tìm thấy dữ liệu raw!")
        return

    df_raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    if len(df_raw) > 0:
        if COL_MA_HS_RAW not in df_raw.columns:
            print(f"  ✗ Dữ liệu raw thiếu cột '{COL_MA_HS_RAW}'")
            if not use_cluster_id:
                return
        else:
            df_tax[COL_MA_HS_TAX] = df_tax[COL_MA_HS_TAX].astype(str).str.strip()
            df_raw[COL_MA_HS_RAW] = df_raw[COL_MA_HS_RAW].astype(str).str.strip()
            print(f"  → Đang tiền xử lý {len(df_raw)} dòng raw...")
            df_raw['_clean'] = df_raw[COL_TEN_HANG].apply(clean_product_name)
            df_raw['_tokenized_str'] = df_raw['_clean'].apply(lambda x: ' '.join(tokenize_vi(x)))
    else:
        df_tax[COL_MA_HS_TAX] = df_tax[COL_MA_HS_TAX].astype(str).str.strip()

    # ── 3. TF-IDF xuyên nhóm theo từng mã HS ────────────────
    print(f"\n[3/4] Trích xuất keyword (TF-IDF xuyên nhóm)...")

    results = [''] * len(df_tax)
    hs_codes = df_tax[COL_MA_HS_TAX].unique()

    for hs_code in hs_codes:
        tax_mask = df_tax[COL_MA_HS_TAX] == hs_code
        tax_indices = df_tax[tax_mask].index.tolist()

        print(f"\n  Mã HS: {hs_code} → {len(tax_indices)} nhóm")

        group_docs = {}
        fallback_seeds = {}

        for idx in tax_indices:
            row = df_tax.loc[idx]
            lop_2 = str(row.get(COL_LOP_2, '')).strip()
            lop_1 = str(row.get(COL_LOP_1, '')).strip()
            seed_text = lop_2 if (lop_2 and lop_2.lower() not in ('nan', '0')) else lop_1
            fallback_seeds[idx] = seed_text

            # Ưu tiên: dùng Cluster_ID để lấy đúng sản phẩm
            cluster_id = row.get('Cluster_ID', None)
            if use_cluster_id and cluster_id is not None and not pd.isna(cluster_id):
                key = (str(hs_code).strip(), int(cluster_id))
                prods = cluster_raw_map.get(key, [])
                group_docs[idx] = prods
            elif len(df_raw) > 0 and COL_TEN_HANG in df_raw.columns:
                # Fallback: regex match theo tên Lớp 2
                subset_raw = df_raw[df_raw[COL_MA_HS_RAW] == hs_code]
                seeds = [s for s in seed_text.lower().split() if len(s) > 2]
                if seeds:
                    pattern = '|'.join(re.escape(s) for s in seeds)
                    matched = subset_raw[
                        subset_raw[COL_TEN_HANG].str.contains(pattern, case=False, na=False)
                    ]['_tokenized_str'].tolist()
                    group_docs[idx] = matched
                else:
                    group_docs[idx] = []
            else:
                group_docs[idx] = []

        # Trích xuất keyword bằng Purity-Weighted AI logic
        kw_map = extract_keywords_ai(group_docs, top_n=top_n, fallback_seeds=fallback_seeds)

        for idx, kw in kw_map.items():
            results[list(df_tax.index).index(idx)] = kw
            # Log ngắn gọn
            lop2_name = str(df_tax.loc[idx, COL_LOP_2])[:40]
            print(f"    [{lop2_name}] → {kw[:60]}")

    # ── 4. Lưu kết quả ───────────────────────────────────────
    print(f"\n[4/4] Lưu kết quả...")
    df_tax[COL_KEYWORD] = results

    # Thử lưu, nếu file đang mở trong Excel → lưu sang tên khác
    save_path = output_path
    try:
        df_tax.to_excel(save_path, index=False)
    except PermissionError:
        # Tự động đổi tên để tránh conflict
        base, ext = os.path.splitext(output_path)
        save_path = f"{base}_new{ext}"
        print(f"  ⚠ File gốc đang mở trong Excel, lưu sang: {os.path.basename(save_path)}")
        df_tax.to_excel(save_path, index=False)

    print("\n" + "=" * 60)
    print(f"  ✅ HOÀN TẤT!")
    print(f"  → Kết quả: {save_path}")
    print(f"  → {len(df_tax)} dòng đã có keyword phân biệt")
    print("=" * 60)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Trích xuất Keyword phân biệt bằng TF-IDF xuyên nhóm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python keyword_extractor.py --phan-loai phan_loai_7020.xlsx --nk 7020-NK.xlsx --xk 7020-XK.xlsx -o result.xlsx
  python keyword_extractor.py  # File mặc định
        """
    )
    parser.add_argument('--phan-loai', default=DEFAULT_PHAN_LOAI,
                        help=f'File phân loại (mặc định: {os.path.basename(DEFAULT_PHAN_LOAI)})')
    parser.add_argument('--nk', default=DEFAULT_NHAP_KHAU,
                        help=f'File raw NK (mặc định: {os.path.basename(DEFAULT_NHAP_KHAU)})')
    parser.add_argument('--xk', default=DEFAULT_XUAT_KHAU,
                        help=f'File raw XK (mặc định: {os.path.basename(DEFAULT_XUAT_KHAU)})')
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT,
                        help=f'File kết quả (mặc định: {os.path.basename(DEFAULT_OUTPUT)})')
    parser.add_argument('--top-n', type=int, default=12,
                        help='Số keyword mỗi nhóm (mặc định: 12)')

    args = parser.parse_args()

    extract_keywords(
        phan_loai_path=args.phan_loai,
        nk_path=args.nk,
        xk_path=args.xk,
        output_path=args.output,
        top_n=args.top_n,
    )


if __name__ == '__main__':
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    main()
