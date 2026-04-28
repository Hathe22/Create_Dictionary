# -*- coding: utf-8 -*-
"""
keyword_extractor.py — Trích xuất Keyword cho file phân loại hải quan
=====================================================================
Đọc file phân loại (đã có Mã HS, Dòng SP, Loại, Lớp 1, Lớp 2),
kết hợp với dữ liệu raw (NK/XK) để trích xuất từ khóa phổ biến nhất
cho mỗi dòng phân loại.

Cách dùng:
    python keyword_extractor.py --phan-loai phan_loai.xlsx --nk nhap_khau.xlsx --xk xuat_khau.xlsx --output result.xlsx
    python keyword_extractor.py  # Dùng file mặc định trong thư mục hiện tại
"""

import pandas as pd
import re
import os
import sys
import argparse
from pyvi import ViTokenizer
from collections import Counter

# ===========================================================================
# CẤU HÌNH MẶC ĐỊNH
# ===========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PHAN_LOAI = os.path.join(BASE_DIR, 'phan_loai.xlsx')
DEFAULT_NHAP_KHAU = os.path.join(BASE_DIR, 'nhap_khau.xlsx')
DEFAULT_XUAT_KHAU = os.path.join(BASE_DIR, 'xuat_khau.xlsx')
DEFAULT_OUTPUT = os.path.join(BASE_DIR, 'phan_loai_co_keyword.xlsx')

# Tên cột trong file Phân Loại
COL_MA_HS_TAX = 'Mã HS'
COL_LOP_1 = 'Lớp 1'
COL_LOP_2 = 'Lớp 2'
COL_KEYWORD = 'Keyword'

# Tên cột trong file Raw
COL_MA_HS_RAW = 'HS_Code'
COL_TEN_HANG = 'Detailed_Product'

# Các từ dừng tiếng Việt (Stopwords)
VI_STOPWORDS = {
    'của', 'và', 'các', 'có', 'là', 'được', 'cho', 'trong', 'với', 'không',
    'những', 'một', 'từ', 'cùng', 'khi', 'đó', 'thì', 'ở', 'đến', 'này',
    'bằng', 'theo', 'như', 'tại', 'vào', 'phải', 'về', 'lại', 'thêm', 'ra',
    'nếu', 'hơn', 'chưa', 'nên', 'vẫn', 'để', 'mà', 'sau', 'nào', 'chỉ',
    'loại', 'hiệu', 'mới', 'dùng', 'tên', 'chi', 'tiết', 'hàng', 'nhãn', 'model'
}


def clean_and_tokenize(text):
    """Làm sạch và tách từ tiếng Việt"""
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(
        r'[^a-záàảãạăắằẳẵặâấầẩẫậpéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ0-9\s]+',
        ' ', text
    )
    tokens = ViTokenizer.tokenize(text).split()
    clean_tokens = [t.replace('_', ' ') for t in tokens
                    if t not in VI_STOPWORDS and len(t) > 1]
    return clean_tokens


def load_raw_file(file_path):
    """Đọc file raw Excel, tự động tìm dòng header."""
    if not os.path.exists(file_path):
        return None
    print(f"  → Đang đọc file: {os.path.basename(file_path)}...")
    try:
        df_check = pd.read_excel(file_path, header=None, nrows=20)
        header_row = 0
        for i, row in df_check.iterrows():
            row_str = " ".join([str(val) for val in row.values])
            if COL_MA_HS_RAW in row_str or COL_TEN_HANG in row_str:
                header_row = i
                break

        df = pd.read_excel(file_path, header=header_row)
        df.columns = [str(c).strip() for c in df.columns]
        print(f"    ✓ {len(df)} dòng")
        return df
    except Exception as e:
        print(f"    ✗ Lỗi khi đọc: {e}")
        return None


def extract_keywords(phan_loai_path, nk_path, xk_path, output_path):
    """
    Pipeline trích xuất keyword.
    
    Args:
        phan_loai_path: file phân loại (đã có Mã HS, Lớp 1, Lớp 2)
        nk_path: file raw nhập khẩu
        xk_path: file raw xuất khẩu  
        output_path: file kết quả
    """
    print("=" * 60)
    print("  TRÍCH XUẤT KEYWORD CHO PHÂN LOẠI")
    print("=" * 60)

    # ── 1. Đọc file phân loại ────────────────────────────────
    print("\n[1/4] Đọc file phân loại...")
    if not os.path.exists(phan_loai_path):
        print(f"  ✗ Không tìm thấy: {phan_loai_path}")
        return

    df_taxonomy = pd.read_excel(phan_loai_path)
    # Chuẩn hóa tên cột: xóa khoảng trắng thừa
    df_taxonomy.columns = [str(c).strip() for c in df_taxonomy.columns]
    print(f"  ✓ {len(df_taxonomy)} dòng phân loại")
    print(f"  Các cột: {', '.join(df_taxonomy.columns)}")

    # Kiểm tra cột bắt buộc
    if COL_MA_HS_TAX not in df_taxonomy.columns:
        print(f"  ✗ Không tìm thấy cột '{COL_MA_HS_TAX}'")
        print(f"  Gợi ý: Kiểm tra tên cột trong file Excel")
        return

    # ── 2. Đọc dữ liệu raw ──────────────────────────────────
    print("\n[2/4] Đọc dữ liệu raw...")
    raw_frames = []

    df_nk = load_raw_file(nk_path)
    if df_nk is not None:
        raw_frames.append(df_nk)

    df_xk = load_raw_file(xk_path)
    if df_xk is not None:
        raw_frames.append(df_xk)

    if not raw_frames:
        print("  ✗ Không tìm thấy dữ liệu raw!")
        return

    df_raw = pd.concat(raw_frames, ignore_index=True)

    if COL_MA_HS_RAW not in df_raw.columns:
        print(f"  ✗ Dữ liệu raw thiếu cột '{COL_MA_HS_RAW}'")
        return

    # Chuẩn hóa Mã HS
    df_taxonomy[COL_MA_HS_TAX] = df_taxonomy[COL_MA_HS_TAX].astype(str).str.strip()
    df_raw[COL_MA_HS_RAW] = df_raw[COL_MA_HS_RAW].astype(str).str.strip()

    # ── 3. Trích xuất keyword ─────────────────────────────────
    print(f"\n[3/4] Trích xuất keyword cho {len(df_taxonomy)} dòng...")
    results = []
    total = len(df_taxonomy)

    for i, row in df_taxonomy.iterrows():
        ma_hs = row.get(COL_MA_HS_TAX, '')
        lop_1 = str(row.get(COL_LOP_1, '')).lower().strip()
        lop_2 = str(row.get(COL_LOP_2, '')).lower().strip()

        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"  Tiến độ: {i + 1}/{total}")

        # Lọc raw data cùng mã HS
        subset_raw = df_raw[df_raw[COL_MA_HS_RAW] == ma_hs]

        # Tạo seed keywords từ Lớp 2 (hoặc Lớp 1)
        seed_text = lop_2 if (lop_2 != 'nan' and lop_2 != '0' and lop_2 != '') else lop_1
        seeds = [s for s in seed_text.split() if len(s) > 2]

        # Tìm sản phẩm có chứa seed keywords
        matched_descriptions = []
        if seeds and COL_TEN_HANG in subset_raw.columns:
            pattern = '|'.join(re.escape(s) for s in seeds)
            matched_descriptions = subset_raw[
                subset_raw[COL_TEN_HANG].str.contains(pattern, case=False, na=False)
            ][COL_TEN_HANG].tolist()

        # Tokenize và đếm từ khóa phổ biến
        all_tokens = []
        for desc in matched_descriptions:
            all_tokens.extend(clean_and_tokenize(desc))

        if all_tokens:
            counts = Counter(all_tokens)
            top_keywords = [word for word, count in counts.most_common(12)]
            results.append(", ".join(top_keywords))
        else:
            results.append(seed_text if seed_text != 'nan' else "")

    # ── 4. Lưu kết quả ───────────────────────────────────────
    print(f"\n[4/4] Lưu kết quả...")
    df_taxonomy[COL_KEYWORD] = results
    df_taxonomy.to_excel(output_path, index=False)

    print("\n" + "=" * 60)
    print(f"  ✅ HOÀN TẤT!")
    print(f"  → Kết quả: {output_path}")
    print(f"  → {len(df_taxonomy)} dòng đã có keyword")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Trích xuất Keyword cho file phân loại hải quan',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python keyword_extractor.py --phan-loai phan_loai_7020.xlsx --nk 7020-NK.xlsx --xk 7020-XK.xlsx -o result.xlsx
  python keyword_extractor.py  # Dùng file mặc định (phan_loai.xlsx, nhap_khau.xlsx, xuat_khau.xlsx)
        """
    )
    parser.add_argument('--phan-loai', default=DEFAULT_PHAN_LOAI,
                        help=f'File phân loại (mặc định: {os.path.basename(DEFAULT_PHAN_LOAI)})')
    parser.add_argument('--nk', default=DEFAULT_NHAP_KHAU,
                        help=f'File raw nhập khẩu (mặc định: {os.path.basename(DEFAULT_NHAP_KHAU)})')
    parser.add_argument('--xk', default=DEFAULT_XUAT_KHAU,
                        help=f'File raw xuất khẩu (mặc định: {os.path.basename(DEFAULT_XUAT_KHAU)})')
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT,
                        help=f'File kết quả (mặc định: {os.path.basename(DEFAULT_OUTPUT)})')

    args = parser.parse_args()

    extract_keywords(
        phan_loai_path=args.phan_loai,
        nk_path=args.nk,
        xk_path=args.xk,
        output_path=args.output,
    )


if __name__ == '__main__':
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    main()
