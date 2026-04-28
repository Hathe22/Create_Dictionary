# -*- coding: utf-8 -*-
"""
run_pipeline.py — Điều phối toàn bộ pipeline tạo từ điển hải quan
=================================================================
Chạy lần lượt: auto_classify → (review thủ công) → keyword_extractor

Cách dùng:
    python run_pipeline.py --hs 7020          # Tự tìm file trong folder raw
    python run_pipeline.py --hs 7020 --step 1 # Chỉ chạy bước phân loại
    python run_pipeline.py --hs 7020 --step 2 # Chỉ chạy bước keyword (sau khi đã review)
"""

import os
import sys
import argparse
import glob

# ===========================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ===========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.normpath(os.path.join(
    BASE_DIR, '..', 'Dữ liệu hải quan', 'raw Th12.2025'
))

# Tên Dòng SP theo mã HS chính
DONG_SP_MAP = {
    '9617': 'SP BÌNH/PHÍCH',
    '7020': 'SP THỦY TINH',
    '8539': 'SP ĐÈN/BÓNG ĐÈN',
    '9405': 'SP ĐÈN/THIẾT BỊ CHIẾU SÁNG',
    '85167910': 'SP THIẾT BỊ ĐIỆN',
}


def find_raw_files(hs_code, raw_dir=RAW_DIR):
    """Tự động tìm file NK/XK cho mã HS trong folder raw."""
    nk_path = None
    xk_path = None

    if not os.path.exists(raw_dir):
        print(f"  ⚠ Folder raw không tồn tại: {raw_dir}")
        return nk_path, xk_path

    for f in os.listdir(raw_dir):
        f_lower = f.lower()
        if hs_code.lower() in f_lower and f.endswith('.xlsx') and not f.startswith('~$'):
            full_path = os.path.join(raw_dir, f)
            if '-nk-' in f_lower or 'nk' in f_lower.split('-'):
                nk_path = full_path
            elif '-xk-' in f_lower or 'xk' in f_lower.split('-'):
                xk_path = full_path

    return nk_path, xk_path


def run_step1(hs_code, nk_path, xk_path, dong_sp, eps, min_samples):
    """Bước 1: Tự động phân loại bằng clustering."""
    from auto_classify import auto_classify

    output_draft = os.path.join(BASE_DIR, f'draft_phan_loai_{hs_code}.xlsx')

    print(f"\n{'='*60}")
    print(f"  BƯỚC 1: TỰ ĐỘNG PHÂN LOẠI — Mã HS {hs_code}")
    print(f"{'='*60}")
    print(f"  NK: {nk_path or 'Không có'}")
    print(f"  XK: {xk_path or 'Không có'}")
    print(f"  Dòng SP: {dong_sp}")
    print(f"  Output: {output_draft}")

    auto_classify(
        nk_path=nk_path,
        xk_path=xk_path,
        dong_sp=dong_sp,
        output_path=output_draft,
        eps=eps,
        min_samples=min_samples,
    )

    print(f"\n  📋 TIẾP THEO:")
    print(f"  1. Mở file: {output_draft}")
    print(f"  2. Review sheet 'Phân loại' — chỉnh sửa Loại, Lớp 1, Lớp 2")
    print(f"  3. Lưu lại thành: phan_loai_{hs_code}.xlsx")
    print(f"  4. Chạy: python run_pipeline.py --hs {hs_code} --step 2")


def run_step2(hs_code, nk_path, xk_path):
    """Bước 2: Trích xuất keyword sau khi đã review."""
    from keyword_extractor import extract_keywords

    phan_loai_path = os.path.join(BASE_DIR, f'phan_loai_{hs_code}.xlsx')
    output_path = os.path.join(BASE_DIR, f'phan_loai_co_keyword_{hs_code}.xlsx')

    print(f"\n{'='*60}")
    print(f"  BƯỚC 2: TRÍCH XUẤT KEYWORD — Mã HS {hs_code}")
    print(f"{'='*60}")

    if not os.path.exists(phan_loai_path):
        print(f"  ✗ Không tìm thấy: {phan_loai_path}")
        print(f"  → Bạn đã hoàn thành review và lưu file chưa?")
        print(f"  → File cần có tên: phan_loai_{hs_code}.xlsx")
        return

    print(f"  Input: {phan_loai_path}")
    print(f"  NK: {nk_path or 'Không có'}")
    print(f"  XK: {xk_path or 'Không có'}")

    extract_keywords(
        phan_loai_path=phan_loai_path,
        nk_path=nk_path or '',
        xk_path=xk_path or '',
        output_path=output_path,
    )

    print(f"\n  🎉 KẾT QUẢ CUỐI CÙNG: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline tạo từ điển phân loại hải quan',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quy trình:
  Bước 1: python run_pipeline.py --hs 7020 --step 1
          → Tạo draft_phan_loai_7020.xlsx (tự động clustering)
          → Mở Excel, review và chỉnh sửa
          → Lưu thành phan_loai_7020.xlsx

  Bước 2: python run_pipeline.py --hs 7020 --step 2
          → Trích xuất keyword
          → Kết quả: phan_loai_co_keyword_7020.xlsx

  Cả 2:   python run_pipeline.py --hs 7020
          → Chạy bước 1, dừng lại chờ review
        """
    )
    parser.add_argument('--hs', required=True, help='Mã HS chính (vd: 7020, 9617)')
    parser.add_argument('--step', type=int, choices=[1, 2],
                        help='Chỉ chạy 1 bước (1=phân loại, 2=keyword). Mặc định: chạy bước 1')
    parser.add_argument('--nk', help='File NK (tự tìm nếu không chỉ định)')
    parser.add_argument('--xk', help='File XK (tự tìm nếu không chỉ định)')
    parser.add_argument('--dong-sp', help='Tên dòng sản phẩm (tự tra cứu nếu không chỉ định)')
    parser.add_argument('--eps', type=float, default=0.45, help='Ngưỡng DBSCAN (mặc định: 0.45)')
    parser.add_argument('--min-samples', type=int, default=2, help='Min samples DBSCAN (mặc định: 2)')
    parser.add_argument('--raw-dir', default=RAW_DIR, help=f'Thư mục chứa file raw')

    args = parser.parse_args()

    hs_code = args.hs

    # Tự tìm file raw nếu không chỉ định
    nk_path = args.nk
    xk_path = args.xk
    if not nk_path and not xk_path:
        nk_path, xk_path = find_raw_files(hs_code, args.raw_dir)

    # Tự tra cứu Dòng SP
    dong_sp = args.dong_sp or DONG_SP_MAP.get(hs_code, f'SP {hs_code}')

    print(f"\n  🏭 PIPELINE TẠO TỪ ĐIỂN — Mã HS: {hs_code}")
    print(f"  Dòng SP: {dong_sp}")
    if nk_path:
        print(f"  NK: {os.path.basename(nk_path)}")
    if xk_path:
        print(f"  XK: {os.path.basename(xk_path)}")

    step = args.step

    if step == 2:
        run_step2(hs_code, nk_path, xk_path)
    else:
        # Mặc định hoặc step=1: chạy bước 1
        run_step1(hs_code, nk_path, xk_path, dong_sp, args.eps, args.min_samples)


if __name__ == '__main__':
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    main()
