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
# Trỏ vào thư mục 'Raw' nằm trong Create_Dictionary
RAW_DIR = os.path.normpath(os.path.join(BASE_DIR, 'Raw'))

# Tên Dòng SP theo mã HS chính
DONG_SP_MAP = {
    '9617': 'SP BÌNH/PHÍCH',
    '7020': 'SP THỦY TINH',
    '8539': 'SP ĐÈN/BÓNG ĐÈN',
    '9405': 'SP ĐÈN/THIẾT BỊ CHIẾU SÁNG',
    '85167910': 'SP THIẾT BỊ ĐIỆN GIA DỤNG',
}


def find_raw_files(hs_code, raw_dir=RAW_DIR):
    """
    Tìm file NK/XK trong nhiều thư mục có thể chứa raw data.
    Tìm theo mã đầy đủ, sau đó theo 4 số đầu.
    """
    # Các thư mục có thể chứa file raw, ưu tiên thư mục gần nhất
    candidate_dirs = [
        raw_dir,  # Create_Dictionary/Raw
        os.path.normpath(os.path.join(BASE_DIR, '..', 'Dữ liệu hải quan', 'raw Th12.2025')),
    ]

    nk_path, xk_path = None, None
    # Tìm theo mã HS đầy đủ trước, sau đó 4 số đầu
    search_codes = [hs_code]
    if len(hs_code) > 4:
        search_codes.append(hs_code[:4])

    for search_dir in candidate_dirs:
        if not os.path.exists(search_dir):
            continue
        for code in search_codes:
            if nk_path and xk_path:
                break
            for f in sorted(os.listdir(search_dir)):
                if f.startswith('~$') or not f.lower().endswith('.xlsx'):
                    continue
                f_lower = f.lower()
                if code.lower() not in f_lower:
                    continue
                full_path = os.path.join(search_dir, f)
                if not nk_path and '-nk-' in f_lower:
                    nk_path = full_path
                    print(f"  → NK tìm thấy: {f}")
                elif not xk_path and '-xk-' in f_lower:
                    xk_path = full_path
                    print(f"  → XK tìm thấy: {f}")
        if nk_path and xk_path:
            break

    if not nk_path and not xk_path:
        print(f"  ⚠ Không tìm thấy file raw cho mã HS: {hs_code}")
        print(f"    Đượng dẫn đã tìm: {[d for d in candidate_dirs if os.path.exists(d)]}")

    return nk_path, xk_path


def run_step1(hs_code, nk_path, xk_path, dong_sp, eps, min_samples, use_llm=False):
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
        use_llm=use_llm,
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

    draft_path = os.path.join(BASE_DIR, f'draft_phan_loai_{hs_code}.xlsx')

    extract_keywords(
        phan_loai_path=phan_loai_path,
        nk_path=nk_path or '',
        xk_path=xk_path or '',
        output_path=output_path,
        draft_path=draft_path,
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
    parser.add_argument('--eps', type=float, default=0.65,
                        help='Ngưỡng DBSCAN — càng cao càng ít cluster (mặc định: 0.65)')
    parser.add_argument('--min-samples', type=int, default=5,
                        help='Số SP tối thiểu mỗi cluster, nhỏ hơn → OUTLIER (mặc định: 5)')
    parser.add_argument('--raw-dir', default=RAW_DIR, help=f'Thư mục chứa file raw')
    parser.add_argument('--use-llm', action='store_true',
                        help='Dùng Gemini LLM để đặt tên Lớp 2 chính xác hơn')

    args = parser.parse_args()

    hs_input = args.hs

    if hs_input.lower() == 'all':
        # Chỉ chạy theo mã HS chính (DONG_SP_MAP), mỗi dòng SP = 1 file draft
        hs_list = sorted(DONG_SP_MAP.keys())
        print(f"🚀 CHẠY BATCH: {len(hs_list)} dòng sản phẩm → {hs_list}")
    else:
        hs_list = [hs_input]

    for hs_code in hs_list:
        # Tự tìm file raw nếu không chỉ định
        nk_path, xk_path = args.nk, args.xk
        if not nk_path and not xk_path:
            nk_path, xk_path = find_raw_files(hs_code, args.raw_dir)

        # Tự tra cứu Dòng SP
        dong_sp = args.dong_sp or DONG_SP_MAP.get(hs_code, f'SP {hs_code}')

        print(f"\n" + "="*60)
        print(f"  🏭 PIPELINE — Mã HS: {hs_code}")
        print(f"  Dòng SP: {dong_sp}")
        print("="*60)

        if args.step == 2:
            run_step2(hs_code, nk_path, xk_path)
        else:
            run_step1(hs_code, nk_path, xk_path, dong_sp, args.eps, args.min_samples, args.use_llm)


if __name__ == '__main__':
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    main()
