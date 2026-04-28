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
    '100', 'hàng mới', 'mới 100',
    # Mã quốc gia / vùng lãnh thổ thường lẫn trong dữ liệu
    'us', 'uk', 'kr', 'jp', 'de', 'fr', 'it', 'au', 'ca', 'eu',
    # Nhãn hiệu phổ biến trong dữ liệu hải quan
    'stanley', 'owala', 'zojirushi', 'lock', 'locknlock', 'tupperware',
    'fuji', 'tyeso', 'outin', 'elemental', 'cherry', 'mkb', 'mkr', 'mcs',
    'mcz', 'mct', 'med', 'btl', 'sb',
    # Đơn vị / ký hiệu kỹ thuật không mang nghĩa phân loại
    'pcs', 'set', 'unit', 'pc', 'oz', 'ml', 'mm', 'cm',
    'made', 'use', 'size', 'type', 'part',
}

# Regex kiểm tra token có chứa ký tự tiếng Việt / chữ cái Latin không
_VALID_TOKEN_RE = re.compile(
    r'[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]{2,}'
)

# Stopwords riêng cho việc đặt tên Nhãn (Lớp 2) — chặt hơn VI_STOPWORDS nhiều
# Không dùng cho tokenize/TF-IDF keyword, chỉ dùng để lọc tên cluster
LABEL_STOPWORDS = VI_STOPWORDS | {
    # Từ mô tả chung, không phân loại được
    'chất', 'liệu', 'thể', 'điện', 'ngoài', 'bên', 'lớp', 'thương',
    'cách', 'giữa', 'tích', 'hoàn', 'chỉnh', 'gồm', 'kết', 'hợp',
    'mặt', 'tổng', 'dòng', 'đặc', 'chuyên', 'thông', 'dụng',
    'đầu', 'trên', 'dưới', 'trước', 'trong', 'ký', 'đê', 'phân',
    'phối', 'nhà', 'làm', 'đang', 'quốc', 'mbình',
    # Tính từ kích thước / mức độ
    'cao', 'dài', 'rộng', 'dày', 'mỏng', 'nhỏ', 'lớn', 'đơn', 'kép',
    # Màu sắc (không phân biệt danh mục)
    'đen', 'trắng', 'xám', 'xanh', 'đỏ', 'vàng', 'hồng', 'tím', 'nâu', 'bạc',
    'ngà', 'sẫm', 'nhạt', 'đậm', 'màu', 'sắc',
    # Đơn vị đo lường / thông số kỹ thuật
    'oz', 'ml', 'lít', 'lit', 'liter', 'litre', 'g', 'kg',
    'w', 'v', 'watt', 'volt', 'wh', 'kwh', 'ac', 'dc',
    # Từ dữ liệu dạng mã / ghép sai
    'mới100', 'hiêu', 'slo', 'đê', 'quy', 'kiểu',
    'khôngcó', 'thươnghiệu', 'khônghiệu', 'bìnhgiữnhiệt',
    'chấtliệubằngthépkhôngrỉ', 'dungtích', 'vỏbằng',
    # Từ tiếng Anh chung trong hải quan (không phân biệt danh mục)
    'vacuum', 'steel', 'stainless', 'bottle', 'cup', 'mug', 'thermos',
    'body', 'lid', 'cap', 'inner', 'outer', 'bottom', 'clear', 'logo',
    'food', 'grade', 'bpa', 'free', 'with', 'without', 'double', 'wall',
    'flask', 'tumbler', 'water', 'insulated', 'travel', 'handle',
    # Tên công ty / brand nước ngoài thường gặp
    'worthington', 'dachengco', 'wolfpak', 'kaxifei', 'shandongco',
    'lebenlang', 'qihu', 'lhc', 'vhc', 'commerce', 'containers',
    'serial', 'allcho', 'kaiyo', 'ruby', 'elk', 'products',
    'guangzhou', 'zhejiang', 'industry', 'main', 'huizhou',
    'revomax', 'dwf', 'tresette', 'zhengzheng', 'shang',
    'shengyuan', 'inochi', 'ember', 'xile', 'hoycom',
    'adventure', 'quencher', 'rna', 'tumb', 'qnchr',
    'hankie', 'urban', 'outfitters', 'sprngblssms',
    'nonvac', 'disney', 'sports',
    # Mã ký hiệu ngắn thường lẫn vào
    'ky', 'dt', 'rb', 'db', 'wd', 'wt', 'th', 'sb', 'kr', 'us', 'jp',
    'psg', 'ign', 'ptr', 'hsymbl', 'hsty', 'ctg', 'mcx', 'mea', 'mtr',
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

    # === Mã 8539: SP Bóng đèn ===
    # -- Đèn pha gắn kín (853910) --
    '85391010': 'Đèn pha gắn kín — dùng cho xe có động cơ',
    '85391090': 'Đèn pha gắn kín — loại khác',

    # -- Bóng đèn ha-lo-gien vonfram (853921) --
    '85392120': 'Bóng đèn ha-lo-gien vonfram — thiết bị y tế',
    '85392130': 'Bóng đèn ha-lo-gien vonfram — xe có động cơ',
    '85392140': 'Bóng đèn ha-lo-gien vonfram — phản xạ',
    '85392190': 'Bóng đèn ha-lo-gien vonfram — loại khác',

    # -- Bóng đèn dây tóc ≤200W, >100V (853922) --
    '85392220': 'Bóng đèn dây tóc ≤200W — thiết bị y tế',
    '85392231': 'Bóng đèn dây tóc — chiếu sáng trang trí ≤60W',
    '85392232': 'Bóng đèn dây tóc — chiếu sáng trang trí >60W',
    '85392233': 'Bóng đèn dây tóc — chiếu sáng gia dụng',
    '85392239': 'Bóng đèn dây tóc ≤200W — loại khác',
    '85392291': 'Bóng đèn dây tóc — chiếu sáng trang trí ≤60W (nhóm khác)',
    '85392293': 'Bóng đèn dây tóc — chiếu sáng gia dụng (nhóm khác)',
    '85392299': 'Bóng đèn dây tóc ≤200W — loại khác (nhóm khác)',

    # -- Bóng đèn dây tóc loại khác (853929) --
    '85392910': 'Bóng đèn dây tóc — thiết bị y tế',
    '85392920': 'Bóng đèn dây tóc — xe có động cơ',
    '85392930': 'Bóng đèn dây tóc — phản xạ',
    '85392941': 'Bóng đèn flash / cỡ nhỏ — thiết bị y tế',
    '85392949': 'Bóng đèn flash / cỡ nhỏ — loại khác',
    '85392950': 'Bóng đèn dây tóc >200W ≤300W, >100V',
    '85392960': 'Bóng đèn dây tóc ≤200W, ≤100V',
    '8539296010': 'Bóng đèn phòng nổ hai sợi đốt — đèn thợ mỏ',
    '8539296090': 'Bóng đèn dây tóc ≤200W ≤100V — loại khác',
    '85392990': 'Bóng đèn dây tóc — loại khác',

    # -- Bóng đèn huỳnh quang ca-tốt nóng (853931) --
    '85393110': 'Bóng đèn huỳnh quang — ống dùng cho đèn com-pắc',
    '85393120': 'Bóng đèn huỳnh quang — ống thẳng',
    '85393130': 'Bóng đèn huỳnh quang com-pắc có chấn lưu lắp liền',
    '85393190': 'Bóng đèn huỳnh quang ca-tốt nóng — loại khác',

    # -- Bóng đèn hơi thủy ngân / natri / ha-lo-gien kim loại (853932) --
    '85393200': 'Bóng đèn hơi thủy ngân / natri / ha-lo-gien kim loại',

    # -- Bóng đèn phóng điện khác (853939) --
    '85393910': 'Bóng đèn phóng điện — ống dùng cho đèn com-pắc',
    '85393920': 'Bóng đèn CCFL — màn hình dẹt',
    '85393940': 'Bóng đèn CCFL — loại khác',
    '85393990': 'Bóng đèn phóng điện — loại khác',
    '8539399010': 'Đèn ống phóng điện — trang trí / công cộng',
    '8539399020': 'Bóng đèn phóng điện — xe có động cơ / xe đạp',
    '8539399090': 'Bóng đèn phóng điện — loại khác',

    # -- Bóng đèn tia cực tím / hồng ngoại / hồ quang (853941/853949) --
    '85394100': 'Bóng đèn hồ quang',
    '85394900': 'Bóng đèn tia cực tím / hồng ngoại',

    # -- Nguồn sáng LED (853951/853952) --
    '85395100': 'Mô-đun LED',
    '8539510010': 'Mô-đun LED — dùng cho đèn chiếu sáng',
    '8539510020': 'Mô-đun LED — dùng cho xe có động cơ',
    '8539510090': 'Mô-đun LED — loại khác',
    '85395210': 'Bóng đèn LED — đầu đèn ren xoáy',
    '85395290': 'Bóng đèn LED — loại khác',

    # -- Bộ phận (853990) --
    '85399010': 'Bộ phận bóng đèn — nắp / đui nhôm huỳnh quang',
    '85399020': 'Bộ phận bóng đèn — dùng cho xe có động cơ',
    '85399030': 'Bộ phận mô-đun LED',
    '8539903010': 'Bộ phận mô-đun LED — dùng cho đèn chiếu sáng',
    '8539903090': 'Bộ phận mô-đun LED — loại khác',
    '85399090': 'Bộ phận bóng đèn — loại khác',

    # === Mã 9405: SP Đèn / Bộ đèn ===
    # -- Bộ đèn chùm, trần, tường (940511 LED / 940519 khác) --
    '94051110': 'Bộ đèn LED — đèn phòng mổ',
    '94051191': 'Bộ đèn LED — đèn rọi',
    '94051199': 'Bộ đèn LED — loại khác (trần/tường)',
    '94051910': 'Bộ đèn loại khác — đèn phòng mổ',
    '94051991': 'Bộ đèn loại khác — đèn rọi',
    '94051992': 'Bộ đèn loại khác — đèn huỳnh quang',
    '94051999': 'Bộ đèn loại khác — loại khác (trần/tường)',

    # -- Đèn bàn, đèn giường, đèn cây (940521 LED / 940529 khác) --
    '94052110': 'Đèn bàn/giường/cây LED — đèn phòng mổ',
    '94052190': 'Đèn bàn/giường/cây LED — loại khác',
    '9405219010': 'Đèn bàn/giường/cây LED — đèn sân khấu',
    '9405219090': 'Đèn bàn/giường/cây LED — loại khác',
    '94052910': 'Đèn bàn/giường/cây loại khác — đèn phòng mổ',
    '94052990': 'Đèn bàn/giường/cây loại khác — loại khác',
    '9405299010': 'Đèn bàn/giường/cây — đèn sân khấu',
    '9405299090': 'Đèn bàn/giường/cây — loại khác',

    # -- Dây chiếu sáng Nô-en (940531 LED / 940539 khác) --
    '94053100': 'Dây chiếu sáng Nô-en LED',
    '94053900': 'Dây chiếu sáng Nô-en loại khác',

    # -- Đèn quang điện LED (940541) --
    '94054110': 'Đèn LED — đèn pha',
    '94054120': 'Đèn LED — đèn rọi',
    '94054130': 'Đèn LED — tín hiệu sân bay / đường sắt / tàu thủy',
    '94054140': 'Đèn LED — chiếu sáng công cộng / ngoài trời',
    '94054190': 'Đèn LED — loại khác',
    '9405419010': 'Đèn LED — đèn sân khấu',
    '9405419090': 'Đèn LED — loại khác',

    # -- Đèn điện LED loại khác (940542) --
    '94054210': 'Đèn điện LED khác — đèn pha',
    '94054220': 'Đèn điện LED khác — đèn rọi',
    '94054230': 'Đèn điện LED khác — tín hiệu sân bay / đường sắt',
    '94054240': 'Đèn LED — báo hiệu thiết bị gia dụng 85.16',
    '94054250': 'Đèn điện LED khác — chiếu sáng công cộng',
    '94054260': 'Đèn điện LED khác — chiếu sáng ngoài trời',
    '94054290': 'Đèn điện LED khác — loại khác',
    '9405429010': 'Đèn điện LED khác — đèn sân khấu',
    '9405429090': 'Đèn điện LED khác — loại khác',

    # -- Đèn điện loại khác (940549) --
    '94054910': 'Đèn điện loại khác — đèn pha',
    '94054920': 'Đèn điện loại khác — đèn rọi',
    '94054930': 'Đèn điện loại khác — tín hiệu sân bay / đường sắt',
    '94054940': 'Đèn điện loại khác — báo hiệu thiết bị gia dụng',
    '94054950': 'Đèn điện loại khác — chiếu sáng công cộng',
    '94054960': 'Đèn điện loại khác — chiếu sáng ngoài trời',
    '94054990': 'Đèn điện loại khác — loại khác',
    '9405499010': 'Đèn điện loại khác — đèn sân khấu',
    '9405499090': 'Đèn điện loại khác — loại khác',

    # -- Đèn không hoạt động bằng điện (940550) --
    '94055011': 'Đèn dầu bằng đồng — nghi lễ tôn giáo',
    '94055019': 'Đèn dầu — loại khác',
    '94055040': 'Đèn bão',
    '94055050': 'Đèn thợ mỏ / khai thác đá',
    '94055090': 'Đèn không điện — loại khác',

    # -- Biển hiệu chiếu sáng (940561 LED / 940569 khác) --
    '94056110': 'Biển hiệu LED — cảnh báo / tên đường / giao thông',
    '94056190': 'Biển hiệu LED — loại khác',
    '94056910': 'Biển hiệu loại khác — cảnh báo / tên đường / giao thông',
    '94056990': 'Biển hiệu loại khác — loại khác',

    # -- Bộ phận bằng thủy tinh (940591) --
    '94059110': 'Bộ phận thủy tinh — đèn phòng mổ',
    '94059120': 'Bộ phận thủy tinh — đèn rọi',
    '94059140': 'Bộ phận thủy tinh — chao đèn / thông phong',
    '94059150': 'Bộ phận thủy tinh — đèn pha',
    '94059190': 'Bộ phận thủy tinh — loại khác',

    # -- Bộ phận bằng plastic (940592) --
    '94059210': 'Bộ phận plastic — đèn phòng mổ',
    '94059220': 'Bộ phận plastic — đèn rọi',
    '94059230': 'Bộ phận plastic — đèn pha',
    '94059290': 'Bộ phận plastic — loại khác',

    # -- Bộ phận loại khác (940599) --
    '94059910': 'Bộ phận đèn — chụp đèn vải',
    '94059920': 'Bộ phận đèn — chụp đèn vật liệu khác',
    '94059930': 'Bộ phận đèn — của đèn dầu 9405.50.11/9405.50.19',
    '94059940': 'Bộ phận đèn — của đèn pha / đèn rọi',
    '94059950': 'Bộ phận đèn — gốm / sứ / kim loại',
    '94059990': 'Bộ phận đèn — loại khác',

    # === Mã 8516: SP Dụng cụ điện gia dụng (chú trọng 85167910 - Ấm đun nước) ===
    # -- Đun nước nóng (851610) --
    '85161011': 'Bình thủy điện (water dispenser) gia dụng',
    '85161019': 'Dụng cụ đun nước nóng tức thời / dự trữ — loại khác',
    '85161030': 'Dụng cụ đun nước nóng kiểu nhúng',

    # -- Làm nóng không gian (851621/851629) --
    '85162100': 'Dụng cụ điện làm nóng không gian — bức xạ giữ nhiệt',
    '85162900': 'Dụng cụ điện làm nóng không gian — loại khác',

    # -- Dụng cụ làm tóc / sấy tay (851631/851632/851633) --
    '85163100': 'Máy sấy tóc',
    '85163200': 'Dụng cụ làm tóc khác (máy uốn, kẹp...)',
    '85163300': 'Máy sấy khô tay',

    # -- Bàn là điện (851640) --
    '85164010': 'Bàn là điện — dùng hơi nước công nghiệp',
    '85164090': 'Bàn là điện — loại khác',

    # -- Lò vi sóng / lò nướng / bếp (851650/851660) --
    '85165000': 'Lò vi sóng',
    '85166010': 'Nồi cơm điện',
    '85166090': 'Lò nướng / bếp điện — loại khác',

    # -- Dụng cụ nhiệt điện khác (851671/851672/851679) --
    '85167100': 'Dụng cụ pha chè / cà phê',
    '85167200': 'Lò nướng bánh (toaster)',
    '85167910': 'Ấm đun nước điện',
    '85167990': 'Dụng cụ nhiệt điện gia dụng khác',

    # -- Điện trở đốt nóng (851680) — linh kiện --
    '85168010': 'Điện trở đốt nóng — máy đúc chữ / lò công nghiệp',
    '85168030': 'Điện trở đốt nóng — thiết bị gia dụng',
    '85168090': 'Điện trở đốt nóng — loại khác',

    # -- Bộ phận (851690) — linh kiện --
    '85169021': 'Bộ phận thiết bị điện — tấm toả nhiệt gia dụng',
    '85169029': 'Bộ phận thiết bị điện — loại khác (nhóm sấy/lò)',
    '85169030': 'Bộ phận của thiết bị đun nước nóng 8516.10',
    '85169040': 'Bộ phận điện trở đốt nóng — máy đúc chữ',
    '85169090': 'Bộ phận thiết bị điện — loại khác',
}

# Từ khóa nhận diện Linh kiện (LK)
LK_KEYWORDS = [
    'linh kiện', 'phụ tùng', 'bộ phận', 'nắp', 'vòi', 'thân', 'đáy',
    'phôi', 'gioăng', 'vòng', 'đệm', 'vít', 'ốc', 'trục', 'khuôn', 'viền',
    'vỏ', 'lõi', 'tấm', 'phụ kiện', 'mảnh', 'miếng', 'nút', 'đầu',
    'tay cầm', 'ruột', 'cán', 'khung', 'mặt nạ', 'chân', 'đế',
    'ống hút', 'nút bấm', 'vòng đệm', 'đui', 'chấn lưu', 'driver', 'pcba',
    'mạch', 'bảng mạch', 'module', 'mô-đun'
]

# Ánh xạ mã HS → Loại mặc định (NC: Nội chính/Thành phẩm, LK: Linh kiện)
HS_TYPE_MAP = {
    # 9617
    '96170010': 'NC',
    '96170020': 'LK',

    # 7020
    '70200011': 'LK',  # Khuôn
    '70200019': 'LK',  # Khuôn
    '70200020': 'LK',  # Ống thạch anh
    '70200030': 'LK',  # Ruột phích
    '70200040': 'LK',  # Ống chân không
    '70200090': 'NC',
    '7020009010': 'NC',
    '7020009090': 'NC',

    # 8539 — Bóng đèn pha, dây tóc, huỳnh quang, LED → thành phẩm (NC)
    '85391010': 'NC', '85391090': 'NC',
    '85392120': 'NC', '85392130': 'NC', '85392140': 'NC', '85392190': 'NC',
    '85392220': 'NC', '85392231': 'NC', '85392232': 'NC', '85392233': 'NC',
    '85392239': 'NC', '85392291': 'NC', '85392293': 'NC', '85392299': 'NC',
    '85392910': 'NC', '85392920': 'NC', '85392930': 'NC',
    '85392941': 'NC', '85392949': 'NC', '85392950': 'NC',
    '85392960': 'NC', '8539296010': 'NC', '8539296090': 'NC', '85392990': 'NC',
    '85393110': 'NC', '85393120': 'NC', '85393130': 'NC', '85393190': 'NC',
    '85393200': 'NC',
    '85393910': 'NC', '85393920': 'NC', '85393940': 'NC', '85393990': 'NC',
    '8539399010': 'NC', '8539399020': 'NC', '8539399090': 'NC',
    '85394100': 'NC', '85394900': 'NC',
    # LED module: là linh kiện cấu thành đèn → LK
    '85395100': 'LK', '8539510010': 'LK', '8539510020': 'LK', '8539510090': 'LK',
    # Bóng đèn LED thành phẩm → NC
    '85395210': 'NC', '85395290': 'NC',
    # Bộ phận → LK
    '85399010': 'LK', '85399020': 'LK', '85399030': 'LK',
    '8539903010': 'LK', '8539903090': 'LK', '85399090': 'LK',

    # 9405 — Đèn thành phẩm → NC
    '94051110': 'NC', '94051191': 'NC', '94051199': 'NC',
    '94051910': 'NC', '94051991': 'NC', '94051992': 'NC', '94051999': 'NC',
    '94052110': 'NC', '94052190': 'NC', '9405219010': 'NC', '9405219090': 'NC',
    '94052910': 'NC', '94052990': 'NC', '9405299010': 'NC', '9405299090': 'NC',
    '94053100': 'NC', '94053900': 'NC',
    '94054110': 'NC', '94054120': 'NC', '94054130': 'NC', '94054140': 'NC',
    '94054190': 'NC', '9405419010': 'NC', '9405419090': 'NC',
    '94054210': 'NC', '94054220': 'NC', '94054230': 'NC', '94054240': 'NC',
    '94054250': 'NC', '94054260': 'NC', '94054290': 'NC',
    '9405429010': 'NC', '9405429090': 'NC',
    '94054910': 'NC', '94054920': 'NC', '94054930': 'NC', '94054940': 'NC',
    '94054950': 'NC', '94054960': 'NC', '94054990': 'NC',
    '9405499010': 'NC', '9405499090': 'NC',
    '94055011': 'NC', '94055019': 'NC', '94055040': 'NC',
    '94055050': 'NC', '94055090': 'NC',
    '94056110': 'NC', '94056190': 'NC', '94056910': 'NC', '94056990': 'NC',
    # 9405 Bộ phận → LK
    '94059110': 'LK', '94059120': 'LK', '94059140': 'LK', '94059150': 'LK', '94059190': 'LK',
    '94059210': 'LK', '94059220': 'LK', '94059230': 'LK', '94059290': 'LK',
    '94059910': 'LK', '94059920': 'LK', '94059930': 'LK',
    '94059940': 'LK', '94059950': 'LK', '94059990': 'LK',

    # 8516 — Dụng cụ điện thành phẩm → NC
    '85161011': 'NC', '85161019': 'NC', '85161030': 'NC',
    '85162100': 'NC', '85162900': 'NC',
    '85163100': 'NC', '85163200': 'NC', '85163300': 'NC',
    '85164010': 'NC', '85164090': 'NC',
    '85165000': 'NC',
    '85166010': 'NC', '85166090': 'NC',
    '85167100': 'NC', '85167200': 'NC', '85167910': 'NC', '85167990': 'NC',
    # 8516 Điện trở + Bộ phận → LK
    '85168010': 'LK', '85168030': 'LK', '85168090': 'LK',
    '85169021': 'LK', '85169029': 'LK', '85169030': 'LK',
    '85169040': 'LK', '85169090': 'LK',
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


def _is_valid_cluster_token(token):
    """Kiểm tra token có đủ chất lượng để làm tên Lớp 2 không."""
    token_lower = token.lower()
    # Loại bỏ stopwords riêng cho nhãn (chặt hơn)
    if token_lower in LABEL_STOPWORDS:
        return False
    # Loại token bắt đầu bằng số (vd: '12oz', '350ml', '10', '107')
    if re.match(r'^\d', token):
        return False
    # Yêu cầu ít nhất 3 ký tự (giảm tiếng lẳng nhằng của từ 2 chữ cái)
    if len(token) < 3:
        return False
    # Loại token chứa số + chữ kiểu mã SKU (vd: "360ml", "18l", "40oz")
    if re.fullmatch(r'\d+[a-z]+|[a-z]+\d+', token_lower):
        return False
    # Yêu cầu có ít nhất 3 ký tự chữ cái liên tiếp
    if not _VALID_TOKEN_RE.search(token_lower):
        return False
    return True


def get_cluster_name(products, raw_descriptions=None, top_n=4):
    """
    Lấy tên nhóm có ý nghĩa từ các sản phẩm trong cluster.
    Chiến lược:
      1. Gom top-N từ khóa hợp lệ (lọc qua _is_valid_cluster_token)
      2. Nếu < 2 từ hợp lệ → dùng tên sản phẩm thực gần nhất (ngắn nhất, sạch nhất)
    """
    all_words = []
    for prod in products:
        words = prod.split()
        all_words.extend([w for w in words if _is_valid_cluster_token(w)])

    if all_words:
        counter = Counter(all_words)
        top_words = [word for word, _ in counter.most_common(top_n)]
        if len(top_words) >= 2:
            return ' '.join(top_words)

    # Fallback: dùng tên sản phẩm thực
    # Lấy các từ hợp lệ từ mô tả gốc → ghép thành tên ngắn gọn
    if raw_descriptions:
        candidates = []
        for desc in raw_descriptions:
            cleaned = re.sub(r'^[^#]*#\s*&?\s*', '', str(desc))  # bỏ mã SKU đầu
            cleaned = re.sub(r'#.*$', '', cleaned).strip()        # bỏ phần sau #
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            # Lọc chỉ giữ từ hợp lệ (không số, không brand, không stopword)
            valid_words = [w for w in cleaned.lower().split()
                          if _is_valid_cluster_token(w)]
            if len(valid_words) >= 2:
                candidates.append(' '.join(valid_words[:6]))
        if candidates:
            return min(candidates, key=len)[:60]

    return "Chưa phân loại"


def detect_type(hs_code, cluster_name, samples):
    """
    Tự động nhận diện Loại (NC/LK) dựa trên mã HS và từ khóa.

    Logic:
      - HS_TYPE_MAP là nguồn chính xác nhất nếu có.
      - Tuy nhiên, nếu có ≥ 2 từ khóa LK rõ ràng trong tên hàng thực
        (ví dụ: "gioang" + "phiêu") thì override sang LK dù HS_TYPE_MAP nói NC.
      - OUTLIER ('?') không được sử dụng nữa — luôn trả về NC hoặc LK.
    """
    # Ghép tên cluster + tất cả mẫu để quét từ khóa
    text_to_check = (cluster_name + ' ' + ' '.join(samples)).lower()

    # Đếm số từ khóa LK xuất hiện
    lk_hits = sum(1 for kw in LK_KEYWORDS if kw in text_to_check)

    # Nếu có bằng chứng LK mạnh (≥2 từ khóa) → override bất kể HS_TYPE_MAP
    if lk_hits >= 2:
        return 'LK'

    # Tra HS_TYPE_MAP làm mặc định
    if hs_code in HS_TYPE_MAP:
        return HS_TYPE_MAP[hs_code]

    # Nếu có bất kỳ 1 từ khóa LK nào (khi không có trong HS_TYPE_MAP)
    if lk_hits >= 1:
        return 'LK'

    return 'NC'  # Mặc định là Nội chính

from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVec


def get_cluster_names_tfidf(clusters_data, top_n=4):
    """
    Đặt tên phân biệt cho tất cả cluster của 1 mã HS bằng TF-IDF xuyên nhóm.

    Args:
        clusters_data: dict {label → {'prods': [...], 'raw': [...]}}
            - prods: tokenized strings (1 string per product)
            - raw: original product names
        top_n: số từ dung để tạo tên cluster

    Returns:
        dict {label → cluster_name_str}
    """
    labels = list(clusters_data.keys())

    # Gom tất cả token của mỗi cluster thành 1 document
    documents = []
    for lbl in labels:
        tokens = clusters_data[lbl]['prods']
        # Lọc token hợp lệ trước khi gộp
        valid_tokens = [t for token_str in tokens for t in token_str.split()
                        if _is_valid_cluster_token(t)]
        documents.append(' '.join(valid_tokens))

    # Nếu chỉ có 1 cluster → dùng fallback
    if len(labels) <= 1:
        result = {}
        for i, lbl in enumerate(labels):
            result[lbl] = get_cluster_name(
                clusters_data[lbl]['prods'],
                raw_descriptions=clusters_data[lbl]['raw'],
                top_n=top_n
            )
        return result

    # Thay document rỗng bằng placeholder
    docs_for_tfidf = [d if d.strip() else '.' for d in documents]

    try:
        vectorizer = _TfidfVec(
            tokenizer=lambda x: x.split(),
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=0.80,   # Từ xuất hiện trong >80% cluster → không phân biệt được
            max_features=5000,
        )
        tfidf_matrix = vectorizer.fit_transform(docs_for_tfidf)
        feature_names = vectorizer.get_feature_names_out()
    except Exception:
        # Fallback nếu TF-IDF lỗi
        return {lbl: get_cluster_name(
            clusters_data[lbl]['prods'],
            raw_descriptions=clusters_data[lbl]['raw'],
            top_n=top_n
        ) for lbl in labels}

    # Regex kiểm tra từ có chứa ký tự tiếng Việt (ưu tiên hơn từ Latin thuần)
    _VI_CHAR_RE = re.compile(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]')

    result = {}
    for pos, lbl in enumerate(labels):
        row = tfidf_matrix[pos]
        scores = sorted(zip(row.indices, row.data), key=lambda x: x[1], reverse=True)

        vi_words   = []  # Từ tiếng Việt (ưu tiên cao)
        latin_words = [] # Từ Latin có nghĩa (ưu tiên thấp hơn)

        for feat_idx, score in scores:
            word = feature_names[feat_idx]
            # Chỉ lấy 1-gram và phải hợp lệ
            if ' ' not in word and _is_valid_cluster_token(word):
                if _VI_CHAR_RE.search(word):
                    vi_words.append(word)
                else:
                    latin_words.append(word)
            if len(vi_words) >= top_n:
                break

        # Ghép: ưu tiên từ tiếng Việt, bổ sung Latin nếu thiếu
        top_words = vi_words[:top_n]
        if len(top_words) < 2:
            top_words += latin_words[:top_n - len(top_words)]

        if len(top_words) >= 2:
            # Sắp xếp lại: từ phổ biến nhất trong cluster đứng đầu (đọc tự nhiên hơn)
            doc = documents[pos]
            word_counts = Counter(doc.split())
            top_words.sort(key=lambda w: word_counts.get(w, 0), reverse=True)
            result[lbl] = ' '.join(top_words)
        else:
            # Fallback: dùng sản phẩm thực
            result[lbl] = get_cluster_name(
                clusters_data[lbl]['prods'],
                raw_descriptions=clusters_data[lbl]['raw'],
                top_n=top_n
            )

    return result


def merge_duplicate_clusters(all_rows):
    """
    Gộp các cluster có tên Lớp 2 giống nhau (cùng mã HS).
    → Giảm số hàng lặp, giữ cluster có nhiều SP nhất làm đại diện.
    """
    if not all_rows:
        return all_rows

    df = pd.DataFrame(all_rows)

    # Chuẩn hóa Cluster_ID: -1 là OUTLIER, giữ nguyên
    # Gộp theo (Mã HS, Lớp 2)
    merged = []
    for (ma_hs, lop_2), group in df.groupby(['Mã HS', 'Lớp 2'], sort=False):
        # Lấy row có nhiều SP nhất làm đại diện
        best_row = group.loc[group['Số lượng SP'].idxmax()].copy()
        # Tổng số lượng SP
        best_row['Số lượng SP'] = group['Số lượng SP'].sum()
        # Nếu nhiều cluster được gộp, ghi chú vào Mô tả mẫu
        if len(group) > 1:
            samples = group['Mô tả mẫu'].dropna().tolist()[:3]
            best_row['Mô tả mẫu'] = ' | '.join(str(s)[:60] for s in samples)
        merged.append(best_row.to_dict())

    return merged


def auto_classify(nk_path=None, xk_path=None, dong_sp='', output_path='draft_phan_loai.xlsx',
                  eps=0.45, min_samples=2, use_llm=False):
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

        # Thu thập tất cả cluster trước, sau đó đặt tên bằng TF-IDF xuyên nhóm
        unique_labels = sorted(set(labels))
        clusters_data = {}
        for label in unique_labels:
            cluster_mask = subset['_cluster'] == label
            cluster_prods = subset[cluster_mask]['_tokenized'].tolist()
            cluster_raw   = subset[cluster_mask]['Detailed_Product'].tolist()
            clusters_data[label] = {
                'prods':  cluster_prods,
                'raw':    cluster_raw,
                'count':  len(cluster_prods),
                'sample': str(cluster_raw[0])[:120] if cluster_raw else '',
            }

        # Đặt tên phân biệt bằng TF-IDF xuyên nhóm (OUTLIER xử lý riêng)
        non_outlier_data = {lbl: v for lbl, v in clusters_data.items() if lbl != -1}
        cluster_names = get_cluster_names_tfidf(non_outlier_data, top_n=4)

        if use_llm:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                print("  [!] Canh bao: Tham so --use-llm duoc bat nhung chua set bien moi truong GROQ_API_KEY. Se bo qua LLM.")
            else:
                try:
                    from llm_labeler import label_clusters_batch
                    cluster_names = label_clusters_batch(cluster_names, non_outlier_data, api_key=api_key)
                except Exception as e:
                    print(f"  [!] Loi chay LLM: {e}. Se dung ten TF-IDF goc.")

        # Xử lý OUTLIER riêng
        if -1 in clusters_data:
            outlier_name = get_cluster_name(
                clusters_data[-1]['prods'],
                raw_descriptions=clusters_data[-1]['raw'],
                top_n=3
            )
            cluster_names[-1] = f"[OUTLIER] {outlier_name}"

        # Tạo rows
        for label in unique_labels:
            info = clusters_data[label]
            lop_2_suggest = cluster_names.get(label, 'Chưa phân loại')
            loai = detect_type(hs_code, lop_2_suggest, info['raw'][:5])

            all_rows.append({
                'Mã HS':       hs_code,
                'Dòng SP':     dong_sp,
                'Loại':        loai,
                'Lớp 1':       lop_1_default,
                'Lớp 2':       lop_2_suggest,
                'Keyword':     '',
                'Cluster_ID':  int(label),
                'Số lượng SP': info['count'],
                'Mô tả mẫu':  info['sample'],
            })
            status = "OUTLIER" if label == -1 else f"Cluster {label}"
            print(f"    {status}: {info['count']} sp → \"{lop_2_suggest}\"")

    # ── 4. Gộp cluster trùng tên + tạo DataFrame ─────────────
    print(f"\n[4/5] Tạo file kết quả...")

    before_merge = len(all_rows)
    all_rows = merge_duplicate_clusters(all_rows)
    after_merge = len(all_rows)
    if before_merge > after_merge:
        print(f"  → Gộp cluster trùng tên: {before_merge} → {after_merge} nhóm")

    df_result = pd.DataFrame(all_rows)
    # Sắp xếp: Mã HS → Số lượng SP giảm dần (nhóm lớn lên đầu dễ review)
    df_result = df_result.sort_values(
        ['Mã HS', 'Số lượng SP'], ascending=[True, False]
    ).reset_index(drop=True)

    # ── 5. Xuất Excel ─────────────────────────────────────────
    print(f"\n[5/5] Xuất file: {output_path}")

    save_path = output_path
    try:
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            df_export = df_result[['Keyword', 'Mã HS', 'Dòng SP', 'Loại', 'Lớp 1', 'Lớp 2', 'Cluster_ID']].copy()
            df_export.to_excel(writer, sheet_name='Phân loại', index=False)
            df_result.to_excel(writer, sheet_name='Chi tiết Cluster', index=False)
            df_raw_export = df_raw[['HS_Code', 'Detailed_Product', '_clean', '_cluster']].copy()
            df_raw_export.columns = ['Mã HS', 'Tên hàng gốc', 'Đã làm sạch', 'Cluster_ID']
            df_raw_export.to_excel(writer, sheet_name='Raw + Cluster', index=False)
    except PermissionError:
        base, ext = os.path.splitext(output_path)
        save_path = f"{base}_new{ext}"
        print(f"  ⚠ File gốc đang mở trong Excel → lưu sang: {os.path.basename(save_path)}")
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            df_export = df_result[['Keyword', 'Mã HS', 'Dòng SP', 'Loại', 'Lớp 1', 'Lớp 2', 'Cluster_ID']].copy()
            df_export.to_excel(writer, sheet_name='Phân loại', index=False)
            df_result.to_excel(writer, sheet_name='Chi tiết Cluster', index=False)
            df_raw_export = df_raw[['HS_Code', 'Detailed_Product', '_clean', '_cluster']].copy()
            df_raw_export.columns = ['Mã HS', 'Tên hàng gốc', 'Đã làm sạch', 'Cluster_ID']
            df_raw_export.to_excel(writer, sheet_name='Raw + Cluster', index=False)

    print("\n" + "=" * 60)
    print(f"  ✅ HOÀN TẤT!")
    print(f"  → File kết quả: {save_path}")
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
