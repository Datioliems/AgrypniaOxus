# -*- coding: utf-8 -*-
"""Sinh BẢNG PHÂN CÔNG CÔNG VIỆC (.docx) — khớp đúng Mục lục báo cáo IS54A."""
import docx
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

doc = docx.Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(13)

h = doc.add_paragraph(); h.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = h.add_run("BẢNG PHÂN CÔNG CÔNG VIỆC"); r.bold = True; r.font.size = Pt(14)
doc.add_paragraph(
    "Tổng phần trăm đóng góp của các thành viên = 100%. "
    "Mọi mục trong quyển báo cáo (kể cả phần đầu, Tài liệu tham khảo) "
    "đều xuất hiện trong bảng dưới đây."
).italic = True

COLS = ["Họ tên – Mã sinh viên", "Quyển báo cáo", "Sản phẩm (mã nguồn)", "% – Ký tên"]
WID  = [Cm(3.2), Cm(6.4), Cm(6.6), Cm(2.3)]

members = [
    # ── Hằng: Chương 2 ──────────────────────────────────────────────────────
    (
        "Đỗ Vũ Việt Hằng\nMSSV: 26A404____",
        [
            "CHƯƠNG 2: CHUẨN BỊ DỮ LIỆU",
            "2.1. Mô hình CNN phân loại trạng thái mắt (MRL Eye Dataset)",
            "  2.1.1. Thu thập dữ liệu",
            "  2.1.2. Tiền xử lý dữ liệu",
            "2.2. Mô hình CNN phân loại trạng thái ngáp (Yawn Dataset)",
            "  2.2.1. Thu thập dữ liệu",
            "  2.2.2. Tiền xử lý dữ liệu",
            "2.3. Mô hình YOLOv11 phát hiện đối tượng (Datio_yolo Dataset)",
            "  2.3.1. Thu thập dữ liệu",
            "  2.3.2. Tiền xử lý dữ liệu",
            "2.4. Mô hình YOLO26 phát hiện đối tượng (Datio_yolo Dataset)",
            "  2.4.1. Thu thập dữ liệu",
            "  2.4.2. Tiền xử lý dữ liệu",
            "2.5. Mô hình RT-DETR phát hiện đối tượng (Datio_yolo Dataset)",
            "  2.5.1. Thu thập dữ liệu",
            "  2.5.2. Tiền xử lý dữ liệu",
            "2.6. Mô hình RF-DETR phát hiện đối tượng (Datio_yolo Dataset)",
            "  2.6.1. Thu thập dữ liệu",
            "  2.6.2. Tiền xử lý dữ liệu",
            "Tài liệu tham khảo (Chương 2)",
        ],
        [
            "Thu thập & tiền xử lý dữ liệu:",
            "- tools/pipeline_visual.py (10 bước tiền xử lý)",
            "- data_pipeline/preprocess.py",
            "- colab_extract_frames.ipynb (cắt frame video)",
            "- tools/step8_realdata_train.py (crop MediaPipe + trộn dataset)",
            "- tools/process_real_video.py",
            "- tools/build_report_ch2_full.js",
        ],
        "25%",
    ),

    # ── Quang: Chương 3.2 – 3.5 ──────────────────────────────────────────────
    (
        "(Họ tên) Quang\nMSSV: 26A404____",
        [
            "CHƯƠNG 3: XÂY DỰNG MÔ HÌNH (mục 3.2 – 3.5)",
            "3.2. Lựa chọn thuật toán và mô hình",
            "  3.2.1. Chiến lược phân chia dữ liệu theo subject",
            "  3.2.2. Kiến trúc các nhánh mô hình",
            "  3.2.3. Trọng số lớp (class weight) để tăng recall",
            "  3.2.4. Xử lý lỗi chuẩn hoá phát hiện trong quá trình huấn luyện",
            "3.3. Cấu hình huấn luyện mô hình",
            "  3.3.1. Cấu hình phần cứng",
            "  3.3.2. Cấu hình siêu tham số",
            "  3.3.3. Quá trình huấn luyện và kết quả",
            "3.4. So sánh hai phương pháp chia tập dữ liệu",
            "  3.4.1. Hai phương pháp chia",
            "  3.4.2. Điểm khác nhau giữa hai phương pháp",
            "  3.4.3. Vì sao chia theo subject phù hợp hơn",
            "  3.4.4. Kết quả so sánh thực nghiệm",
            "  3.4.5. Phân tích và kết luận",
            "3.5. Xây dựng các mô hình phát hiện vật thể",
            "  3.5.1. Bài toán phát hiện vật thể và bộ dữ liệu sáu lớp",
            "  3.5.2. Họ mô hình một giai đoạn YOLO",
            "  3.5.3. Họ mô hình DETR dựa trên transformer",
            "  3.5.4. Cấu hình huấn luyện bốn mô hình",
            "Tài liệu tham khảo (Chương 3)",
        ],
        [
            "Xây dựng & huấn luyện mô hình:",
            "- train_cnn_eye.py / .ipynb (CNN mắt)",
            "- train_cnn_yawn.py / .ipynb (CNN ngáp)",
            "- colab_yolo11s.py, colab_yolo26_training.py",
            "- colab_rtdetr.py, colab_rfdetr_6class.ipynb",
            "- notebooks/AllInOne_Pipeline.ipynb",
            "- app/.../TfliteDrowsinessClassifier.kt",
            "- app/.../YawnClassifier.kt, YoloDetector.kt",
            "- tools/build_report_ch3.js",
        ],
        "25%",
    ),

    # ── Thu Hoài: Phần đầu + Chương 1 + Chương 3.1 ──────────────────────────
    (
        "(Họ tên) Thu Hoài\nMSSV: 26A404____",
        [
            "DANH MỤC TỪ VIẾT TẮT",
            "LỜI CAM ĐOAN",
            "LỜI CẢM ƠN",
            "CHƯƠNG 1: GIỚI THIỆU BÀI TOÁN",
            "1.1. Giới thiệu bài toán",
            "  1.1.1. Phát biểu bài toán",
            "  1.1.2. Input của bài toán",
            "  1.1.3. Output của bài toán",
            "1.2. Ứng dụng của bài toán",
            "  1.2.1. Một số ứng dụng thực tế của bài toán",
            "  1.2.2. Giới hạn phạm vi ứng dụng của đề tài",
            "1.3. Khảo sát các bài làm liên quan",
            "  1.3.1. Các công trình nghiên cứu nước ngoài",
            "  1.3.2. Các công trình nghiên cứu trong nước",
            "  1.3.3. Nhận xét tổng hợp",
            "Chương 3: 3.1. Trích chọn đặc trưng",
            "  3.1.1. Đặc trưng hình học EAR và MAR",
            "  3.1.2. Đặc trưng học sâu với mạng nơ-ron tích chập (CNN)",
            "  3.1.3. Đặc trưng thời gian từ chuỗi EAR",
            "  3.1.4. Đặc trưng end-to-end của các mô hình phát hiện vật thể",
            "Tài liệu tham khảo (Chương 1)",
        ],
        [
            "Trích chọn đặc trưng & cơ chế cảnh báo:",
            "- app/.../DrowsinessAnalyzer.kt (EAR/MAR từ landmark)",
            "- app/.../AlertController.kt (âm thanh binaural beta §1.3)",
            "- streamlit_app.py — _ear / _mar / _make_landmarker",
            "- streamlit_app.py — make_binaural (beta 18 Hz)",
            "- tools/build_abbreviations.ipynb",
        ],
        "25%",
    ),

    # ── Nguyễn Tuấn Đạt: Phụ lục đầu + Chương 4 ─────────────────────────────
    (
        "Nguyễn Tuấn Đạt\nMSSV: 26A404____",
        [
            "BẢNG PHÂN CÔNG CÔNG VIỆC",
            "MINH CHỨNG CỘNG TÁC LÀM VIỆC",
            "CHƯƠNG 4: ĐÁNH GIÁ VÀ SO SÁNH CÁC MÔ HÌNH",
            "4.1. Độ đo đánh giá",
            "  4.1.1. Ma trận nhầm lẫn và các thành phần cơ bản",
            "  4.1.2. Accuracy",
            "  4.1.3. Precision và Recall",
            "  4.1.4. F1-score, Macro F1 và Weighted F1",
            "  4.1.5. IoU, AP và mAP (phát hiện vật thể)",
            "4.2. Thực nghiệm và kết quả đánh giá",
            "  4.2.1. Thiết lập thực nghiệm",
            "  4.2.2. Kết quả cụm mô hình phát hiện vật thể",
            "  4.2.3. Kết quả so sánh chiến lược chia dữ liệu cho CNN",
            "4.3. Demo sản phẩm trên thiết bị Android",
            "  4.3.1. Nhận xét tổng hợp và đánh giá chung",
            "TÀI LIỆU THAM KHẢO",
        ],
        [
            "Đánh giá, ứng dụng & triển khai:",
            "- tools/evaluate_combined.py (chỉ số đầy đủ)",
            "- evaluation_report.ipynb, colab_compare_all.ipynb",
            "- app/.../MainActivity.kt (tích hợp, fusion CNN-primary)",
            "- app/.../AnalyticsActivity.kt (báo cáo + khung giờ buồn ngủ)",
            "- app/.../EmergencyContactActivity.kt",
            "- app/.../EmergencyDispatcher.kt, UiKit.kt",
            "- streamlit_app.py (demo web), Dockerfile",
            "- STREAMLIT_DEPLOY.md",
            "- tools/build_report_ch4.js, tools/merge_report.py",
        ],
        "25%",
    ),
]

# ── Build table ───────────────────────────────────────────────────────────────
tbl = doc.add_table(rows=1, cols=4)
tbl.style = "Table Grid"
tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

for i, c in enumerate(tbl.rows[0].cells):
    c.width = WID[i]
    p = c.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(COLS[i]); run.bold = True


def fill(cell, lines, center=False, bold_first=False):
    cell.paragraphs[0].text = ""
    for j, ln in enumerate(lines):
        p = cell.paragraphs[0] if j == 0 else cell.add_paragraph()
        if center:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(ln)
        run.font.size = Pt(11)
        if bold_first and j == 0:
            run.bold = True


for name, report, code, pct in members:
    row = tbl.add_row().cells
    for i in range(4):
        row[i].width = WID[i]
    fill(row[0], name.split("\n"), bold_first=True)
    fill(row[1], report, bold_first=True)
    fill(row[2], code, bold_first=True)
    fill(row[3], [pct, "", "(Ký tên)"], center=True)

doc.add_paragraph()
note = doc.add_paragraph()
note.add_run("Ghi chú: ").bold = True
note.add_run(
    "Điền đầy đủ Họ tên + Mã sinh viên của Quang và Thu Hoài; "
    "điều chỉnh tỉ lệ % theo thực tế đóng góp sao cho tổng = 100%. "
    "Mục lục trong bảng này khớp đúng với file báo cáo hợp nhất "
    "outputs/BaoCao_HopNhat_DrowsyDriver.docx."
).italic = True

import os
os.makedirs("outputs", exist_ok=True)
doc.save("outputs/PhanCong_CongViec.docx")
print("OK -> outputs/PhanCong_CongViec.docx")
