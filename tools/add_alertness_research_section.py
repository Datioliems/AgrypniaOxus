from __future__ import annotations

from pathlib import Path
from shutil import copyfile

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "Bao_cao_IS54A_DrowsyDriverAndroid_theo_khung.docx"
TARGET = ROOT / "outputs" / "Bao_cao_IS54A_DrowsyDriverAndroid_bo_sung_am_thanh_dashboard.docx"


FONT_NAME = "Times New Roman"
FONT_SIZE = Pt(13)
SPACING = Pt(3)


def apply_run_style(run, bold: bool = False, italic: bool = False):
    run.font.name = FONT_NAME
    run.font.size = FONT_SIZE
    run.bold = bold
    run.italic = italic


def style_paragraph(paragraph):
    paragraph.paragraph_format.space_before = SPACING
    paragraph.paragraph_format.space_after = SPACING
    for run in paragraph.runs:
        apply_run_style(run)


def add_heading(doc: Document, text: str, level: int = 1):
    paragraph = doc.add_paragraph()
    paragraph.style = doc.styles["Normal"]
    paragraph.paragraph_format.space_before = SPACING
    paragraph.paragraph_format.space_after = SPACING
    run = paragraph.add_run(text)
    apply_run_style(run, bold=True)
    if level == 1:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return paragraph


def add_paragraph(doc: Document, text: str, bold_prefix: str | None = None):
    paragraph = doc.add_paragraph()
    paragraph.style = doc.styles["Normal"]
    paragraph.paragraph_format.space_before = SPACING
    paragraph.paragraph_format.space_after = SPACING
    if bold_prefix and text.startswith(bold_prefix):
        run = paragraph.add_run(bold_prefix)
        apply_run_style(run, bold=True)
        rest = paragraph.add_run(text[len(bold_prefix) :])
        apply_run_style(rest)
    else:
        run = paragraph.add_run(text)
        apply_run_style(run)
    return paragraph


def style_table(table):
    table.style = "Table Grid"
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = SPACING
                paragraph.paragraph_format.space_after = SPACING
                for run in paragraph.runs:
                    apply_run_style(run)


def add_table(doc: Document, headers: list[str], rows: list[list[str]]):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for index, header in enumerate(headers):
        run = table.rows[0].cells[index].paragraphs[0].add_run(header)
        apply_run_style(run, bold=True)
    for row_values in rows:
        row = table.add_row()
        for index, value in enumerate(row_values):
            run = row.cells[index].paragraphs[0].add_run(value)
            apply_run_style(run)
    style_table(table)
    return table


def set_normal_style(doc: Document):
    normal = doc.styles["Normal"]
    normal.font.name = FONT_NAME
    normal.font.size = FONT_SIZE
    normal.paragraph_format.space_before = SPACING
    normal.paragraph_format.space_after = SPACING


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source report not found: {SOURCE}")

    copyfile(SOURCE, TARGET)
    doc = Document(TARGET)
    set_normal_style(doc)

    doc.add_page_break()
    add_heading(doc, "PHẦN BỔ SUNG: ÂM THANH CẢNH BÁO, CAN THIỆP AN TOÀN VÀ DASHBOARD GIẤC NGỦ", 1)

    add_heading(doc, "1. Cơ sở nghiên cứu về âm thanh và trạng thái tỉnh táo", 2)
    add_paragraph(
        doc,
        "Các nghiên cứu về âm thanh cảnh báo cho thấy âm thanh có thể tạo phản xạ chú ý tức thời, giúp người lái nhận ra nguy cơ và chuyển sự chú ý về nhiệm vụ lái xe. Tuy nhiên, hiện chưa có bằng chứng chắc chắn rằng một tần số đơn lẻ có thể làm con người hết buồn ngủ ngay lập tức. Vì vậy, trong đề tài này, âm thanh được xem là cơ chế cảnh báo và kích hoạt hành động an toàn, không phải biện pháp thay thế giấc ngủ.",
    )
    add_paragraph(
        doc,
        "Nghiên cứu về tín hiệu cảnh báo đa âm và biến điệu tần số của Haas và Casali cho thấy mức âm lượng, dạng xung và khoảng cách giữa các xung ảnh hưởng đến cảm nhận khẩn cấp và thời gian phát hiện tín hiệu. Các tài liệu hướng dẫn human factors cho hệ thống cảnh báo trong xe cũng khuyến nghị dùng âm thanh ngắn, rõ, đủ nổi bật trong môi trường ồn và không gây quá tải nhận thức.",
    )
    add_paragraph(
        doc,
        "Một nhóm nghiên cứu khác về binaural beats/gamma 40 Hz cho thấy khả năng tác động đến chú ý hoặc trạng thái nhận thức trong một số bài kiểm tra, nhưng bằng chứng còn giới hạn và không trực tiếp chứng minh hiệu quả an toàn khi lái xe. Do đó, 40 Hz chỉ nên được trình bày như hướng nghiên cứu mở rộng hoặc chế độ hỗ trợ tỉnh táo khi xe đã dừng, không dùng như cảnh báo chính khi đang điều khiển phương tiện.",
    )

    add_table(
        doc,
        ["Nhóm bằng chứng", "Kết luận chính", "Cách áp dụng trong đề tài"],
        [
            [
                "Âm thanh cảnh báo trong hệ thống an toàn",
                "Tín hiệu rõ, ngắn, có nhịp lặp và đủ âm lượng giúp người dùng phát hiện nguy cơ nhanh hơn.",
                "Dùng chuỗi beep/voice warning theo cấp độ, ưu tiên dải dễ nghe khoảng 500-3000 Hz và tránh âm quá lớn kéo dài.",
            ],
            [
                "Binaural/gamma 40 Hz",
                "Có một số nghiên cứu về chú ý và EEG, nhưng chưa đủ để coi là biện pháp tỉnh táo tức thời khi lái xe.",
                "Đưa vào phần nghiên cứu mở rộng; không dùng headphone khi lái; không thay thế yêu cầu nghỉ ngơi.",
            ],
            [
                "Khuyến nghị an toàn drowsy driving",
                "NHTSA, CDC/NIOSH và AASM đều nhấn mạnh ngủ đủ, dừng xe an toàn, caffeine và ngủ ngắn khi buồn ngủ.",
                "Khi cảnh báo nặng hoặc lặp lại, hệ thống phải khuyến nghị dừng xe thay vì chỉ phát âm thanh.",
            ],
        ],
    )

    add_heading(doc, "2. Cơ chế cảnh báo và can thiệp đề xuất", 2)
    add_paragraph(
        doc,
        "Cơ chế đề xuất sử dụng âm thanh theo cấp độ. Ở mức nhẹ, hệ thống phát âm ngắn để thu hút chú ý. Ở mức trung bình, hệ thống lặp cảnh báo và yêu cầu người lái xác nhận đã tỉnh táo. Ở mức nguy hiểm, hệ thống chuyển sang khuyến nghị dừng xe, nghỉ ngắn và có thể thông báo cho người thân nếu người dùng đã cho phép từ trước.",
    )

    add_table(
        doc,
        ["Cấp độ", "Điều kiện kích hoạt gợi ý", "Âm thanh/can thiệp", "Mục tiêu an toàn"],
        [
            [
                "Cấp 1 - Nhắc nhở",
                "Mắt nhắm liên tục 1,2-1,7 giây hoặc EAR thấp nhiều frame liên tiếp.",
                "Beep ngắn 800-1200 Hz trong 200-300 ms, hiển thị Awake/Eyes closed.",
                "Kéo chú ý về màn hình/đường đi, hạn chế báo động quá mạnh.",
            ],
            [
                "Cấp 2 - Cảnh báo",
                "DROWSY kéo dài trên 2 giây hoặc có nhiều cảnh báo trong 60 giây.",
                "Âm xen kẽ 1000 Hz và 2500 Hz, lặp 3 chu kỳ; voice: 'Bạn có dấu hiệu buồn ngủ'.",
                "Tạo phản ứng khẩn cấp vừa đủ, yêu cầu người lái thay đổi hành vi.",
            ],
            [
                "Cấp 3 - Nguy hiểm",
                "Tổng thời gian buồn ngủ vượt 15 giây trong 5 phút, hoặc 3 lần cảnh báo cấp 2 trong 10 phút.",
                "Âm cảnh báo mạnh hơn, yêu cầu dừng xe; đề xuất cà phê + ngủ ngắn 15-20 phút; nếu không xác nhận, gửi vị trí cho liên hệ tin cậy.",
                "Không cố 'đánh thức' người lái bằng âm thanh đơn thuần, mà chuyển sang hành động giảm rủi ro.",
            ],
        ],
    )

    add_paragraph(
        doc,
        "Ngưỡng thời gian trên là cấu hình đề xuất cho demo và cần hiệu chỉnh bằng dữ liệu thực tế. Trong sản phẩm thực tế, âm lượng phải được giới hạn để tránh gây giật mình quá mức hoặc ảnh hưởng thính giác; đồng thời cần có cơ chế tắt/bật, xác nhận và ghi log tối thiểu.",
    )

    add_heading(doc, "3. Trang dashboard giấc ngủ và phân tích thời điểm buồn ngủ", 2)
    add_paragraph(
        doc,
        "Bên cạnh app realtime, đề tài có thể bổ sung một dashboard web/Streamlit để phân tích lịch sử buồn ngủ theo thời gian trong ngày. Dữ liệu đầu vào có thể lấy từ event log của Android, gồm thời điểm cảnh báo, trạng thái Awake/Drowsy/Yawning, thời lượng mắt nhắm, mức độ cảnh báo, FPS/latency và vị trí GPS nếu người dùng cho phép.",
    )
    add_paragraph(
        doc,
        "Dashboard nên hiển thị heatmap 24 giờ, số lần cảnh báo theo khung giờ, tổng thời lượng buồn ngủ, khuyến nghị thời điểm nghỉ, và mức rủi ro theo ngày. Chức năng này giúp người dùng nhận ra các khung giờ thường xuyên buồn ngủ, ví dụ sau bữa trưa, cuối ca làm việc, ban đêm hoặc sau khi ngủ không đủ.",
    )
    add_paragraph(
        doc,
        "Tính năng thông báo người thân nên được thiết kế theo nguyên tắc privacy-by-design: người dùng tự cấu hình liên hệ tin cậy, tự cho phép chia sẻ vị trí, không gửi video khuôn mặt mặc định, chỉ gửi sự kiện nguy hiểm và vị trí gần nhất. Khi cảnh báo cấp 3 xảy ra và người lái không phản hồi trong một khoảng thời gian, hệ thống có thể gửi tin nhắn như: 'Người lái có dấu hiệu buồn ngủ nghiêm trọng, vị trí gần nhất: ...'.",
    )

    add_table(
        doc,
        ["Thành phần dashboard", "Dữ liệu cần dùng", "Ý nghĩa"],
        [
            ["Heatmap 24 giờ", "timestamp, alert_level", "Tìm khung giờ buồn ngủ lặp lại."],
            ["Sleep risk score", "số cảnh báo, thời lượng drowsy, yawning", "Tóm tắt rủi ro trong ngày/tuần."],
            ["Khuyến nghị nghỉ", "mức cảnh báo, thời gian lái liên tục", "Đề xuất dừng xe, ngủ ngắn hoặc đổi tài xế."],
            ["Trusted contact", "GPS, event type, xác nhận người dùng", "Gửi vị trí khi cảnh báo nguy hiểm và không phản hồi."],
        ],
    )

    add_heading(doc, "4. Tài liệu tham khảo bổ sung", 2)
    references = [
        "NHTSA. Drowsy Driving: Avoid Falling Asleep Behind the Wheel. https://www.nhtsa.gov/risky-driving/drowsy-driving",
        "CDC/NIOSH. Driver Fatigue on the Job. https://www.cdc.gov/niosh/motor-vehicle/driver-fatigue/index.html",
        "American Academy of Sleep Medicine. Confronting Drowsy Driving: The AASM Perspective. https://pmc.ncbi.nlm.nih.gov/articles/PMC4623133/",
        "Haas, E. C., & Casali, J. G. (1993). The Perceived Urgency and Detection Time of Multi-Tone and Frequency-Modulated Warning Signals. Proceedings of the Human Factors and Ergonomics Society Annual Meeting.",
        "Reedijk, S. A., Bolders, A., & Hommel, B. More attentional focusing through binaural beats: evidence from the global-local task. https://pmc.ncbi.nlm.nih.gov/articles/PMC5233742/",
        "Ultralytics/vehicle safety related implementation references are used only for engineering deployment; medical and road-safety claims rely on NHTSA, CDC/NIOSH and AASM.",
    ]
    for ref in references:
        add_paragraph(doc, ref)

    doc.save(TARGET)
    print(TARGET)


if __name__ == "__main__":
    main()
