from __future__ import annotations

from pathlib import Path
from shutil import copyfile

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "Bao_cao_IS54A_DrowsyDriverAndroid_bo_sung_am_thanh_dashboard.docx"
TARGET = ROOT / "outputs" / "Bao_cao_IS54A_DrowsyDriverAndroid_bo_sung_am_thanh_lien_tuc_dashboard.docx"

FONT_NAME = "Times New Roman"
FONT_SIZE = Pt(13)
SPACING = Pt(3)


def apply_run_style(run, bold: bool = False, italic: bool = False):
    run.font.name = FONT_NAME
    run.font.size = FONT_SIZE
    run.bold = bold
    run.italic = italic


def add_heading(doc: Document, text: str, centered: bool = False):
    paragraph = doc.add_paragraph()
    paragraph.style = doc.styles["Normal"]
    paragraph.paragraph_format.space_before = SPACING
    paragraph.paragraph_format.space_after = SPACING
    if centered:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    apply_run_style(run, bold=True)
    return paragraph


def add_paragraph(doc: Document, text: str):
    paragraph = doc.add_paragraph()
    paragraph.style = doc.styles["Normal"]
    paragraph.paragraph_format.space_before = SPACING
    paragraph.paragraph_format.space_after = SPACING
    run = paragraph.add_run(text)
    apply_run_style(run)
    return paragraph


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
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = SPACING
                paragraph.paragraph_format.space_after = SPACING
                for run in paragraph.runs:
                    if run.bold is None:
                        apply_run_style(run)
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
    add_heading(doc, "PHẦN BỔ SUNG 2: CƠ CHẾ NỀN KÍCH THÍCH ÂM THANH LIÊN TỤC CÓ GIỚI HẠN", centered=True)

    add_heading(doc, "1. Lý do không dùng âm thanh liên tục vô hạn")
    add_paragraph(
        doc,
        "Ý tưởng phát một tần số nền để giữ người lái tỉnh táo cần được trình bày thận trọng. Âm thanh có thể làm tăng chú ý trong thời gian ngắn, nhưng không thể đảo ngược thiếu ngủ sinh lý. Nếu người lái đã xuất hiện microsleep hoặc buồn ngủ không kiểm soát, giải pháp an toàn không phải là cố duy trì lái xe bằng âm thanh, mà là dừng xe tại vị trí an toàn, dùng caffeine nếu phù hợp và ngủ ngắn.",
    )
    add_paragraph(
        doc,
        "Ngoài ra, âm thanh liên tục quá lớn hoặc kéo dài có thể gây khó chịu, giảm khả năng nghe tín hiệu giao thông, gây habituation làm người dùng quen dần với cảnh báo, và tạo rủi ro thính giác. CDC/NIOSH khuyến nghị giới hạn phơi nhiễm tiếng ồn nghề nghiệp ở 85 dBA trung bình trong 8 giờ; khi âm lượng tăng, thời gian tiếp xúc cho phép giảm nhanh. Vì vậy, cơ chế này chỉ nên dùng như một lớp can thiệp khẩn cấp, có thời lượng tối đa và có nút dừng rõ ràng.",
    )

    add_heading(doc, "2. Bằng chứng liên quan")
    add_paragraph(
        doc,
        "Trong môi trường lái xe mô phỏng, Lin và cộng sự khảo sát các cảnh báo âm thanh liên tục và dạng tone burst ở 500 Hz, 1750 Hz và 3000 Hz. Kết quả tóm tắt của nghiên cứu cho thấy cảnh báo âm thanh cải thiện phản ứng lái và đặc tính phổ của âm thanh có ảnh hưởng đến hiệu quả cảnh báo. Điều này hỗ trợ việc thử nghiệm các dải tần nghe được, nhưng không chứng minh rằng một âm thanh nền có thể thay thế nghỉ ngơi.",
    )
    add_paragraph(
        doc,
        "Các nghiên cứu về âm thanh báo thức và sleep inertia cho thấy âm thanh có giai điệu/nhịp có thể liên quan đến cảm nhận tỉnh táo sau khi thức dậy. Tuy nhiên, bối cảnh đó khác với lái xe đang buồn ngủ. Vì vậy, đề tài chỉ dùng bằng chứng này để thiết kế âm thanh dễ nhận biết, ít gây hoảng loạn hơn, không dùng để khẳng định hiệu quả y học.",
    )

    add_table(
        doc,
        ["Dải/chế độ", "Cách dùng đề xuất", "Lý do", "Giới hạn an toàn"],
        [
            [
                "500 Hz - low tone",
                "Âm nền rất ngắn hoặc tone burst chậm khi cảnh báo cấp trung bình.",
                "Tần số thấp hơn, ít sắc gắt hơn, có thể nghe rõ trong cabin.",
                "Không dùng kéo dài một tần số đơn vì dễ quen và gây khó chịu.",
            ],
            [
                "1750 Hz - mid tone",
                "Dải chính cho chế độ nền kích thích vì dễ nghe trên loa điện thoại/cabin.",
                "Nằm trong vùng tai người nhạy và đã được khảo sát trong nghiên cứu lái xe mô phỏng.",
                "Giới hạn 30-60 giây mỗi lần, âm lượng dưới mức gây khó chịu.",
            ],
            [
                "3000 Hz - high tone",
                "Dùng xen kẽ ngắn để tăng độ khẩn cấp, không chạy liên tục lâu.",
                "Tạo cảm giác nổi bật hơn trong môi trường có nhiễu.",
                "Có thể chói tai, nên chỉ dùng theo nhịp ngắn và cho phép tắt.",
            ],
            [
                "Infrasound/ultrasound",
                "Không dùng trong prototype.",
                "Loa điện thoại/dashcam thường không phát đúng; bằng chứng ứng dụng an toàn cho lái xe không đủ.",
                "Có thể gây khó chịu hoặc không kiểm soát được mức tiếp xúc.",
            ],
        ],
    )

    add_heading(doc, "3. Cơ chế đề xuất trong hệ thống")
    add_paragraph(
        doc,
        "Cơ chế nền kích thích âm thanh chỉ bật khi hệ thống xác định người lái đang ở mức nguy hiểm: ví dụ tổng thời gian DROWSY vượt 15 giây trong 5 phút, hoặc có từ 3 cảnh báo cấp 2 trong 10 phút. Khi bật, hệ thống phát một vòng âm thanh có nhịp ở dải 1750 Hz, xen kẽ 500 Hz và 3000 Hz để tránh quen âm. Vòng âm thanh tự tắt sau 30-60 giây, hoặc tắt ngay khi người lái xác nhận đã dừng xe/an toàn.",
    )
    add_paragraph(
        doc,
        "Nếu sau 60 giây người lái vẫn tiếp tục có dấu hiệu buồn ngủ, hệ thống không tăng âm vô hạn mà chuyển sang cấp an toàn: hiển thị yêu cầu dừng xe, phát thông báo giọng nói, ghi log sự kiện, và gửi vị trí cho liên hệ tin cậy nếu người dùng đã cấp quyền. Đây là điểm quan trọng để tránh biến hệ thống cảnh báo thành công cụ khuyến khích lái xe khi đã thiếu ngủ nghiêm trọng.",
    )

    add_table(
        doc,
        ["Điều kiện", "Hành động âm thanh", "Hành động hệ thống"],
        [
            [
                "DROWSY ngắn",
                "Beep/tone burst 1000-1750 Hz.",
                "Hiển thị cảnh báo cấp 1, không bật nền liên tục.",
            ],
            [
                "DROWSY lặp lại",
                "Bật nền kích thích 1750 Hz có nhịp, xen kẽ 500/3000 Hz trong tối đa 30 giây.",
                "Yêu cầu người lái xác nhận hoặc tìm nơi dừng.",
            ],
            [
                "Không phản hồi hoặc tiếp tục buồn ngủ sau 60 giây",
                "Dừng nền liên tục, chuyển sang voice alert rõ ràng.",
                "Khuyến nghị dừng xe; nếu được cấp quyền, gửi vị trí cho liên hệ tin cậy.",
            ],
        ],
    )

    add_heading(doc, "4. Tài liệu tham khảo bổ sung")
    refs = [
        "Lin, C.-T., Huang, T.-Y., Liang, W.-C., Chiu, T.-T., Chao, C.-F., Hsu, S.-H., & Ko, L.-W. (2009). Assessing Effectiveness of Various Auditory Warning Signals in Maintaining Drivers' Attention in Virtual Reality-Based Driving Environments. Perceptual and Motor Skills, 108(3), 825-835. https://doi.org/10.2466/pms.108.3.825-835",
        "McFarlane, S. J., Garcia, J. E., Verhagen, D. S., & Dyer, A. G. (2020). Alarm tones, music and their elements: Analysis of reported waking sounds to counteract sleep inertia. PLOS ONE, 15(1), e0215788. https://doi.org/10.1371/journal.pone.0215788",
        "CDC/NIOSH. Noise-Induced Hearing Loss. https://www.cdc.gov/niosh/noise/about/noise.html",
        "NHTSA. Drowsy Driving: Avoid Falling Asleep Behind the Wheel. https://www.nhtsa.gov/risky-driving/drowsy-driving",
        "CDC/NIOSH. Driver Fatigue on the Job. https://www.cdc.gov/niosh/motor-vehicle/driver-fatigue/index.html",
    ]
    for ref in refs:
        add_paragraph(doc, ref)

    doc.save(TARGET)
    print(TARGET)


if __name__ == "__main__":
    main()
