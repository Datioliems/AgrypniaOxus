// Sinh Chương 3 (Xây dựng mô hình) — Word đúng format IS54A.
const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, LevelFormat, WidthType, BorderStyle, ShadingType } = require("docx");
const IMG = "outputs/report_imgs"; const img = (n) => fs.readFileSync(`${IMG}/${n}`);

const body = (t) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 312 }, indent: { firstLine: 720 }, children: [new TextRun(t)] });
const H = (lv, t) => new Paragraph({ heading: [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3, HeadingLevel.HEADING_4][lv], numbering: { reference: "hd", level: lv }, spacing: { before: 240, after: 120 }, children: [new TextRun({ text: t, bold: true })] });
const figure = (f, cap, w, h) => [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, children: [new ImageRun({ type: f.endsWith(".png") ? "png" : "jpg", data: img(f), transformation: { width: w, height: h } })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new TextRun({ text: cap, italics: true, size: 24 })] }),
];
const tcap = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, children: [new TextRun({ text: t, italics: true, size: 24 })] });
const code = (lines) => new Paragraph({ spacing: { before: 80, after: 120 }, shading: { type: ShadingType.CLEAR, fill: "F2F2F2" },
  border: { top: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" }, bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" }, left: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" }, right: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" } },
  children: lines.map((l, i) => new TextRun({ text: l, font: "Consolas", size: 20, break: i ? 1 : 0 })) });

const cell = (t, head, w) => new TableCell({ width: { size: w, type: WidthType.DXA }, shading: head ? { type: ShadingType.CLEAR, fill: "D5E8F0" } : undefined, margins: { top: 50, bottom: 50, left: 80, right: 80 }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: t, bold: !!head, size: 22 })] })] });
const W = [1700, 1500, 1100, 1000, 1100, 1100, 1860];
const row = (cells, head) => new TableRow({ children: cells.map((c, i) => cell(c, head, W[i])) });
const hpTable = new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: W, rows: [
  row(["Mô hình", "Pretrained", "Epochs", "Batch", "imgsz", "LR", "Optimizer/Ghi chú"], true),
  row(["CNN mắt/ngáp", "Không", "20-30", "16-32", "64", "1e-3", "Adam, dropout 0.4"]),
  row(["YOLOv11s", "COCO", "60", "16", "640", "1e-2", "SGD, cos_lr"]),
  row(["YOLO26s", "COCO", "60", "16", "640", "auto", "MuSGD (NMS-free)"]),
  row(["RT-DETR-L", "COCO", "10", "8", "640", "1e-4", "AdamW (transformer)"]),
  row(["RF-DETR-S", "DINOv2", "10", "4", "560", "1e-4", "grad_accum=4"]),
]});

const refs = [
  "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.",
  "Jocher, G., et al. (2024). Ultralytics YOLO11. https://docs.ultralytics.com",
  "Khanam, R., & Hussain, M. (2024). YOLOv11: An overview of the key architectural enhancements. arXiv:2410.17725.",
  "LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.",
  "Robinson, I., et al. (2025). RF-DETR: A SOTA real-time object detection model. Roboflow.",
  "Soukupová, T., & Čech, J. (2016). Real-time eye blink detection using facial landmarks. CVWW.",
  "Zhao, Y., et al. (2024). DETRs beat YOLOs on real-time object detection (RT-DETR). CVPR.",
];

const doc = new Document({
  numbering: { config: [{ reference: "hd", levels: [
    { level: 0, format: LevelFormat.DECIMAL, text: "CHƯƠNG %1:", alignment: AlignmentType.START, start: 3 },
    { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.START },
    { level: 2, format: LevelFormat.DECIMAL, text: "%1.%2.%3.", alignment: AlignmentType.START },
    { level: 3, format: LevelFormat.LOWER_LETTER, text: "%4.", alignment: AlignmentType.START },
  ] }] },
  styles: { default: { document: { run: { font: "Times New Roman", size: 26 } } }, paragraphStyles: [
    { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: "Times New Roman", size: 32, bold: true, color: "1F3864" }, paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
    { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: "Times New Roman", size: 28, bold: true, color: "2E5496" }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: "Times New Roman", size: 26, bold: true }, paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
    { id: "Heading4", name: "Heading 4", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: "Times New Roman", size: 26, bold: true, italics: true }, paragraph: { spacing: { before: 120, after: 60 }, outlineLevel: 3 } },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children: [
      H(0, " XÂY DỰNG MÔ HÌNH"),
      body("Chương này trình bày quá trình trích chọn đặc trưng, lựa chọn và xây dựng các mô hình giải quyết bài toán, cũng như cấu hình huấn luyện. Hệ thống được thiết kế theo kiến trúc lai (hybrid): kết hợp đặc trưng hình học có thể giải thích được với mạng học sâu, đồng thời so sánh nhiều họ kiến trúc phát hiện vật thể nhằm lựa chọn phương án tối ưu cho thiết bị di động. Hình 3.1 mô tả tổng quan kiến trúc hệ thống chạy hoàn toàn trên thiết bị, không phụ thuộc kết nối đám mây."),
      ...figure("arch_pipeline.png", "Hình 3.1. Kiến trúc tổng quan của hệ thống phát hiện buồn ngủ chạy trên thiết bị", 600, 229),

      H(1, "Trích chọn đặc trưng"),
      body("Dữ liệu của bài toán là dữ liệu hình ảnh, do đó việc trích chọn đặc trưng đóng vai trò then chốt. Đề tài khai thác đồng thời hai hướng trích xuất đặc trưng bổ trợ cho nhau: đặc trưng học sâu được mạng tích chập tự học từ dữ liệu, và đặc trưng hình học được tính toán thủ công từ các điểm mốc khuôn mặt."),
      H(2, "Đặc trưng học sâu từ mạng tích chập"),
      body("Mạng nơ-ron tích chập (Convolutional Neural Network) tự động học các đặc trưng phân cấp từ ảnh thông qua các bộ lọc tích chập (kernel) trượt trên ảnh để phát hiện cạnh, góc, kết cấu ở lớp nông và các bộ phận phức tạp hơn ở lớp sâu (LeCun, Bengio, & Hinton, 2015). So với phương pháp thiết kế đặc trưng thủ công, CNN có khả năng học biểu diễn tối ưu trực tiếp từ dữ liệu, nhờ đó phân biệt tốt trạng thái mắt mở và mắt nhắm ngay cả khi điều kiện ánh sáng thay đổi."),
      H(2, "Đặc trưng hình học EAR và MAR"),
      body("Song song với đặc trưng học sâu, hệ thống tính các đặc trưng hình học là Tỷ lệ khung hình mắt (EAR) và miệng (MAR) từ các điểm mốc do MediaPipe cung cấp, theo công thức của Soukupová và Čech (2016). Đây là phương pháp nằm ngoài chương trình học cơ bản; ưu điểm là tốc độ tính toán rất nhanh, không cần huấn luyện và có khả năng giải thích trực tiếp (giá trị EAR giảm khi mắt nhắm). Việc kết hợp hai loại đặc trưng cho phép xây dựng cơ chế hợp nhất vừa chính xác vừa an toàn, trình bày ở mục 3.2.4."),

      H(1, "Lựa chọn thuật toán và mô hình"),
      body("Bộ dữ liệu ở mức điển hình, đủ lớn để phân chia theo tỷ lệ 60% huấn luyện, 20% xác thực và 20% kiểm thử mà không cần kiểm định chéo. Riêng với tập dữ liệu tự thu thập có quy mô nhỏ, nhóm áp dụng tăng cường dữ liệu và trọng số lớp (class weight) để giảm thiểu hiện tượng quá khớp và mất cân bằng. Đề tài triển khai bốn nhóm mô hình rồi so sánh để chọn phương án tối ưu."),
      H(2, "Mạng tích chập phân loại trạng thái"),
      body("Mô hình phân loại trạng thái mắt và ngáp được xây dựng dưới dạng mạng tích chập gọn nhẹ, gồm ba khối tích chập với số bộ lọc tăng dần, kết hợp chuẩn hóa theo lô (Batch Normalization) và lớp gộp cực đại (Max Pooling), sau đó là lớp gộp trung bình toàn cục (Global Average Pooling) và lớp kết nối đầy đủ có Dropout để chống quá khớp. Kiến trúc này được minh họa ở Hình 3.2 và mã nguồn xây dựng được trình bày trong đoạn mã bên dưới."),
      ...figure("arch_cnn.png", "Hình 3.2. Kiến trúc mạng tích chập phân loại trạng thái mắt và ngáp", 600, 142),
      code(["model = Sequential([", "  Conv2D(32,3,activation='relu',padding='same'), BatchNormalization(), MaxPool2D(),",
            "  Conv2D(64,3,activation='relu',padding='same'), BatchNormalization(), MaxPool2D(),",
            "  Conv2D(128,3,activation='relu',padding='same'), GlobalAveragePooling2D(),",
            "  Dense(128,activation='relu'), Dropout(0.4), Dense(2,activation='softmax')])"]),
      H(2, "Các mô hình phát hiện vật thể YOLO"),
      body("Để phát hiện và định vị đồng thời vùng mắt, miệng theo sáu lớp, nhóm sử dụng hai kiến trúc YOLO mới nhất là YOLOv11 và YOLO26 (Jocher và cộng sự, 2024; Khanam & Hussain, 2024). Các mô hình được nạp trọng số tiền huấn luyện trên tập COCO rồi tinh chỉnh (fine-tune) trên bộ dữ liệu sáu lớp thông qua thư viện Ultralytics. YOLO26 đặc biệt ở chỗ có đầu dự đoán một-một (end-to-end), loại bỏ bước hậu xử lý NMS, giúp triển khai gọn hơn trên thiết bị."),
      H(2, "Các mô hình Transformer thời gian thực"),
      body("Bên cạnh YOLO, nhóm thử nghiệm hai kiến trúc dựa trên Transformer là RT-DETR (Zhao và cộng sự, 2024) và RF-DETR (Robinson và cộng sự, 2025). Khác với YOLO dùng anchor và NMS, các mô hình DETR sử dụng bộ giải mã Transformer để dự đoán trực tiếp tập hợp đối tượng nhờ cơ chế chú ý toàn cục, qua đó định vị tốt hơn trong các trường hợp khuôn mặt bị che khuất hoặc nghiêng. RF-DETR sử dụng backbone DINOv2 tiền huấn luyện, cho phép hội tụ nhanh chỉ với số epoch nhỏ."),
      H(2, "Cơ chế hợp nhất quyết định"),
      body("Hệ thống triển khai cơ chế hợp nhất lấy mạng tích chập làm chủ đạo: khi CNN đủ tin cậy, kết quả của CNN được dùng để quyết định trạng thái; ngược lại, hệ thống quay về sử dụng đặc trưng EAR/MAR như cơ chế dự phòng. Đặc biệt, để bảo đảm an toàn, nếu đặc trưng hình học cũng khẳng định trạng thái buồn ngủ thì cảnh báo vẫn được kích hoạt nhằm không bỏ sót trường hợp nguy hiểm. Kết hợp với tầng làm mượt theo thời gian (temporal), cơ chế này giúp giảm cả báo động giả lẫn bỏ sót."),

      H(1, "Cấu hình huấn luyện mô hình"),
      H(2, "Cấu hình phần cứng"),
      body("Quá trình huấn luyện các mô hình phát hiện vật thể và Transformer được thực hiện trên nền tảng Google Colab với GPU NVIDIA Tesla T4 (16 GB). Các mạng tích chập gọn nhẹ được huấn luyện trên máy cá nhân (CPU/GPU rời) do quy mô tham số nhỏ. Mô hình sau huấn luyện được chuyển sang định dạng TensorFlow Lite để chạy suy luận trực tiếp trên thiết bị Android."),
      H(2, "Cấu hình siêu tham số"),
      body("Bảng 3.1 tổng hợp các siêu tham số chính của từng mô hình. Các giá trị được lựa chọn dựa trên khuyến nghị của thư viện và đặc thù dữ liệu: dữ liệu nhỏ dùng số epoch và độ tăng cường phù hợp để tránh quá khớp; Transformer dùng kích thước lô nhỏ do tiêu tốn bộ nhớ lớn hơn mạng tích chập."),
      tcap("Bảng 3.1. Cấu hình siêu tham số của các mô hình trong đề tài."),
      hpTable, body(""),

      new Paragraph({ spacing: { before: 200 }, children: [new TextRun({ text: "TÀI LIỆU THAM KHẢO (CHƯƠNG 3)", bold: true })] }),
      ...refs.map((r) => new Paragraph({ spacing: { after: 60 }, indent: { left: 720, hanging: 720 }, children: [new TextRun({ text: r, size: 24 })] })),
    ],
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync("outputs/BaoCao_Chuong3.docx", b); console.log("OK -> outputs/BaoCao_Chuong3.docx"); });
