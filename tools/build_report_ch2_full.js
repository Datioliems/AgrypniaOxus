// Gộp TOÀN BỘ Chương 2 (2.1 MRL + 2.2 Yawn + 2.3 YOLOv11 + 2.4 YOLO26 + 2.5 RT-DETR + 2.6 RF-DETR)
// -> Word đúng format IS54A. Tiểu mục đậm -> heading cấp 4 (a. b. c.).
const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, LevelFormat, WidthType, BorderStyle, ShadingType } = require("docx");
const IMG = "outputs/report_imgs";
const img = (n) => fs.readFileSync(`${IMG}/${n}`);

const body = (t) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 312 }, indent: { firstLine: 720 }, children: [new TextRun(t)] });
const lead = (it, rest) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 312 }, indent: { firstLine: 720 }, children: [new TextRun({ text: it, italics: true }), new TextRun(rest)] });
const H = (lv, t) => new Paragraph({ heading: [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3, HeadingLevel.HEADING_4][lv], numbering: { reference: "hd", level: lv }, spacing: { before: lv < 2 ? 240 : 160, after: 100 }, children: [new TextRun({ text: t, bold: true })] });
const tcap = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 140, after: 60 }, children: [new TextRun({ text: t, bold: true, size: 24 })] });
const code = (lines) => new Paragraph({ spacing: { before: 80, after: 120 }, shading: { type: ShadingType.CLEAR, fill: "F4F4F4" },
  border: { top: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" }, bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" }, left: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" }, right: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" } },
  children: lines.flatMap((l, i) => [new TextRun({ text: l, font: "Consolas", size: 19, break: i ? 1 : 0 })]) });
const center = (t, mono) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 120 }, children: [new TextRun(mono ? { text: t, font: "Consolas", size: 20 } : { text: t, italics: true })] });
const eqfig = (file, w, h) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100, after: 140 }, children: [new ImageRun({ type: "png", data: img(file), transformation: { width: w, height: h } })] });
const figc = (file, cap, w, h) => [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(`outputs/pipeline_steps/${file}`), transformation: { width: w, height: h } })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new TextRun({ text: cap, italics: true, size: 24 })] }),
];
const refP = (t) => new Paragraph({ spacing: { after: 80 }, indent: { left: 600, hanging: 600 }, children: [new TextRun({ text: t, size: 24 })] });

// ───────── Bảng ─────────
const cell = (t, head, w, span) => new TableCell(Object.assign({ width: { size: w, type: WidthType.DXA }, shading: head ? { type: ShadingType.CLEAR, fill: "D5E8F0" } : undefined, margins: { top: 40, bottom: 40, left: 70, right: 70 }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: t, bold: !!head, size: 21 })] })] }, span ? { rowSpan: span } : {}));
const mkTable = (widths, rows) => new Table({ width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths, rows: rows.map((r, ri) => new TableRow({ children: r.map((c, ci) => cell(c, ri === 0, widths[ci])) })) });

// Bảng 2.1 — MRL split (rowSpan cho cột phương pháp)
const W21 = [3000, 1900, 1500, 1500, 1460];
const t21 = new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: W21, rows: [
  new TableRow({ children: ["Phương pháp chia", "Tập dữ liệu", "eyes_closed", "eyes_open", "Tổng"].map((t, i) => cell(t, true, W21[i])) }),
  new TableRow({ children: [cell("Chia ngẫu nhiên theo ảnh (Random Splitting)", false, W21[0], 3), cell("Train", false, W21[1]), cell("29.254", false, W21[2]), cell("30.174", false, W21[3]), cell("59.428", false, W21[4])] }),
  new TableRow({ children: [cell("Validation", false, W21[1]), cell("8.481", false, W21[2]), cell("8.498", false, W21[3]), cell("16.979", false, W21[4])] }),
  new TableRow({ children: [cell("Test", false, W21[1]), cell("4.211", false, W21[2]), cell("4.280", false, W21[3]), cell("8.491", false, W21[4])] }),
  new TableRow({ children: [cell("Chia theo đối tượng (Subject-based Splitting)", false, W21[0], 3), cell("Train", false, W21[1]), cell("24.102", false, W21[2]), cell("28.500", false, W21[3]), cell("52.602", false, W21[4])] }),
  new TableRow({ children: [cell("Validation", false, W21[1]), cell("10.006", false, W21[2]), cell("11.589", false, W21[3]), cell("21.595", false, W21[4])] }),
  new TableRow({ children: [cell("Test", false, W21[1]), cell("7.838", false, W21[2]), cell("2.863", false, W21[3]), cell("10.701", false, W21[4])] }),
]});

const t22 = mkTable([3120, 3120, 3120], [
  ["Thuộc tính", "Trước tiền xử lý", "Sau tiền xử lý"],
  ["Kích thước ảnh", "Ngẫu nhiên (20×20 đến 80×80)", "Cố định 64×64×3"],
  ["Kiểu dữ liệu", "uint8 (0–255)", "float32 (0.0–1.0)"],
  ["Nhãn", "Chuỗi trong tên file", "Vector One-hot [1,0]/[0,1]"],
  ["Cách nạp", "Toàn bộ vào RAM", "Theo batch, song song từ đĩa"],
  ["Tập Train", "Ảnh tĩnh", "Biến đổi ngẫu nhiên mỗi epoch"],
  ["Trọng số lớp", "Bằng nhau", "eyes_closed ưu tiên ×1.3"],
]);
const t23 = mkTable([2340, 2340, 2340, 2340], [
  ["Tập dữ liệu", "no_yawn", "yawn", "Tổng"],
  ["Train", "2.072", "2.022", "4.094"],
  ["Validation", "259", "252", "511"],
  ["Test", "260", "254", "514"],
  ["Tổng cộng", "2.591", "2.528", "5.119"],
]);
const t24 = mkTable([3120, 3120, 3120], [
  ["Thuộc tính", "Trước tiền xử lý", "Sau tiền xử lý"],
  ["Phạm vi ảnh", "Toàn bộ khuôn mặt RGB", "Chỉ vùng miệng 64×64×3"],
  ["Kích thước", "Không đồng nhất", "Cố định 64×64"],
  ["Kiểu dữ liệu", "uint8 [0–255]", "float32 [0.0–1.0]"],
  ["Nhãn", "Tên thư mục", "Vector One-hot"],
  ["Đặc trưng bổ trợ", "Không có", "MAR (tỉ lệ khung miệng)"],
]);
const t25 = mkTable([1560, 4200, 3600], [
  ["Chỉ số", "Tên lớp", "Ý nghĩa"],
  ["0", "close_eyeL", "Mắt trái nhắm"],
  ["1", "close_eyeR", "Mắt phải nhắm"],
  ["2", "no_yawn", "Miệng bình thường (không ngáp)"],
  ["3", "open_eyeL", "Mắt trái mở"],
  ["4", "open_eyeR", "Mắt phải mở"],
  ["5", "yawn", "Miệng đang ngáp"],
]);
const W26 = [1500, 900, 1180, 1180, 1100, 1180, 1180, 1040, 1100];
const t26 = mkTable(W26, [
  ["Tập", "Ảnh", "close_eyeL", "close_eyeR", "no_yawn", "open_eyeL", "open_eyeR", "yawn", "Tổng box"],
  ["Train", "1.008", "322", "330", "647", "532", "543", "359", "2.733"],
  ["Valid", "440", "196", "202", "77", "195", "198", "364", "1.232"],
]);

const refs = [
  '[1] A. Singha, "MRL Eye Dataset," Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset',
  '[2] R. Fusek, "Pupil Localization Using Geodesic Distance," in Proc. Int. Conf. Advanced Concepts for Intelligent Vision Systems, 2018; MRL Eye Dataset, MRL Media Lab, VŠB-TUO. Available: http://mrl.cs.vsb.cz/eyedataset',
  '[3] Python Software Foundation, "re — Regular expression operations," Python 3 Documentation. Available: https://docs.python.org/3/library/re.html',
  '[4] TensorFlow Team, "tf.data: Build TensorFlow input pipelines," TensorFlow Documentation. Available: https://www.tensorflow.org/guide/data',
  '[5] C. Shorten and T. M. Khoshgoftaar, "A survey on Image Data Augmentation for Deep Learning," Journal of Big Data, vol. 6, no. 1, p. 60, 2019.',
  '[6] M. Buda, A. Maki, and M. A. Mazurowski, "A systematic study of the class imbalance problem in convolutional neural networks," Neural Networks, vol. 106, pp. 249–259, 2018.',
  '[7] Kaggle, "Yawn / Drowsiness Detection Dataset," Kaggle Datasets. Available: https://www.kaggle.com/datasets',
  '[8] C. Lugaresi et al., "MediaPipe: A Framework for Building Perception Pipelines," arXiv preprint arXiv:1906.08172, 2019.',
  '[9] B. Dwyer, J. Nelson, T. Hansen, et al., "Roboflow (Version 1.0)," 2024. Available: https://roboflow.com',
  '[10] G. Jocher and J. Qiu, "Ultralytics YOLO11," Ultralytics, 2024. Available: https://docs.ultralytics.com',
  '[11] A. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, "YOLOv4: Optimal Speed and Accuracy of Object Detection," arXiv:2004.10934, 2020.',
  '[12] Ultralytics, "YOLO26: NMS-free End-to-End Object Detection," Ultralytics, 2025. Available: https://docs.ultralytics.com',
  '[13] N. Carion et al., "End-to-End Object Detection with Transformers," in Proc. ECCV, 2020, pp. 213–229.',
  '[14] Y. Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection," in Proc. IEEE/CVF CVPR, 2024.',
  '[15] I. Robinson, P. Skalski, et al., "RF-DETR: A SOTA Real-Time Object Detection Model," Roboflow, 2025.',
  '[16] T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context," in Proc. ECCV, 2014, pp. 740–755.',
];

const doc = new Document({
  numbering: { config: [{ reference: "hd", levels: [
    { level: 0, format: LevelFormat.DECIMAL, text: "CHƯƠNG %1:", alignment: AlignmentType.START, start: 2 },
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
      H(0, " CHUẨN BỊ DỮ LIỆU"),
      body("Chương này trình bày chi tiết về quy trình chuẩn bị dữ liệu cho từng mô hình được nhóm sử dụng trong hệ thống phát hiện và cảnh báo trạng thái buồn ngủ của tài xế. Do nhóm tiếp cận bài toán theo hướng so sánh nhiều mô hình học máy, mỗi mô hình sử dụng bộ dữ liệu và chiến lược tiền xử lý khác nhau, nên chương này được tổ chức theo cấu trúc lấy mô hình làm đơn vị chính, trong đó mỗi mô hình có hai mục con riêng biệt là thu thập dữ liệu và tiền xử lý dữ liệu. Nội dung bám sát các bước triển khai trong mã nguồn thực tế, đi sâu phân tích cơ sở dữ liệu, các vấn đề cần làm sạch từ dữ liệu thu thập được và chi tiết hóa các phương pháp tiền xử lý nâng cao."),

      // ════════ 2.1 MRL ════════
      H(1, "Mô hình CNN phân loại trạng thái mắt (MRL Eye Dataset)"),
      body("Mô hình đầu tiên trong hệ thống là mạng nơ-ron tích chập (Convolutional Neural Network, CNN) được xây dựng để phân loại trạng thái mắt của tài xế thành hai lớp: mắt nhắm (eyes_closed) và mắt mở (eyes_open). Đây là thành phần cốt lõi quyết định tính chính xác của cảnh báo buồn ngủ. Bộ dữ liệu được lựa chọn cho mô hình này là MRL Eye Dataset, một kho dữ liệu ảnh mắt hồng ngoại chuyên biệt được tải về từ nền tảng Kaggle [1]. Việc lựa chọn mô hình CNN kết hợp với bộ dữ liệu MRL xuất phát từ đặc điểm ảnh hồng ngoại hoạt động tốt trong mọi điều kiện ánh sáng, phù hợp với bối cảnh ứng dụng trong cabin xe vào ban đêm hoặc ánh sáng yếu."),

      H(2, "Thu thập dữ liệu"),
      H(3, "Nguồn gốc và đặc điểm của bộ dữ liệu"),
      body("MRL Eye Dataset được xây dựng bởi Phòng thí nghiệm Truyền thông MRL (MRL Media Lab), Đại học Kỹ thuật Ostrava, Cộng hòa Séc và được công bố tại địa chỉ chính thức của phòng thí nghiệm [2]. Bộ dữ liệu sau đó được cộng đồng học máy đăng tải và chuẩn hóa lại trên nền tảng Kaggle [1], nơi nhóm tải về để sử dụng. Đây là một kho dữ liệu chuyên biệt lớn, chứa tổng cộng 84.898 hình ảnh mắt được cắt ra từ các chuỗi video ghi hình của 37 đối tượng thử nghiệm khác nhau (gồm cả nam và nữ). Tất cả hình ảnh được thu thập bằng cảm biến hồng ngoại chủ động, nhằm bao quát đầy đủ các điều kiện thực tế như góc xoay của đầu, trạng thái đeo kính, hiện tượng phản chiếu ánh sáng trên tròng kính và khoảng cách từ camera đến khuôn mặt. Do đặc thù ảnh hồng ngoại, toàn bộ dữ liệu ở định dạng ảnh thang độ xám (grayscale) đơn kênh, thể hiện rõ nét cấu trúc hình học của con ngươi và mí mắt ngay cả trong điều kiện ban đêm."),
      H(3, "Phương pháp gán nhãn dữ liệu"),
      body("Nhóm hoàn toàn không phải thực hiện gán nhãn thủ công cho bộ dữ liệu này. Điểm đặc biệt của MRL Eye Dataset là nhãn phân loại đã được các tác giả mã hóa sẵn và lưu trữ trực tiếp vào cấu trúc tên tệp tin theo một quy ước nghiêm ngặt, giúp loại bỏ hoàn toàn sai số chủ quan. Định dạng tên tệp tin có cấu trúc chuẩn như sau:"),
      center("s0021_00019_0_0_0_0_0_01.png", true),
      body("Tên tệp tin chứa chính xác tám trường thông tin thuộc tính của ảnh mắt, phân tách bởi dấu gạch dưới. Trường 1 là mã định danh đối tượng (ví dụ s0021 là đối tượng thứ 21). Trường 2 là chỉ số camera. Trường 3 là giới tính. Trường 4 là trạng thái đeo kính. Trường 5, quan trọng nhất, là trạng thái mắt: giá trị 0 ứng với mắt nhắm (eyes_closed) và giá trị 1 ứng với mắt mở (eyes_open). Trường 6 là trạng thái phản chiếu ánh sáng trên kính. Trường 7 là điều kiện ánh sáng môi trường. Trường 8 là chỉ số khung hình."),
      body("Dựa trên quy ước đặt tên có sẵn này, nhóm chỉ việc xây dựng hệ thống trích xuất nhãn tự động bằng biểu thức chính quy trong Python. Nếu trường thứ 5 bằng 0, ảnh được xếp vào lớp eyes_closed (index 0); nếu bằng 1, ảnh được xếp vào lớp eyes_open (index 1). Thứ tự này trùng khớp với thứ tự alphabet mà thư viện Keras tự sắp xếp khi quét thư mục, bảo đảm sự nhất quán giữa phương thức gán nhãn và cơ chế nạp dữ liệu khi huấn luyện."),
      H(3, "Tổ chức lưu trữ và quan sát dữ liệu"),
      body("Sau khi tải về, bộ dữ liệu được giải nén và tổ chức tại thư mục dataset trên Google Colab, với cấu trúc phân cấp theo tập dữ liệu và lớp phân loại như sau:"),
      code(["/content/dataset/", "    train/  eyes_closed/   eyes_open/", "    val/    eyes_closed/   eyes_open/", "    test/   eyes_closed/   eyes_open/"]),
      body("Qua quan sát và thống kê thực tế, nhóm phát hiện ba đặc điểm và vấn đề lớn cần xử lý. Thứ nhất là sự không đồng nhất về độ phân giải: các ảnh mắt có kích thước dao động rất lớn, từ cỡ 20×20 pixel đến 80×80 pixel, khiến mô hình không thể tiếp nhận trực tiếp dưới dạng tensor đầu vào cố định. Thứ hai là hiện tượng nhiễu phản chiếu ánh sáng hồng ngoại: với đối tượng đeo kính, ánh sáng hồng ngoại tạo ra điểm sáng trắng đè lên con ngươi, làm lu mờ ranh giới mí mắt, đòi hỏi mô hình phải học được đặc trưng cấu trúc hình học của toàn bộ vùng mắt thay vì chỉ dựa vào độ sáng tối của đồng tử."),
      body("Thứ ba, và quan trọng nhất, là hiện tượng rò rỉ dữ liệu (data leakage) do phân chia ngẫu nhiên. Bộ MRL có 84.898 ảnh nhưng chỉ đến từ 37 người, nên mỗi người trung bình có hơn 2.000 khung hình; các khung hình liên tiếp của cùng một người có mức độ tương đồng gần như tuyệt đối. Nếu phân chia ngẫu nhiên theo từng ảnh (Random Splitting), ảnh của cùng một đối tượng sẽ xuất hiện song song ở cả ba tập train, validation và test. Lúc này CNN có thể đạt độ chính xác ảo xấp xỉ 98,5% không phải nhờ học đặc trưng mở/nhắm mắt, mà nhờ ghi nhớ khuôn mặt người đó. Khi triển khai trên một tài xế mới hoàn toàn, hiệu năng sẽ sụt giảm nghiêm trọng. Đây là vấn đề điển hình của các bộ dữ liệu thu thập theo đối tượng (subject-based datasets)."),
      body("Để giải quyết, nhóm áp dụng chiến lược phân chia theo đối tượng (Subject-based Splitting): thay vì xáo trộn từng ảnh, hệ thống gom toàn bộ ảnh theo mã đối tượng (s0001 đến s0037), xáo trộn danh sách 37 người này rồi phân chia theo tỉ lệ 70/20/10. Kết quả là tập huấn luyện chứa toàn bộ ảnh của 26 người đầu, tập kiểm định chứa ảnh của 7 người tiếp theo và tập kiểm thử chứa ảnh của 4 người cuối. Với cách này, mô hình bắt buộc phải phân loại trạng thái mắt của những người chưa từng xuất hiện trong quá trình học, phản ánh đúng năng lực tổng quát hóa khi triển khai thực tế. Bảng 2.1 thống kê chi tiết số lượng mẫu theo từng phương pháp chia."),
      tcap("Bảng 2.1. Thống kê số lượng mẫu phân chia theo hai phương pháp (tỉ lệ 70/20/10)."),
      t21, body(""),
      body("Một quan sát đáng chú ý trong Bảng 2.1 là trong khi Random Splitting luôn duy trì tỉ lệ cân bằng gần 50/50 giữa hai lớp ở mọi tập (do tính ngẫu nhiên), Subject-based Splitting lại tạo ra tập kiểm thử mất cân bằng đáng kể: 7.838 ảnh mắt nhắm so với chỉ 2.863 ảnh mắt mở (xấp xỉ 73/27). Đây là hệ quả tự nhiên của việc 4 người ngẫu nhiên thuộc tập test tình cờ có nhiều ảnh mắt nhắm hơn. Đó là một hạn chế của Subject-based Splitting khi số đối tượng nhỏ (chỉ 37 người), nhưng không làm giảm giá trị khoa học của phương pháp vì mục tiêu cốt lõi vẫn là đánh giá trên người hoàn toàn mới. Kết quả so sánh hiệu năng giữa hai phương pháp sẽ được trình bày chi tiết trong Chương 4."),

      H(2, "Tiền xử lý dữ liệu"),
      H(3, "Kỹ thuật đã học và thực hành trên lớp"),
      body("Nhóm kỹ thuật đầu tiên bao gồm các phép biến đổi cơ bản đã được học trong khóa học, áp dụng đồng nhất cho toàn bộ tập dữ liệu trước khi đưa vào mạng CNN."),
      lead("Thay đổi kích thước ảnh (Resizing): ", "Tất cả hình ảnh mắt được đưa về kích thước cố định 64×64 pixel thông qua phép nội suy song tuyến tính (bilinear interpolation) do hàm tf.image.resize xử lý. Kích thước 64×64 nhỏ đủ để mô hình CNN nhẹ hoạt động ở tốc độ cao trên Android, nhưng đủ lớn để bảo toàn các đặc trưng hình học quan trọng như đường biên mí mắt, vùng lòng trắng và con ngươi."),
      lead("Chuẩn hóa giá trị điểm ảnh (Normalization): ", "Sau khi resize, giá trị mỗi điểm ảnh trong khoảng nguyên [0, 255] được chuyển sang số thực và chia cho 255.0, đưa về đoạn [0.0, 1.0]. Phép chuẩn hóa này triệt tiêu chênh lệch biên độ lớn, giúp thuật toán Adam tính gradient ổn định và hội tụ nhanh hơn. Điểm kỹ thuật quan trọng là chuẩn hóa được thực hiện bên ngoài mô hình Keras (trong luồng tf.data), không đặt lớp Rescaling bên trong mô hình. Lý do là khi xuất sang TensorFlow Lite chạy trên Android, nếu chuẩn hóa được nhúng vào mô hình mà ứng dụng vẫn chuẩn hóa thêm một lần nữa sẽ gây lỗi chuẩn hóa hai lần và cho kết quả sai hoàn toàn. Cách chuẩn hóa ngoài mô hình bảo đảm nhất quán tuyệt đối giữa pipeline huấn luyện Python và pipeline suy luận Kotlin trên Android."),
      lead("Mã hóa nhãn dạng One-hot (One-hot Encoding): ", "Các nhãn lớp được chuyển thành vector nhị phân bằng hàm to_categorical của Keras: lớp eyes_closed (index 0) ứng với [1, 0] và eyes_open (index 1) ứng với [0, 1]. Dạng mã hóa này cần thiết để kết hợp với hàm mất mát categorical_crossentropy và lớp đầu ra Dense(2, activation='softmax')."),
      ...figc("step4_resize.png", "Hình 2.1. Minh họa phép thay đổi kích thước vùng quan tâm về 64×64 pixel", 450, 175),
      ...figc("step8_normalize.png", "Hình 2.2. Minh họa phép chuẩn hóa điểm ảnh từ [0, 255] về [0, 1]", 450, 191),
      H(3, "Kỹ thuật chưa học trên lớp"),
      body("Nhóm kỹ thuật thứ hai bao gồm các phương pháp nâng cao chưa được đề cập trong chương trình học, được nhóm tự nghiên cứu và áp dụng nhằm giải quyết các vấn đề đặc thù của bộ MRL."),
      lead("Trích xuất mã đối tượng bằng biểu thức chính quy (Regular Expression): ", "Để phân chia theo đối tượng, hệ thống cần nhận diện ảnh nào thuộc về người nào từ tên tệp. Kỹ thuật Regex [3] được áp dụng để trích xuất tiền tố mã đối tượng dạng s#### một cách chính xác và hiệu quả:"),
      code([
        "import re",
        "all_items = []   # (tên_lớp, mã_subject, đường_dẫn_ảnh)",
        "for split in ['train','val','valid','test']:",
        "    for c in CLASSES:",
        "        d = f'{SRC}/{split}/{c}'",
        "        if not os.path.isdir(d): continue",
        "        for fn in os.listdir(d):",
        "            m = re.match(r'(s\\d+)', fn)      # bắt tiền tố s####",
        "            subj = m.group(1) if m else fn",
        "            all_items.append((c, subj, os.path.join(d, fn)))",
        "n_subj = len(set(s for _, s, _ in all_items))",
      ]),
      body("Đoạn mã trên gom toàn bộ 84.898 ảnh từ mọi thư mục vào một danh sách thống nhất all_items, mỗi phần tử lưu đủ ba thông tin: tên lớp, mã đối tượng và đường dẫn tuyệt đối. Việc thu gom về một danh sách duy nhất trước khi chia lại là cần thiết để phép chia theo đối tượng được thực hiện trên toàn bộ dữ liệu, không bị giới hạn bởi cách phân chia sẵn có của bộ gốc."),
      lead("Xây dựng pipeline tải ảnh động bằng tf.data: ", "Thay vì tải toàn bộ 84.898 ảnh vào RAM, hệ thống xây dựng pipeline tải ảnh theo nhu cầu bằng tf.data.Dataset.from_tensor_slices [4]. Cơ chế này chỉ lưu danh sách đường dẫn vào bộ nhớ, còn ảnh thực tế được đọc trực tiếp từ đĩa và giải mã ngay trong vòng lặp huấn luyện:"),
      code([
        "def make_ds(items, training):",
        "    paths  = [p for (c, s, p) in items]",
        "    labels = tf.keras.utils.to_categorical(",
        "        [CLASSES.index(c) for (c, s, p) in items], num_classes=2)",
        "    ds = tf.data.Dataset.from_tensor_slices((paths, labels))",
        "    if training:",
        "        ds = ds.shuffle(min(len(paths), 10000), seed=SEED)",
        "    def load(path, label):",
        "        img = tf.io.decode_image(tf.io.read_file(path), channels=3,",
        "                                 expand_animations=False)",
        "        img = tf.image.resize(img, (IMG, IMG))",
        "        img = tf.cast(img, tf.float32) / 255.0      # chuẩn hóa [0,1]",
        "        img.set_shape((IMG, IMG, 3)); return img, label",
        "    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)",
        "    if training:",
        "        ds = ds.map(lambda x, y: (aug(x, training=True), y),",
        "                    num_parallel_calls=tf.data.AUTOTUNE)",
        "    return ds.prefetch(tf.data.AUTOTUNE)",
      ]),
      body("Tham số num_parallel_calls=tf.data.AUTOTUNE cho phép TensorFlow tự phân bổ số luồng đọc ảnh song song theo phần cứng. Hàm prefetch(tf.data.AUTOTUNE) bảo đảm batch tiếp theo đã sẵn sàng trên CPU trong khi GPU đang xử lý batch hiện tại, loại bỏ thời gian chờ I/O. Một lưu ý quan trọng: tập kiểm thử không kích hoạt shuffle để bảo đảm thứ tự nhãn thật y_true khớp với thứ tự dự đoán y_pred khi tính ma trận nhầm lẫn."),
      lead("Tăng cường dữ liệu thời gian thực (Real-time Data Augmentation): ", "Nhằm hạn chế quá khớp (overfitting), lớp tăng cường dữ liệu được tích hợp như một bước biến đổi ngẫu nhiên trong pipeline, chỉ kích hoạt cho tập Train và tự vô hiệu hóa khi đánh giá [5]:"),
      code([
        "aug = tf.keras.Sequential([",
        "    tf.keras.layers.RandomFlip('horizontal'),      # lật ngang",
        "    tf.keras.layers.RandomRotation(0.05),          # xoay ±5%",
        "    tf.keras.layers.RandomZoom(0.10),              # thu phóng ±10%",
        "    tf.keras.layers.RandomContrast(0.20),          # tương phản ±20%",
        "    tf.keras.layers.RandomBrightness(0.20, value_range=(0.0, 1.0)),",
        "], name='augmentation')",
      ]),
      body("Phép lật ngang mô phỏng khác biệt giữa mắt trái và mắt phải. Xoay nhẹ mô phỏng nghiêng đầu của tài xế. Thay đổi độ sáng và tương phản mô phỏng biến đổi chiếu sáng trong cabin. Thu phóng mô phỏng thay đổi khoảng cách camera. Toàn bộ được áp dụng sau bước batch hóa để tối ưu hiệu năng GPU."),
      ...figc("step6_augment.png", "Hình 2.3. Minh họa các phép tăng cường dữ liệu: lật ngang, xoay, tăng giảm độ sáng", 470, 110),
      lead("Tính toán trọng số lớp tùy biến (Custom Class Weight): ", "Trong bài toán cảnh báo buồn ngủ, bỏ sót một trường hợp nhắm mắt thật (False Negative của lớp eyes_closed) nguy hiểm hơn nhiều so với cảnh báo nhầm. Do đó nhóm ưu tiên tối đa hóa Recall của lớp eyes_closed bằng cách gán trọng số huấn luyện lớn hơn, tính động theo tần suất lớp kết hợp hệ số ưu tiên 1.3 [6]:"),
      eqfig("eq_classweight.png", 470, 60),
      body("trong đó N_train là tổng số mẫu tập Train, N_closed và N_open là số mẫu mỗi lớp. Hệ số 1.3 là siêu tham số chọn theo thực nghiệm: quá lớn sẽ khiến mô hình cảnh báo nhầm liên tục, quá nhỏ sẽ không cải thiện được Recall. Ở mức 1.3, mô hình đạt cân bằng giữa độ nhạy phát hiện buồn ngủ và tỉ lệ báo động giả chấp nhận được khi chạy thực tế trên Android."),
      H(3, "Lý do cần thiết của quy trình tiền xử lý và so sánh trước sau"),
      body("Toàn bộ quy trình tiền xử lý được thiết kế nhằm giải quyết ba vấn đề được phát hiện khi quan sát dữ liệu. Resizing về 64×64 giải quyết sự không đồng nhất kích thước và bảo đảm tốc độ suy luận cao trên chip di động. Chuẩn hóa điểm ảnh giải quyết vấn đề gradient không ổn định. Data Augmentation giúp mô hình học được đặc trưng bất biến trước nhiễu phản chiếu kính và thay đổi ánh sáng. Custom Class Weight bảo đảm độ nhạy phát hiện an toàn. Bảng 2.2 so sánh trạng thái dữ liệu trước và sau khi đi qua toàn bộ pipeline."),
      tcap("Bảng 2.2. So sánh trạng thái dữ liệu trước và sau tiền xử lý của mô hình phân loại mắt."),
      t22, body(""),

      // ════════ 2.2 YAWN ════════
      H(1, "Mô hình CNN phân loại trạng thái ngáp (Yawn Dataset)"),
      body("Mô hình thứ hai trong hệ thống là một mạng nơ-ron tích chập độc lập, đảm nhiệm việc phân loại trạng thái miệng của tài xế thành hai lớp: đang ngáp (yawn) và không ngáp (no_yawn). Ngáp là một dấu hiệu sinh lý xuất hiện sớm của trạng thái mệt mỏi, thường diễn ra trước khi tài xế rơi vào trạng thái nhắm mắt vi mô, do đó phát hiện ngáp bổ trợ quan trọng cho mô hình mắt ở mục 2.1, tạo thành cơ chế cảnh báo hai tầng. Khác với bộ MRL hồng ngoại đơn kênh, bộ dữ liệu ngáp là ảnh màu (RGB), phản ánh đúng tín hiệu thu được từ camera thông thường trong khoang lái vào ban ngày."),
      H(2, "Thu thập dữ liệu"),
      H(3, "Nguồn gốc và đặc điểm của bộ dữ liệu"),
      body("Bộ dữ liệu phân loại ngáp được nhóm tải về từ nền tảng Kaggle [7], là tập hợp các hình ảnh khuôn mặt người ở hai trạng thái ngáp và không ngáp, thu thập trong nhiều điều kiện ánh sáng và góc chụp khác nhau. Toàn bộ dữ liệu ở định dạng ảnh màu ba kênh, kích thước không đồng nhất, bao gồm cả ảnh chính diện lẫn nghiêng nhẹ, một số có đeo kính hoặc có râu, nhằm tăng tính đa dạng. Sau khi tải về và thống kê, bộ dữ liệu gồm tổng cộng 5.119 ảnh, phân bố tương đối cân bằng giữa hai lớp như trình bày trong Bảng 2.3."),
      tcap("Bảng 2.3. Thống kê số lượng mẫu của bộ dữ liệu phân loại ngáp."),
      t23, body(""),
      H(3, "Phương pháp gán nhãn dữ liệu"),
      body("Tương tự bộ MRL, nhóm không phải gán nhãn thủ công cho bộ ngáp. Nhãn được mã hóa trực tiếp thông qua cấu trúc thư mục: toàn bộ ảnh thuộc một lớp được đặt trong một thư mục con mang đúng tên lớp đó (yawn hoặc no_yawn). Đây là quy ước gán nhãn theo thư mục (folder-based labeling) được Keras hỗ trợ trực tiếp qua tham số class_names, trong đó thứ tự alphabet (no_yawn < yawn) quy định chỉ số lớp: no_yawn là index 0 và yawn là index 1. Việc nhãn đã được tổ chức sẵn theo thư mục giúp loại bỏ sai số chủ quan và bảo đảm nhất quán với cơ chế nạp dữ liệu."),
      H(3, "Tổ chức lưu trữ và quan sát dữ liệu"),
      body("Bộ dữ liệu được tổ chức theo cấu trúc tương tự bộ MRL, gồm ba tập train, val, test, mỗi tập chứa hai thư mục lớp no_yawn và yawn. Qua quan sát thực tế, nhóm nhận thấy hai đặc điểm cần xử lý. Thứ nhất là đối tượng quan tâm chỉ là vùng miệng nhưng ảnh lại chứa toàn bộ khuôn mặt: nếu đưa nguyên ảnh khuôn mặt vào CNN, mô hình sẽ phải học thêm nhiều đặc trưng nhiễu không liên quan đến hành vi ngáp (mắt, tóc, nền phía sau), làm loãng tín hiệu và dễ học nhầm. Yêu cầu đặt ra là phải khoanh vùng và cắt riêng vùng miệng trước khi huấn luyện."),
      body("Thứ hai là sự khác biệt về phương thức (modality) so với dữ liệu suy luận thực tế: dữ liệu Kaggle là ảnh tĩnh chụp trong nhiều bối cảnh, trong khi khi triển khai trên Android, đầu vào lại là vùng miệng cắt theo điểm mốc khuôn mặt từ luồng camera. Sự chênh lệch này đòi hỏi quy trình cắt vùng miệng lúc huấn luyện phải đồng nhất với lúc suy luận, nếu không mô hình sẽ gặp vấn đề lệch phân bố dữ liệu (out-of-distribution) như phân tích trong Chương 4."),
      H(2, "Tiền xử lý dữ liệu"),
      H(3, "Kỹ thuật đã học và thực hành trên lớp"),
      body("Nhóm áp dụng đồng bộ các phép biến đổi cơ bản giống mô hình mắt để bảo đảm nhất quán giữa hai nhánh phân loại. Tất cả ảnh vùng miệng được đưa về kích thước cố định 64×64 pixel bằng nội suy song tuyến tính, sau đó giá trị điểm ảnh nguyên [0, 255] được chuyển sang số thực và chia cho 255.0 để chuẩn hóa về [0.0, 1.0]. Nhãn được mã hóa One-hot, trong đó no_yawn ứng với [1, 0] và yawn ứng với [0, 1], kết hợp categorical_crossentropy và Dense(2, activation='softmax'). Phép chuẩn hóa cũng được đặt ngoài mô hình (trong tf.data) thay vì nhúng lớp Rescaling, vì lý do tương thích với pipeline suy luận TensorFlow Lite đã giải thích ở mục 2.1."),
      H(3, "Kỹ thuật chưa học trên lớp"),
      lead("Trích xuất vùng miệng bằng điểm mốc khuôn mặt MediaPipe (Facial Landmark ROI Extraction): ", "Đây là kỹ thuật cốt lõi và phức tạp nhất của nhánh ngáp. Thay vì huấn luyện trên cả khuôn mặt, nhóm dùng mô hình FaceLandmarker của thư viện MediaPipe [8] để phát hiện 478 điểm mốc khuôn mặt, sau đó chọn cụm điểm bao quanh viền môi để xác định khung chữ nhật vùng miệng và cắt riêng. Quy trình này được áp dụng đồng nhất cho cả dữ liệu huấn luyện lẫn luồng camera lúc suy luận:"),
      code([
        "import mediapipe as mp",
        "from mediapipe.tasks import python as mpp",
        "from mediapipe.tasks.python import vision",
        "opts = vision.FaceLandmarkerOptions(",
        "    base_options=mpp.BaseOptions(model_asset_path='face_landmarker.task'),",
        "    running_mode=vision.RunningMode.IMAGE, num_faces=1)",
        "LMK = vision.FaceLandmarker.create_from_options(opts)",
        "MOUTH = [61, 291, 0, 17, 13, 14]   # góc trái/phải + môi trên/dưới",
        "def crop_mouth(img, pad_x=0.5, pad_y=0.5):",
        "    res = LMK.detect(mp.Image(image_format=mp.ImageFormat.SRGB,",
        "          data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))",
        "    if not res.face_landmarks: return None",
        "    lm = res.face_landmarks[0]; h, w = img.shape[:2]",
        "    xs = [lm[i].x*w for i in MOUTH]; ys = [lm[i].y*h for i in MOUTH]",
        "    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)",
        "    bw, bh = x2-x1, y2-y1",
        "    x1, x2 = int(x1-bw*pad_x), int(x2+bw*pad_x)",
        "    y1, y2 = int(y1-bh*pad_y), int(y2+bh*pad_y)",
        "    crop = img[max(0,y1):y2, max(0,x1):x2]",
        "    return cv2.resize(crop, (64, 64)) if crop.size else None",
      ]),
      body("Hệ số đệm (pad_x, pad_y bằng 0.5) mở rộng khung cắt ra 50% mỗi chiều so với viền môi, nhằm giữ lại phần cằm và cơ quanh miệng — những vùng co giãn rõ rệt khi ngáp, giúp mô hình phân biệt tốt hơn giữa ngáp thật và việc chỉ mở miệng nói chuyện."),
      ...figc("step3_roi.png", "Hình 2.4. Minh họa cắt vùng mắt và miệng từ điểm mốc khuôn mặt MediaPipe", 470, 132),
      lead("Đặc trưng hình học bổ trợ MAR (Mouth Aspect Ratio): ", "Song song với CNN, nhóm tính thêm tỉ lệ khung hình miệng (MAR) trực tiếp từ các điểm mốc, theo công thức:"),
      eqfig("eq_mar.png", 230, 69),
      body("Giá trị MAR tăng vọt khi miệng há rộng theo chiều dọc lúc ngáp. Đặc trưng hình học này đóng vai trò lớp an toàn dự phòng: khi CNN không đủ tin cậy, hệ thống vẫn có thể dựa vào ngưỡng MAR để phát hiện ngáp; cơ chế hợp nhất sẽ được trình bày trong Chương 3."),
      lead("Pipeline tf.data, tăng cường dữ liệu và trọng số lớp: ", "Nhóm tái sử dụng kiến trúc pipeline tf.data tải ảnh động, lớp tăng cường dữ liệu thời gian thực (RandomFlip, RandomRotation, RandomBrightness, RandomContrast, RandomZoom) chỉ kích hoạt cho tập Train, và cơ chế trọng số lớp như mục 2.1. Do bộ dữ liệu ngáp khá cân bằng (xấp xỉ 50/50), trọng số lớp ở đây gần như bằng nhau, nhưng vẫn được giữ lại để bảo đảm tính tổng quát của quy trình huấn luyện chung cho cả hai mô hình CNN."),
      body("Bảng 2.4 so sánh trạng thái dữ liệu của mô hình ngáp trước và sau tiền xử lý."),
      tcap("Bảng 2.4. So sánh trạng thái dữ liệu trước và sau tiền xử lý của mô hình phân loại ngáp."),
      t24, body(""),

      // ════════ 2.3 YOLOv11 ════════
      H(1, "Mô hình YOLOv11 phát hiện đối tượng (Datio_yolo Dataset)"),
      body("Bắt đầu từ mục này, hệ thống chuyển từ bài toán phân loại ảnh sang bài toán phát hiện đối tượng (object detection), trong đó mô hình không chỉ phân loại trạng thái mà còn phải định vị chính xác vị trí của từng mắt và miệng trong khung hình bằng các khung bao (bounding box). Mô hình đầu tiên theo hướng này là YOLOv11 [10], phiên bản mới của dòng YOLO nổi tiếng về tốc độ và độ chính xác. Vì cả bốn mô hình phát hiện (YOLOv11, YOLO26, RT-DETR, RF-DETR) đều dùng chung một bộ dữ liệu nên mục 2.3.1 sẽ mô tả đầy đủ bộ dữ liệu này, các mục 2.4–2.6 chỉ nêu điểm khác biệt về định dạng và tiền xử lý của từng mô hình."),
      H(2, "Thu thập dữ liệu"),
      H(3, "Nguồn gốc và đặc điểm của bộ dữ liệu"),
      body("Bộ dữ liệu phát hiện đối tượng mang tên Datio_yolo do chính nhóm xây dựng và quản lý trên nền tảng Roboflow [9] (không gian làm việc nguyen-tuan-dat, giấy phép CC BY 4.0). Đây là tập ảnh màu chụp khuôn mặt tài xế trong khoang lái, được gán nhãn theo sáu lớp đối tượng tương ứng với trạng thái của từng mắt và miệng. Bộ dữ liệu gồm 1.448 ảnh với tổng cộng 3.965 khung bao đối tượng, được phân chia sẵn thành hai tập huấn luyện và kiểm định. Sáu lớp đối tượng và ý nghĩa được trình bày trong Bảng 2.5."),
      tcap("Bảng 2.5. Ý nghĩa sáu lớp đối tượng của bộ dữ liệu Datio_yolo."),
      t25, body(""),
      ...figc("step0_raw.png", "Hình 2.5. Một số ảnh thô RGB tiêu biểu của bộ dữ liệu Datio_yolo", 450, 242),
      H(3, "Phương pháp gán nhãn dữ liệu"),
      body("Khác với hai bộ dữ liệu phân loại ở trên (vốn có nhãn sẵn), bộ Datio_yolo được nhóm tự gán nhãn thủ công bằng công cụ chú thích trực tuyến của Roboflow. Với mỗi ảnh, nhóm vẽ một khung bao quanh từng đối tượng quan tâm (mắt trái, mắt phải, miệng) và chọn lớp tương ứng. Sau khi hoàn tất, Roboflow xuất nhãn theo định dạng YOLO: mỗi ảnh đi kèm một tệp văn bản .txt cùng tên, mỗi dòng mô tả một đối tượng theo cấu trúc gồm chỉ số lớp và bốn tọa độ khung bao đã chuẩn hóa về đoạn [0, 1]:"),
      center("class_id   x_center   y_center   width   height", true),
      body("Việc chuẩn hóa tọa độ theo tỉ lệ chiều rộng và chiều cao ảnh giúp nhãn độc lập với kích thước ảnh gốc, nên mô hình có thể huấn luyện trên ảnh ở nhiều độ phân giải khác nhau mà không cần chỉnh sửa nhãn."),
      H(3, "Tổ chức lưu trữ và quan sát dữ liệu"),
      body("Sau khi tải về qua thư viện Roboflow, bộ dữ liệu có cấu trúc gồm các thư mục train, valid, mỗi thư mục chia thành images (ảnh) và labels (nhãn), kèm tệp cấu hình data.yaml khai báo đường dẫn, số lớp và tên lớp:"),
      code(["train/  images/*.jpg   labels/*.txt", "valid/  images/*.jpg   labels/*.txt", "data.yaml  -> nc: 6", "  names: [close_eyeL, close_eyeR, no_yawn,", "          open_eyeL, open_eyeR, yawn]"]),
      body("Qua thống kê thực tế (Bảng 2.6), nhóm phát hiện hai đặc điểm cần lưu ý. Thứ nhất, số khung bao lớn hơn số ảnh (2.733 box trên 1.008 ảnh huấn luyện), phản ánh đúng bản chất bài toán phát hiện đối tượng: mỗi ảnh thường chứa nhiều đối tượng cùng lúc (hai mắt và một miệng) — điểm khác biệt căn bản so với bài toán phân loại ở mục 2.1 và 2.2, nơi mỗi ảnh chỉ mang một nhãn. Thứ hai, có sự mất cân bằng phân bố lớp giữa hai tập: lớp no_yawn chiếm tới 647 box ở tập train nhưng chỉ 77 box ở tập valid, trong khi lớp yawn lại nhiều hơn ở tập valid. Sự lệch này cần được lưu ý khi diễn giải các chỉ số đánh giá theo lớp ở Chương 4."),
      tcap("Bảng 2.6. Phân bố khung bao theo lớp trên bộ dữ liệu Datio_yolo."),
      t26, body(""),
      ...figc("step1_ingest.png", "Hình 2.6. Kiểm kê và phân bố kích thước ảnh của bộ dữ liệu Datio_yolo", 380, 220),
      ...figc("step2_quality.png", "Hình 2.7. Lọc chất lượng ảnh theo độ nét (Laplacian) và độ sáng", 440, 210),
      ...figc("step5_eda.png", "Hình 2.8. Phân bố số lượng nhãn theo sáu lớp đối tượng (EDA)", 420, 223),
      ...figc("step7_split.png", "Hình 2.9. Phân chia tập huấn luyện, kiểm định và kiểm thử", 380, 236),
      H(2, "Tiền xử lý dữ liệu"),
      H(3, "Kỹ thuật đã học và thực hành trên lớp"),
      body("Các phép tiền xử lý cơ bản như thay đổi kích thước ảnh và chuẩn hóa giá trị điểm ảnh vẫn được áp dụng, nhưng đối với YOLO các bước này được thư viện Ultralytics tự động thực hiện bên trong pipeline huấn luyện. Cụ thể, mọi ảnh đầu vào được đưa về kích thước 640×640 pixel và giá trị điểm ảnh được chuẩn hóa về [0, 1] trước khi vào mạng. Nhóm chỉ cần khai báo siêu tham số imgsz=640, phần còn lại do framework đảm nhiệm."),
      H(3, "Kỹ thuật chưa học trên lớp"),
      lead("Tải dữ liệu bằng Roboflow API: ", "Bộ dữ liệu được tải về và đồng bộ định dạng tự động thông qua thư viện Roboflow, bảo đảm nhãn luôn đúng chuẩn YOLO mà không cần thao tác thủ công:"),
      code(["from roboflow import Roboflow", "rf = Roboflow(api_key='<API_KEY>')", "project = rf.workspace('nguyen-tuan-dat').project('datio_yolo')", "dataset = project.version(1).download('yolov11')   # đúng định dạng YOLOv11"]),
      lead("Co giãn giữ tỉ lệ (Letterbox Resizing): ", "Khác với resize thông thường làm méo ảnh, YOLO dùng kỹ thuật letterbox — co giãn ảnh giữ nguyên tỉ lệ khung hình rồi chèn viền xám vào phần thiếu để đạt kích thước vuông 640×640. Cách này bảo toàn hình dạng thật của mắt và miệng, tránh làm sai lệch đặc trưng hình học vốn rất quan trọng để phân biệt mắt mở và mắt nhắm."),
      lead("Tăng cường dữ liệu trực tuyến đặc thù của YOLO: ", "YOLO áp dụng một tập kỹ thuật tăng cường dữ liệu mạnh ngay trong huấn luyện, nổi bật là Mosaic [11] — ghép bốn ảnh khác nhau thành một ảnh lớn để mô hình học được nhiều bối cảnh và tỉ lệ đối tượng trong một lần truyền, cùng với MixUp, biến đổi không gian màu HSV, lật ngang và dịch chuyển ngẫu nhiên. Toàn bộ được điều khiển tự động và tắt dần ở các epoch cuối để mô hình hội tụ ổn định."),
      lead("Học chuyển giao từ trọng số COCO (Transfer Learning): ", "Do bộ dữ liệu chỉ có hơn một nghìn ảnh, huấn luyện từ đầu sẽ dễ quá khớp. Nhóm khởi tạo mô hình từ trọng số yolo11s.pt đã được huấn luyện trước trên bộ COCO (80 lớp đối tượng phổ thông), sau đó tinh chỉnh lại trên sáu lớp của bài toán:"),
      code(["from ultralytics import YOLO", "model = YOLO('yolo11s.pt')               # nạp trọng số COCO", "model.train(data=f'{dataset.location}/data.yaml',", "            epochs=60, imgsz=640, batch=16)"]),
      ...figc("step9_10_manifest.png", "Hình 2.10. Bảng kê dữ liệu (manifest) và kiểm tra tính toàn vẹn của bộ dữ liệu", 360, 259),

      // ════════ 2.4 YOLO26 ════════
      H(1, "Mô hình YOLO26 phát hiện đối tượng (Datio_yolo Dataset)"),
      body("YOLO26 là phiên bản cải tiến với điểm đột phá là cơ chế dự đoán đầu cuối không cần hậu xử lý NMS (Non-Maximum Suppression), giúp đơn giản hóa khâu triển khai và giảm độ trễ trên thiết bị di động [12]. Mô hình sử dụng chung bộ dữ liệu Datio_yolo đã mô tả ở mục 2.3.1."),
      H(2, "Thu thập dữ liệu"),
      body("Nhóm sử dụng đúng tập ảnh và nhãn đã tự gán ở mục 2.3.1, điểm khác biệt duy nhất nằm ở định dạng xuất từ Roboflow. Để tương thích với kiến trúc YOLO26, dữ liệu được tải về theo định dạng yolo26:"),
      code(["dataset26 = project.version(1).download('yolo26')"]),
      body("Về bản chất, nhãn vẫn theo chuẩn YOLO (chỉ số lớp kèm bốn tọa độ chuẩn hóa), chỉ khác ở cách Roboflow tổ chức tệp cấu hình data.yaml cho đúng yêu cầu của phiên bản. Trong quá trình thực hiện, nhóm gặp tình huống tệp data.yaml thiếu khóa train, đã xử lý bằng cách dò cấu trúc thư mục và tự sinh lại tệp cấu hình hợp lệ trước khi huấn luyện."),
      H(2, "Tiền xử lý dữ liệu"),
      body("Các bước tiền xử lý cơ bản (resize 640, chuẩn hóa, letterbox, tăng cường Mosaic) hoàn toàn giống YOLOv11 vì cùng hệ sinh thái Ultralytics. Điểm khác biệt cốt lõi nằm ở cơ chế gán nhãn huấn luyện: YOLO26 áp dụng chiến lược gán một-một (one-to-one assignment) cho phép loại bỏ bước NMS, đồng thời sử dụng bộ tối ưu MuSGD. Về phía dữ liệu, điều này không đòi hỏi thao tác tiền xử lý thủ công bổ sung, nhưng cho phép mô hình xuất trực tiếp tập khung bao cuối cùng mà không cần lọc trùng lặp, thuận lợi khi chuyển sang TensorFlow Lite cho Android."),
      code(["from ultralytics import YOLO", "model = YOLO('yolo26s.pt')", "model.train(data=f'{dataset26.location}/data.yaml',", "            epochs=60, imgsz=640, batch=16)   # không cần cấu hình NMS"]),

      // ════════ 2.5 RT-DETR ════════
      H(1, "Mô hình RT-DETR phát hiện đối tượng (Datio_yolo Dataset)"),
      body("RT-DETR (Real-Time Detection Transformer) [14] tiếp cận bài toán phát hiện đối tượng theo hướng hoàn toàn khác: thay vì dùng lưới ô và anchor như YOLO, mô hình sử dụng bộ giải mã Transformer với cơ chế chú ý toàn cục để dự đoán trực tiếp một tập đối tượng. Mô hình vẫn dùng chung bộ Datio_yolo."),
      H(2, "Thu thập dữ liệu"),
      body("RT-DETR trong hệ sinh thái Ultralytics tiêu thụ trực tiếp định dạng nhãn YOLO, do đó nhóm tái sử dụng đúng bộ dữ liệu và tệp data.yaml đã chuẩn bị cho YOLOv11 ở mục 2.3.1 mà không cần chuyển đổi định dạng. Điều này cho phép so sánh công bằng giữa các kiến trúc trên cùng một dữ liệu đầu vào."),
      H(2, "Tiền xử lý dữ liệu"),
      body("Các bước resize 640, chuẩn hóa và letterbox vẫn được áp dụng. Tuy nhiên, do kiến trúc Transformer có nhu cầu dữ liệu lớn hơn và hội tụ chậm hơn các mô hình dựa trên tích chập, nhóm điều chỉnh chiến lược huấn luyện cho phù hợp: sử dụng bộ tối ưu AdamW với tốc độ học nhỏ, kích thước lô nhỏ hơn để phù hợp bộ nhớ, và đặc biệt dựa nhiều vào học chuyển giao từ trọng số tiền huấn luyện do dữ liệu chỉ hơn một nghìn ảnh. Do giới hạn tài nguyên tính toán, số epoch của RT-DETR được giới hạn ở mức 10, nên kết quả phản ánh năng lực của mô hình trong điều kiện huấn luyện ngắn."),
      code(["from ultralytics import RTDETR", "model = RTDETR('rtdetr-l.pt')           # trọng số tiền huấn luyện", "model.train(data=f'{dataset.location}/data.yaml',", "            epochs=10, imgsz=640, batch=8)"]),
      body("Một đặc điểm quan trọng về mặt dữ liệu là RT-DETR không sử dụng anchor và không cần bước NMS: mô hình dùng thuật toán so khớp Hungary (Hungarian matching) để ghép mỗi dự đoán với đúng một đối tượng thực, nên không cần các bước tiền xử lý liên quan đến anchor như các mô hình phát hiện truyền thống [13]."),

      // ════════ 2.6 RF-DETR ════════
      H(1, "Mô hình RF-DETR phát hiện đối tượng (Datio_yolo Dataset)"),
      body("RF-DETR [15] là mô hình Transformer thời gian thực do Roboflow phát triển, sử dụng backbone DINOv2 được tiền huấn luyện theo phương pháp tự giám sát, nhờ đó hội tụ nhanh với số epoch nhỏ. Đây là mô hình duy nhất trong nhóm yêu cầu chuyển đổi định dạng dữ liệu, nên phần thu thập có điểm khác biệt rõ rệt."),
      H(2, "Thu thập dữ liệu"),
      body("Khác với ba mô hình trên dùng định dạng YOLO, RF-DETR yêu cầu nhãn ở định dạng COCO [16] — một chuẩn lưu nhãn dưới dạng tệp JSON thay vì nhiều tệp văn bản rời. Vì vậy, từ cùng một bộ ảnh Datio_yolo, nhóm tải về bản xuất theo định dạng COCO:"),
      code(["dataset_coco = project.version(1).download('coco')"]),
      body("Mỗi tập dữ liệu khi đó đi kèm một tệp _annotations.coco.json duy nhất, mô tả toàn bộ nhãn của tập theo ba thành phần chính: danh sách ảnh (images), danh sách chú thích đối tượng (annotations) và danh sách lớp (categories). Một điểm khác biệt quan trọng so với YOLO là tọa độ khung bao trong COCO được lưu ở dạng tuyệt đối theo pixel với cấu trúc [x_góc-trái-trên, y_góc-trái-trên, width, height], thay vì dạng tâm đã chuẩn hóa như YOLO. Sự khác biệt này được thư viện RF-DETR xử lý nội bộ khi nạp dữ liệu."),
      H(2, "Tiền xử lý dữ liệu"),
      body("Về tiền xử lý cơ bản, RF-DETR đưa ảnh về độ phân giải chia hết cho 56 (nhóm chọn 560×560) do ràng buộc kích thước của backbone DINOv2, kèm chuẩn hóa theo trung bình và độ lệch chuẩn của bộ ImageNet — điểm khác với cách chia 255 đơn giản của các mô hình trước. Về kỹ thuật nâng cao, do hạn chế bộ nhớ GPU khi huấn luyện mô hình Transformer lớn, nhóm áp dụng tích lũy gradient (gradient accumulation): thay vì cập nhật trọng số sau mỗi lô nhỏ, hệ thống cộng dồn gradient qua nhiều lô rồi mới cập nhật một lần, qua đó mô phỏng kích thước lô lớn hơn mà không vượt quá dung lượng bộ nhớ:"),
      code(["from rfdetr import RFDETRSmall", "model = RFDETRSmall()                    # backbone DINOv2 tiền huấn luyện", "model.train(dataset_dir=dataset_coco.location,", "            epochs=10, batch_size=4, grad_accum_steps=4)  # lô hiệu dụng = 16"]),
      body("Tương tự RT-DETR, RF-DETR dựa trên cơ chế dự đoán tập hợp của họ DETR nên không cần anchor và NMS, giúp quy trình tiền xử lý nhãn gọn nhẹ, tập trung chủ yếu vào việc bảo đảm tính đúng đắn của tệp chú thích COCO."),

      // ════════ TLTK ════════
      new Paragraph({ pageBreakBefore: true, spacing: { after: 120 }, children: [new TextRun({ text: "TÀI LIỆU THAM KHẢO CHƯƠNG 2", bold: true, size: 28 })] }),
      ...refs.map(refP),
    ],
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync("outputs/BaoCao_Chuong2.docx", b); console.log("OK -> outputs/BaoCao_Chuong2.docx", b.length, "bytes"); });
