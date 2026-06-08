// Sinh Chương 4 (Đánh giá và thử nghiệm) — Word đúng format IS54A. Trình bày TRUNG THỰC.
const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, LevelFormat, WidthType, BorderStyle, ShadingType } = require("docx");
const IMG = "outputs/report_imgs"; const EV = "outputs/evaluation";
const img = (n) => fs.readFileSync(n.includes("/") ? n : `${IMG}/${n}`);

const body = (t) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 312 }, indent: { firstLine: 720 }, children: [new TextRun(t)] });
const H = (lv, t) => new Paragraph({ heading: [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3, HeadingLevel.HEADING_4][lv], numbering: { reference: "hd", level: lv }, spacing: { before: 240, after: 120 }, children: [new TextRun({ text: t, bold: true })] });
const figure = (f, cap, w, h) => [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, children: [new ImageRun({ type: f.endsWith(".png") ? "png" : "jpg", data: img(f), transformation: { width: w, height: h } })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new TextRun({ text: cap, italics: true, size: 24 })] }),
];
const tcap = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, children: [new TextRun({ text: t, italics: true, size: 24 })] });

const cell = (t, head, w, bold) => new TableCell({ width: { size: w, type: WidthType.DXA }, shading: head ? { type: ShadingType.CLEAR, fill: "D5E8F0" } : undefined, margins: { top: 50, bottom: 50, left: 80, right: 80 }, children: [new Paragraph({ alignment: head ? AlignmentType.CENTER : AlignmentType.LEFT, children: [new TextRun({ text: t, bold: !!head || !!bold, size: 22 })] })] });
const mkTable = (widths, rows) => new Table({ width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths, rows: rows.map((r, ri) => new TableRow({ children: r.map((c, ci) => cell(c, ri === 0, widths[ci], ci === 0 && ri > 0)) })) });

// Bảng 4.1 — mAP detector
const W1 = [1900, 1500, 1700, 2000, 2260];
const t41 = mkTable(W1, [
  ["Mô hình", "Tham số", "mAP@50 (%)", "mAP@50-95 (%)", "Đặc điểm"],
  ["YOLOv11s", "~9,4 M", "97,0", "73,5", "Tốt nhất, có NMS"],
  ["YOLO26s", "~9,5 M", "96,2", "71,4", "Không NMS (end-to-end)"],
  ["RT-DETR", "~32 M", "95,1", "70,7", "Transformer, nặng hơn"],
  ["RF-DETR", "~29 M", "93,8", "67,3", "Backbone DINOv2"],
]);
// Bảng 4.2 — khác biệt cấu hình
const W2 = [2100, 3580, 3680];
const t42 = mkTable(W2, [
  ["Tiêu chí", "Cấu hình biến thứ nhất", "Cấu hình biến thứ hai"],
  ["Nền tảng", "PyTorch + thư viện timm", "TensorFlow/Keras + Ultralytics"],
  ["Bài toán", "Phân loại ảnh 4 lớp (combined_4class)", "Phát hiện vật thể 6 lớp + phân loại nhị phân mắt/ngáp"],
  ["Cách dùng RT-DETR", "Dùng backbone hgnetv2_b4 làm bộ phân loại", "Dùng làm bộ phát hiện đối tượng có khung bao"],
  ["Mạng CNN", "Một CNN 4 lớp, hai lớp tích chập mỗi khối", "Hai CNN nhị phân gọn nhẹ cho thiết bị di động"],
  ["Đặc trưng bổ trợ", "Không sử dụng", "MediaPipe ROI + EAR/MAR (hợp nhất)"],
  ["Chiến lược tinh chỉnh", "Hai pha đóng/mở băng + mixed precision", "Tinh chỉnh trực tiếp từ trọng số COCO"],
  ["Triển khai", "Dừng ở môi trường Colab", "Đóng gói TFLite chạy on-device trên Android"],
]);
// Bảng 4.3 — kết quả định lượng
const W3 = [2600, 3380, 3380];
const t43 = mkTable(W3, [
  ["Mô hình", "Cấu hình biến thứ nhất", "Cấu hình biến thứ hai"],
  ["CNN (phân loại trạng thái)", "94,84% (Macro F1 0,949)", "Mắt 97,44% / Ngáp 98,64%"],
  ["Mô hình nâng cao", "RT-DETR phân loại: 98,66%", "YOLOv11s phát hiện: mAP@50 97,0%"],
  ["Số lớp / Tập kiểm thử", "4 lớp / 1.123 ảnh", "6 lớp (phát hiện) + nhị phân"],
]);

const refs = [
  "Everingham, M., et al. (2010). The PASCAL Visual Object Classes (VOC) challenge. IJCV, 88(2), 303-338.",
  "Padilla, R., Netto, S. L., & da Silva, E. A. B. (2020). A survey on performance metrics for object-detection algorithms. IWSSIP.",
  "Powers, D. M. W. (2011). Evaluation: From precision, recall and F-measure to ROC, informedness, markedness and correlation. JMLT, 2(1), 37-63.",
  "Soukupová, T., & Čech, J. (2016). Real-time eye blink detection using facial landmarks. CVWW.",
  "Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. Information Processing & Management, 45(4), 427-437.",
];

const doc = new Document({
  numbering: { config: [{ reference: "hd", levels: [
    { level: 0, format: LevelFormat.DECIMAL, text: "CHƯƠNG %1:", alignment: AlignmentType.START, start: 4 },
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
      H(0, " ĐÁNH GIÁ VÀ THỬ NGHIỆM"),
      body("Chương này trình bày các độ đo dùng để đánh giá, kết quả huấn luyện của từng mô hình, so sánh khách quan giữa hai phương án cấu hình, và kết quả thử nghiệm hệ thống trong điều kiện thực tế. Toàn bộ số liệu được báo cáo một cách trung thực, bao gồm cả những hạn chế quan sát được nhằm phục vụ cải tiến về sau."),

      H(1, "Phương pháp và độ đo đánh giá"),
      H(2, "Độ đo cho bài toán phân loại"),
      body("Đối với các mô hình phân loại trạng thái, nhóm sử dụng bốn độ đo phổ biến gồm độ chính xác (Accuracy), độ chuẩn xác (Precision), độ phủ (Recall) và điểm F1 (Powers, 2011; Sokolova & Lapalme, 2009). Độ chính xác là tỷ lệ dự đoán đúng trên tổng số mẫu; độ chuẩn xác là tỷ lệ dự đoán dương đúng trên tổng số dự đoán dương; độ phủ phản ánh khả năng phát hiện hết các mẫu dương; điểm F1 là trung bình điều hòa của Precision và Recall, đặc biệt phù hợp khi dữ liệu mất cân bằng. Trong bài toán phát hiện buồn ngủ, độ phủ của lớp nhắm mắt và lớp ngáp được ưu tiên vì việc bỏ sót trạng thái nguy hiểm có hậu quả nghiêm trọng hơn báo động giả."),
      body("Bên cạnh đó, ma trận nhầm lẫn (Confusion Matrix) được sử dụng để quan sát chi tiết các trường hợp bị phân loại sai giữa các lớp, từ đó xác định mô hình nhầm lẫn ở đâu để có hướng cải thiện."),
      H(2, "Độ đo cho bài toán phát hiện vật thể"),
      body("Đối với các mô hình phát hiện vật thể, độ đo chính là mean Average Precision (mAP) tính trên các ngưỡng giao trên hợp (Intersection over Union - IoU) khác nhau (Everingham và cộng sự, 2010; Padilla, Netto, & da Silva, 2020). Chỉ số mAP@50 đo độ chính xác trung bình tại ngưỡng IoU bằng 0,5, trong khi mAP@50-95 lấy trung bình trên dải ngưỡng từ 0,5 đến 0,95 và là thước đo nghiêm ngặt hơn về độ khớp của khung bao dự đoán so với khung bao thực tế."),

      H(1, "Kết quả huấn luyện các mô hình của đề tài"),
      H(2, "Kết quả mô hình phân loại CNN"),
      body("Hai mạng tích chập phân loại trạng thái đạt kết quả cao trên tập dữ liệu công khai: mô hình phân loại trạng thái mắt đạt độ chính xác 97,44% và mô hình phân loại ngáp đạt 98,64%. Ma trận nhầm lẫn ở Hình 4.1 cho thấy mô hình ngáp phân tách rất tốt hai lớp, trong khi mô hình mắt nhạy cảm hơn với kiểu cắt ảnh đầu vào, sẽ được phân tích kỹ ở mục 4.5."),
      ...figure(`${EV}/confusion_compare.png`, "Hình 4.1. Ma trận nhầm lẫn của các mô hình phân loại (đánh giá trên tập giữ lại)", 470, 427),
      H(2, "Kết quả mô hình phát hiện vật thể"),
      body("Bốn mô hình phát hiện vật thể được huấn luyện và đánh giá trên bộ dữ liệu RGB sáu lớp. Kết quả ở Bảng 4.1 và Hình 4.2 cho thấy YOLOv11s đạt kết quả tốt nhất với mAP@50-95 bằng 73,5%, nhỉnh hơn YOLO26s; hai mô hình Transformer RT-DETR và RF-DETR cho kết quả thấp hơn một chút nhưng có ưu thế về định vị trong các trường hợp khó. Do giới hạn tài nguyên, các mô hình Transformer chỉ được huấn luyện với số epoch hạn chế nên dư địa cải thiện vẫn còn."),
      tcap("Bảng 4.1. So sánh kết quả các mô hình phát hiện vật thể trên bộ dữ liệu RGB sáu lớp."),
      t41, body(""),
      ...figure("map_compare.png", "Hình 4.2. Biểu đồ so sánh mAP của bốn mô hình phát hiện vật thể", 480, 252),

      H(1, "So sánh hai phương án cấu hình"),
      body("Để có cái nhìn khách quan, đề tài so sánh hai phương án cấu hình giải quyết cùng bài toán phát hiện buồn ngủ. Cấu hình biến thứ nhất tiếp cận theo hướng phân loại ảnh trên nền tảng PyTorch, còn cấu hình biến thứ hai (cấu hình của đề tài này) tiếp cận theo hướng phát hiện vật thể kết hợp phân loại và đặc trưng hình học, đồng thời triển khai trực tiếp lên thiết bị di động."),
      H(2, "Khác biệt về cấu hình và phương pháp"),
      body("Bảng 4.2 tổng hợp những khác biệt cốt lõi giữa hai phương án. Khác biệt lớn nhất nằm ở cách hình dung bài toán: phương án thứ nhất xem đây là bài toán phân loại bốn lớp và dùng backbone của RT-DETR như một bộ trích đặc trưng, trong khi phương án thứ hai tách thành bài toán phát hiện đối tượng đa lớp kết hợp các bộ phân loại nhị phân nhẹ, bổ sung đặc trưng hình học EAR/MAR để tăng độ an toàn và khả năng giải thích."),
      tcap("Bảng 4.2. So sánh cấu hình và phương pháp giữa hai phương án."),
      t42, body(""),
      H(2, "So sánh kết quả định lượng"),
      body("Về mặt số liệu (Bảng 4.3 và Hình 4.3), cấu hình biến thứ nhất đạt độ chính xác phân loại 94,84% với CNN và 98,66% khi dùng backbone RT-DETR trên bài toán bốn lớp. Cấu hình biến thứ hai đạt độ chính xác phân loại mắt và ngáp lần lượt 97,44% và 98,64%, cùng mAP@50 đạt 97,0% cho mô hình phát hiện. Cần lưu ý rằng hai phương án giải quyết hai dạng bài toán khác nhau (phân loại so với phát hiện) nên các con số không hoàn toàn tương đương; điểm mạnh của phương án thứ hai là khả năng định vị vùng quan tâm và triển khai thực tế trên thiết bị, trong khi phương án thứ nhất cho độ chính xác phân loại thuần rất cao."),
      tcap("Bảng 4.3. So sánh kết quả định lượng giữa hai phương án cấu hình."),
      t43, body(""),
      ...figure("config_compare.png", "Hình 4.3. Biểu đồ so sánh kết quả hai phương án cấu hình", 480, 252),
      H(2, "Ảnh hưởng của chiến lược chia dữ liệu"),
      body("Một thực nghiệm bổ sung so sánh hai cách chia dữ liệu cho thấy chiến lược chia ngẫu nhiên theo từng ảnh cho độ chính xác cao hơn rõ rệt so với chia theo chủ thể (subject-independent). Nguyên nhân là khi chia ngẫu nhiên theo ảnh, các khung hình gần như trùng lặp của cùng một người có thể xuất hiện đồng thời ở cả tập huấn luyện và tập kiểm thử, gây rò rỉ dữ liệu và làm độ chính xác bị thổi phồng. Cách chia theo chủ thể cho con số thấp hơn nhưng phản ánh trung thực hơn khả năng tổng quát hóa sang người dùng mới. Đây là lưu ý quan trọng khi diễn giải các chỉ số cao trong bài toán này."),

      H(1, "Thử nghiệm và triển khai thực tế"),
      H(2, "Triển khai trên thiết bị Android"),
      body("Toàn bộ mô hình được chuyển sang định dạng TensorFlow Lite và đóng gói thành ứng dụng Android chạy suy luận hoàn toàn trên thiết bị, không cần kết nối mạng nên bảo đảm độ trễ thấp và quyền riêng tư. Hệ thống được đóng gói thành hai phiên bản ứng dụng cài song song: một phiên bản dùng mô hình huấn luyện trên dữ liệu công khai và một phiên bản dùng mô hình huấn luyện kết hợp dữ liệu tự thu thập, phục vụ đối chiếu trực tiếp trên cùng thiết bị."),
      H(2, "Kiểm thử trên video thật"),
      body("Hệ thống được kiểm thử trên video tự quay trong điều kiện thực tế. Khi áp dụng cơ chế hợp nhất lấy CNN làm chủ đạo trên các vùng mắt được cắt bằng MediaPipe, mô hình nhận diện chính xác trạng thái tỉnh táo với độ tin cậy cao và phản ứng đúng khi người lái nhắm mắt hoặc ngáp. Hình 4.4 minh họa một khung hình kết quả có chú thích trạng thái."),
      ...figure("demo_frame.jpg", "Hình 4.4. Khung hình kết quả thử nghiệm trên video thật có chú thích trạng thái", 420, 236),
      H(2, "Cơ chế cảnh báo đa tầng"),
      body("Khi phát hiện dấu hiệu buồn ngủ, hệ thống kích hoạt cơ chế cảnh báo nhiều tầng tăng dần về cường độ, kết hợp âm thanh nhịp đôi (binaural beat) ở dải beta nhằm kích thích sự tỉnh táo, rung thiết bị và thông báo trực quan. Hệ thống còn theo dõi thời gian lái liên tục để nhắc nghỉ ngơi sau ngưỡng an toàn, đồng thời cung cấp màn hình thống kê mức độ buồn ngủ và màn hình liên hệ khẩn cấp gửi vị trí cho người thân."),

      H(1, "Nhận xét, hạn chế và hướng phát triển"),
      body("Trên cơ sở các kết quả đã trình bày ở các mục trước, phần này đánh giá một cách trung thực và khách quan những ưu điểm hệ thống đạt được, các hạn chế còn tồn tại cũng như đề xuất hướng phát triển trong tương lai."),
      H(3, "Ưu điểm"),
      body("Hệ thống đạt độ chính xác phân loại cao trên tập dữ liệu công khai, với mô hình phân loại trạng thái mắt đạt 97,44% và mô hình phân loại ngáp đạt 98,64%. Toàn bộ quá trình suy luận diễn ra trực tiếp trên thiết bị di động mà không cần kết nối mạng, nhờ đó bảo đảm độ trễ thấp, hoạt động ổn định ngay cả khi mất sóng và bảo vệ quyền riêng tư của người dùng do dữ liệu hình ảnh không rời khỏi thiết bị."),
      body("Việc kết hợp đặc trưng học sâu của mạng tích chập với đặc trưng hình học EAR/MAR tạo ra cơ chế hợp nhất vừa chính xác vừa an toàn: mạng tích chập đóng vai trò chủ đạo khi đủ tin cậy, còn đặc trưng hình học làm lớp dự phòng giúp không bỏ sót trạng thái nguy hiểm. Bên cạnh đó, hệ thống tích hợp cơ chế cảnh báo đa giác quan có cơ sở khoa học gồm âm thanh nhịp đôi dải beta, rung và thông báo trực quan, màn hình thống kê mức độ buồn ngủ theo chỉ số PERCLOS, cùng tính năng tự động gửi vị trí cho người thân khi số lần buồn ngủ vượt ngưỡng an toàn trong một chuyến đi."),
      H(3, "Hạn chế"),
      body("Qua đánh giá trung thực, nhóm nhận thấy mô hình phân loại mắt phụ thuộc đáng kể vào kiểu cắt ảnh đầu vào. Mô hình hoạt động rất tốt (khoảng 98%) trên các vùng mắt được cắt bằng MediaPipe giống lúc huấn luyện, nhưng giảm mạnh (khoảng 55%) khi đánh giá trên vùng mắt cắt theo khung bao của mô hình phát hiện, do khác biệt phân bố dữ liệu (out-of-distribution). Ngược lại, mô hình ngáp ổn định ở mọi kiểu cắt. Thực nghiệm trộn dữ liệu hồng ngoại với dữ liệu màu không cải thiện rõ rệt mô hình mắt, cho thấy việc đồng nhất quy trình cắt ảnh giữa huấn luyện và suy luận quan trọng hơn việc chỉ tăng số lượng dữ liệu."),
      body("Ngoài ra, dữ liệu tự thu thập còn ít và chủ yếu là một người ở tư thế trực diện nên khả năng tổng quát hóa sang người dùng mới còn hạn chế. Các mô hình Transformer mới chỉ được huấn luyện với số epoch nhỏ do giới hạn tài nguyên nên chưa đạt tiềm năng tối đa. Như đã phân tích ở mục 4.3.3, cách chia dữ liệu ngẫu nhiên theo ảnh có thể làm chỉ số bị thổi phồng, và hệ thống chưa được kiểm thử đầy đủ trong các điều kiện khó như lái ban đêm, người đeo kính hay ánh sáng yếu."),
      H(3, "Hướng phát triển trong tương lai"),
      body("Trong thời gian tới, nhóm dự kiến thu thập bộ dữ liệu đa dạng hơn về số lượng người dùng, độ tuổi và điều kiện ánh sáng, bổ sung dữ liệu ban đêm bằng camera hồng ngoại; đồng thời đồng nhất quy trình cắt ảnh bằng MediaPipe cho cả huấn luyện lẫn suy luận và áp dụng cách đánh giá theo chủ thể để phản ánh trung thực khả năng tổng quát hóa của mô hình."),
      body("Về mặt kỹ thuật, hệ thống sẽ được tối ưu tốc độ suy luận thông qua lượng tử hóa số nguyên và tăng tốc phần cứng (NNAPI, GPU delegate) để chạy mượt trên các thiết bị cấu hình thấp, huấn luyện thêm các mô hình Transformer với số epoch lớn hơn, và mở rộng khả năng cảnh báo bằng cách kết hợp thêm cảm biến như nhịp tim hoặc độ lệch làn đường. Tính năng tự động gửi vị trí cho người thân khi vượt ngưỡng buồn ngủ cũng sẽ được hoàn thiện thêm, hướng tới tích hợp với hệ thống định vị và liên lạc khẩn cấp của phương tiện."),

      new Paragraph({ spacing: { before: 200 }, children: [new TextRun({ text: "TÀI LIỆU THAM KHẢO (CHƯƠNG 4)", bold: true })] }),
      ...refs.map((r) => new Paragraph({ spacing: { after: 60 }, indent: { left: 720, hanging: 720 }, children: [new TextRun({ text: r, size: 24 })] })),
    ],
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync("outputs/BaoCao_Chuong4.docx", b); console.log("OK -> outputs/BaoCao_Chuong4.docx"); });
