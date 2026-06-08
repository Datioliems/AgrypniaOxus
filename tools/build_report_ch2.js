// Sinh Chương 2 (Chuẩn bị dữ liệu) — Word đúng format IS54A.
// TNR 13pt · thụt đầu dòng 1.27cm · heading đa cấp CHƯƠNG X / X.Y. / X.Y.Z / a.b.c.
// Caption bảng TRÊN (có dấu .) · caption hình DƯỚI (không dấu .) · đoạn văn liên tục · APA.
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, LevelFormat, WidthType, BorderStyle, ShadingType,
} = require("docx");

const IMG = "outputs/report_imgs";
const img = (n) => fs.readFileSync(`${IMG}/${n}`);

// ── Đoạn văn thân bài: thụt đầu dòng 1.27cm (720 twips), căn đều ──
const body = (text) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 120, line: 312 },
  indent: { firstLine: 720 },
  children: [new TextRun(text)],
});

// ── Heading đa cấp (dùng numbering reference "hd") ──
const H = (level, text) => new Paragraph({
  heading: [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3, HeadingLevel.HEADING_4][level],
  numbering: { reference: "hd", level },
  spacing: { before: 240, after: 120 },
  children: [new TextRun({ text, bold: true })],
});

// ── Hình: ảnh + caption DƯỚI (không dấu .) ──
const figure = (file, caption, w, h) => [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 },
    children: [new ImageRun({ type: file.endsWith(".png") ? "png" : "jpg", data: img(file),
      transformation: { width: w, height: h } })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
    children: [new TextRun({ text: caption, italics: true, size: 24 })] }),
];

// ── Caption bảng TRÊN (có dấu .) ──
const tableCaption = (text) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
  children: [new TextRun({ text, italics: true, size: 24 })] });

// ── Code capture (Consolas, nền xám) ──
const code = (lines) => new Paragraph({
  spacing: { before: 80, after: 120 }, shading: { type: ShadingType.CLEAR, fill: "F2F2F2" },
  border: { top: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" }, bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" },
            left: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" }, right: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" } },
  children: lines.flatMap((l, i) => [new TextRun({ text: l, font: "Consolas", size: 20, break: i ? 1 : 0 })]),
});

// ── Bảng thống kê dữ liệu MRL ──
const dataRow = (cells, head) => new TableRow({ children: cells.map((c) => new TableCell({
  width: { size: 2340, type: WidthType.DXA },
  shading: head ? { type: ShadingType.CLEAR, fill: "D5E8F0" } : undefined,
  margins: { top: 60, bottom: 60, left: 100, right: 100 },
  children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: c, bold: !!head })] })],
})) });
const dataTable = new Table({
  width: { size: 9360, type: WidthType.DXA }, columnWidths: [2340, 2340, 2340, 2340],
  rows: [
    dataRow(["Tập dữ liệu", "eyes_closed", "eyes_open", "Tổng"], true),
    dataRow(["Train (60%)", "25.167", "25.770", "50.937"]),
    dataRow(["Validation (20%)", "8.389", "8.591", "16.980"]),
    dataRow(["Test (20%)", "8.390", "8.591", "16.981"]),
    dataRow(["Tổng cộng", "41.946", "42.952", "84.898"], true),
  ],
});

const doc = new Document({
  numbering: { config: [{ reference: "hd", levels: [
    { level: 0, format: LevelFormat.DECIMAL, text: "CHƯƠNG %1:", alignment: AlignmentType.START, start: 2 },
    { level: 1, format: LevelFormat.DECIMAL, text: "%1.%2.", alignment: AlignmentType.START },
    { level: 2, format: LevelFormat.DECIMAL, text: "%1.%2.%3.", alignment: AlignmentType.START },
    { level: 3, format: LevelFormat.LOWER_LETTER, text: "%4.", alignment: AlignmentType.START },
  ] }] },
  styles: { default: { document: { run: { font: "Times New Roman", size: 26 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Times New Roman", size: 32, bold: true, color: "1F3864" }, paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Times New Roman", size: 28, bold: true, color: "2E5496" }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Times New Roman", size: 26, bold: true }, paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
      { id: "Heading4", name: "Heading 4", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Times New Roman", size: 26, bold: true, italics: true }, paragraph: { spacing: { before: 120, after: 60 }, outlineLevel: 3 } },
    ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children: [
      H(0, " CHUẨN BỊ DỮ LIỆU"),
      body("Chương này trình bày toàn bộ quá trình chuẩn bị dữ liệu cho hệ thống phát hiện và cảnh báo trạng thái buồn ngủ của tài xế, bao gồm việc thu thập dữ liệu từ nhiều nguồn khác nhau, phương pháp gán nhãn, tổ chức và thống kê dữ liệu, cũng như các kỹ thuật tiền xử lý và trích xuất đặc trưng. Do đặc thù triển khai trực tiếp trên thiết bị di động, dữ liệu được lựa chọn và xử lý theo hướng vừa bảo đảm tính tổng quát hóa của mô hình, vừa bám sát điều kiện thực tế của luồng ảnh từ camera điện thoại."),

      H(1, "Thu thập dữ liệu"),
      body("Để giải quyết bài toán một cách hiệu quả, nhóm nghiên cứu kết hợp sử dụng cả dữ liệu công khai nhằm bảo đảm quy mô và tính tổng quát hóa, lẫn dữ liệu tự thu thập nhằm tinh chỉnh mô hình cho phù hợp với môi trường triển khai thực tế trên thiết bị Android."),

      H(2, "Dữ liệu có sẵn"),
      body("Nguồn dữ liệu công khai chủ đạo là MRL Eye Dataset của Phòng thí nghiệm Media Research Lab, Đại học Kỹ thuật Ostrava (Fusek, 2018). Đây là tập dữ liệu quy mô lớn chuyên biệt cho bài toán nhận diện trạng thái mắt, gồm 84.898 ảnh vùng mắt của 37 đối tượng được ghi lại bằng cảm biến hồng ngoại trong nhiều điều kiện chiếu sáng, góc chụp và có/không đeo kính. Toàn bộ ảnh đã được gán nhãn sẵn theo hai trạng thái cơ bản là mắt mở và mắt nhắm, bảo đảm sự đa dạng về sinh trắc học và môi trường thu nhận."),
      body("Bên cạnh đó, để bổ sung dữ liệu ảnh màu RGB ba kênh phù hợp với luồng camera điện thoại, nhóm sử dụng thêm bộ dữ liệu sáu lớp trên nền tảng Roboflow Universe, gồm các ảnh khuôn mặt tài xế thật được gán nhãn hộp giới hạn (bounding box) cho từng vùng mắt trái, mắt phải và miệng theo sáu lớp: close_eyeL, close_eyeR, open_eyeL, open_eyeR, yawn và no_yawn. Khác với MRL vốn là ảnh hồng ngoại đã cắt sẵn, bộ này cung cấp ảnh màu trong điều kiện thực tế (ánh sáng buồng lái, đeo kính râm), được dùng vừa cho mô hình phát hiện vật thể (YOLO/Transformer), vừa làm nguồn ảnh màu để trích xuất ROI cho mạng tích chập."),

      H(2, "Dữ liệu tự thu thập"),
      body("Mặc dù dữ liệu có sẵn rất lớn, chúng thường được thu thập qua camera hồng ngoại hoặc webcam cố định trên máy tính, khác biệt đáng kể so với camera trước của điện thoại thông minh. Để thu hẹp khoảng cách triển khai, nhóm tiến hành tự thu thập một tập dữ liệu nhỏ trực tiếp từ thiết bị Android — thiết bị mục tiêu của hệ thống. Quá trình thu thập sử dụng camera trước của điện thoại đặt ở vị trí tương tự giá đỡ trên ô tô, quay các video clip ngắn từ 10 đến 20 giây ở nhiều kịch bản gồm: tỉnh táo và chớp mắt bình thường, nhắm mắt liên tục trong 2–3 giây, mắt lờ đờ nhắm chậm mở chậm, và đang ngáp hoặc mô phỏng trạng thái ngáp."),
      body("Từ các đoạn video đã quay, nhóm trích xuất khung hình với tần suất 2–5 khung trên giây. Các khung hình mờ hoặc trùng lặp quá nhiều được loại bỏ nhằm bảo đảm tính đa dạng và giảm nhiễu. Hình 2.1 minh họa một số khung hình màu RGB tiêu biểu trong tập dữ liệu thô trước khi xử lý."),
      ...figure("step0_raw.png", "Hình 2.1. Một số khung hình màu RGB tiêu biểu trong tập dữ liệu thô", 560, 280),

      H(2, "Phương pháp gán nhãn"),
      body("Với dữ liệu MRL, dữ liệu gốc vốn được đánh nhãn theo trạng thái sinh lý; nhóm ánh xạ lại thành hai nhãn chuẩn hóa cho bài toán phân loại nhị phân là eyes_open (mắt mở) và eyes_closed (mắt nhắm). Với dữ liệu tự thu thập, nhóm áp dụng hai phương án gán nhãn song song."),
      body("Phương án thứ nhất sử dụng công cụ Roboflow để gán nhãn bán tự động (label assist): mô hình phát hiện được dùng để tự động vẽ hộp giới hạn cho vùng người, mắt và miệng, sau đó người gán nhãn kiểm tra và hiệu chỉnh lại. Cách làm này tạo ra nhãn dạng hộp giới hạn phục vụ trực tiếp cho các mô hình phát hiện vật thể. Hình 2.2 minh họa kết quả gán nhãn tự động bằng Roboflow trên một khung hình tài xế thật."),
      ...figure("roboflow_label.jpg", "Hình 2.2. Gán nhãn bán tự động bằng Roboflow (vùng person, eye, mouth kèm độ tin cậy)", 520, 293),
      body("Phương án thứ hai tận dụng chính bước trích xuất vùng quan tâm (ROI) trong quy trình tiền xử lý: khung hình được đưa qua MediaPipe Face Landmarker để định vị khuôn mặt, sau đó cắt riêng vùng mắt và vùng miệng; nhãn của các vùng cắt được suy ra trực tiếp từ chủ đích của từng video (ví dụ toàn bộ khung hình từ video nhắm mắt được gán eyes_closed), nhờ đó giảm đáng kể công sức gán nhãn thủ công từng khung. Hình 2.3 minh họa kết quả cắt vùng mắt và miệng từ một ảnh khuôn mặt màu."),
      ...figure("step3_roi.png", "Hình 2.3. Trích xuất ROI mắt trái, mắt phải và miệng từ ảnh khuôn mặt bằng MediaPipe", 600, 150),

      H(2, "Tổ chức và thống kê dữ liệu"),
      body("Dữ liệu sau khi thu thập và gán nhãn được tổ chức theo cấu trúc thư mục tiêu chuẩn cho bài toán phân loại ảnh, phân chia thành ba tập là Huấn luyện (Train) chiếm 60%, Xác thực (Validation) chiếm 20% và Kiểm thử (Test) chiếm 20%. Qua quan sát và thống kê, tập dữ liệu đạt sự cân bằng tương đối tốt giữa hai lớp, phù hợp để huấn luyện mô hình học sâu mà không gặp hiện tượng mất cân bằng lớp (class imbalance). Bảng 2.1 trình bày số lượng mẫu sau khi chuẩn hóa."),
      tableCaption("Bảng 2.1. Thống kê số lượng mẫu của bộ dữ liệu MRL Eye sau khi chuẩn hóa và phân chia."),
      dataTable,
      body(""),

      H(1, "Tiền xử lý dữ liệu và trích xuất đặc trưng"),
      body("Dữ liệu hình ảnh thô không thể đưa trực tiếp vào mô hình mà cần trải qua quá trình tiền xử lý và trích xuất đặc trưng. Đề tài kết hợp cả các kỹ thuật xử lý ảnh cơ bản đã được học trên lớp và công nghệ thị giác máy tính hiện đại chưa nằm trong chương trình giảng dạy cơ bản."),

      H(2, "Các kỹ thuật tiền xử lý ảnh cơ bản"),
      body("Nhóm kỹ thuật này đóng vai trò làm sạch và chuẩn hóa đầu vào cho mạng nơ-ron tích chập. Trước hết, toàn bộ ảnh vùng quan tâm (ROI vùng mắt) được đồng nhất kích thước về chuẩn 64×64 điểm ảnh nhằm bảo đảm tính nhất quán cho tensor đầu vào của mô hình. Tiếp theo, hình ảnh được đọc dưới dạng ba kênh màu RGB thay vì ảnh xám; mặc dù ảnh mắt có thể dùng ảnh xám để giảm khối lượng tính toán, việc giữ RGB giúp mô hình tận dụng tốt hơn các thuật toán tăng cường dữ liệu và phù hợp với luồng ảnh màu thực tế từ camera điện thoại."),
      body("Cuối cùng, các giá trị điểm ảnh ban đầu nằm trong dải [0, 255] được chia cho 255 để đưa về dải [0, 1]. Việc đưa dữ liệu về cùng một thang đo nhỏ giúp thuật toán tối ưu Gradient Descent hội tụ nhanh và ổn định hơn, đồng thời hạn chế hiện tượng bùng nổ gradient trong quá trình huấn luyện (Ioffe & Szegedy, 2015). Đáng chú ý, lớp chuẩn hóa này được thực hiện ở cấp độ pipeline xử lý trên Android thay vì gắn thẳng vào tệp mô hình TFLite, nhằm tối ưu bộ nhớ cho ứng dụng di động. Đoạn mã dưới đây minh họa thao tác chuẩn hóa được áp dụng trong dự án."),
      code(["# Tiền xử lý: BGR -> RGB, /255 đưa về [0,1], đầu vào CNN 64x64x3",
            "x = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype('float32') / 255.0",
            "x = cv2.resize(x, (64, 64))[None]   # (1, 64, 64, 3)"]),
      body("Hình 2.4 so sánh phân bố giá trị điểm ảnh trước và sau khi chuẩn hóa, cho thấy dải giá trị được nén từ [0, 255] về [0, 1]."),
      ...figure("step8_normalize.png", "Hình 2.4. Phân bố giá trị điểm ảnh trước (trái) và sau khi chuẩn hóa về [0, 1] (phải)", 520, 189),
      body("Bên cạnh đó, để tăng tính đa dạng và khả năng tổng quát hóa, nhóm áp dụng tăng cường dữ liệu (data augmentation) gồm lật ngang, xoay nhẹ, thay đổi độ sáng và phóng to ngẫu nhiên. Hình 2.5 minh họa một số biến thể tăng cường sinh ra từ cùng một ảnh gốc."),
      ...figure("step6_augment.png", "Hình 2.5. Các biến thể tăng cường dữ liệu từ một ảnh ROI gốc", 540, 116),

      H(2, "Kỹ thuật trích xuất đặc trưng nâng cao"),
      body("Ngoài mô hình học sâu, đề tài áp dụng một phương pháp trích xuất đặc trưng hình học chưa nằm trong chương trình học cơ bản nhằm xây dựng hệ thống chạy thời gian thực trên Android, đó là sử dụng MediaPipe Face Landmarker để tính toán các chỉ số EAR và MAR. MediaPipe Face Landmarker là giải pháp thị giác máy tính mã nguồn mở của Google (Lugaresi và cộng sự, 2019; Kartynnik và cộng sự, 2019), có khả năng trích xuất 468 điểm mốc trên khuôn mặt dưới dạng tọa độ ba chiều theo thời gian thực với độ trễ rất thấp ngay trên thiết bị di động."),
      body("Dựa vào các điểm mốc thu được quanh vùng mắt và môi, nhóm áp dụng công thức của Soukupová và Čech (2016) để tính Tỷ lệ khung hình mắt (Eye Aspect Ratio, EAR) và Tỷ lệ khung hình miệng (Mouth Aspect Ratio, MAR). EAR là tỷ lệ giữa khoảng cách chiều dọc và chiều ngang của mắt; khi mắt nhắm lại, giá trị EAR giảm tiệm cận về 0. Đoạn mã sau minh họa cách tính EAR từ sáu điểm mốc của mỗi mắt được áp dụng trong dự án."),
      code(["# EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|)  (Soukupová & Čech, 2016)",
            "def ear(p):",
            "    v = dist(p[1], p[5]) + dist(p[2], p[4])",
            "    h = 2.0 * dist(p[0], p[3]) + 1e-6",
            "    return v / h"]),
      body("Kỹ thuật này giải quyết xuất sắc bài toán xác định vị trí đối tượng (object localization). Việc cắt vùng mắt dựa trên tọa độ của MediaPipe, thay vì dùng các bộ dò khuôn mặt truyền thống như Haar Cascade, giúp hệ thống xử lý chính xác ngay cả khi tài xế nghiêng đầu, đồng thời tạo ra một hệ thống trí tuệ nhân tạo có thể giải thích được (Explainable AI) khi đối chiếu kết quả EAR hình học với dự đoán của mạng tích chập. Đây cũng chính là phương án gán nhãn thứ hai đã trình bày ở mục 2.1.3."),

      H(1, "Quan sát dữ liệu và các vấn đề cần làm sạch"),
      body("Quá trình khảo sát cho thấy dữ liệu thô tồn tại một số vấn đề cần xử lý. Thứ nhất, các khung hình trích từ video có hiện tượng trùng lặp cao do hai khung liền kề gần như giống nhau, dễ gây rò rỉ dữ liệu giữa tập huấn luyện và kiểm thử nếu chia tách không cẩn thận. Thứ hai, một số ảnh bị mờ do chuyển động hoặc thiếu sáng, làm giảm chất lượng đặc trưng. Thứ ba, dữ liệu màu RGB và dữ liệu hồng ngoại MRL thuộc hai miền (modality) khác nhau, cần được cân nhắc khi trộn chung để tránh làm nhiễu mô hình."),
      body("Tương ứng, nhóm áp dụng các biện pháp làm sạch gồm: lọc ảnh mờ dựa trên phương sai của toán tử Laplacian và ngưỡng độ sáng trung bình; loại bỏ ảnh trùng lặp bằng băm MD5; và kiểm tra tính toàn vẹn cặp ảnh – nhãn trước khi đưa vào huấn luyện. Những biện pháp này được tổ chức thành quy trình tiền xử lý mười bước, có sinh ảnh kết quả ở từng bước để kiểm chứng trực quan."),

      new Paragraph({ spacing: { before: 200 }, children: [new TextRun({ text: "TÀI LIỆU THAM KHẢO (CHƯƠNG 2)", bold: true })] }),
      ...[
        "Fusek, R. (2018). Pupil localization using geodesic distance. MRL Eye Dataset, Media Research Lab, VSB – Technical University of Ostrava. http://mrl.cs.vsb.cz/eyedataset",
        "Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. ICML.",
        "Kartynnik, Y., Ablavatski, A., Grishchenko, I., & Grundmann, M. (2019). Real-time facial surface geometry from monocular video on mobile GPUs. CVPR Workshops.",
        "Lugaresi, C., et al. (2019). MediaPipe: A framework for building perception pipelines. arXiv:1906.08172.",
        "Soukupová, T., & Čech, J. (2016). Real-time eye blink detection using facial landmarks. 21st Computer Vision Winter Workshop (CVWW).",
      ].map((r) => new Paragraph({ spacing: { after: 60 }, indent: { left: 720, hanging: 720 }, children: [new TextRun({ text: r, size: 24 })] })),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("outputs/BaoCao_Chuong2.docx", buf);
  console.log("OK -> outputs/BaoCao_Chuong2.docx");
});
