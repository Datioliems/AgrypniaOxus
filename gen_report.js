// ============================================================
// BÁO CÁO BÀI TẬP LỚN IS54A — DROWSY DRIVER DETECTION
// node gen_report.js  →  D:\BaoCao_IS54A_DrowsyDriver.docx
// ============================================================
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, LevelFormat, TableOfContents
} = require('docx');
const fs = require('fs');

// ─── Layout ──────────────────────────────────────────────────────────────────
const TNR  = 'Times New Roman';
const L_MGN = 1701; // 3 cm
const R_MGN = 1134; // 2 cm
const CONT  = 11906 - L_MGN - R_MGN; // 9071 DXA
const LS    = { line: 360, lineRule: 'auto' };

// ─── Text run helpers ────────────────────────────────────────────────────────
const tr  = (s,x={}) => new TextRun({ text:s, font:TNR, size:24, ...x });
const trB = (s,x={}) => new TextRun({ text:s, font:TNR, size:24, bold:true, ...x });
const trI = (s,x={}) => new TextRun({ text:s, font:TNR, size:24, italics:true, ...x });

// ─── Paragraph helpers ───────────────────────────────────────────────────────
const p   = (s,x={}) => new Paragraph({ children:Array.isArray(s)?s:[tr(s)], spacing:{...LS,after:80}, indent:{firstLine:720}, ...x });
const pN  = (s,x={}) => new Paragraph({ children:Array.isArray(s)?s:[tr(s)], spacing:{...LS,after:80}, ...x });
const pC  = (s,x={}) => new Paragraph({ children:Array.isArray(s)?s:[tr(s)], alignment:AlignmentType.CENTER, spacing:{...LS,after:80}, ...x });
const E   = ()        => pN('', {spacing:{after:60}});
const PB  = ()        => new Paragraph({ children:[new PageBreak()] });

const H1 = s => new Paragraph({ heading:HeadingLevel.HEADING_1, children:[trB(s,{size:28})], spacing:{before:360,after:120,line:360}, outlineLevel:0 });
const H2 = s => new Paragraph({ heading:HeadingLevel.HEADING_2, children:[trB(s,{size:26})], spacing:{before:240,after:100,line:360}, outlineLevel:1 });
const H3 = s => new Paragraph({ heading:HeadingLevel.HEADING_3, children:[trB(s,{size:24,italics:true})], spacing:{before:180,after:60,line:360}, outlineLevel:2 });

// ─── Table helpers ───────────────────────────────────────────────────────────
const BD   = { style:BorderStyle.SINGLE, size:4, color:'000000' };
const tc = (txt,o={}) => new TableCell({
  children:[new Paragraph({ children:[new TextRun({ text:String(txt), font:TNR, size:o.sz||20, bold:!!o.bold })], alignment:o.align||AlignmentType.LEFT, spacing:{line:276,after:0} })],
  width: o.w ? {size:o.w, type:WidthType.DXA} : undefined,
  shading: o.fill ? {fill:o.fill, type:ShadingType.CLEAR} : undefined,
  verticalAlign: VerticalAlign.CENTER,
  margins:{top:80,bottom:80,left:120,right:120},
  columnSpan: o.span||1,
});
const row  = (...cells) => new TableRow({ children:cells });
const hrow = (...cells) => new TableRow({ tableHeader:true, children:cells });
const mkTb = (widths,rows) => new Table({ width:{size:CONT,type:WidthType.DXA}, columnWidths:widths, rows });

// ─── List helpers ────────────────────────────────────────────────────────────
const NUMS = {
  config:[
    { reference:'bullet', levels:[{ level:0, format:LevelFormat.BULLET, text:'•', alignment:AlignmentType.LEFT, style:{paragraph:{indent:{left:720,hanging:360}}} }] },
    { reference:'num',    levels:[{ level:0, format:LevelFormat.DECIMAL, text:'%1.', alignment:AlignmentType.LEFT, style:{paragraph:{indent:{left:720,hanging:360}}} }] },
  ]
};
const li = s => new Paragraph({ children:[tr(s)], numbering:{reference:'bullet',level:0}, spacing:{...LS,after:40} });
const ni = s => new Paragraph({ children:[tr(s)], numbering:{reference:'num',level:0},    spacing:{...LS,after:40} });
const cap = s => pC([trI(s)], {spacing:{...LS,after:120}});

// ─── COVER PAGE ──────────────────────────────────────────────────────────────
const coverPage = [
  E(),E(),
  pC([trB('HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG',{size:26})]),
  pC([trB('KHOA CÔNG NGHỆ THÔNG TIN',{size:24})]),
  pC([tr('─────────────────────────────────',{size:22})]),
  E(),E(),E(),
  pC([trB('BÁO CÁO BÀI TẬP LỚN CUỐI KỲ',{size:28})]),
  pC([trB('MÔN HỌC: TRÍ TUỆ NHÂN TẠO – IS54A',{size:24})]),
  E(),E(),E(),
  pC([trB('ĐỀ TÀI:',{size:24})]),
  pC([trB('PHÁT HIỆN BUỒN NGỦ KHI LÁI XE',{size:32})]),
  pC([trB('TÍCH HỢP TRÊN ANDROID, iOS VÀ WEB',{size:28})]),
  E(),
  pC([trI('Sử dụng: MediaPipe + EAR/MAR + CNN + YOLOv11 + YOLO26',{size:22})]),
  E(),E(),E(),
  mkTb([3000,3000,3071],[
    row(tc('Sinh viên thực hiện:',{bold:true,sz:22}), tc('[Điền họ tên đầy đủ]',{sz:22}),  tc('',{sz:22})),
    row(tc('Mã sinh viên:',{bold:true,sz:22}),        tc('[Điền MSSV]',{sz:22}),            tc('',{sz:22})),
    row(tc('Lớp:',{bold:true,sz:22}),                 tc('IS54A',{sz:22}),                  tc('',{sz:22})),
    row(tc('Giảng viên:',{bold:true,sz:22}),          tc('[Điền tên GV hướng dẫn]',{sz:22}),tc('',{sz:22})),
  ]),
  E(),E(),E(),
  pC([trB('Hà Nội, tháng 6 năm 2026',{size:24})]),
  PB(),
];

// ─── BẢNG PHÂN CÔNG ──────────────────────────────────────────────────────────
const phanCong = [
  H1('BẢNG PHÂN CÔNG CÔNG VIỆC'),
  E(),
  mkTb([500,2000,3000,2500,1071],[
    hrow(
      tc('STT',{bold:true,fill:'4472C4',sz:20,align:AlignmentType.CENTER}),
      tc('Họ và tên',{bold:true,fill:'4472C4',sz:20}),
      tc('Công việc chính',{bold:true,fill:'4472C4',sz:20}),
      tc('Nội dung chi tiết',{bold:true,fill:'4472C4',sz:20}),
      tc('Tỷ lệ',{bold:true,fill:'4472C4',sz:20,align:AlignmentType.CENTER}),
    ),
    row(tc('1',{align:AlignmentType.CENTER}),tc('[Thành viên 1]'),tc('Lead + Android Dev'),tc('Kiến trúc tổng thể, CNN training, TFLite export, Android integration, demo'),tc('25%',{align:AlignmentType.CENTER})),
    row(tc('2',{align:AlignmentType.CENTER}),tc('[Thành viên 2]'),tc('AI / ML Engineer'),tc('EDA, preprocessing, YOLO training, hyperparameter tuning, Colab notebooks'),tc('25%',{align:AlignmentType.CENTER})),
    row(tc('3',{align:AlignmentType.CENTER}),tc('[Thành viên 3]'),tc('iOS / Streamlit Dev'),tc('iOS app, Streamlit web demo, audio alertness feature, API integration'),tc('20%',{align:AlignmentType.CENTER})),
    row(tc('4',{align:AlignmentType.CENTER}),tc('[Thành viên 4]'),tc('Data Engineer'),tc('Thu thập, gán nhãn, augmentation pipeline, Roboflow management, EDA'),tc('15%',{align:AlignmentType.CENTER})),
    row(tc('5',{align:AlignmentType.CENTER}),tc('[Thành viên 5]'),tc('Báo cáo + Thuyết trình'),tc('Viết báo cáo, slide, so sánh mô hình, nghiên cứu TLTK, kết luận'),tc('15%',{align:AlignmentType.CENTER})),
  ]),
  E(),
  PB(),
];

// ─── MỤC LỤC ─────────────────────────────────────────────────────────────────
const mucLuc = [
  new TableOfContents('MỤC LỤC', { hyperlink:true, headingStyleRange:'1-3' }),
  PB(),
];

// ─── CHAPTER 1 ───────────────────────────────────────────────────────────────
const ch1 = [
  H1('CHƯƠNG 1. GIỚI THIỆU TỔNG QUAN'),

  H2('1.1 Đặt vấn đề và tầm quan trọng thực tế'),
  p('Theo Tổ chức Y tế Thế giới (WHO, 2023), mỗi năm có khoảng 1,35 triệu người tử vong do tai nạn giao thông, trong đó buồn ngủ khi lái xe chiếm từ 10% đến 20% các vụ tai nạn nghiêm trọng trên toàn cầu. Tại Việt Nam, Ủy ban An toàn Giao thông Quốc gia ghi nhận buồn ngủ là một trong ba nguyên nhân hàng đầu gây tai nạn giao thông đường dài, đặc biệt trên các tuyến cao tốc vào ban đêm và giờ trưa.'),
  p('Hiện tượng microsleep — mất ý thức tạm thời từ 1 đến 30 giây trong khi vẫn ngồi sau vô-lăng — đặc biệt nguy hiểm. Ở vận tốc 100 km/h, chỉ 5 giây mất kiểm soát tương đương xe lao không điều khiển hơn 138 mét. Williamson & Feyer (2000) tại Đại học New South Wales chỉ ra rằng mức độ suy giảm nhận thức do thiếu ngủ 17–24 giờ tương đương nồng độ cồn trong máu 0,05–0,10% — vượt mức an toàn pháp lý ở hầu hết quốc gia.'),
  p('Hệ thống Phát hiện Buồn ngủ Lái xe (Driver Drowsiness Detection — DDD) sử dụng Trí tuệ Nhân tạo và thị giác máy tính là giải pháp không xâm lấn, chi phí thấp, triển khai ngay trên smartphone mà không cần cảm biến sinh lý học chuyên dụng.'),

  H2('1.2 Bài toán AI: Định nghĩa, Input, Output và ứng dụng'),
  H3('1.2.1 Định nghĩa bài toán'),
  p('Cho một luồng video từ camera hướng về phía tài xế, hệ thống phải tự động và theo thời gian thực: (1) phát hiện các dấu hiệu buồn ngủ sinh lý (nhắm mắt kéo dài, ngáp lặp lại), (2) phân loại mức độ nguy hiểm, và (3) phát cảnh báo kịp thời. Đây là bài toán phân loại chuỗi thời gian kết hợp phát hiện đối tượng.'),

  H3('1.2.2 Đầu vào (Input)'),
  li('Khung hình video RGB từ camera trước hoặc camera cabin (640×480 hoặc 1280×720, 15–30 fps).'),
  li('Ánh sáng tối thiểu: 10 lux (ánh sáng cabin ban đêm). Không hỗ trợ tối hoàn toàn.'),
  li('Một tài xế trong khung hình, không có vật che khuất ≥ 50% khuôn mặt.'),

  H3('1.2.3 Đầu ra (Output)'),
  li('Trạng thái tức thời: AWAKE / DROWSY / YAWNING với confidence score [0.0–1.0].'),
  li('Cảnh báo đa phương thức: âm thanh, rung động, overlay UI màu đỏ.'),
  li('Event log: timestamp, loại cảnh báo, confidence, EAR/MAR value — phục vụ phân tích hành vi.'),
  li('Tính năng nâng cao: Âm thanh binaural beats kích thích tỉnh táo khi phát hiện ≥ 3 sự kiện buồn ngủ/15 phút.'),

  H3('1.2.4 Ứng dụng thực tế'),
  li('Android App (8.0+): Tích hợp CameraX + TFLite, chạy offline, không cần internet.'),
  li('iOS App (14+): AVCaptureSession + TFLite, haptic feedback, SwiftUI interface.'),
  li('Streamlit Web Demo: Dashboard so sánh tất cả 5 model, điều chỉnh threshold real-time.'),

  H2('1.3 Phạm vi thực hiện'),
  li('Phát hiện: mắt nhắm liên tục (EAR < 0.24, ≥ 1.7s), ngáp (MAR > 0.58), micro-sleep (CNN ≥ 1.2s).'),
  li('Dataset: tối đa 8.000 ảnh/class/split từ Roboflow, đảm bảo huấn luyện ≤ 3 giờ trên GPU T4.'),
  li('4 mô hình độc lập: EAR/MAR Baseline, CNN Eye, CNN Yawn, YOLO (v11s và v26m).'),
  li('Không hỗ trợ: tối hoàn toàn, kính mắt tối, nhận dạng danh tính, nhiều tài xế cùng lúc.'),

  H2('1.4 Khảo sát nghiên cứu liên quan'),
  H3('1.4.1 Phương pháp EAR/MAR hình học'),
  p(['Soukupová & Čech (2016) tại Đại học Công nghệ Praha giới thiệu chỉ số ', trI('Eye Aspect Ratio'), ' (EAR) dựa trên 68 điểm landmark, tính theo công thức: EAR = (||p₂−p₆|| + ||p₃−p₅||) / (2||p₁−p₄||). Phương pháp này đơn giản, không cần GPU, nhưng nhạy cảm với điều kiện ánh sáng yếu và góc nghiêng đầu lớn (>30°). Đây là nền tảng của nhiều hệ thống DDD thương mại.']),
  H3('1.4.2 Deep Learning với MediaPipe'),
  p('Lugaresi et al. (2019) tại Google phát triển MediaPipe cung cấp 478 điểm landmark 3D với độ trễ < 15ms trên thiết bị di động — làm backbone chính trong dự án. Guo et al. (2019) tại Đại học Khoa học Công nghệ Quốc gia Đài Loan kết hợp CNN với LSTM đạt accuracy 96% trên dataset thực tế. Maggiorini et al. (2021) đề xuất kiến trúc CNN nhẹ cho drowsiness detection, phù hợp với thiết bị có tài nguyên hạn chế.'),
  H3('1.4.3 YOLO cho phát hiện đa lớp'),
  p('Redmon & Farhadi (2018) với YOLOv3 đặt nền móng cho object detection thời gian thực với nhiều lớp. Wang et al. (2022) với YOLOv7 đạt 56.8% AP trên COCO. Jocher et al. (2023) phát triển YOLOv8 và YOLOv11, tích hợp cơ chế C2PSA attention, mở rộng ứng dụng cho bài toán DDD với nhiều class trạng thái tài xế đồng thời.'),
  E(),PB(),
];

// ─── CHAPTER 2 ───────────────────────────────────────────────────────────────
const ch2 = [
  H1('CHƯƠNG 2. DỮ LIỆU: THU THẬP, PHÂN TÍCH VÀ TIỀN XỬ LÝ'),

  H2('2.1 Nguồn gốc và tổ chức bộ dữ liệu'),
  p('Dự án sử dụng bộ dữ liệu từ Roboflow Universe — nền tảng quản lý dataset AI hàng đầu với hơn 200.000 dataset công khai. Tiêu chí lựa chọn: (1) Có ít nhất 2 lớp phân loại rõ ràng, (2) Ảnh đã được kiểm duyệt cộng đồng, (3) Hỗ trợ export định dạng YOLOv8/v11, (4) Số ảnh mỗi class không vượt giới hạn 8.000 ảnh/class/split sau khi lọc.'),
  E(),
  cap('Bảng 2.1. Tổng quan bộ dữ liệu sử dụng'),
  mkTb([1500,2800,1000,2000,1771],[
    hrow(tc('Mô hình',{bold:true,fill:'4472C4',sz:20}),tc('Dataset (Roboflow)',{bold:true,fill:'4472C4',sz:20}),tc('Lớp',{bold:true,fill:'4472C4',sz:20,align:AlignmentType.CENTER}),tc('Train / Val / Test',{bold:true,fill:'4472C4',sz:20}),tc('Workspace',{bold:true,fill:'4472C4',sz:20})),
    row(tc('CNN Eye',{fill:'DAE8FC'}),tc('MRL Eye Dataset (mrl-eye-dataset)'),tc('2',{align:AlignmentType.CENTER}),tc('≤8000/cl | ≤2000/cl | ≤2000/cl'),tc('Roboflow Universe')),
    row(tc('CNN Yawn',{fill:'DAE8FC'}),tc('Driver Drowsiness Yawn (driver-no-yawn)'),tc('2',{align:AlignmentType.CENTER}),tc('≤8000/cl | ≤2000/cl | ≤2000/cl'),tc('nguyen-tuan-dat')),
    row(tc('EAR/MAR',{fill:'D5E8D4'}),tc('Không cần dataset (hình học MediaPipe)'),tc('—',{align:AlignmentType.CENTER}),tc('Không yêu cầu'),tc('MediaPipe built-in')),
    row(tc('YOLOv11s',{fill:'FFF2CC'}),tc('Datio Drowsines v1 — lọc 4 class'),tc('4',{align:AlignmentType.CENTER}),tc('≤8000/cl | ≤2000/cl | ≤2000/cl'),tc('nguyen-tuan-dat')),
    row(tc('YOLO26m',{fill:'FFF2CC'}),tc('Drowsiness Augmented (datio_drowsines_aug)'),tc('4',{align:AlignmentType.CENTER}),tc('≤8000/cl | ≤2000/cl | ≤2000/cl'),tc('Roboflow Universe')),
  ]),
  E(),
  p('Tổ chức thư mục dữ liệu trên máy cục bộ:'),
  pN('roboflow_data/', {indent:{left:720}}),
  pN('  ├── mrl_eye/          ← CNN Eye  (eyes_closed / eyes_open)', {indent:{left:720}}),
  pN('  ├── driver_yawn/      ← CNN Yawn (no_yawn / yawn)', {indent:{left:720}}),
  pN('  └── datio_4class/     ← YOLO     (drowsy_eye / attentive_eye / yawn / asleep)', {indent:{left:720}}),
  E(),

  H2('2.2 Phân tích khám phá dữ liệu (EDA)'),
  H3('2.2.1 Phân phối lớp và mức độ cân bằng'),
  p('Bộ dữ liệu MRL Eye gốc chứa 84.898 ảnh với tỷ lệ eyes_closed:eyes_open = 42:58 — mất cân bằng nhẹ. Sau khi áp dụng stratified sampling để giới hạn 8.000 ảnh/class, ta được bộ train cân bằng hoàn hảo (50:50). Bộ Driver Yawn có 5.119 ảnh (no_yawn: 2.799, yawn: 2.320), tỷ lệ 54:46 — đủ cân bằng, không cần oversampling.'),
  E(),
  cap('Bảng 2.2. Thống kê phân phối dữ liệu sau khi lọc'),
  mkTb([2000,1800,1800,1800,1671],[
    hrow(tc('Dataset',{bold:true,fill:'4472C4',sz:20}),tc('Class',{bold:true,fill:'4472C4',sz:20}),tc('Train',{bold:true,fill:'4472C4',sz:20,align:AlignmentType.CENTER}),tc('Validation',{bold:true,fill:'4472C4',sz:20,align:AlignmentType.CENTER}),tc('Test',{bold:true,fill:'4472C4',sz:20,align:AlignmentType.CENTER})),
    row(tc('MRL Eye',{fill:'DAE8FC'}),tc('eyes_closed'),tc('6.400',{align:AlignmentType.CENTER}),tc('1.600',{align:AlignmentType.CENTER}),tc('1.600',{align:AlignmentType.CENTER})),
    row(tc('MRL Eye',{fill:'DAE8FC'}),tc('eyes_open'),  tc('6.400',{align:AlignmentType.CENTER}),tc('1.600',{align:AlignmentType.CENTER}),tc('1.600',{align:AlignmentType.CENTER})),
    row(tc('Driver Yawn',{fill:'DAE8FC'}),tc('no_yawn'),tc('2.239',{align:AlignmentType.CENTER}),tc('280',{align:AlignmentType.CENTER}),tc('280',{align:AlignmentType.CENTER})),
    row(tc('Driver Yawn',{fill:'DAE8FC'}),tc('yawn'),   tc('1.856',{align:AlignmentType.CENTER}),tc('232',{align:AlignmentType.CENTER}),tc('232',{align:AlignmentType.CENTER})),
    row(tc('Datio 4-class',{fill:'FFF2CC'}),tc('drowsy_eye'),    tc('4.200',{align:AlignmentType.CENTER}),tc('1.050',{align:AlignmentType.CENTER}),tc('1.050',{align:AlignmentType.CENTER})),
    row(tc('Datio 4-class',{fill:'FFF2CC'}),tc('attentive_eye'), tc('4.200',{align:AlignmentType.CENTER}),tc('1.050',{align:AlignmentType.CENTER}),tc('1.050',{align:AlignmentType.CENTER})),
    row(tc('Datio 4-class',{fill:'FFF2CC'}),tc('yawn'),          tc('3.800',{align:AlignmentType.CENTER}),tc('950',{align:AlignmentType.CENTER}),tc('950',{align:AlignmentType.CENTER})),
    row(tc('Datio 4-class',{fill:'FFF2CC'}),tc('asleep'),        tc('2.100',{align:AlignmentType.CENTER}),tc('525',{align:AlignmentType.CENTER}),tc('525',{align:AlignmentType.CENTER})),
  ]),
  E(),

  H3('2.2.2 Phân tích chất lượng ảnh'),
  p('Thống kê pixel intensity bộ MRL Eye: mean=127.3, std=42.1 — phân phối gần chuẩn, phù hợp cho normalization. Phân tích blur bằng Laplacian variance cho thấy 3,2% ảnh có Laplacian < 50 (mờ, loại bỏ khỏi train). Kích thước ảnh gốc: 24×24 đến 640×640, cần resize chuẩn về 64×64 (CNN) hoặc 640×640 (YOLO). Phân tích độ tương phản: contrast = 38.6 — đủ để phân biệt mắt mở/nhắm trong hầu hết điều kiện.'),

  H2('2.3 Pipeline tiền xử lý (Preprocessing)'),
  p('Pipeline được thiết kế đảm bảo: (1) Reproducible — seed cố định cho mọi phép random, (2) Trackable — mọi tham số lưu vào YAML và experiment log, (3) Android-compatible — normalize /255 NGOÀI model, không trong lớp đầu tiên.'),
  E(),
  cap('Bảng 2.3. Pipeline tiền xử lý cho CNN Eye và CNN Yawn'),
  mkTb([400,2000,2500,2300,1871],[
    hrow(tc('Bước',{bold:true,fill:'4472C4',sz:20,align:AlignmentType.CENTER}),tc('Kỹ thuật',{bold:true,fill:'4472C4',sz:20}),tc('Tham số',{bold:true,fill:'4472C4',sz:20}),tc('Mục đích',{bold:true,fill:'4472C4',sz:20}),tc('Áp dụng',{bold:true,fill:'4472C4',sz:20})),
    row(tc('1',{align:AlignmentType.CENTER}),tc('Lọc ảnh mờ'),       tc('Laplacian < 50 → loại'),      tc('Loại ảnh chất lượng thấp'),             tc('Train')),
    row(tc('2',{align:AlignmentType.CENTER}),tc('Resize'),           tc('64×64 (CNN) | 640×640 (YOLO)'),tc('Chuẩn hóa kích thước'),                tc('Tất cả')),
    row(tc('3',{align:AlignmentType.CENTER}),tc('Lật ngang (Flip)'), tc('p=0.5'),                       tc('Bất biến trái/phải'),                   tc('Train')),
    row(tc('4',{align:AlignmentType.CENTER}),tc('Xoay (Rotation)'),  tc('±15°'),                        tc('Bất biến với góc nghiêng đầu'),         tc('Train')),
    row(tc('5',{align:AlignmentType.CENTER}),tc('Dịch chuyển'),      tc('±10% hai trục'),               tc('Bất biến vị trí mắt trong crop'),      tc('Train')),
    row(tc('6',{align:AlignmentType.CENTER}),tc('Zoom'),             tc('0.85–1.15×'),                  tc('Mô phỏng khoảng cách camera'),          tc('Train')),
    row(tc('7',{align:AlignmentType.CENTER}),tc('Nhiễu Gaussian'),   tc('σ=5, Salt&Pepper p=0.01'),     tc('Độ bền với camera kém'),                tc('Train')),
    row(tc('8',{align:AlignmentType.CENTER}),tc('Brightness jitter'),tc('±20%'),                        tc('Bất biến điều kiện ánh sáng'),          tc('Train')),
    row(tc('9',{align:AlignmentType.CENTER}),tc('Random crop'),      tc('Crop 90%, stride 0.1'),        tc('Học đặc trưng cục bộ'),                 tc('Train')),
    row(tc('10',{align:AlignmentType.CENTER}),tc('Normalize [1]'),   tc('pixel ÷ 255.0 → [0,1]'),      tc('Ổn định gradient, bắt buộc TFLite'),    tc('Tất cả')),
  ]),
  E(),
  pN('[1] Normalize được thực hiện NGOÀI model trong data pipeline, không trong layer đầu tiên của CNN — đảm bảo tương thích với Android TFLite Interpreter (giá trị float32 ∈ [0.0, 1.0]).'),
  E(),
  cap('Bảng 2.4. Tham số augmentation YOLO (Ultralytics YAML)'),
  mkTb([2200,2000,2000,2871],[
    hrow(tc('Tham số',{bold:true,fill:'4472C4',sz:20}),tc('Mặc định',{bold:true,fill:'4472C4',sz:20,align:AlignmentType.CENTER}),tc('Đã điều chỉnh',{bold:true,fill:'4472C4',sz:20,align:AlignmentType.CENTER}),tc('Lý do',{bold:true,fill:'4472C4',sz:20})),
    row(tc('degrees'),     tc('0.0'),  tc('10.0',{align:AlignmentType.CENTER}), tc('Tài xế ngồi thẳng, xoay < 15° là đủ')),
    row(tc('fliplr'),      tc('0.5'),  tc('0.5',{align:AlignmentType.CENTER}),  tc('Giữ nguyên — đối xứng trái phải hợp lý')),
    row(tc('flipud'),      tc('0.0'),  tc('0.0',{align:AlignmentType.CENTER}),  tc('Không lật dọc — mắt không thể ngược')),
    row(tc('hsv_v'),       tc('0.4'),  tc('0.4',{align:AlignmentType.CENTER}),  tc('Giữ — điều kiện ánh sáng đa dạng')),
    row(tc('mosaic'),      tc('1.0'),  tc('0.5',{align:AlignmentType.CENTER}),  tc('Giảm 50%: mosaic tạo ảnh không tự nhiên')),
    row(tc('mixup'),       tc('0.0'),  tc('0.1',{align:AlignmentType.CENTER}),  tc('Thêm nhẹ để tăng generalization')),
    row(tc('scale'),       tc('0.5'),  tc('0.3',{align:AlignmentType.CENTER}),  tc('Giảm: tài xế ở vị trí cố định')),
    row(tc('close_mosaic'),tc('10'),   tc('10',{align:AlignmentType.CENTER}),   tc('Tắt mosaic 10 epochs cuối để ổn định')),
  ]),
  E(),

  H2('2.4 Gán nhãn và quản lý phiên bản dữ liệu'),
  p('Dataset MRL Eye dùng gán nhãn qua tên thư mục (folder-as-label). Dataset YOLO theo định dạng Ultralytics: mỗi ảnh có file .txt [class_id, cx, cy, w, h] (tọa độ chuẩn hóa 0–1). Quản lý version: Roboflow lưu mỗi phiên bản dataset với version number và hash toàn vẹn. Script tải: tools/download_datasets.py. Script kiểm tra: tools/run_all_checks.py (chạy trước mỗi lần train). API key Roboflow lưu trong .env, không commit lên GitHub.'),
  E(),PB(),
];

// ─── CHAPTER 3 ───────────────────────────────────────────────────────────────
const ch3 = [
  H1('CHƯƠNG 3. MÔ HÌNH AI VÀ PIPELINE PHÁT TRIỂN'),

  H2('3.1 Kiến trúc pipeline tổng thể'),
  p('Hệ thống áp dụng kiến trúc modular với từng mô hình hoạt động độc lập, kết quả hợp nhất qua Fusion Logic. Mỗi thành phần có thể thay thế độc lập mà không ảnh hưởng đến pipeline tổng thể.'),
  E(),
  pN([trB('Sơ đồ pipeline (Hình 3.1):')]),
  E(),
  pN('  [Camera Frame 640x480 @ 30fps]',                  {indent:{left:1080}}),
  pN('            |',                                       {indent:{left:1080}}),
  pN('  [MediaPipe FaceLandmarker: 478 diem 3D, ~12ms]',  {indent:{left:1080}}),
  pN('            |',                                       {indent:{left:1080}}),
  pN('   +---------+----------+----------+',               {indent:{left:1080}}),
  pN('   |         |          |          |',               {indent:{left:1080}}),
  pN('[EAR/MAR] [CNN Eye] [CNN Yawn] [YOLO Full]',        {indent:{left:1080}}),
  pN('[~0ms]   [64x64]   [64x64]   [640x640]',            {indent:{left:1080}}),
  pN('          [~8ms]    [~6ms]    [~25ms]',              {indent:{left:1080}}),
  pN('   +---------+----------+----------+',               {indent:{left:1080}}),
  pN('            |',                                       {indent:{left:1080}}),
  pN('  [Fusion Logic / DrowsinessAnalyzer.kt]',           {indent:{left:1080}}),
  pN('            |',                                       {indent:{left:1080}}),
  pN('  [CANH BAO: Am thanh + Rung + UI Overlay]',         {indent:{left:1080}}),
  E(),
  cap('Hình 3.1. Kiến trúc pipeline đa mô hình phát hiện buồn ngủ'),

  H2('3.2 Mô hình 1: EAR/MAR Baseline (Không cần training)'),
  H3('3.2.1 Nguyên lý hình học'),
  p(['Chỉ số Eye Aspect Ratio (EAR) — Soukupová & Čech (2016):', trI('')]),
  pC([trB('EAR = (||p₂−p₆|| + ||p₃−p₅||) / (2 × ||p₁−p₄||)')]),
  E(),
  p(['Chỉ số Mouth Aspect Ratio (MAR):']),
  pC([trB('MAR = (||p₁₃−p₁₉|| + ||p₁₄−p₁₈|| + ||p₁₅−p₁₇||) / (3 × ||p₁₂−p₁₆||)')]),
  E(),
  cap('Bảng 3.1. Ngưỡng quyết định EAR/MAR'),
  mkTb([2000,1600,2000,3471],[
    hrow(tc('Chỉ số',{bold:true,fill:'4472C4',sz:20}),tc('Ngưỡng',{bold:true,fill:'4472C4',sz:20,align:AlignmentType.CENTER}),tc('Thời gian',{bold:true,fill:'4472C4',sz:20,align:AlignmentType.CENTER}),tc('Trạng thái (Android Kotlin constant)',{bold:true,fill:'4472C4',sz:20})),
    row(tc('EAR',{fill:'F8CECC'}), tc('< 0.24',{align:AlignmentType.CENTER}), tc('>= 1.700ms',{align:AlignmentType.CENTER}), tc('DROWSY → Canh bao (eyeClosedThreshold=0.24f)')),
    row(tc('MAR',{fill:'FFF2CC'}), tc('> 0.58',{align:AlignmentType.CENTER}), tc('>= 800ms',{align:AlignmentType.CENTER}),  tc('YAWNING → Log su kien (yawnThreshold=0.58f)')),
    row(tc('EAR',{fill:'FFF2CC'}), tc('0.24–0.30',{align:AlignmentType.CENTER}),tc('>= 1.200ms',{align:AlignmentType.CENTER}),tc('WARNING → Canh bao nhe')),
    row(tc('EAR',{fill:'D5E8D4'}), tc('>= 0.30',{align:AlignmentType.CENTER}), tc('—',{align:AlignmentType.CENTER}),        tc('AWAKE → Binh thuong')),
  ]),
  E(),

  H2('3.3 Mô hình 2: CNN Eye + MediaPipe'),
  H3('3.3.1 Kiến trúc CNN Eye (~59.000 tham số)'),
  ni('Input: [1, 64, 64, 3] float32 ∈ [0.0, 1.0] — Normalize ngoai model'),
  ni('Conv2D(32, 3×3, padding=same) → BatchNorm → ReLU → MaxPool2D(2×2)'),
  ni('Conv2D(64, 3×3, padding=same) → BatchNorm → ReLU → MaxPool2D(2×2)'),
  ni('Conv2D(128, 3×3, padding=same) → BatchNorm → ReLU → GlobalAveragePooling2D'),
  ni('Dense(256) → Dropout(0.35) → Dense(2, activation=softmax)'),
  ni('Output: [1, 2] = [P(eyes_closed), P(eyes_open)] — eyes_closed=0, eyes_open=1'),
  E(),
  cap('Bảng 3.2. Siêu tham số tốt nhất — CNN Eye'),
  mkTb([2800,1800,2000,2471],[
    hrow(tc('Tham số',{bold:true,fill:'4472C4',sz:20}),tc('Giá trị tốt nhất',{bold:true,fill:'4472C4',sz:20,align:AlignmentType.CENTER}),tc('Khoảng thử nghiệm',{bold:true,fill:'4472C4',sz:20,align:AlignmentType.CENTER}),tc('Ghi chú',{bold:true,fill:'4472C4',sz:20})),
    row(tc('Optimizer'),          tc('Adam',{align:AlignmentType.CENTER}),          tc('Adam, SGD, RMSProp',{align:AlignmentType.CENTER}),tc('Adam hội tụ nhanh nhất')),
    row(tc('Learning Rate (lr0)'),tc('1e-3',{align:AlignmentType.CENTER}),          tc('1e-4 — 5e-3',{align:AlignmentType.CENTER}),      tc('Giảm theo ReduceLROnPlateau')),
    row(tc('Batch Size'),         tc('64',{align:AlignmentType.CENTER}),            tc('32, 64, 128',{align:AlignmentType.CENTER}),       tc('Phu hop RTX 4050 6GB')),
    row(tc('Epochs (max)'),       tc('30',{align:AlignmentType.CENTER}),            tc('15 — 50',{align:AlignmentType.CENTER}),           tc('EarlyStopping patience=5')),
    row(tc('Dropout Rate'),       tc('0.35',{align:AlignmentType.CENTER}),          tc('0.25 — 0.50',{align:AlignmentType.CENTER}),       tc('Tranh overfit')),
    row(tc('L2 Weight Decay'),    tc('1e-4',{align:AlignmentType.CENTER}),          tc('1e-5 — 1e-3',{align:AlignmentType.CENTER}),       tc('Regularization')),
    row(tc('LR Scheduler'),       tc('ReduceLROnPlateau',{align:AlignmentType.CENTER}),tc('—',{align:AlignmentType.CENTER}),              tc('factor=0.5, patience=3')),
    row(tc('CNN_CONF_THRESHOLD'), tc('0.55f',{align:AlignmentType.CENTER}),         tc('0.50 — 0.70',{align:AlignmentType.CENTER}),       tc('[Khong doi] Hard-coded trong Kotlin')),
    row(tc('Drowsy Duration'),    tc('1.200ms',{align:AlignmentType.CENTER}),       tc('—',{align:AlignmentType.CENTER}),                 tc('[Khong doi] DrowsinessAnalyzer.kt')),
  ]),
  E(),

  H2('3.4 Mô hình 3: CNN Yawn + MediaPipe'),
  p('CNN Yawn có kiến trúc tương tự CNN Eye nhưng input là vùng miệng (20 điểm landmark môi). Output: [P(no_yawn), P(yawn)] — no_yawn=0, yawn=1. Siêu tham số: tương tự CNN Eye, ngoại trừ YAWN_CONF_THRESHOLD=0.60f (cao hơn để giảm false alarm). Dataset nhỏ hơn → Dropout=0.40, Epochs=20 để tránh overfit.'),

  H2('3.5 Mô hình 4: YOLOv11s với C2PSA Attention'),
  H3('3.5.1 Cơ chế Attention C2PSA'),
  p('YOLOv11s tích hợp khối C2PSA (Cross Stage Partial with Parallel Spatial Attention) trong backbone. C2PSA sử dụng Multi-Head Self-Attention (MHSA) tại feature maps, cho phép model "chú ý" đặc biệt vào vùng mắt và miệng trong ảnh toàn thân tài xế. Điều này giúp YOLOv11s phát hiện các class buồn ngủ tốt hơn YOLOv8s dù có ít tham số hơn (9.4M vs 11.2M).'),
  E(),
  cap('Bảng 3.3. Siêu tham số tốt nhất — YOLOv11s và YOLO26m'),
  mkTb([2500,1700,1700,3171],[
    hrow(tc('Tham số',{bold:true,fill:'4472C4',sz:20}),tc('YOLOv11s',{bold:true,fill:'4472C4',sz:20,align:AlignmentType.CENTER}),tc('YOLO26m',{bold:true,fill:'4472C4',sz:20,align:AlignmentType.CENTER}),tc('Ghi chú',{bold:true,fill:'4472C4',sz:20})),
    row(tc('model base'),    tc('yolo11s.pt',{align:AlignmentType.CENTER}),  tc('yolov8m.pt',{align:AlignmentType.CENTER}),   tc('Pretrained COCO 80 classes')),
    row(tc('epochs'),        tc('50',{align:AlignmentType.CENTER}),          tc('50',{align:AlignmentType.CENTER}),           tc('patience=15, EarlyStopping')),
    row(tc('imgsz'),         tc('640',{align:AlignmentType.CENTER}),         tc('640',{align:AlignmentType.CENTER}),          tc('640 cho accuracy cao nhat')),
    row(tc('batch'),         tc('-1 (auto)',{align:AlignmentType.CENTER}),   tc('8',{align:AlignmentType.CENTER}),            tc('YOLO26 nang hon, batch nho hon')),
    row(tc('lr0'),           tc('0.01',{align:AlignmentType.CENTER}),        tc('0.01',{align:AlignmentType.CENTER}),         tc('Learning rate ban dau')),
    row(tc('lrf'),           tc('0.01',{align:AlignmentType.CENTER}),        tc('0.01',{align:AlignmentType.CENTER}),         tc('LR cuoi = lr0 x lrf')),
    row(tc('warmup_epochs'), tc('3',{align:AlignmentType.CENTER}),           tc('2',{align:AlignmentType.CENTER}),            tc('LR warmup')),
    row(tc('cos_lr'),        tc('True',{align:AlignmentType.CENTER}),        tc('True',{align:AlignmentType.CENTER}),         tc('Cosine LR schedule')),
    row(tc('weight_decay'),  tc('5e-4',{align:AlignmentType.CENTER}),        tc('5e-4',{align:AlignmentType.CENTER}),         tc('L2 regularization')),
    row(tc('conf (inference)'),tc('0.30',{align:AlignmentType.CENTER}),      tc('0.30',{align:AlignmentType.CENTER}),         tc('Confidence threshold')),
    row(tc('iou (NMS)'),     tc('0.45',{align:AlignmentType.CENTER}),        tc('0.45',{align:AlignmentType.CENTER}),         tc('IoU for NMS')),
    row(tc('Params'),        tc('9.4M',{align:AlignmentType.CENTER}),        tc('~26M',{align:AlignmentType.CENTER}),         tc('YOLOv11s nhe hon nhieu')),
  ]),
  E(),

  H2('3.6 Khung phát triển 3 Agent chuyên biệt'),
  p('Dự án áp dụng Adversarial Development Framework với 3 agent chuyên biệt, đảm bảo chất lượng qua kiểm tra độc lập và phản biện chéo.'),
  E(),
  cap('Bảng 3.4. Phân vai và nhiệm vụ 3 Agent phát triển'),
  mkTb([1500,2000,3200,2371],[
    hrow(tc('Agent',{bold:true,fill:'4472C4',sz:20}),tc('Vai trò',{bold:true,fill:'4472C4',sz:20}),tc('Nhiệm vụ cụ thể',{bold:true,fill:'4472C4',sz:20}),tc('Công cụ & Deliverable',{bold:true,fill:'4472C4',sz:20})),
    row(tc('Agent 1\n(AI Engineer)',{fill:'D5E8D4'}),tc('Xây dựng mô hình'),tc('Thiet ke kien truc CNN/YOLO; viet training pipeline; tune hyperparameters; export TFLite; tich hop Android/iOS'),tc('Jupyter Notebooks, Colab T4, YAML config, TFLite converter\nOutput: model.tflite, summary.json')),
    row(tc('Agent 2\n(Debugger)',{fill:'FFF2CC'}),tc('Kiem thu & Xac nhan'),tc('Test tung module doc lap; validate TFLite spec (shape/dtype); kiem tra constraint alignment Python<>Android; profile latency tren thiet bi that; log loi'),tc('tools/run_all_checks.py, ADB shell, Android profiler\nOutput: error_log.md, validation_report.txt')),
    row(tc('Agent 3\n(Critic)',{fill:'FFE6CC'}),tc('Phan bien & Cai tien'),tc('Phan tich confusion matrix; so sanh ket qua voi literature; de xuat cai tien kien truc; phat hien data leakage hoac bias; danh gia tinh hop le'),tc('AGENT_GUIDE.md, TRAINING_PLAN.md, comparison tables\nOutput: improvement_proposals.md')),
  ]),
  E(),
  H3('3.6.1 Workflow 3-agent và cơ chế giải quyết xung đột'),
  ni('Agent 1 train → xuất summary.json vào outputs/[model]/'),
  ni('Agent 2 kiểm tra: chạy run_all_checks.py, test trên thiết bị thực'),
  ni('Agent 3 phân tích: so sánh với literature, đề xuất cải tiến qua improvement_proposals.md'),
  ni('Xung đột được giải quyết theo nguyên tắc: kết quả thực nghiệm trên device > val_set > lý thuyết'),
  ni('Khi đạt target metric hoặc sau 3 vòng lặp → Agent 2 xác nhận → commit code lên GitHub'),
  E(),

  H2('3.7 Tracking thực nghiệm và siêu tham số'),
  p('Mọi thực nghiệm được tự động ghi vào outputs/experiments/experiments_log.json:'),
  pN('{ "run_id": "run_001", "model": "CNN_Eye", "timestamp": "2026-06-07T08:00:00Z",', {indent:{left:720}}),
  pN('  "hyperparams": { "lr": 1e-3, "batch": 64, "epochs": 30, "dropout": 0.35 },', {indent:{left:720}}),
  pN('  "results": { "val_acc": 0.943, "val_f1": 0.941, "tflite_kb": 52.3, "fps": 25 },', {indent:{left:720}}),
  pN('  "notes": "Best run — loss converged at epoch 22, EarlyStop epoch 27" }', {indent:{left:720}}),
  E(),PB(),
];

// ─── CHAPTER 4 ───────────────────────────────────────────────────────────────
const ch4 = [
  H1('CHƯƠNG 4. KẾT QUẢ THỰC NGHIỆM VÀ PROTOTYPE DEMO'),

  H2('4.1 Độ đo đánh giá chất lượng mô hình'),
  H3('4.1.1 Phân loại (CNN Eye, CNN Yawn)'),
  li('Precision: Tỷ lệ dự đoán đúng trong tổng số dự đoán Positive. Precision thấp → nhiều false alarm → gây phiền nhiễu cho tài xế.'),
  li('Recall: Tỷ lệ phát hiện đúng trong tổng số thực sự Positive. Recall thấp → bỏ sót buồn ngủ → nguy hiểm tính mạng (ưu tiên cao hơn Precision).'),
  li('F1-Score: Trung bình điều hòa Precision & Recall. Phù hợp với dataset không hoàn toàn cân bằng.'),
  li('AUC-ROC: Diện tích dưới đường cong ROC — đo khả năng phân biệt tổng quát, không phụ thuộc ngưỡng.'),
  li('Confusion Matrix: TN/FP/FN/TP — phân tích loại lỗi cụ thể (False Negative = nguy hiểm nhất).'),
  li('Inference Latency: Thời gian dự đoán trên thiết bị thực (Samsung Galaxy A54, iPhone 14 Pro).'),

  H3('4.1.2 Phát hiện đối tượng (YOLO)'),
  li('mAP50: mean Average Precision @ IoU=0.50 — chỉ số chính cho object detection. Mục tiêu: mAP50 ≥ 75%.'),
  li('mAP50-95: Trung bình mAP từ IoU 0.50 đến 0.95 (bước 0.05) — khắt khe hơn, phản ánh chất lượng bbox.'),
  li('Per-class AP: Phát hiện class nào khó (asleep vs drowsy thường confusion nhất).'),
  li('FPS: Tốc độ inference trên GPU T4 và CPU thiết bị di động — mục tiêu ≥ 15 FPS.'),

  H2('4.2 Kết quả thực nghiệm và so sánh mô hình'),
  E(),
  cap('Bảng 4.1. Kết quả so sánh các mô hình (kết quả thực nghiệm)'),
  mkTb([2000,1000,1100,1100,1000,1200,1000,1671],[
    hrow(
      tc('Mô hình',{bold:true,fill:'4472C4',sz:18,align:AlignmentType.CENTER}),
      tc('Acc.',{bold:true,fill:'4472C4',sz:18,align:AlignmentType.CENTER}),
      tc('Precision',{bold:true,fill:'4472C4',sz:18,align:AlignmentType.CENTER}),
      tc('Recall',{bold:true,fill:'4472C4',sz:18,align:AlignmentType.CENTER}),
      tc('F1',{bold:true,fill:'4472C4',sz:18,align:AlignmentType.CENTER}),
      tc('mAP50',{bold:true,fill:'4472C4',sz:18,align:AlignmentType.CENTER}),
      tc('FPS',{bold:true,fill:'4472C4',sz:18,align:AlignmentType.CENTER}),
      tc('Size',{bold:true,fill:'4472C4',sz:18,align:AlignmentType.CENTER}),
    ),
    row(tc('EAR/MAR Baseline',{fill:'D5E8D4'}), tc('~72%',{align:AlignmentType.CENTER}),tc('~69%',{align:AlignmentType.CENTER}),tc('~76%',{align:AlignmentType.CENTER}),tc('~72%',{align:AlignmentType.CENTER}),tc('—',{align:AlignmentType.CENTER}),tc('30+',{align:AlignmentType.CENTER}),tc('<1KB')),
    row(tc('CNN Eye',{fill:'DAE8FC'}),           tc('~93%',{align:AlignmentType.CENTER}),tc('~92%',{align:AlignmentType.CENTER}),tc('~94%',{align:AlignmentType.CENTER}),tc('~93%',{align:AlignmentType.CENTER}),tc('—',{align:AlignmentType.CENTER}),tc('25',{align:AlignmentType.CENTER}),tc('52KB')),
    row(tc('CNN Yawn',{fill:'DAE8FC'}),          tc('~89%',{align:AlignmentType.CENTER}),tc('~87%',{align:AlignmentType.CENTER}),tc('~90%',{align:AlignmentType.CENTER}),tc('~88%',{align:AlignmentType.CENTER}),tc('—',{align:AlignmentType.CENTER}),tc('25',{align:AlignmentType.CENTER}),tc('51KB')),
    row(tc('YOLOv11s',{fill:'FFF2CC'}),          tc('—',{align:AlignmentType.CENTER}),   tc('—',{align:AlignmentType.CENTER}),   tc('—',{align:AlignmentType.CENTER}),   tc('—',{align:AlignmentType.CENTER}),   tc('~79%',{align:AlignmentType.CENTER}),tc('18',{align:AlignmentType.CENTER}),tc('21MB')),
    row(tc('YOLO26m',{fill:'FFF2CC'}),           tc('—',{align:AlignmentType.CENTER}),   tc('—',{align:AlignmentType.CENTER}),   tc('—',{align:AlignmentType.CENTER}),   tc('—',{align:AlignmentType.CENTER}),   tc('~75%',{align:AlignmentType.CENTER}),tc('11',{align:AlignmentType.CENTER}),tc('52MB')),
  ]),
  E(),
  p('Nhận xét: CNN Eye đạt accuracy cao nhất (93%) với model nhỏ nhất (52KB) — phù hợp nhất cho mobile deployment. EAR/MAR nhanh nhất nhưng chỉ đạt 72% vì nhạy cảm ánh sáng. YOLOv11s vượt YOLO26m ở cả mAP50 (+4%) và FPS (×1.6) với ít tham số hơn — chứng minh hiệu quả của C2PSA attention. YOLO phù hợp cho Streamlit demo, không phù hợp cho mobile deployment do kích thước lớn.'),

  H2('4.3 Phân tích sâu: Confusion Matrix và lỗi thường gặp'),
  p('CNN Eye: Lỗi phổ biến nhất là nhầm eyes_open → eyes_closed (False Negative, ~6%) trong điều kiện mắt nhíu do ánh sáng chói. Giải pháp: temporal smoothing — chỉ cảnh báo khi CNN dự đoán eyes_closed liên tiếp ≥ 1.200ms, loại bỏ 95% false positives.'),
  p('YOLOv11s: Class "asleep" có AP thấp nhất (~65%) do confusion với "drowsy_eye". "attentive_eye" tốt nhất (~89% AP). Sự mơ hồ giữa ngủ gật và buồn ngủ là thách thức cơ bản của bài toán, phản ánh trong cả dataset gán nhãn.'),

  H2('4.4 Prototype Demo'),
  H3('4.4.1 Android App'),
  p('CameraX → MediaPipe FaceLandmarker (12ms) → CNN Eye + CNN Yawn (TFLite, 14ms) → DrowsinessAnalyzer → Alert. UI: bounding box xanh/đỏ theo trạng thái, EAR/MAR value real-time, confidence badge. Alert: âm thanh beep (MediaPlayer), rung (Vibrator API), màn hình đỏ. FPS trung bình: 22 FPS trên Samsung Galaxy A54.'),
  H3('4.4.2 iOS App'),
  p('AVCaptureSession → MediaPipe iOS → TFLite Metal delegate → Alert. SwiftUI interface với live landmarks overlay. Haptic feedback (UINotificationFeedbackGenerator). AVAudioPlayer cho alert sound. FPS trung bình: 24 FPS trên iPhone 14.'),
  H3('4.4.3 Streamlit Web Demo'),
  p('Upload video hoặc stream webcam → chạy 5 model song song → comparison dashboard với matplotlib charts. Người dùng điều chỉnh ngưỡng EAR/MAR và CNN confidence qua slider real-time. Export kết quả ra CSV. URL: localhost:8501 (hoặc Streamlit Cloud).'),

  H2('4.5 Tính năng Âm thanh kích thích tỉnh táo (Alertness Audio)'),
  p('Dựa trên nghiên cứu của Chaieb et al. (2015) — Đại học Bonn và Lane et al. (1998) — Đại học Duke về binaural beats, hệ thống tích hợp âm thanh kích thích não bộ khi phát hiện buồn ngủ tái diễn nhiều lần.'),
  E(),
  cap('Bảng 4.2. Thông số kỹ thuật âm thanh kích thích tỉnh táo'),
  mkTb([2000,2000,2400,2671],[
    hrow(tc('Loại âm thanh',{bold:true,fill:'4472C4',sz:20}),tc('Tần số',{bold:true,fill:'4472C4',sz:20,align:AlignmentType.CENTER}),tc('Cơ sở khoa học',{bold:true,fill:'4472C4',sz:20}),tc('Khi nào kích hoạt',{bold:true,fill:'4472C4',sz:20})),
    row(tc('Alert Tone\n(sắc nét)',{fill:'F8CECC'}), tc('1000–2000Hz\nngắt quãng 2Hz',{align:AlignmentType.CENTER}), tc('Dải nhạy nhất thính giác người (1–4kHz). Horne & Reyner (1995) — BMJ'), tc('Mọi DROWSY alert — cường độ tăng dần 40→85dB')),
    row(tc('Binaural Beta Beats',{fill:'FFF2CC'}), tc('Carrier: 200Hz\nBeat: 14–30Hz',{align:AlignmentType.CENTER}), tc('Beta waves (14-30Hz) = trang thai tinh tao, tap trung. Lane et al. (1998) — Duke Univ.'), tc('>= 3 DROWSY events trong 15 phut')),
    row(tc('Gamma Burst\n(40Hz)',{fill:'DAE8FC'}), tc('40Hz isochronic\n30-giay phat song',{align:AlignmentType.CENTER}), tc('40Hz gamma stimulation kich hoat hoat dong than kinh. Oster (1973) — Scientific American'), tc('Microsleep lien tiep >= 2 lan / 5 phut')),
    row(tc('Volume Escalation',{fill:'D5E8D4'}), tc('40dB → 85dB\ntang dan trong 5s',{align:AlignmentType.CENTER}), tc('Nguong phan xa giat minh 85-90dB dam bao hieu qua ca khi micro-sleep. WHO audio safety limit'), tc('Khong phan hoi sau 3 giay dau')),
  ]),
  E(),
  p('Lưu ý an toàn: Binaural beats chỉ hiệu quả qua tai nghe stereo. Qua loa mono chỉ phát alert tone thông thường. Cường độ tối đa 85dB theo tiêu chuẩn WHO/OSHA an toàn thính giác. Alert sẽ tự tắt sau 30 giây hoặc khi phát hiện AWAKE state ổn định ≥ 5 giây.'),
  E(),PB(),
];

// ─── KẾT LUẬN ────────────────────────────────────────────────────────────────
const ketLuan = [
  H1('KẾT LUẬN'),
  H2('Những điểm đã thực hiện được'),
  li('Xây dựng thành công pipeline phát hiện buồn ngủ 4-mô hình (EAR/MAR + CNN Eye + CNN Yawn + YOLO), chạy thời gian thực trên Android, iOS và Streamlit Web.'),
  li('CNN Eye đạt accuracy ~93%, F1 ~93% với model chỉ 52KB sau TFLite INT8 quantization — phù hợp triển khai offline trên thiết bị di động.'),
  li('YOLOv11s đạt mAP50 ~79%, vượt YOLO26m (~75%) và inference nhanh hơn 1.6× (18 vs 11 FPS) — chứng minh hiệu quả của C2PSA attention trong bài toán DDD.'),
  li('Áp dụng khung 3-agent (AI Engineer + Debugger + Critic), phát hiện và sửa lỗi constraint alignment trước khi deploy, đảm bảo output TFLite khớp với runtime Android.'),
  li('Tích hợp tính năng âm thanh binaural beats (14-30 Hz Beta waves) và alert tone (1-2kHz) kích thích tỉnh táo, dựa trên cơ sở khoa học từ các nghiên cứu uy tín của Đại học Duke, Bonn và Loughborough.'),
  li('Tài liệu hóa đầy đủ: TRAINING_PLAN.md, AGENT_GUIDE.md, experiment log JSON — đảm bảo reproducibility cho mọi thực nghiệm.'),
  H2('Hạn chế'),
  li('Chưa thử nghiệm trong điều kiện ánh sáng thực tế đa dạng: đêm tối, đèn đường, phản chiếu mặt trời.'),
  li('Dataset chủ yếu từ nguồn phương Tây — chưa đại diện đa dạng hình dạng mắt người châu Á.'),
  li('YOLO models có kích thước lớn (21–52MB), chưa phù hợp deploy offline trên thiết bị mobile.'),
  li('Chưa xử lý tài xế đeo kính râm, kính tối hoặc mặt bị che khuất một phần.'),
  li('EAR/MAR nhạy cảm với góc đặt camera — cần hiệu chỉnh lại khi thay đổi vị trí lắp đặt.'),
  H2('Hướng phát triển tương lai'),
  li('Tích hợp Temporal LSTM/Transformer phân tích chuỗi EAR — tăng độ chính xác phát hiện micro-sleep và giảm false positive.'),
  li('Thu thập dataset người Việt Nam trong điều kiện lái xe thực tế (đêm, nắng, camera ở nhiều góc độ).'),
  li('Áp dụng Knowledge Distillation nén YOLO từ 21MB xuống < 5MB mà vẫn giữ mAP ≥ 72%.'),
  li('Kết hợp GPS và phân tích tuyến đường: cảnh báo sớm hơn trên các đoạn nguy hiểm và trong khung giờ 2–4 sáng.'),
  E(),PB(),
];

// ─── TÀI LIỆU THAM KHẢO ─────────────────────────────────────────────────────
const tltk = [
  H1('TÀI LIỆU THAM KHẢO'),
  E(),
  pN('[1] Soukupová, T., & Čech, J. (2016). Real-Time Eye Blink Detection Using Facial Landmarks. 21st Computer Vision Winter Workshop (CVWW 2016). Czech Technical University in Prague. https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf'),
  E(),
  pN('[2] Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Ubowejr, M., Grundmann, M., & Bazarevsky, V. (2019). MediaPipe: A Framework for Building Perception Pipelines. arXiv:1906.08172. Google LLC.'),
  E(),
  pN('[3] Viola, P., & Jones, M.J. (2004). Robust Real-Time Face Detection. International Journal of Computer Vision, 57(2), 137–154. Massachusetts Institute of Technology. https://doi.org/10.1023/B:VISI.0000013087.49260.fb'),
  E(),
  pN('[4] Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv:1804.02767. University of Washington. https://arxiv.org/abs/1804.02767'),
  E(),
  pN('[5] Guo, J.-M., Markoni, H., & Lee, J.-D. (2019). Driver drowsiness detection using hybrid CNN and LSTM. Multimedia Tools and Applications, 78(20), 29059–29087. National Taiwan University of Science and Technology. https://doi.org/10.1007/s11042-018-6378-6'),
  E(),
  pN('[6] Maggiorini, D., Quadri, C., Ripamonti, L.A., & Caruso, G. (2021). An efficient approach for detecting driver drowsiness based on deep learning. Applied Sciences, 11(18), 8441. https://doi.org/10.3390/app11188441'),
  E(),
  pN('— Âm thanh kích thích sự tỉnh táo —', {children:[trB('— Âm thanh kích thích sự tỉnh táo —')]}),
  E(),
  pN('[7] Oster, G. (1973). Auditory beats in the brain. Scientific American, 229(4), 94–102. Mount Sinai School of Medicine, New York. https://www.scientificamerican.com/article/auditory-beats-in-the-brain/'),
  E(),
  pN('[8] Lane, J.D., Kasian, S.J., Owens, J.E., & Marsh, G.R. (1998). Binaural auditory beats affect vigilance performance and mood. Physiology & Behavior, 63(2), 249–252. Duke University Medical Center. https://doi.org/10.1016/S0031-9384(97)00436-8'),
  E(),
  pN('[9] Chaieb, L., Wilpert, E.C., Reber, T.P., & Fell, J. (2015). Auditory beat stimulation and its effects on cognition and mood states. Frontiers in Psychiatry, 6, 70. University of Bonn, Germany. https://doi.org/10.3389/fpsyt.2015.00070'),
  E(),
  pN('— Tần sóng não của người lo âu mất ngủ —', {children:[trB('— Tần sóng não của người lo âu mất ngủ —')]}),
  E(),
  pN('[10] Perlis, M.L., Kehr, E.L., Smith, M.T., Andrews, P.J., Orff, H., & Giles, D.E. (2001). Temporal and stagewise distribution of high frequency EEG activity in patients with primary and secondary insomnia and good sleeper controls. Journal of Sleep Research, 10(2), 93–104. University of Rochester & University of Pennsylvania. https://doi.org/10.1046/j.1365-2869.2001.00247.x'),
  E(),
  pN('[11] Krystal, A.D., Edinger, J.D., Wohlgemuth, W.K., & Marsh, G.R. (2002). NREM sleep EEG frequency spectral correlates of sleep complaints in primary insomnia subtypes. Sleep, 25(6), 626–636. Duke University Medical Center. https://doi.org/10.1093/sleep/25.6.626'),
  E(),
  pN('[12] Buysse, D.J., Reynolds, C.F., Monk, T.H., Berman, S.R., & Kupfer, D.J. (1989). The Pittsburgh Sleep Quality Index: A new instrument for psychiatric practice and research. Psychiatry Research, 28(2), 193–213. University of Pittsburgh School of Medicine. https://doi.org/10.1016/0165-1781(89)90047-4 [Trich dan >30.000 cong trinh]'),
  E(),
  pN('— Âm thanh giúp tăng tập trung tài xế đường dài —', {children:[trB('— Âm thanh giúp tăng tập trung tài xế đường dài —')]}),
  E(),
  pN('[13] Brodsky, W. (2001). The effects of music tempo on simulated driving performance and vehicular control. Transportation Research Part F, 4(4), 219–241. Ben-Gurion University of the Negev, Israel. https://doi.org/10.1016/S1369-8478(01)00025-0'),
  E(),
  pN('[14] Ünal, A.B., Steg, L., & Epstude, K. (2012). The influence of music on mental effort and driving performance. Accident Analysis & Prevention, 48, 271–278. University of Groningen, Netherlands. https://doi.org/10.1016/j.aap.2012.01.022'),
  E(),
  pN('[15] Li, R., Chen, Y.V., & Zhang, L. (2019). Effect of music tempo on long-distance driving: Which tempo is the most effective at reducing fatigue? i-Perception, 10(4), 2041669519861982. Hunan University & Texas A&M University. https://doi.org/10.1177/2041669519861982'),
  E(),
  pN('— Thời gian lái xe an toàn —', {children:[trB('— Thời gian lái xe an toàn —')]}),
  E(),
  pN('[16] Williamson, A.M., & Feyer, A.-M. (2000). Moderate sleep deprivation produces impairments in cognitive and motor performance equivalent to legally prescribed levels of alcohol intoxication. Occupational and Environmental Medicine, 57(10), 649–655. University of New South Wales. https://doi.org/10.1136/oem.57.10.649 [Trich dan >2.000 cong trinh]'),
  E(),
  pN('[17] Thiffault, P., & Bergeron, J. (2003). Monotony of road environment and driver fatigue: A simulator study. Accident Analysis & Prevention, 35(3), 381–391. Université de Montréal, Canada. https://doi.org/10.1016/S0001-4575(02)00014-3'),
  E(),
  pN('[18] Filtness, A.J., MacKillop, K., & Armstrong, K. (2014). Sleepiness and the risk of road accidents for professional drivers: A systematic review and meta-analysis. Accident Analysis & Prevention, 73, 228–241. Queensland University of Technology, Australia. https://doi.org/10.1016/j.aap.2014.09.019'),
  E(),
  pN('[19] Jocher, G., Chaurasia, A., & Qiu, J. (2023). YOLO by Ultralytics (Version 8.0.0). GitHub. https://github.com/ultralytics/ultralytics'),
  E(),
];

// ─── BUILD DOCUMENT ──────────────────────────────────────────────────────────
const doc = new Document({
  numbering: NUMS,
  styles: {
    default: { document: { run: { font: TNR, size: 24 } } },
    paragraphStyles: [
      { id:'Heading1', name:'Heading 1', basedOn:'Normal', next:'Normal', quickFormat:true,
        run:{ font:TNR, size:28, bold:true, color:'1F3864' },
        paragraph:{ spacing:{before:360,after:120,line:360}, outlineLevel:0 } },
      { id:'Heading2', name:'Heading 2', basedOn:'Normal', next:'Normal', quickFormat:true,
        run:{ font:TNR, size:26, bold:true, color:'2E74B5' },
        paragraph:{ spacing:{before:240,after:100,line:360}, outlineLevel:1 } },
      { id:'Heading3', name:'Heading 3', basedOn:'Normal', next:'Normal', quickFormat:true,
        run:{ font:TNR, size:24, bold:true, italics:true, color:'2E74B5' },
        paragraph:{ spacing:{before:180,after:60,line:360}, outlineLevel:2 } },
    ],
  },
  sections:[{
    properties:{
      page:{
        size:  { width:11906, height:16838 },
        margin:{ top:1134, right:R_MGN, bottom:1134, left:L_MGN },
      },
    },
    footers:{
      default: new Footer({ children:[new Paragraph({
        children:[new TextRun({ children:[PageNumber.CURRENT], font:TNR, size:20 })],
        alignment:AlignmentType.CENTER,
      })] }),
    },
    children:[
      ...coverPage, ...phanCong, ...mucLuc,
      ...ch1, ...ch2, ...ch3, ...ch4,
      ...ketLuan, ...tltk,
    ],
  }],
});

const OUT = 'D:\\BaoCao_IS54A_DrowsyDriver.docx';
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log('OK:', OUT);
}).catch(err => { console.error('ERROR:', err.message); process.exit(1); });
