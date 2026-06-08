# 🎨 UI Development Plan — AgrypniaOxus

> Ánh xạ thiết kế Stitch (8 màn hình) ↔ code hiện tại ↔ việc cần làm.
> Nguồn: `design_stitch/` (HTML + ảnh mockup) · báo cáo: `CK_AI_baocao.docx`

## Hệ thiết kế (design system)
- **Phong cách:** ThoughtLab — dark cinematic, "crimson flare in obsidian"
- **Màu chính:** Signal Red `#fc1c46` trên nền đen `#000`, chữ trắng/xám `#ccc`
- **Font:** sui (thay bằng **Inter**), headline cực lớn (198px), body 15-17px
- **App hiện tại:** Jetpack **Compose** (`MainActivity.kt`) + overlay tùy biến (`StatusOverlayView.kt`)

## Ánh xạ 8 màn hình → trạng thái

| # | Màn hình Stitch | Chức năng | Code hiện có | Trạng thái |
|---|-----------------|-----------|--------------|-----------|
| 1 | **Splash / Chào mừng** | logo + vào app | — | ⬜ TODO (mới) |
| 2 | **Detection / Phát hiện** | camera + overlay EAR/MAR + trạng thái | `MainActivity` + `StatusOverlayView` | ✅ **CHẠY ĐƯỢC** (core) |
| 3 | **Dashboard phân tích** | tổng quan phiên lái | `DriverSessionTracker` | 🟡 có data, UI cần làm |
| 4 | **Analytics / Thống kê** | lịch sử đoạn đường buồn ngủ | `EventLogger` | 🟡 có data, UI cần làm |
| 5 | **Alert khẩn cấp** | đỏ + âm thanh + "Tôi ổn" | `AlertController` | 🟡 logic có, UI cần làm |
| 6 | **Settings / Cài đặt** | độ nhạy AI | hằng số `DrowsinessConstants` | ⬜ TODO |
| 7 | **Thư viện âm thanh** | chọn âm cảnh báo | `AlertController` | ⬜ TODO |
| 8 | **Người thân / Emergency contact** | gọi/nhắn người thân | — | ⬜ TODO (mới) |

## Đánh giá thực tế (cho deadline 3h)
- ❌ **KHÔNG kịp** code 8 màn hình Compose từ thiết kế trong 3h (cần vài ngày).
- ✅ **NÊN làm ngay (cho báo cáo):**
  1. App **chạy được** (màn Detection) → demo trên điện thoại.
  2. Dùng **ảnh mockup `screen.png`** làm hình "Thiết kế giao diện" trong báo cáo — trông rất chuyên nghiệp.
  3. (Tùy chọn, ~15 phút) đổi **màu theme** Compose sang đỏ-đen Stitch → app giống thiết kế ngay.
- 📅 **Sau deadline:** code dần 8 màn hình theo `design_stitch/*/code.html` (đã có sẵn HTML+Tailwind để tham chiếu).

## Lộ trình code UI (sau deadline)
1. Tạo `ui/theme/` (Color.kt, Type.kt) theo design system Stitch.
2. Mỗi màn hình Stitch → 1 Composable (tham chiếu `code.html` tương ứng).
3. Điều hướng: NavHost (Splash → Detection → Dashboard/Analytics/Settings).
4. Giữ logic AI hiện có (`DrowsinessAnalyzer`, classifiers) — chỉ thay lớp UI.

## File tham chiếu
- `design_stitch/.../code.html` — HTML+Tailwind mỗi màn (convert sang Compose)
- `design_stitch/.../screen.png` — ảnh mockup (dùng cho báo cáo)
- `design_stitch/.../design_9.md` + `agrypniaoxus/DESIGN.md` — tokens màu/font/spacing
