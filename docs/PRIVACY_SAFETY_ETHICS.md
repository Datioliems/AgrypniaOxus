# Privacy, Safety, And Ethics

## Why This Matters

The project uses a camera pointed at a driver's face. That makes privacy and
safety considerations part of the technical design, not an optional appendix.

## Privacy Risks

- Face video is sensitive personal data.
- Continuous recording may reveal behavior, location context, and identity.
- If used by an organization, drivers may feel monitored or pressured.
- A false alert can be annoying; a missed alert can be dangerous.

## Mitigations In This Project

| Risk | Mitigation |
|---|---|
| Raw face video exposure | Process frames on-device by default |
| Excessive storage | Store event logs instead of video |
| Lack of transparency | Show visible camera status and alert state |
| False sense of safety | Clearly present the app as a prototype aid |
| Dataset bias | Report dataset limitations and test conditions |

## Recommended Event Log

The Android prototype includes a minimal local CSV logger for `DROWSY` and
`YAWNING` states. It keeps event data only, not video. If logging is extended,
keep it minimal:

```text
timestamp
state
confidence
EAR
MAR
FPS
device_id or anonymous session_id
```

Avoid by default:

- full video;
- face screenshots;
- GPS;
- audio recording;
- identity information.

## Report Wording

Use this paragraph in the limitations or future-work section:

> Vi he thong su dung camera huong vao khuon mat tai xe, nhom uu tien xu ly
> truc tiep tren thiet bi va khong luu video khuon mat mac dinh. Neu trien khai
> trong to chuc, he thong can co thong bao ro rang, su dong y cua nguoi dung,
> chinh sach luu tru du lieu va chi nen luu log su kien toi thieu thay vi video
> lien tuc.

## Safety Wording

Use this paragraph when presenting:

> He thong chi la cong cu ho tro canh bao som, khong thay the trach nhiem nghi
> ngoi cua tai xe va khong phai he thong an toan da duoc chung nhan cho xe
> thuong mai. Truoc khi dung thuc te can kiem thu co kiem soat trong nhieu dieu
> kien anh sang, goc camera va nhom nguoi dung khac nhau.

## Ethical Position

The project should be framed as:

- assistive, not punitive;
- privacy-preserving by default;
- transparent about limitations;
- evaluated with safety-oriented metrics;
- designed to reduce harm, not to surveil drivers continuously.
