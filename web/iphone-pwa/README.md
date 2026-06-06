# Drowsy Guard iPhone PWA

Day la ban PWA hoan chinh hon cho iPhone Safari. No khong phai app iOS native,
nhung co the dung de demo tren iPhone qua HTTPS/ngrok.

## Tinh nang

- Camera realtime tren iPhone Safari khi chay bang HTTPS.
- Snapshot fallback neu Safari chan realtime camera.
- MediaPipe Face Landmarker JS de tinh EAR/MAR realtime.
- CNN Eye da export sang JSON web de phan loai eyes_closed / eyes_open tren crop mat.
- Canh bao Awake / Eyes closing / Yawning / Drowsy alert.
- Che do Sound On/Off cho iPhone Safari.
- Am thanh canh bao va nen kich thich 1750 Hz co gioi han.
- Event log luu localStorage.
- Mini dashboard theo khung gio.
- Lay vi tri va chia se canh bao cho lien he tin cay.
- Export CSV log.
- Add to Home Screen tren iPhone.

## Chay local

Terminal 1:

```bat
cd D:\2026.AI\DrowsyDriverAndroid
python tools\run_iphone_pwa_server.py --host 0.0.0.0 --port 8502
```

Mo tren laptop:

```text
http://127.0.0.1:8502/web/iphone-pwa/
```

## Chay tren iPhone bang ngrok HTTPS

Terminal 2:

```bat
ngrok http 8502
```

Ngrok se hien link:

```text
https://something.ngrok-free.dev
```

Tren iPhone mo Safari:

```text
https://something.ngrok-free.dev/web/iphone-pwa/index.html?v=7
```

Sau do:

1. Bam `Bat am thanh` mot lan de iPhone cho phep app phat canh bao.
2. Bam `Beep` de test loa. Neu khong nghe, tang volume va tat Silent Mode.
3. Bam `Mo camera`.
4. Bam `Tai MediaPipe`.
5. Bam `Tai CNN` neu dong CNN Eye chua hien `ready`.
6. Neu camera bi chan, dung `Chup anh fallback`.
7. Bam Share -> Add to Home Screen neu muon cai nhu app PWA.

## Luu y am thanh tren iPhone

iPhone Safari chan autoplay audio, nen app khong the tu phat am neu nguoi dung
chua bam nut bat am thanh. Sau khi bam `Bat am thanh`, cac beep canh bao va
nen 1750 Hz moi duoc phep chay trong phien do.

## Vi sao phai HTTPS

iPhone Safari yeu cau secure context cho `getUserMedia`. Link HTTP LAN nhu
`http://10.10.x.x:8502/...` co the xem giao dien nhung khong dam bao mo realtime
camera. Ngrok cung cap HTTPS nen phu hop de test nhanh.

## Gioi han

- PWA dung MediaPipe EAR/MAR baseline, chua thay the Android native realtime.
- CNN/TFLite hien van duoc uu tien trong Android native.
- Neu can san pham iPhone that su, can viet iOS native bang Swift/AVFoundation.
- Canh bao am thanh chi la ho tro ngan han; neu drowsy lap lai, phai dung xe va nghi.
