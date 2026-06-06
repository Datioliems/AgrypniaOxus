# CodeRabbit Review Report

Generated: 2026-06-07

## Ket luan

Chua the chay CodeRabbit review tren toan bo folder code vi moi truong hien tai bi chan truoc buoc cai/chay CLI.

Day khong phai la danh sach loi do CodeRabbit phat hien trong source code. CodeRabbit review chua duoc thuc thi thanh cong.

## Pham vi du kien

Repo:

```text
D:/2026.AI/DrowsyDriverAndroid
```

Yeu cau:

```text
Kiem tra lai toan bo folder code va lap danh sach loi bang CodeRabbit.
```

## Trang thai kiem tra dieu kien

| Hang muc | Ket qua |
|---|---|
| Git repository | OK |
| CodeRabbit CLI tren PowerShell | FAIL |
| Lenh cai CodeRabbit bang install.sh | FAIL |
| WSL kha dung | PARTIAL |
| Distro WSL dev nhu Ubuntu/Debian | FAIL |
| CodeRabbit review | NOT RUN |

## Loi chan

### CR-BLOCK-01 - Chua cai CodeRabbit CLI tren Windows PowerShell

Lenh da chay:

```powershell
coderabbit --version
```

Ket qua:

```text
coderabbit : The term 'coderabbit' is not recognized as the name of a cmdlet,
function, script file, or operable program.
```

Tac dong:

```text
Khong the chay coderabbit review --agent.
```

### CR-BLOCK-02 - Lenh cai chuan can `sh`, nhung PowerShell khong co `sh`

Lenh da chay:

```powershell
curl.exe -fsSL https://cli.coderabbit.ai/install.sh | sh
```

Ket qua:

```text
sh : The term 'sh' is not recognized as the name of a cmdlet, function,
script file, or operable program.
```

Tac dong:

```text
Khong the cai CodeRabbit CLI bang install script trong PowerShell hien tai.
```

### CR-BLOCK-03 - WSL hien tai chi co `docker-desktop`, khong co Ubuntu/Debian dev distro

Lenh da chay:

```powershell
wsl -l -v
```

Ket qua:

```text
NAME              STATE    VERSION
docker-desktop    Running  2
```

Tac dong:

```text
Khong co WSL distro phu hop de cai curl/unzip/git va CodeRabbit CLI.
```

## Huong khac phuc

1. Cai Ubuntu cho WSL:

```powershell
wsl --install -d Ubuntu
```

2. Mo Ubuntu WSL va cai prerequisites:

```bash
sudo apt update
sudo apt install -y curl unzip git
```

3. Di chuyen vao repo tu WSL:

```bash
cd /mnt/d/2026.AI/DrowsyDriverAndroid
```

4. Cai CodeRabbit CLI:

```bash
curl -fsSL https://cli.coderabbit.ai/install.sh | sh
source ~/.bashrc
coderabbit --version
```

5. Dang nhap agent mode:

```bash
coderabbit auth login --agent
coderabbit auth status --agent
```

6. Chay review:

```bash
coderabbit review --agent
```

## Ghi chu

- Theo tai lieu CodeRabbit, CodeRabbit CLI tren Windows nen chay qua WSL.
- Khi CodeRabbit CLI kha dung va auth thanh cong, chay lai review de tao danh sach loi that su tu CodeRabbit.
- Do CodeRabbit review chua chay, file nay chi ghi loi moi truong/CLI, khong tu y gan cac loi code la ket qua CodeRabbit.

