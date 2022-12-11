## Docker で GPU を使う

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

### 1. グラフィックボードのドライバをインストール

ホスト（WSL2 Ubuntu）にLinuxドライバをインストール
https://www.nvidia.com/en-us/drivers/unix/

```bash
$ sudo sh NVIDIA-Linux-x86_64-525.60.11.run
```

### 2. Docker のインストール

Docker Desktop でなくて Ubuntu の Docker

https://docs.docker.com/engine/install/ubuntu/

```bash
$ curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
```

#### Windows 11 WSL2 では systemd が使用できる

```
$ vi /etc/wsl.conf
[boot]
systemd = true
```

### 3. NVIDIA-docker2のインストール

```bash
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
$ sudo apt-get update
$ sudo apt-get install -y nvidia-docker2
$ sudo systemctl restart docker
```

