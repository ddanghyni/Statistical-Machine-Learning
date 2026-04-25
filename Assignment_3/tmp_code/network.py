"""
============================================================================
 network.py — Autoencoder 아키텍처 (논문 Table 1 그대로)
============================================================================
 논문: Bergmann et al. (2019),
       "Improving Unsupervised Defect Segmentation by Applying
        Structural Similarity to Autoencoders"

 [아키텍처 요약]
 ---------------------------------------------------------------------------
 Encoder:  128×128×C  →  1×1×d   (strided conv 로 점진적 다운샘플)
 Decoder:  1×1×d      →  128×128×C (transposed conv 로 점진적 업샘플)

 "Decoder 는 Encoder 의 reversed version" (논문 원문).
 활성화: LeakyReLU(slope=0.2).  마지막 층만 linear (활성화 없음).

 [논문 Table 1 — Encoder 구조]
 ┌───────┬──────────┬────────┬────────┬─────────┐
 │ Layer │ Out Size │ Kernel │ Stride │ Padding │
 ├───────┼──────────┼────────┼────────┼─────────┤
 │ Input │128×128×C │   -    │   -    │    -    │
 │ Conv1 │ 64×64×32 │  4×4   │   2    │    1    │  ← 해상도 절반
 │ Conv2 │ 32×32×32 │  4×4   │   2    │    1    │  ← 해상도 절반
 │ Conv3 │ 32×32×32 │  3×3   │   1    │    1    │  ← 해상도 유지 (정제)
 │ Conv4 │ 16×16×64 │  4×4   │   2    │    1    │
 │ Conv5 │ 16×16×64 │  3×3   │   1    │    1    │
 │ Conv6 │  8×8×128 │  4×4   │   2    │    1    │
 │ Conv7 │  8×8×64  │  3×3   │   1    │    1    │
 │ Conv8 │  8×8×32  │  3×3   │   1    │    1    │
 │ Conv9 │  1×1×d   │  8×8   │   1    │    0    │  ← bottleneck!
 └───────┴──────────┴────────┴────────┴─────────┘

 [왜 strided conv 인가?]
   MaxPooling 은 "어떤 값을 살릴지" 규칙이 고정 (max) 이지만,
   stride=2 conv 는 "어떤 정보를 보존할지" 를 학습 파라미터로 결정.
   → 복원 품질이 더 좋아짐.  GAN 문헌에서도 strided conv 가 대세.

 [왜 Conv9 의 kernel 이 8×8 인가?]
   Conv8 출력이 8×8×32.  여기에 kernel=8, stride=1, padding=0 을 적용하면
   출력 크기 = (8 - 8 + 0) / 1 + 1 = 1.  즉 8×8 공간 전체를 한 번에
   하나의 벡터로 "요약" 하는 것.  사실상 Fully-Connected 와 같지만,
   conv 형태를 유지해서 구현이 깔끔함.
============================================================================
"""

import torch
import torch.nn as nn


# =============================================================================
# 1) Encoder — 이미지를 latent vector z 로 압축
# =============================================================================
class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, z_dim: int = 100):
        """
        Parameters
        ----------
        in_channels : 입력 채널 수.
                      MVTec RGB → 3, grayscale (grid/screw/zipper) → 1.
        z_dim       : bottleneck 차원 d.
                      texture → 100, object → 500 (논문 4.2).
        """
        super().__init__()

        # LeakyReLU(0.2) — slope=0.2 는 논문 지정값.
        # 일반 ReLU 는 음수 입력이 오면 기울기가 0 이 되어 뉴런이 "죽음".
        # Leaky 는 음수 영역에도 작은 기울기(0.2) 를 유지해서 이 문제를 완화.
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # ---- Downsampling block 1 : 128 → 64 ----
        # kernel=4, stride=2, padding=1 → 출력 = (128 + 2*1 - 4) / 2 + 1 = 64
        self.conv1 = nn.Conv2d(in_channels, 32, 4, 2, 1)

        # ---- Downsampling block 2 : 64 → 32 ----
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)

        # ---- Refinement : 32 → 32 (해상도 유지) ----
        # stride=1 이라 해상도 안 변함.  채널도 32→32.
        # "같은 해상도에서 특징을 한 번 더 정제" 하는 역할.
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)

        # ---- Downsampling block 3 : 32 → 16 + 정제 ----
        self.conv4 = nn.Conv2d(32, 64, 4, 2, 1)   # 32→16, 채널 32→64
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)   # 정제

        # ---- Downsampling block 4 : 16 → 8 ----
        self.conv6 = nn.Conv2d(64, 128, 4, 2, 1)  # 16→8, 채널 64→128

        # ---- 채널 축소 : 128 → 64 → 32 ----
        # 공간 해상도는 8×8 고정.  채널만 줄여서 정보를 압축.
        self.conv7 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv8 = nn.Conv2d(64, 32, 3, 1, 1)

        # ---- Bottleneck : 8×8×32 → 1×1×d ----
        # kernel=8 이 8×8 공간 전체를 한 번에 봄.
        # 출력: (B, z_dim, 1, 1)  ← 이게 latent vector z.
        self.conv9 = nn.Conv2d(32, z_dim, 8, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, 128, 128)  →  z: (B, z_dim, 1, 1)"""
        x = self.act(self.conv1(x))   # (B, 32, 64, 64)
        x = self.act(self.conv2(x))   # (B, 32, 32, 32)
        x = self.act(self.conv3(x))   # (B, 32, 32, 32)
        x = self.act(self.conv4(x))   # (B, 64, 16, 16)
        x = self.act(self.conv5(x))   # (B, 64, 16, 16)
        x = self.act(self.conv6(x))   # (B, 128, 8, 8)
        x = self.act(self.conv7(x))   # (B, 64, 8, 8)
        x = self.act(self.conv8(x))   # (B, 32, 8, 8)
        z = self.conv9(x)             # (B, z_dim, 1, 1)  ← linear!
        return z


# =============================================================================
# 2) Decoder — latent vector z 를 원본 해상도 이미지로 복원
# =============================================================================
# Encoder 의 "거울상(reversed version)".
# Conv2d(stride=2)  →  ConvTranspose2d(stride=2)  로 뒤집으면 업샘플 됨.
#
# [ConvTranspose2d 란?]
#   Conv2d 의 역연산 격.  stride=2 로 설정하면 입력보다 2배 큰 출력을 만듦.
#   "deconvolution" 이라고도 부르지만, 수학적으로 정확한 이름은
#   "transposed convolution" 또는 "fractionally-strided convolution".
# =============================================================================
class Decoder(nn.Module):
    def __init__(self, out_channels: int = 3, z_dim: int = 100):
        """out_channels 은 Encoder 의 in_channels 와 같아야 원본과 동일 shape."""
        super().__init__()
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # ---- 1×1×d → 8×8×32  (Bottleneck 펼치기) ----
        self.deconv9 = nn.ConvTranspose2d(z_dim, 32, 8, 1, 0)

        # ---- 채널 확장 : 32 → 64 → 128 ----
        self.deconv8 = nn.ConvTranspose2d(32, 64, 3, 1, 1)
        self.deconv7 = nn.ConvTranspose2d(64, 128, 3, 1, 1)

        # ---- Upsampling block 1 : 8 → 16 + 정제 ----
        self.deconv6 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(64, 64, 3, 1, 1)

        # ---- Upsampling block 2 : 16 → 32 + 정제 ----
        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, 3, 1, 1)

        # ---- Upsampling block 3 : 32 → 64 ----
        self.deconv2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)

        # ---- 마지막 업샘플 64 → 128 + 출력 채널 복원 ----
        # 활성화 없이 linear 출력.  [0,1] 정규화 입력이면 출력도 그 근방.
        self.deconv1 = nn.ConvTranspose2d(32, out_channels, 4, 2, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, z_dim, 1, 1)  →  x_hat: (B, C, 128, 128)"""
        x = self.act(self.deconv9(z))    # (B, 32, 8, 8)
        x = self.act(self.deconv8(x))    # (B, 64, 8, 8)
        x = self.act(self.deconv7(x))    # (B, 128, 8, 8)
        x = self.act(self.deconv6(x))    # (B, 64, 16, 16)
        x = self.act(self.deconv5(x))    # (B, 64, 16, 16)
        x = self.act(self.deconv4(x))    # (B, 32, 32, 32)
        x = self.act(self.deconv3(x))    # (B, 32, 32, 32)
        x = self.act(self.deconv2(x))    # (B, 32, 64, 64)
        x_hat = self.deconv1(x)          # (B, C, 128, 128)  ← linear!
        return x_hat


# =============================================================================
# 3) 전체 AutoEncoder = Encoder + Decoder
# =============================================================================
# 논문 식 (1):  x̂ = D(E(x)) = D(z)
#   z 가 bottleneck.  d << C·H·W 이므로 단순 복사 불가.
#   → 중요한 특징만 z 에 압축하도록 강제됨.
#   → 학습 때 못 본 패턴 (=결함) 은 z 에 담기지 않아 복원 실패
#   → 원본과 복원의 차이 (residual) 로 결함 탐지.
# =============================================================================
class AutoEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, z_dim: int = 100):
        super().__init__()
        self.encoder = Encoder(in_channels, z_dim)
        self.decoder = Decoder(in_channels, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)        # 압축
        x_hat = self.decoder(z)    # 복원
        return x_hat


# =============================================================================
# 직접 실행 시 shape 검증.
#   python network.py
# =============================================================================
if __name__ == "__main__":
    for ch, d in [(3, 100), (1, 100), (3, 500)]:
        net = AutoEncoder(ch, d)
        x = torch.randn(2, ch, 128, 128)
        y = net(x)
        n = sum(p.numel() for p in net.parameters())
        print(f"ch={ch} z_dim={d}  in={tuple(x.shape)} out={tuple(y.shape)}  params={n:,}")
