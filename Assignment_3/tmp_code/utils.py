"""
============================================================================
 utils.py — SSIM 계산, loss 함수, Dataset, residual map, 기타 유틸
============================================================================
 이 파일에 모든 "도구" 를 모아둔다.
 network.py (모델) 와 train.py/test.py (실행) 양쪽에서 import 됨.

 포함 내용:
   1) SSIM 계산 함수  — 논문 식 (5)~(8) 구현
   2) Loss 함수들     — MSE, SSIM, SSIM+L1
   3) Residual map    — 테스트 시 결함 위치 탐지
   4) MVTec Dataset   — Train / Test
   5) 기타 (seed 고정 등)
============================================================================
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image


# ========================================================================
# 1) SSIM 계산 — 논문의 핵심.  식 (5)~(8) 을 conv2d 로 효율적 구현.
# ========================================================================
# [기본 개념]
#   SSIM(p, q) = l(p,q) · c(p,q) · s(p,q)
#   l = luminance  (평균 비교 — 1차 moment)
#   c = contrast   (분산 비교 — 2차 moment)
#   s = structure  (공분산 기반 — 2차 moment, 정규화)
#
# [구현 핵심 트릭]
#   각 픽셀 위치에서 K×K 패치의 (μ, σ², σ_xy) 를 계산해야 하는데,
#   이걸 "가우시안 window 와의 conv2d" 로 벡터화하면 한 번에 전체 이미지에 대해
#   모든 패치 통계량이 나옴.
#     μ_x = conv2d(x, w)              ← 가중 평균
#     σ_x² = conv2d(x², w) - μ_x²    ← E[X²] - (E[X])²
#     σ_xy = conv2d(xy, w) - μ_x·μ_y ← E[XY] - E[X]·E[Y]
#
# [RGB 처리]
#   groups=C (depth-wise conv) 덕에 채널별 독립 SSIM 이 자동 계산됨.
#   최종 .mean() 은 채널 축까지 포함한 평균 → 스칼라.
#
# [경계(boundary) 처리]
#   padding=K//2 의 zero-padding.  이미지 가장자리 K//2 픽셀은 패치의 일부가
#   0 으로 채워져서 μ, σ 가 살짝 편향됨.  하지만 실전에서는:
#     1) 결함이 보통 이미지 중앙 부근에 있음
#     2) 테스트 때 stride 30 으로 슬라이딩 평균하면 편향이 상쇄됨
#   그래서 큰 문제 없음.
# ========================================================================

def _gaussian_1d(size: int, sigma: float) -> torch.Tensor:
    """1D 가우시안 커널 (합=1).  2D 는 이것의 outer product 로 생성."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _create_window(size: int, channel: int, sigma: float = 1.5) -> torch.Tensor:
    """
    depth-wise conv 용 2D 가우시안 커널.
    shape: (channel, 1, K, K).  channel=3 이면 R/G/B 각각에 같은 커널.
    """
    _1d = _gaussian_1d(size, sigma).unsqueeze(1)   # (K, 1)
    _2d = _1d @ _1d.t()                            # (K, K) — separable
    return _2d.expand(channel, 1, size, size).contiguous()


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
    return_map: bool = False,
):
    """
    SSIM 계산.

    Parameters
    ----------
    x, y        : (B, C, H, W),  값 범위 [0, data_range].
    window_size : K.  논문 기본 11.
    data_range  : 픽셀 최댓값.  [0,1] 정규화면 1.0.
    return_map  : True 면 (스칼라, 픽셀별 맵) 튜플 반환.

    Returns
    -------
    return_map=False → 스칼라 SSIM ∈ [-1, 1]
    return_map=True  → (스칼라, (B, C, H, W) SSIM map)
    """
    C = x.shape[1]

    # 안정화 상수 (분모 0 방지).  원 SSIM 논문 (Wang et al. 2004) 공식.
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    pad = window_size // 2
    win = _create_window(window_size, C).to(x.device).type_as(x)

    # ── Step 1: 가중 평균 μ  (conv = 가중 평균이라는 사실을 이용) ──
    mu_x = F.conv2d(x, win, padding=pad, groups=C)
    mu_y = F.conv2d(y, win, padding=pad, groups=C)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy   = mu_x * mu_y

    # ── Step 2: 분산 σ² 과 공분산 σ_xy  (E[X²]-E[X]² 트릭) ──
    sigma_x_sq = F.conv2d(x * x, win, padding=pad, groups=C) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, win, padding=pad, groups=C) - mu_y_sq
    sigma_xy   = F.conv2d(x * y, win, padding=pad, groups=C) - mu_xy

    # ── Step 3: 논문 식 (8)  (α=β=γ=1 일 때 축약형) ──
    #   SSIM(p,q) = (2μ_pμ_q + c1)(2σ_pq + c2)
    #              / { (μ_p² + μ_q² + c1)(σ_p² + σ_q² + c2) }
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = num / den   # (B, C, H, W)

    if return_map:
        return ssim_map.mean(), ssim_map
    return ssim_map.mean()


# ========================================================================
# 2) Loss 함수들
# ========================================================================
# get_loss_fn() 은 loss 이름(str)을 받아서 callable 을 반환.
# train.py 에서  loss_fn = get_loss_fn(cfg.loss, ...)  식으로 사용.
#
# [통계학 노트 — 세 loss 의 의미]
#   MSE  = 픽셀 i.i.d. 가우시안 가정의 MLE.
#          이웃 픽셀 독립 → 구조적 결함 놓침.
#   SSIM = 국소 패치의 (μ, σ², σ_xy) 비교.
#          구조·명암·밝기를 동시에 봄 → 구조적 결함 잡음.
#   SSIM+L1 = SSIM 이 잡기 힘든 "순수 색상 결함" (carpet/color 등)에
#          L1 보조항을 추가해서 보완.
# ========================================================================

def get_loss_fn(loss_name: str, ssim_window: int = 11, ssim_alpha: float = 0.84):
    """
    loss 이름 → callable(x, x_hat) 반환.

    Parameters
    ----------
    loss_name  : "mse" | "ssim_loss" | "ssim_l1_loss"
    ssim_window: SSIM 윈도우 K
    ssim_alpha : SSIM+L1 에서 SSIM 비중  (0.84 = Zhao et al. 2017 권장)
    """
    if loss_name == "mse":
        # 논문 식 (2): L2(x, x̂) = mean( (x - x̂)² )
        def mse_loss(x, x_hat):
            return F.mse_loss(x_hat, x)
        return mse_loss

    elif loss_name == "ssim_loss":
        # L_SSIM = 1 - SSIM(x, x̂).  SSIM ∈ [-1,1] 이므로 loss ∈ [0,2].
        def _ssim_loss(x, x_hat):
            return 1.0 - ssim(x, x_hat, window_size=ssim_window)
        return _ssim_loss

    elif loss_name == "ssim_l1_loss":
        # alpha * (1-SSIM) + (1-alpha) * L1.
        # 구조 민감한 SSIM + 색상 민감한 L1 = 상호 보완.
        def _ssim_l1_loss(x, x_hat):
            l_ssim = 1.0 - ssim(x, x_hat, window_size=ssim_window)
            l_l1 = F.l1_loss(x_hat, x)
            return ssim_alpha * l_ssim + (1.0 - ssim_alpha) * l_l1
        return _ssim_l1_loss

    else:
        raise ValueError(f"Unknown loss: {loss_name}")


# ========================================================================
# 3) Residual map — 테스트 시 결함 위치를 "점수" 로 표현
# ========================================================================
# 학습된 AE 가 정상만 잘 복원하므로,
# 원본 x 와 복원 x̂ 의 차이가 큰 픽셀 = 결함 가능성 높음.
#
# 차이를 어떻게 재느냐에 따라 3가지 방법:
#   "l2"      : (x - x̂)² 채널 평균  → 논문 baseline
#   "ssim"    : 1 - SSIM_map  채널 평균  → 논문 제안
#   "ssim_l1" : SSIM residual + L1 residual 가중합
#
# 출력: (B, 1, H, W).  이걸 threshold 해서 binary segmentation 생성.
# ========================================================================

def compute_residual_map(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    method: str = "ssim",
    window_size: int = 11,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Parameters
    ----------
    method : "l2", "ssim", "ssim_l1"
    alpha  : ssim_l1 에서 SSIM residual 비중

    Returns
    -------
    (B, 1, H, W) residual map.  값이 클수록 이상(=결함 가능성 높음).
    """
    if method == "l2":
        return ((x - x_hat) ** 2).mean(dim=1, keepdim=True)

    elif method == "ssim":
        _, smap = ssim(x, x_hat, window_size=window_size, return_map=True)
        return (1.0 - smap).mean(dim=1, keepdim=True)

    elif method == "ssim_l1":
        _, smap = ssim(x, x_hat, window_size=window_size, return_map=True)
        ssim_res = (1.0 - smap)
        l1_res = (x - x_hat).abs()
        return (alpha * ssim_res + (1.0 - alpha) * l1_res).mean(dim=1, keepdim=True)

    else:
        raise ValueError(f"Unknown residual method: {method}")


# ========================================================================
# 4) MVTec Train Dataset — 정상 이미지에서 랜덤 패치 크롭
# ========================================================================
# 학습 로직 요약:
#   1) train/good/ 의 정상 이미지만 사용  (비지도: 결함 이미지 안 봄)
#   2) 리사이즈 (im_resize × im_resize)
#   3) 랜덤 위치에서 patch_size × patch_size 크롭
#   4) [0, 1] 정규화
#   5) (옵션) flip / rotation augmentation
#   6) __len__ = num_patches (가상 길이) → 한 에폭에 이만큼 샘플링
#
# [통계학적 의미]
#   한 장의 이미지에서 수천 개의 패치를 뽑으므로,
#   사실상 "로컬 정상 분포 p(patch)" 를 비모수적으로 추정하는 것.
#   train 이미지가 수백 장뿐이어도 패치 수만 장 → 사실상 충분한 n.
# ========================================================================

class MVTecTrainDataset(Dataset):
    def __init__(self, cfg):
        """cfg 는 options.py 에서 반환된 argparse Namespace."""
        self.root = Path(cfg.dataset_path) / cfg.name / "train" / "good"
        assert self.root.exists(), f"경로 없음: {self.root}"

        # 정상 이미지 파일 목록 전부 수집
        self.paths = sorted(
            list(self.root.glob("*.png")) + list(self.root.glob("*.jpg"))
        )
        assert len(self.paths) > 0, f"이미지 없음: {self.root}"

        self.im_resize = cfg.im_resize
        self.patch_size = cfg.patch_size
        self.grayscale = cfg.grayscale
        self.num_patches = cfg.num_patches

        # augmentation 설정 — 카테고리 유형별로 달라야 함
        self.do_aug = cfg.do_aug
        self.p_rotate = cfg.p_rotate
        self.p_hflip = cfg.p_horizontal_flip
        self.p_vflip = cfg.p_vertical_flip

    def __len__(self) -> int:
        # 실제 파일 수가 아니라 "에폭당 보여줄 패치 수" 를 반환하는 게 포인트.
        # DataLoader 는 이 값을 기준으로 iteration 을 돌림.
        return self.num_patches

    def _load(self, path: Path) -> np.ndarray:
        """이미지 로드 → 리사이즈 → [0,1] float → (C, H, W) ndarray."""
        img = Image.open(path)
        img = img.convert("L" if self.grayscale else "RGB")
        img = img.resize((self.im_resize, self.im_resize), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[None, :, :]           # (1, H, W)
        else:
            arr = arr.transpose(2, 0, 1)    # (3, H, W)
        return arr

    def _crop(self, arr: np.ndarray) -> np.ndarray:
        """랜덤 (top, left) 에서 patch_size × patch_size 잘라내기."""
        _, H, W = arr.shape
        t = random.randint(0, H - self.patch_size)
        l = random.randint(0, W - self.patch_size)
        return arr[:, t:t + self.patch_size, l:l + self.patch_size]

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        """기하 증강: flip + 90도 단위 회전."""
        if random.random() < self.p_hflip:
            patch = patch[:, :, ::-1].copy()
        if random.random() < self.p_vflip:
            patch = patch[:, ::-1, :].copy()
        if random.random() < self.p_rotate:
            k = random.randint(1, 3)            # 90, 180, 270 도
            patch = np.rot90(patch, k=k, axes=(1, 2)).copy()
        return patch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # (1) 이미지 랜덤 선택 — idx 는 무시하고 매번 다른 이미지에서 뽑음
        path = random.choice(self.paths)
        arr = self._load(path)

        # (2) 랜덤 패치 크롭
        patch = self._crop(arr)

        # (3) 증강 (옵션)
        if self.do_aug:
            patch = self._augment(patch)

        # 비지도 학습이므로 label 은 의미 없지만, (x, label) 형식 유지.
        return torch.from_numpy(patch).float(), 0


# ========================================================================
# 5) MVTec Test Dataset — 전체 이미지 + GT mask 반환
# ========================================================================
# 학습 때와 달리 패치가 아니라 "전체 이미지" 단위로 반환.
# 각 이미지에 대해 (이미지, GT mask, label, 파일경로) 튜플.
#
# 테스트 폴더 구조:
#   test/good/        → label=0 (정상)
#   test/<defect>/    → label=1 (결함), 해당 GT mask 도 같이 로드
# ========================================================================

class MVTecTestDataset(Dataset):
    def __init__(self, cfg):
        cat_dir = Path(cfg.dataset_path) / cfg.name
        self.test_dir = cat_dir / "test"
        self.gt_dir = cat_dir / "ground_truth"
        assert self.test_dir.exists(), f"경로 없음: {self.test_dir}"

        self.im_resize = cfg.im_resize
        self.grayscale = cfg.grayscale

        # (파일경로, 결함유형명, 정상여부) 리스트 구축
        self.samples: List[Tuple[Path, str, bool]] = []
        for sub in sorted(self.test_dir.iterdir()):
            if not sub.is_dir():
                continue
            is_good = (sub.name == "good")
            for p in sorted(sub.glob("*.png")):
                self.samples.append((p, sub.name, is_good))

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: Path) -> np.ndarray:
        img = Image.open(path)
        img = img.convert("L" if self.grayscale else "RGB")
        img = img.resize((self.im_resize, self.im_resize), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[None, :, :]
        else:
            arr = arr.transpose(2, 0, 1)
        return arr

    def _load_mask(self, img_path: Path, defect_type: str) -> np.ndarray:
        """
        GT mask 로드.
        정상 이미지면 all-zero mask 반환.
        결함 이미지면 ground_truth/<defect_type>/<stem>_mask.png 를 읽음.
        """
        if defect_type == "good":
            return np.zeros((1, self.im_resize, self.im_resize), dtype=np.float32)

        # MVTec 공식 네이밍: "<stem>_mask.png"
        stem = img_path.stem
        mask_path = self.gt_dir / defect_type / f"{stem}_mask.png"
        mask = Image.open(mask_path).convert("L")
        # 마스크는 최근접보간 (bilinear 하면 0/255 경계가 흐릿해짐)
        mask = mask.resize((self.im_resize, self.im_resize), Image.NEAREST)
        arr = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
        return arr[None, :, :]

    def __getitem__(self, idx: int):
        img_path, defect_type, is_good = self.samples[idx]
        img = self._load(img_path)
        mask = self._load_mask(img_path, defect_type)
        label = 0 if is_good else 1

        return (
            torch.from_numpy(img).float(),      # (C, H, W)
            torch.from_numpy(mask).float(),     # (1, H, W)
            label,                               # 0 or 1
            str(img_path),                       # 디버깅용 원본 경로
        )


# ========================================================================
# 6) 기타 유틸
# ========================================================================

def set_seed(seed: int):
    """재현성을 위한 전역 시드 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
