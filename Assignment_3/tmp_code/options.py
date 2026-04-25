"""
============================================================================
 options.py — 학습/테스트 설정을 argparse 로 관리 + wandb 연동
============================================================================
 사용 예시:
   python train.py --name bottle --loss ssim_loss --do_aug --p_rotate 0.
   python train.py --name carpet --loss ssim_l1_loss --do_aug
   python test.py  --name bottle --loss ssim_loss

 [이 파일의 역할]
 ---------------------------------------------------------------------------
 argparse 를 써서 커맨드라인에서 실험 설정을 주입받는 "중앙 설정 관리자".
 원 레포 (plutoyuxie/AE-SSIM) 의 options.py 와 같은 위치.

 MVTec AD 는 15 개 카테고리가 있고, 카테고리마다 최적 하이퍼파라미터가 다름.
 그래서 "--name" 만 넣으면 다음 값들이 자동 결정되도록 해 놓음:
   - grayscale 여부 : grid/screw/zipper → True, 나머지 → False
   - latent dim     : texture → 100, object → 500  (논문 4.2 참고)
 물론 --z_dim 300 --grayscale true 같이 수동 override 도 가능.

 [wandb 연동]
 ---------------------------------------------------------------------------
 wandb (Weights & Biases) 는 실험 로그/그래프를 자동으로 웹 대시보드에
 기록해주는 도구. 한 번 써보면 tensorboard 다시는 안 쓰게 됨.
   - train.py 에서 loss 곡선, 복원 이미지 샘플 등을 wandb.log()
   - test.py 에서 AUROC, residual map 이미지 등을 wandb.log()
   - 설치: pip install wandb
   - 로그인: wandb login  (한 번만 하면 됨)

 wandb 를 안 쓰고 싶으면 --no_wandb 플래그만 추가하면 됨.
============================================================================
"""

import argparse
import os
import sys
from pathlib import Path


# =============================================================================
# 1) MVTec 카테고리별 기본 설정
# =============================================================================
# 논문 공식 (Bergmann et al. 2021, IJCV 확장판):
#   "Since gray-scale images are also common in industrial inspection,
#    three object categories (grid, screw, and zipper) are made available
#    solely as single-channel images."
# → 이 세 카테고리는 파일 자체가 1채널로 저장돼 있음.
# =============================================================================
GRAYSCALE_CATEGORIES = {"grid", "screw", "zipper"}

# texture 계열 (5종): 국소 반복 패턴. 위치 무의미. 패치 기반 방법에 유리.
# object 계열 (10종): 정렬된 개별 물체. 위치/포즈 중요.
TEXTURE_CATEGORIES = {"carpet", "grid", "leather", "tile", "wood"}
OBJECT_CATEGORIES  = {"bottle", "cable", "capsule", "hazelnut", "metal_nut",
                      "pill", "screw", "toothbrush", "transistor", "zipper"}


def get_config():
    """커맨드라인 인자를 파싱해서 cfg 네임스페이스 객체로 반환."""

    p = argparse.ArgumentParser(
        description="AE-SSIM for MVTec AD  (with wandb logging)"
    )

    # ── 데이터 ────────────────────────────────────────────────────────
    # dataset_path : MVTec 압축 풀어둔 루트 폴더.
    #   mvtec_ad/
    #   ├── bottle/
    #   │   ├── train/good/
    #   │   ├── test/  (good + defect 폴더들)
    #   │   └── ground_truth/
    #   ├── carpet/
    #   └── ...
    p.add_argument("--dataset_path", type=str, default="./mvtec_ad",
                   help="MVTec AD 루트 경로")
    p.add_argument("--name", type=str, required=True,
                   help="카테고리 이름 (bottle, carpet, screw ...)")
    p.add_argument("--grayscale", type=str, default="auto",
                   choices=["auto", "true", "false"],
                   help="auto: 카테고리명에 따라 자동 결정")

    # ── 이미지 전처리 ─────────────────────────────────────────────────
    # im_resize : 원본 이미지를 이 크기로 리사이즈한 뒤 패치를 잘라냄.
    # patch_size: 오토인코더가 실제로 보는 입력 크기 (128×128).
    #   논문 4.2: "10,000 defect-free patches of size 128×128,
    #              randomly cropped from the given training images"
    p.add_argument("--im_resize", type=int, default=256,
                   help="전체 이미지 리사이즈 크기")
    p.add_argument("--patch_size", type=int, default=128,
                   help="학습용 랜덤 크롭 패치 크기 (논문: 128)")

    # ── 모델 ──────────────────────────────────────────────────────────
    # z_dim: bottleneck 차원 d.
    #   논문: texture d=100, nanofibres d=500.
    #   원 레포: texture d=100, object d=500.
    #   "--z_dim" 을 명시하지 않으면 카테고리 유형에 따라 자동 결정 (후처리에서).
    p.add_argument("--z_dim", type=int, default=100,
                   help="latent 차원. texture=100, object=500 권장")

    # ── 손실 함수 ─────────────────────────────────────────────────────
    # mse         : 픽셀별 L2 (= 가우시안 i.i.d. 가정의 MLE, 논문 3.1.1)
    # ssim_loss   : 1 - SSIM  (논문 3.1.4, 핵심 제안)
    # ssim_l1_loss: alpha*(1-SSIM) + (1-alpha)*L1
    #   원 레포 Discussion: "SSIM is a measure of similarity only between
    #   grayscale images, it cannot handle color defect in some cases.
    #   So here I use SSIM + L1 distance for anomaly segmentation."
    p.add_argument("--loss", type=str, default="ssim_loss",
                   choices=["mse", "ssim_loss", "ssim_l1_loss"],
                   help="mse=L2, ssim_loss=순수SSIM, ssim_l1_loss=SSIM+L1")
    p.add_argument("--ssim_window", type=int, default=11,
                   help="SSIM 윈도우 크기 K (논문: 11)")
    p.add_argument("--ssim_alpha", type=float, default=0.84,
                   help="SSIM+L1 combo 에서 SSIM 비중 (Zhao et al. 2017 권장)")

    # ── 학습 ──────────────────────────────────────────────────────────
    # 논문 4.2: "200 epochs using ADAM with lr=2e-4, weight_decay=1e-5"
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--decay", type=float, default=1e-5,
                   help="Adam weight decay (논문: 1e-5)")
    p.add_argument("--num_patches", type=int, default=10000,
                   help="에폭당 패치 수 (논문: 10,000)")

    # ── 증강 (augmentation) ───────────────────────────────────────────
    # texture: flip + rotate 전부 OK (방향성 없으니까)
    # object:  flip 만 OK, rotate 는 끄는 게 좋음 (글자/포즈 방향이 고정)
    #   원 레포 README 의 카테고리별 커맨드에서 --p_rotate 0. 이 붙은 이유.
    p.add_argument("--do_aug", action="store_true",
                   help="augmentation 활성화")
    p.add_argument("--p_rotate", type=float, default=0.5,
                   help="90도 회전 확률. object 계열은 0 으로")
    p.add_argument("--p_horizontal_flip", type=float, default=0.5)
    p.add_argument("--p_vertical_flip", type=float, default=0.5)

    # ── 테스트 ────────────────────────────────────────────────────────
    # stride : 테스트 시 패치를 겹쳐 뽑을 때 간격.
    #   논문 4.2: "decreased the stride to 30 pixels and averaged
    #   the reconstructed pixel values"
    # threshold : residual map → binary segmentation 할 때 기준.
    p.add_argument("--stride", type=int, default=30,
                   help="테스트 시 슬라이딩 stride (논문: 30)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="segmentation 용 threshold")

    # ── 체크포인트 / 출력 ─────────────────────────────────────────────
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--result_dir", type=str, default="./results")

    # ── wandb ─────────────────────────────────────────────────────────
    # wandb 로 실험 로그를 남기면 좋은 점:
    #   1) loss 곡선을 웹에서 실시간 확인 (서버에서 학습 돌리면서 폰으로 확인 가능)
    #   2) 복원 이미지, residual map 등을 wandb.Image 로 기록 → 슬라이드에 바로 사용
    #   3) 하이퍼파라미터/결과가 한 대시보드에 자동 정리 → 실험 비교 편함
    #   4) 팀원이 "그 실험 결과 어디 있어?" 하면 URL 하나 던져주면 끝
    p.add_argument("--no_wandb", action="store_true",
                   help="wandb 비활성화 (오프라인 모드)")
    p.add_argument("--wandb_project", type=str, default="AE-SSIM-MVTec",
                   help="wandb 프로젝트 이름")
    p.add_argument("--wandb_entity", type=str, default=None,
                   help="wandb 팀/개인 entity (None 이면 기본 계정)")

    # ── 기타 ──────────────────────────────────────────────────────────
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "cpu", "mps"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)

    cfg = p.parse_args()

    # =================================================================
    # 후처리: 카테고리 기반 자동 설정
    # =================================================================
    # (1) grayscale 자동 판단
    # MVTec 파일이 실제로 1채널로 저장된 카테고리: grid, screw, zipper
    if cfg.grayscale == "auto":
        cfg.grayscale = cfg.name.lower() in GRAYSCALE_CATEGORIES
    elif cfg.grayscale == "true":
        cfg.grayscale = True
    else:
        cfg.grayscale = False

    cfg.in_channels = 1 if cfg.grayscale else 3

    # (2) z_dim 자동 판단 — 유저가 직접 안 넣었으면 카테고리 유형으로 결정
    if "--z_dim" not in sys.argv:
        if cfg.name.lower() in OBJECT_CATEGORIES:
            cfg.z_dim = 500   # object = 복잡 → 큰 bottleneck
        else:
            cfg.z_dim = 100   # texture = 단순 반복 → 작은 bottleneck

    # (3) 출력 디렉토리 생성
    # checkpoints/<name>/ 와 results/<name>/ 에 카테고리별로 분리 저장.
    cfg.checkpoint_dir = os.path.join(cfg.checkpoint_dir, cfg.name)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.name)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)

    return cfg


def print_config(cfg):
    """설정값을 터미널에 보기 좋게 출력.  train/test 시작할 때 호출."""
    print("=" * 55)
    print(f"  Category     : {cfg.name}")
    print(f"  Grayscale    : {cfg.grayscale}  (in_channels={cfg.in_channels})")
    print(f"  Image resize : {cfg.im_resize}")
    print(f"  Patch size   : {cfg.patch_size}")
    print(f"  Latent dim   : {cfg.z_dim}")
    print(f"  Loss         : {cfg.loss}")
    if "ssim" in cfg.loss:
        print(f"  SSIM window  : {cfg.ssim_window}")
    if cfg.loss == "ssim_l1_loss":
        print(f"  SSIM alpha   : {cfg.ssim_alpha}")
    print(f"  Epochs       : {cfg.epochs}")
    print(f"  Batch size   : {cfg.batch_size}")
    print(f"  LR / decay   : {cfg.lr} / {cfg.decay}")
    print(f"  Augmentation : {cfg.do_aug}")
    print(f"  wandb        : {'OFF' if cfg.no_wandb else 'ON'}")
    print(f"  Device       : {cfg.device}")
    print(f"  Checkpoint   : {cfg.checkpoint_dir}")
    print(f"  Results      : {cfg.result_dir}")
    print("=" * 55)


# =============================================================================
# 직접 실행 시 설정값 출력만 해서 확인하는 용도.
#   python options.py --name bottle --loss ssim_loss --do_aug
# =============================================================================
if __name__ == "__main__":
    cfg = get_config()
    print_config(cfg)
