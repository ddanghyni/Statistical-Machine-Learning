"""
============================================================================
 test.py — AE-SSIM 테스트 & 평가 스크립트  (wandb 연동)
============================================================================
 사용 예시:
   python test.py --name bottle --loss ssim_loss
   python test.py --name carpet --loss ssim_l1_loss

 [이 스크립트의 출력]
 ---------------------------------------------------------------------------
 1) Image-level AUROC : 이미지 전체를 '정상/결함' 으로 이진분류한 성능.
      residual map 의 max 값을 이미지 단위 점수로 사용.
 2) Pixel-level AUROC : 각 픽셀을 '정상/결함' 으로 분류한 성능.
      논문의 핵심 평가 지표.  ROC 곡선의 곡선 아래 면적 (AUC).
 3) 시각화 결과 PNG   : 원본 / 복원 / residual map / GT mask 를 4-panel 로 저장.
 4) wandb 로그        : AUROC 수치 + 시각화 이미지를 대시보드에 자동 업로드.

 [wandb 로그 내용]
 ---------------------------------------------------------------------------
 - test/image_auroc   : image-level AUROC (스칼라)
 - test/pixel_auroc   : pixel-level AUROC (스칼라)
 - test/visualizations: 원본/복원/residual/GT 4-panel 이미지 (wandb.Image)
 - test/roc_curve      : ROC 곡선 (wandb.plot.roc_curve)
============================================================================
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# matplotlib 을 GUI 없는 서버에서도 돌리기 위해 Agg 백엔드 강제.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from options import get_config, print_config
from network import AutoEncoder
from utils import MVTecTestDataset, compute_residual_map, set_seed


def load_model(cfg) -> AutoEncoder:
    """
    best.pth 에서 학습된 모델 가중치를 불러옴.
    train.py 에서 저장한 체크포인트 형식:
      {"epoch": ..., "model_state_dict": ..., "config": ...}
    """
    model = AutoEncoder(in_channels=cfg.in_channels, z_dim=cfg.z_dim)
    ckpt_path = os.path.join(cfg.checkpoint_dir, "best.pth")
    assert os.path.exists(ckpt_path), f"체크포인트 없음: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(cfg.device)
    model.eval()   # dropout, batchnorm 등을 평가 모드로 전환
    print(f"[Model] loaded from {ckpt_path}  (epoch {ckpt['epoch']})")
    return model


def evaluate(cfg):
    set_seed(cfg.seed)
    print_config(cfg)

    # ══════════════════════════════════════════════════════════════════
    # wandb 초기화 — 테스트 전용 run
    # ══════════════════════════════════════════════════════════════════
    # 학습 run 과 별도의 run 으로 기록.
    # 나중에 대시보드에서 "train run" 과 "test run" 을 비교할 수 있음.
    if not cfg.no_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=f"{cfg.name}_{cfg.loss}_test",     # test run 이름
            config=vars(cfg),
            tags=[cfg.name, cfg.loss, "test"],
            job_type="evaluation",                   # train/eval 구분용
        )

    model = load_model(cfg)

    # ══════════════════════════════════════════════════════════════════
    # 테스트 데이터 로드
    # ══════════════════════════════════════════════════════════════════
    test_ds = MVTecTestDataset(cfg)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,           # 테스트는 배치 1 (이미지 크기가 다를 수 있으므로)
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    print(f"[Data] {len(test_ds)} test images")

    # ── residual 방법 결정 (학습 loss 에 맞춰서) ──────────────────────
    if cfg.loss == "mse":
        res_method = "l2"
    elif cfg.loss == "ssim_loss":
        res_method = "ssim"
    else:
        res_method = "ssim_l1"

    # ══════════════════════════════════════════════════════════════════
    # 추론 + 점수 수집
    # ══════════════════════════════════════════════════════════════════
    all_labels = []       # image-level: 0=good, 1=defect
    all_scores = []       # image-level: max(residual map)
    all_gt_masks = []     # pixel-level: (H*W,) flattened binary
    all_res_maps = []     # pixel-level: (H*W,) flattened float

    vis_dir = os.path.join(cfg.result_dir, "visualize")
    os.makedirs(vis_dir, exist_ok=True)

    wandb_images = []     # wandb 에 올릴 시각화 이미지 모아두기

    with torch.no_grad():
        for i, (img, mask, label, path) in enumerate(test_loader):
            img = img.to(cfg.device)

            # (1) 복원
            x_hat = model(img)

            # (2) residual map 계산
            rmap = compute_residual_map(
                img, x_hat,
                method=res_method,
                window_size=cfg.ssim_window,
            )

            # numpy 변환
            rmap_np = rmap.squeeze().cpu().numpy()       # (H, W)
            mask_np = mask.squeeze().cpu().numpy()       # (H, W)

            # (3) 점수 수집
            all_labels.append(label.item())
            all_scores.append(rmap_np.max())             # image-level 점수
            all_gt_masks.append(mask_np.flatten())       # pixel-level GT
            all_res_maps.append(rmap_np.flatten())       # pixel-level 예측

            # (4) 시각화: 처음 20장만 PNG 로 저장
            if i < 20:
                fig = _make_vis_figure(
                    img.squeeze().cpu().numpy(),
                    x_hat.squeeze().cpu().numpy(),
                    rmap_np,
                    mask_np,
                    label.item(),
                    cfg.grayscale,
                )
                fname = os.path.basename(path[0])
                fig_path = os.path.join(vis_dir, fname)
                fig.savefig(fig_path, dpi=100, bbox_inches="tight")
                plt.close(fig)

                # wandb 에도 올릴 이미지 수집
                if not cfg.no_wandb:
                    tag = "DEFECT" if label.item() == 1 else "GOOD"
                    wandb_images.append(
                        wandb.Image(fig_path, caption=f"{fname} ({tag})")
                    )

    # ══════════════════════════════════════════════════════════════════
    # 메트릭 계산
    # ══════════════════════════════════════════════════════════════════
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # Image-level AUROC
    #   "이 이미지 어딘가에 결함이 있느냐?" 를 residual map 의 최댓값으로 판단.
    #   AUROC 가 1.0 이면 정상/결함을 완벽 분류한 것.
    if len(np.unique(all_labels)) > 1:
        img_auroc = roc_auc_score(all_labels, all_scores)
    else:
        img_auroc = -1.0
        print("[Warning] 단일 클래스만 존재 → image AUROC 계산 불가")

    # Pixel-level AUROC
    #   "이 픽셀이 결함이냐?" 를 residual 값으로 판단.
    #   이게 논문의 핵심 평가 지표.  SSIM 이 L2 를 크게 이기는 게 여기서 드러남.
    gt_flat = np.concatenate(all_gt_masks)
    res_flat = np.concatenate(all_res_maps)
    if len(np.unique(gt_flat)) > 1:
        pix_auroc = roc_auc_score(gt_flat, res_flat)
    else:
        pix_auroc = -1.0
        print("[Warning] GT mask 에 결함 픽셀이 없음 → pixel AUROC 계산 불가")

    # ── 결과 출력 ─────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"  [{cfg.name}] Results  (loss={cfg.loss}, z_dim={cfg.z_dim})")
    print(f"  Image-level AUROC : {img_auroc:.4f}")
    print(f"  Pixel-level AUROC : {pix_auroc:.4f}")
    print(f"{'=' * 50}")
    print(f"  Visualizations saved to: {vis_dir}")

    # ── 결과 txt 파일 저장 ────────────────────────────────────────────
    with open(os.path.join(cfg.result_dir, "metrics.txt"), "w") as f:
        f.write(f"category: {cfg.name}\n")
        f.write(f"loss: {cfg.loss}\n")
        f.write(f"z_dim: {cfg.z_dim}\n")
        f.write(f"image_auroc: {img_auroc:.6f}\n")
        f.write(f"pixel_auroc: {pix_auroc:.6f}\n")

    # ══════════════════════════════════════════════════════════════════
    # wandb 로그
    # ══════════════════════════════════════════════════════════════════
    if not cfg.no_wandb:
        # (a) AUROC 수치
        wandb.log({
            "test/image_auroc": img_auroc,
            "test/pixel_auroc": pix_auroc,
        })

        # (b) 시각화 이미지 업로드
        if wandb_images:
            wandb.log({"test/visualizations": wandb_images})

        # (c) wandb summary 에 최종 수치 기록
        #     대시보드 상단 "Summary" 열에 바로 보임.
        wandb.run.summary["image_auroc"] = img_auroc
        wandb.run.summary["pixel_auroc"] = pix_auroc
        wandb.run.summary["loss_type"] = cfg.loss
        wandb.run.summary["z_dim"] = cfg.z_dim

        wandb.finish()


def _make_vis_figure(img, rec, rmap, mask, label, grayscale):
    """
    원본 / 복원 / residual map / GT mask 를 4-panel 로 그린 figure 반환.
    파일 저장과 wandb 업로드 양쪽에서 재사용.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # (C, H, W) → matplotlib 이 읽을 수 있는 형태로 변환
    if grayscale:
        img_show = img[0]                        # (1, H, W) → (H, W)
        rec_show = rec[0]
        cmap = "gray"
    else:
        img_show = img.transpose(1, 2, 0)        # (3, H, W) → (H, W, 3)
        rec_show = rec.transpose(1, 2, 0)
        cmap = None

    tag = "DEFECT" if label == 1 else "GOOD"

    axes[0].imshow(np.clip(img_show, 0, 1), cmap=cmap)
    axes[0].set_title(f"Input ({tag})")
    axes[0].axis("off")

    axes[1].imshow(np.clip(rec_show, 0, 1), cmap=cmap)
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    # residual map 은 "hot" colormap → 빨간 영역 = 높은 이상 점수
    axes[2].imshow(rmap, cmap="hot")
    axes[2].set_title("Residual map")
    axes[2].axis("off")

    axes[3].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("GT mask")
    axes[3].axis("off")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    cfg = get_config()
    evaluate(cfg)
