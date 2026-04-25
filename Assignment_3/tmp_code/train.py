"""
============================================================================
 train.py — AE-SSIM 학습 스크립트  (wandb 연동)
============================================================================
 사용 예시:
   # ── bottle (RGB, object, z_dim 자동 500) ──
   python train.py --name bottle --loss ssim_loss --do_aug --p_rotate 0.

   # ── carpet (RGB, texture, z_dim 자동 100) ──
   python train.py --name carpet --loss ssim_l1_loss --do_aug

   # ── grid (grayscale 자동) ──
   python train.py --name grid --loss ssim_loss --do_aug

   # ── L2 baseline ──
   python train.py --name bottle --loss mse --do_aug --p_rotate 0.

   # ── wandb 끄고 싶을 때 ──
   python train.py --name bottle --loss ssim_loss --no_wandb

 [학습 흐름 요약]
 ---------------------------------------------------------------------------
 1. 정상 이미지만 로드 (train/good/)
 2. 랜덤 128×128 패치 크롭 + augmentation
 3. AutoEncoder 에 넣어서 복원
 4. loss(x, x̂) 계산 → backward → optimizer step
 5. 매 에폭마다 loss 를 wandb 에 기록
 6. best loss 일 때 체크포인트 저장

 [wandb 로그 내용]
 ---------------------------------------------------------------------------
 - train/loss          : 에폭별 평균 loss 곡선
 - train/epoch         : 현재 에폭
 - train/reconstructions : 에폭 끝마다 복원 이미지 샘플 (wandb.Image)
 - config              : 모든 하이퍼파라미터 (wandb.init 에서 자동 기록)
============================================================================
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from options import get_config, print_config
from network import AutoEncoder
from utils import MVTecTrainDataset, get_loss_fn, set_seed


def train(cfg):
    set_seed(cfg.seed)
    print_config(cfg)

    # ══════════════════════════════════════════════════════════════════
    # wandb 초기화
    # ══════════════════════════════════════════════════════════════════
    # wandb.init() 에 config=vars(cfg) 를 넘기면,
    # 커맨드라인에서 넣은 모든 설정이 wandb 대시보드의 "Config" 탭에 자동 기록됨.
    # 나중에 "이 실험은 z_dim 이 뭐였지?" 할 때 대시보드에서 바로 확인 가능.
    #
    # run name 을 "<카테고리>_<loss>" 로 해두면 대시보드에서 구분이 쉬움.
    # ──────────────────────────────────────────────────────────────────
    if not cfg.no_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=f"{cfg.name}_{cfg.loss}",          # run 이름
            config=vars(cfg),                        # 모든 하이퍼파라미터 기록
            tags=[cfg.name, cfg.loss],               # 필터링용 태그
        )

    # ══════════════════════════════════════════════════════════════════
    # 데이터 준비
    # ══════════════════════════════════════════════════════════════════
    train_ds = MVTecTrainDataset(cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,     # 마지막 불완전 배치 버림 (BatchNorm 등에 안전)
    )
    print(f"[Data] {len(train_ds)} patches/epoch, "
          f"{len(train_loader)} batches/epoch, "
          f"in_channels={cfg.in_channels}")

    # ══════════════════════════════════════════════════════════════════
    # 모델 생성
    # ══════════════════════════════════════════════════════════════════
    model = AutoEncoder(in_channels=cfg.in_channels, z_dim=cfg.z_dim)
    model.to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] AutoEncoder  z_dim={cfg.z_dim}  params={n_params:,}")

    # wandb 에 모델 구조 기록 (optional)
    if not cfg.no_wandb:
        wandb.watch(model, log="gradients", log_freq=100)

    # ══════════════════════════════════════════════════════════════════
    # 옵티마이저 + loss 함수
    # ══════════════════════════════════════════════════════════════════
    # 논문 4.2: "ADAM optimizer with lr=2e-4 and weight decay 1e-5"
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.decay
    )
    loss_fn = get_loss_fn(cfg.loss, cfg.ssim_window, cfg.ssim_alpha)
    print(f"[Loss] {cfg.loss}\n")

    # ══════════════════════════════════════════════════════════════════
    # 학습 루프
    # ══════════════════════════════════════════════════════════════════
    best_loss = float("inf")

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        running = 0.0

        for x, _ in train_loader:
            # ※ 비지도 학습이므로 label (_ 로 버림) 은 사용 안 함.
            x = x.to(cfg.device)

            x_hat = model(x)              # 순전파: 복원 시도
            loss = loss_fn(x, x_hat)      # loss 계산

            optimizer.zero_grad()
            loss.backward()               # 역전파
            optimizer.step()              # 파라미터 업데이트

            running += loss.item() * x.size(0)

        avg_loss = running / len(train_ds)

        # ── wandb 로그: loss 곡선 ─────────────────────────────────────
        if not cfg.no_wandb:
            log_dict = {
                "train/loss": avg_loss,
                "train/epoch": epoch,
            }

            # 매 20 에폭마다 복원 이미지 샘플도 wandb 에 기록.
            # 학습이 진행되면서 복원이 점점 좋아지는 걸 눈으로 확인 가능.
            if epoch % 20 == 0 or epoch == 1:
                with torch.no_grad():
                    sample_x = x[:4].cpu()          # 배치에서 4장만 뽑아서
                    sample_hat = x_hat[:4].cpu()
                    # (B, C, H, W) → wandb 가 읽을 수 있는 이미지 형식으로 변환
                    imgs = []
                    for i in range(min(4, sample_x.size(0))):
                        orig = sample_x[i].permute(1, 2, 0).numpy()    # (H, W, C)
                        recon = sample_hat[i].permute(1, 2, 0).numpy()
                        orig = np.clip(orig, 0, 1)
                        recon = np.clip(recon, 0, 1)
                        # 옆으로 나란히 이어붙여서 비교 이미지 생성
                        pair = np.concatenate([orig, recon], axis=1)
                        import wandb as _wb
                        imgs.append(_wb.Image(
                            pair,
                            caption=f"epoch {epoch} | left=original, right=reconstruction"
                        ))
                    log_dict["train/reconstructions"] = imgs

            wandb.log(log_dict, step=epoch)

        # ── 터미널 출력 (10 에폭마다) ─────────────────────────────────
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{cfg.epochs}  loss={avg_loss:.6f}")

        # ── 체크포인트 저장 (best loss 갱신 시) ───────────────────────
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(cfg.checkpoint_dir, "best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "config": vars(cfg),
            }, save_path)

    # ── 마지막 에폭도 저장 ────────────────────────────────────────────
    save_path = os.path.join(cfg.checkpoint_dir, "last.pth")
    torch.save({
        "epoch": cfg.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "config": vars(cfg),
    }, save_path)

    print(f"\n[Done] best_loss={best_loss:.6f}")
    print(f"  saved: {cfg.checkpoint_dir}/best.pth")
    print(f"  saved: {cfg.checkpoint_dir}/last.pth")

    # ── wandb 마무리 ──────────────────────────────────────────────────
    if not cfg.no_wandb:
        # best 모델을 wandb Artifact 로 업로드 — 나중에 다운로드해서 쓸 수 있음.
        artifact = wandb.Artifact(
            name=f"ae-ssim-{cfg.name}-{cfg.loss}",
            type="model",
        )
        artifact.add_file(os.path.join(cfg.checkpoint_dir, "best.pth"))
        wandb.log_artifact(artifact)

        wandb.finish()


if __name__ == "__main__":
    cfg = get_config()
    train(cfg)
