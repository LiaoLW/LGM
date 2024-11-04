import tyro
import time
import random
import os
import sys

import torch
from core.options import AllConfigs
from core.models import LGM
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file
import neptune

import kiui

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
sys.path.append(os.getcwd())


def train_step(
    accelerator,
    model,
    step,
    step_ratio,
    data,
    optimizer,
    scheduler,
    train_dataloader,
    epoch,
    opt,
    run,
):
    with accelerator.accumulate(model):

        optimizer.zero_grad()

        out = model(data, step_ratio, run=run if opt.neptune else None)
        loss = out["loss"]
        psnr = out["psnr"]
        accelerator.backward(loss)

        # gradient clipping
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

        optimizer.step()
        scheduler.step(loss)
        if opt.neptune:
            run["train/learning_rate"].log(optimizer.param_groups[0]["lr"])

        return out


def save_image(data, out, epoch, opt, run, step, mode="train", save_alphas=False):
    os.makedirs(f"{opt.workspace}/{mode}", exist_ok=True)
    gt_images = (
        data["images_output"].detach().cpu().numpy()
    )  # [B, V, 3, output_size, output_size]
    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(
        -1, gt_images.shape[1] * gt_images.shape[3], 3
    )  # [B*output_size, V*output_size, 3]
    kiui.write_image(
        f"{opt.workspace}/{mode}/images_{epoch}_{step}_gt.jpg",
        gt_images,
    )
    if save_alphas:
        # 获取 gt_alphas 并转换为适当的形状
        gt_alphas = (
            data["masks_output"].detach().cpu().numpy()
        )  # [B, V, 1, output_size, output_size]
        gt_alphas = gt_alphas.transpose(0, 3, 1, 4, 2).reshape(
            -1, gt_alphas.shape[1] * gt_alphas.shape[3], 1
        )  # [B*output_size, V*output_size, 1]

        # 保存 gt_alphas 为图片
        kiui.write_image(
            f"{opt.workspace}/{mode}/gt_alphas_{epoch}_{step}.jpg",
            gt_alphas,
        )
    if epoch % 20 == 0 and opt.neptune:
        run[f"{mode}/images"].append(
            neptune.types.File(f"{opt.workspace}/{mode}/images_{epoch}_{step}_gt.jpg")
        )
        if save_alphas:
            run[f"{mode}/alphas"].append(
                neptune.types.File(
                    f"{opt.workspace}/{mode}/gt_alphas_{epoch}_{step}.jpg"
                )
            )

    # gt_alphas = data['masks_output'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
    # gt_alphas = gt_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, gt_alphas.shape[1] * gt_alphas.shape[3], 1)
    # kiui.write_image(f'{opt.workspace}/{mode}/gt_alphas_{epoch}_{i}.jpg', gt_alphas)

    pred_images = (
        out["images_pred"].detach().cpu().numpy()
    )  # [B, V, 3, output_size, output_size]
    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(
        -1, pred_images.shape[1] * pred_images.shape[3], 3
    )
    kiui.write_image(
        f"{opt.workspace}/{mode}/images_{epoch}_{step}_pred.jpg",
        pred_images,
    )
    if save_alphas:
        pred_alphas = out["alphas_pred"].detach().cpu().numpy()
        pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(
            -1, pred_alphas.shape[1] * pred_alphas.shape[3], 1
        )
        kiui.write_image(
            f"{opt.workspace}/{mode}/pred_alphas_{epoch}_{step}.jpg",
            pred_alphas,
        )
    if epoch % 20 == 0 and opt.neptune:
        run[f"{mode}/images"].append(
            neptune.types.File(f"{opt.workspace}/{mode}/images_{epoch}_{step}_pred.jpg")
        )
        if save_alphas:
            run[f"{mode}/alphas"].append(
                neptune.types.File(
                    f"{opt.workspace}/{mode}/pred_alphas_{epoch}_{step}.jpg"
                )
            )

    # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
    # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
    # kiui.write_image(f'{opt.workspace}/{mode}/pred_alphas_{epoch}_{i}.jpg', pred_alphas)


def main():
    opt = tyro.cli(AllConfigs)
    if opt.resume is not None:
        suffix = "resume_" + os.path.basename(opt.resume)
    else:
        suffix = f"{time.strftime('%Y%m%d_%H%M')}"
    if opt.exp is not None:
        opt.workspace = f"{opt.workspace}/{opt.exp}/{suffix}"
    else:
        opt.workspace = f"{opt.workspace}/{suffix}"
    os.makedirs(opt.workspace, exist_ok=True)

    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        # kwargs_handlers=[ddp_kwargs],
    )

    # model
    model = LGM(opt)

    # neptune
    if opt.neptune:
        run = neptune.init_run(
            project="LGM",
            tags=["".join(suffix.split("_")[-2:])],
            source_files=["**/*.py"],
        )
        run["parameters"] = opt.__dict__

    # # resume
    # if opt.resume is not None:
    #     if opt.resume.endswith('safetensors'):
    #         ckpt = load_file(opt.resume, device='cpu')
    #     else:
    #         ckpt = torch.load(opt.resume, map_location='cpu')

    #     # tolerant load (only load matching shapes)
    #     # model.load_state_dict(ckpt, strict=False)
    #     state_dict = model.state_dict()
    #     for k, v in ckpt.items():
    #         if k in state_dict:
    #             if state_dict[k].shape == v.shape:
    #                 print(f"[INFO] loading {k}")
    #                 state_dict[k].copy_(v)
    #             else:
    #                 accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
    #         else:
    #             accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')

    # data
    if opt.data_mode == "s3":
        from core.provider_objaverse import ObjaverseDataset as Dataset
    else:
        raise NotImplementedError

    train_dataset = Dataset(opt, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = Dataset(opt, training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95)
    )

    # scheduler (per-iteration)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3000, eta_min=1e-6)
    total_steps = opt.num_epochs * len(train_dataloader)
    pct_start = 30 / total_steps
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.98,
        patience=10,
        verbose=True,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=5e-5,
        eps=1e-08,
    )
    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, test_dataloader, scheduler
        )
    )
    # accelerator.register_for_checkpointing(scheduler)
    if opt.resume:
        accelerator.load_state(os.path.join(os.getcwd(), opt.resume))
    for g in optimizer.param_groups:
        g["lr"] = 5e-5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.98,
        patience=10,
        verbose=True,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=5e-5,
        eps=1e-08,
    )
    model, optimizer, train_dataloader, test_dataloader, scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, test_dataloader, scheduler
        )
    )

    if accelerator.is_main_process:
        run["sys/tags"].add("main_process")

    # eval for resume
    if opt.resume:
        with torch.no_grad():
            model.eval()
            total_psnr = 0
            for step, data in enumerate(test_dataloader):
                out = model(data, run=run if opt.neptune else None)
                psnr = out["psnr"]
                if opt.neptune:
                    run["eval/psnr"].log(psnr.item())
                total_psnr += psnr.detach()
            torch.cuda.empty_cache()

            total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
            if accelerator.is_main_process:
                total_psnr /= len(test_dataloader)
                accelerator.print(f"[eval] resume: psnr: {psnr:.4f}")

    # loop
    for epoch in range(opt.num_epochs):
        # train
        model.train()
        total_loss = 0
        total_psnr = 0
        for step, data in enumerate(train_dataloader):
            step_ratio = (epoch + step / len(train_dataloader)) / opt.num_epochs
            out = train_step(
                accelerator,
                model,
                step,
                step_ratio,
                data,
                optimizer,
                scheduler,
                train_dataloader,
                epoch,
                opt,
                run if opt.neptune else None,
            )
            loss = out["loss"]
            psnr = out["psnr"]
            total_loss += loss.detach()
            total_psnr += psnr.detach()

            if accelerator.is_main_process:
                os.makedirs(f"{opt.workspace}/train", exist_ok=True)
                # logging
                if step % 100 == 0:
                    mem_free, mem_total = torch.cuda.mem_get_info()
                    last_lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"[INFO] {step}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {last_lr:.7f} step_ratio: {step_ratio:.4f} loss: {loss.item():.6f}"
                    )

                # save log images
                if step % 1000 == 0:
                    save_image(
                        data,
                        out,
                        epoch,
                        opt,
                        run if opt.neptune else None,
                        step,
                        mode="train",
                        save_alphas=opt.save_alphas,
                    )

        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            accelerator.print(
                f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}"
            )

        # checkpoint
        # if epoch % 10 == 0 or epoch == opt.num_epochs - 1:
        accelerator.wait_for_everyone()
        # accelerator.save_model(model, opt.workspace)
        start_time = time.time()
        accelerator.save_state(opt.workspace)
        accelerator.print(f"[INFO] save state time: {time.time() - start_time:.2f}s")

        # eval
        with torch.no_grad():
            model.eval()
            total_psnr = 0
            for step, data in enumerate(test_dataloader):

                out = model(data, run=run)

                psnr = out["psnr"]
                if opt.neptune:
                    run["eval/psnr"].log(psnr.item())
                total_psnr += psnr.detach()

                # save some images
                if accelerator.is_main_process:
                    save_image(
                        data,
                        out,
                        epoch,
                        opt,
                        run if opt.neptune else None,
                        step,
                        mode="eval",
                        save_alphas=opt.save_alphas,
                    )
            torch.cuda.empty_cache()

            total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
            if accelerator.is_main_process:
                total_psnr /= len(test_dataloader)
                accelerator.print(f"[eval] epoch: {epoch} psnr: {psnr:.4f}")


if __name__ == "__main__":
    main()
