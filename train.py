import time
import torch
import torch.nn as nn
from torchvision import tv_tensors

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from MViT import MViTConfig, MViT
from loader_aug_gpu import MixupCutmixAugmentation, get_loader, clip_spatial_transform, clip_transform_eval

import ray.train.torch
import torch.distributed as dist

def train_one_epoch(model, loader, mixup_cutmix, optimizer, lr_scheduler, criterion, device):
    model.train()

    total_train_loss = torch.tensor(0, device=device)
    total_iter = 0 

    for x, y in loader:
        t0 = time.time()

        optimizer.zero_grad()

        # This is automatically done by Ray's `prepare_data_loader`!
        # x, y = x.to(device), y.to(device) # (B, T, C, H, W), (B,)
        x = tv_tensors.Video(x)
        x = clip_spatial_transform(x) # augment on gpu
        x, y = mixup_cutmix.augment(x, y) # (B, T, C, H, W), (B, 400)
        x = x.permute(0, 2, 1, 3, 4) # (B, C, T, H, W)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(x) # (B, 400)
            
        loss = criterion(logits, y)
        loss.backward() # compute the gradients and synchronize (avg) the grad across all processes
        # grad_scaler.scale(loss).backward()
        # grad_scaler.unscale_(optimizer)

        # gradient clipping, clip the global norm of the gradients at 1.0, prevent exploding gradients
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # grad_scaler.step(optimizer)
        # grad_scaler.update()  # adjusts scale up/down depending on underflow/overflow

        optimizer.step()
        lr_scheduler.step() # update lr

        total_train_loss += loss.detach()
        total_iter += 1

        torch.cuda.synchronize()  # wait for the GPU to finish work, measure time accurately TODO: remove
        t1 = time.time()
        dt = t1 - t0
        print(f'loss: {loss.item():.4f} | grad norm: {norm:.4f} | time elapsed: {dt}')

    avg_loss = total_train_loss / total_iter
    return avg_loss, norm


def evaluate(model, loader, criterion, device):
    model.eval()

    total_val_loss = torch.tensor(0, device=device)
    total_correct = torch.tensor(0, device=device)
    total_samples = torch.tensor(0)
    total_iter = 0

    with torch.no_grad():
        for x, y in loader:
            # x, y = x.to(device), y.to(device) # (B, T, C, H, W), (B,), not needed if using Ray
            x = tv_tensors.Video(x)
            x = clip_transform_eval(x)

            x = x.permute(0, 2, 1, 3, 4) # (B, C, T, H, W)
            logits = model(x) # (B, 400)
            loss = criterion(logits, y)

            pred = logits.argmax(dim=1) # (B,)
            total_correct += (pred == y).sum()
            total_val_loss += loss
            total_samples += torch.tensor(y.size(1))
            total_iter += 1

    val_loss = total_val_loss / total_iter

    return val_loss, total_correct, total_samples


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_epoch = 300
    lr_warmup_iters = 100 # TODO
    lr_total_iters = 100000 # TODO

    config = MViTConfig()
    raw_model = MViT(config)

    # prepare the model, wrap it in ray for distributed processing
    # prepare_model move the model to the corrrect GPU and wraps it in DistributedDataParallel (DDP)
    # Ray basically does what we'll usually do for DDP. With prepare_model(model, move_to_device=True, parallel_strategy='ddp')
    # Ray detects the Local Rank 'os.environ["LOCAL_RANK"]', .to(device), and  DDP(model, device_ids=[ddp_local_rank])
    model = ray.train.torch.prepare_model(raw_model)
    # model.to(device) # This is done by `prepare_model`
    model = torch.compile(model)

    optimizer = model.configure_optimizers(weight_decay=5e-2, learning_rate=1.6e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_val = nn.CrossEntropyLoss()

    train_loader = get_loader() # TODO
    #if ray.train.get_context().get_world_rank() == 0: # master process
    val_loader = get_loader() # TODO
    # prepare train_loader for distributed sampling
    # prepare_data_loader wrap the dataset in torch's DistributedSampler, which chunks up
    # the dataset for each card, + DataLoader. Equivalent to
    # sampler = DistributedSampler(dataset) if is_distributed else None
    # loader = DataLoader(dataset, shuffle=(sampler is None), sampler=sampler)
    # And with move_to_device=True (the default), it sets up the loader 
    # to move data to the correct GPU automatically
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    val_loader = ray.train.torch.prepare_data_loader(val_loader)

    mixup_cutmix_aug = MixupCutmixAugmentation()

    # linear warmup lr scheduler, linearly increase from 1/100th of LR to full LR
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=lr_warmup_iters)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(lr_total_iters-lr_warmup_iters), eta_min=1.6e-5)
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[lr_warmup_iters])

    # use tensor float32 precision for all matrix multiplications
    # faster than the normal float32 matmul. But all variables (tensors) are still stored in float32
    torch.set_float32_matmul_precision("high")

    # we'll try not to use a grad scaler as BFLOAT16 has the same dynamic range as FP32
    # and checking the entire gradient for NaNs or Infs creates big overhead (requires GPU to CPU synchronization)
    # scaler = torch.amp.GradScaler("cuda")

    train_loss_plot = []
    val_loss_plot = []
    val_acc_plot =[]

    best_val_acc = 0

    for epoch in range(num_epoch):
        if ray.train.get_context().get_world_size() > 1:
            # same as torch's DistributedSampler + DataLoader pattern,
            # it makes shuffling work properly across multiple epochs
            train_loader.sampler.set_epoch(epoch) 

        train_loss, grad_norm = train_one_epoch(model, 
                                     train_loader, 
                                     mixup_cutmix_aug, 
                                     optimizer, 
                                     lr_scheduler, 
                                     criterion, 
                                     device)
        
        # all reduce has to happen outisde the master process, all workers need to all reduce
        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        train_loss = train_loss.item() 

        val_loss, total_correct, total_samples = evaluate(model, val_loader, criterion_val, device)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss = val_loss.item()
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM) 
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)  

        val_acc = total_correct.item() / total_samples.item()

        if ray.train.get_context().get_world_rank() == 0: # master process

            # Report metrics and optionally save a checkpoint
            ray.train.report({"epoch": epoch, "train loss": train_loss, "val loss": val_loss, "val acc": val_acc}) 

            current_lr = optimizer.param_groups[0]['lr']
            train_loss_plot.append(train_loss)
            val_loss_plot.append(val_loss)
            val_acc_plot.append(val_acc)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }

            # save best checkpoint
            status = "Not Best"
            if val_acc > best_val_acc:
                status = "Best"
                best_val_acc = val_acc
                torch.save(checkpoint, 'best_checkpoint.pt')

            # save periodic checkpoint
            if epoch % 10 == 0:
                torch.save(checkpoint, f'epoch_{epoch}.pt')

            print(f'epoch: {epoch} | lr: {current_lr:.4f} | train loss:{train_loss:.4f} | gradient norm: {grad_norm:.4f} | val loss:{val_loss:.4f} | val acc:{val_acc:.4f} | {status}')


if __name__ == "__main__":
     scaling_config = ray.train.ScalingConfig(
         num_workers=8, # one worker per GPU
         use_gpu=True, # training will be done on GPUs (1 GPU per worker by default)
         resources_per_worker={"CPU":12, "GPU":1}, # 12 CPU and 1 GPU per woker
     )

     # PyTorch Distributed supports multiple backends for communicating tensors across workers. 
     # By default Ray Train will use NCCL when use_gpu=True                                       
     trainer = ray.train.torch.TorchTrainer(
         train,
         scaling_config=scaling_config
     )
     
     # Launches the Ray Train controller to run training on workers
     # it launches workers, runs train(), collect all report() calls
     # applied checkpoint rules if any, and return Result object with 
     # restored checkpoints and metrics
     result = trainer.fit()

     # result.metrics
