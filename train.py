import os
import importlib
import time
import torch
import torch.nn as nn
from torchvision import tv_tensors

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from MViT import MViTConfig, MViT
from loader.loader_aug_gpu import MixupCutmixAugmentation, get_loader, clip_spatial_aug, clip_transform_eval

import ray.train.torch
import torch.distributed as dist

from torch.profiler import profile, record_function, ProfilerActivity

def train_one_epoch_with_profiler(model, loader, mixup_cutmix, optimizer, lr_scheduler, criterion, device):
    model.train()

    total_train_loss = torch.tensor(0.0, device=device)
    total_iter = 0 

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True, # tracks the the shape of the tensors for every operation executed in your model.
        profile_memory=True, # useful for checking VRAM usage, which layers or operations are consuming the most VRAM.
        with_stack=True # record the Python call stack for every single operator (e.g. conv2d, matmul) executed during the trace.
    ) as prof:

        for i, (x, y) in enumerate(loader):
            # creates a clear block in the profiler output, distinguish data loading, augmentations, 
            # and the actual model forward/backward pass.
            with record_function("train_step"):
                optimizer.zero_grad()

                # augment on gpu
                x, y  = clip_spatial_aug(x, y) # (B, T, C, H, W), (B,)
                x, y = mixup_cutmix.augment(x, y) # (B, T, C, H, W), (B, 400)
                x = x.permute(0, 2, 1, 3, 4) # (B, C, T, H, W)

                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits = model(x) # (B, 400)
                    
                loss = criterion(logits, y)
                loss.backward() # compute the gradients and synchronize (avg) the grad across all processes
                norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step() # update lr

                total_train_loss += loss.detach()
                total_iter += 1

                # step the profiler internal state
                prof.step()
            
            # stop profiling (train iter) after 10 steps
            if i >= 10:
                break

    # master process print and save the profiler result
    if ray.train.get_context().get_world_rank() == 0:
        print("--- TOP GPU OPERATIONS ---")
        # includes both the time spent on the operation itself and the time spent on all its 
        # child kernels, as it represents the total duration the GPU was occupied by that entire 
        # branch of the execution graph.
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        # sum of the self-time for that specific operation, 
        # representing the total CPU time spent purely inside that function's own code
        # Self-time represents the duration spent executing only the code within a specific 
        # function itself, excluding the time spent waiting for any other functions (child calls) 
        # that it triggers.
        print("\n--- TOP CPU TIME TOTAL ---")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
        # save a trace file to the working directory, view it in https://ui.perfetto.dev or chrome://tracing
        prof.export_chrome_trace("trace.json")

    avg_loss = total_train_loss / total_iter
    return avg_loss, norm


def train_one_epoch(model, loader, mixup_cutmix, optimizer, lr_scheduler, criterion, device):
    model.train()

    total_train_loss = torch.tensor(0.0, device=device)
    total_iter = 0 

    for x, y in loader:
        t0 = time.time()

        optimizer.zero_grad()

        # This is automatically done by Ray's `prepare_data_loader`!
        # non_blocking allows CPU to keep working while the GPU copies the data, asynchronous. Makes sure pin_memory=True
        # Ray should be doing non_block already.
        # x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True) # (B/2, T, C, H, W), (B/2,)

        # augment on gpu
        x, y  = clip_spatial_aug(x, y) # (B, T, C, H, W), (B,)
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
        gpu_num = ray.train.get_context().get_world_rank()
        print(f'gpu: {gpu_num} | loss: {loss.item():.4f} | grad norm: {norm:.4f} | time elapsed: {dt}')

    avg_loss = total_train_loss / total_iter
    return avg_loss, norm


def evaluate(model, loader, criterion, device):
    model.eval()

    total_val_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device)
    total_samples = torch.tensor(0, device=device)
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
            total_samples += torch.tensor(y.size(0), device=device)
            total_iter += 1

    val_loss = total_val_loss / total_iter

    return val_loss, total_correct, total_samples


def train():

    # set global configs

    # due to the different stochastic depth value in every block, torch compiler looks at 
    # every block and recompile. The default recompile limit is 8, and we have 16 blocks.
    # so we have to increase the limit, its a global setting
    # torch._dynamo.config.recompile_limit = 64 # can't be serialized for Ray to send to cuda
    importlib.import_module("torch._dynamo.config").recompile_limit = 64

    # use tensor float32 precision for all matrix multiplications
    # faster than the normal float32 matmul. But all variables (tensors) are still stored in float32
    torch.set_float32_matmul_precision("high")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = MViTConfig()
    raw_model = MViT(config)

    # prepare the model, wrap it in ray for distributed processing
    # prepare_model move the model to the corrrect GPU and wraps it in DistributedDataParallel (DDP)
    # Ray basically does what we'll usually do for DDP. With prepare_model(model, move_to_device=True, parallel_strategy='ddp')
    # Ray detects the Local Rank 'os.environ["LOCAL_RANK"]', .to(device), and  DDP(model, device_ids=[ddp_local_rank])
    model = ray.train.torch.prepare_model(raw_model)
    # model.to(device) # This is done by `prepare_model`

    bs = 16
    # configure optimizer after the model being moved to cuda so it has gpu version of the tensors 
    optimizer = model.module.configure_optimizers(weight_decay=5e-2, learning_rate=1.6e-3*bs/512) # TODO

    # good practice to compile last, so it sees the optimizer (and the model's final state) before 
    # freezing the graph representation. Especially we are setting up different weight decay on param 
    # groups in configure_optimizers
    # Can't wrap the Ray model in torch.compile, it becomes un pick-able (unserializable) which creates
    # an issue for Ray. Instead we'll just compile the MViT code while leaving the wrapper clean
    model.module = torch.compile(model.module)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_val = nn.CrossEntropyLoss()

    csv_path = "/blue/uf-dsi/rongguan.gu/MViT/val_mini.csv"
    video_dir = "/blue/uf-dsi/rongguan.gu/MViT/k400/val"
    train_loader = get_loader(csv_path=csv_path, video_dir=video_dir, batch_size=4, num_workers=14) # TODO
    #if ray.train.get_context().get_world_rank() == 0: # master process
    val_loader = get_loader(csv_path=csv_path, video_dir=video_dir, batch_size=4, num_workers=14) # TODO
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

    num_epoch = 200
    iters_per_epoch = len(train_loader)
    lr_warmup_iters = 30 * iters_per_epoch
    lr_total_iters = 200 * iters_per_epoch 

    # linear warmup lr scheduler, linearly increase from 1/100th of LR to full LR
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=lr_warmup_iters)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(lr_total_iters-lr_warmup_iters), eta_min=1.6e-5)
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[lr_warmup_iters])

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

        # Report metrics and optionally save a checkpoint
        ray.train.report({"epoch": epoch, "train loss": train_loss, "val loss": val_loss, "val acc": val_acc}) 

        if ray.train.get_context().get_world_rank() == 0: # master process

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
                # torch.save(checkpoint, 'best_checkpoint.pt') # TODO

            # save periodic checkpoint
            if epoch % 100 == 0: # TODO
                # torch.save(checkpoint, f'epoch_{epoch}.pt')
                pass

            print(f'epoch: {epoch} | lr: {current_lr:.4f} | train loss:{train_loss:.4f} | gradient norm: {grad_norm:.4f} | val loss:{val_loss:.4f} | val acc:{val_acc:.4f} | {status}')


if __name__ == "__main__":

     ray.init( # suppress non critical ray messages
        runtime_env={
            "env_vars": {
                "RAY_LOG_TO_STDERR": "0",
                "RAY_LOGGER_LEVEL": "ERROR",
            }
        }
     )

     scaling_config = ray.train.ScalingConfig( # TODO
         num_workers=2, # one worker per GPU
         use_gpu=True, # training will be done on GPUs (1 GPU per worker by default)
         resources_per_worker={"CPU":14, "GPU":1}, # 14 CPU and 1 GPU per woker
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
