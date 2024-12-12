import os
import math
import torch
import logging
import argparse
import datasets
import diffusers
import accelerate
import transformers
import torch.nn.functional as F

from pathlib import Path
from tqdm.auto import tqdm
from accelerate import Accelerator
from imv_dataset import ImageVideoDataset
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from diffusers.optimization import get_scheduler
from accelerate.utils import ProjectConfiguration, set_seed

### Loading models
from typing import Tuple, Dict, List
from diffusers.utils import load_image
from torch.utils.data import DataLoader
from transformers.utils import ContextManagers
from transformers import T5Tokenizer, T5EncoderModel
from diffusers.image_processor import VaeImageProcessor
from inference import load_vae, load_unet, load_scheduler
from ltx_video.models.autoencoders.vae_encode import (
    get_vae_size_scale_factor,
    vae_decode,
    vae_encode,
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder

logger = get_logger(__name__, log_level="INFO")
model_path = '/workspace/model'


## Training args
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="The data json where the data will be loaded from",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Path to the folder with videos",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    ) 
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--use_8bit_adam", 
        type=bool,
        default=True,
        help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--use_came",
        type=bool,
        default=False,
        help="whether to use came",
    )
    parser.add_argument(
        "--allow_tf32",
        type=bool,
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    ) 
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=50,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument("--frame_rate", type=int, default=24, help="frame rate for training")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args

@torch.no_grad()
def prepare_conditioning(
    video: torch.Tensor,
    vae = None,
    patchifier = None,
    vae_per_channel_normalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare the conditioning data for the video generation. If an input media item is provided, encode it
    and set the conditioning_mask to indicate which tokens to condition on. Input media item should have
    the same height and width as the generated video.

    Args:
        media_items (torch.Tensor): media items to condition on (images or videos)
        num_frames (int): number of frames to generate
        height (int): height of the generated video
        width (int): width of the generated video
        method (ConditioningMethod, optional): conditioning method to use. Defaults to ConditioningMethod.UNCONDITIONAL.
        vae_per_channel_normalize (bool, optional): whether to normalize the input to the VAE per channel. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: the conditioning latents and the conditioning mask
    """
    media_items = video.permute(0, 2, 1, 3, 4) #b, c, f, h, w
    #print("media", media_items.shape)
    num_frames = media_items.shape[2]
    mid_idx = num_frames//2 - 1
    

    # Encode the input video and repeat to the required number of frame-tokens
    start_init_latents = vae_encode(
        media_items[:, :, 0:1, :, :].to(dtype=vae.dtype, device=vae.device),
        vae,
        vae_per_channel_normalize=vae_per_channel_normalize,
    ).float()

    mid_init_latents = vae_encode(
        media_items[:, :,mid_idx:mid_idx+1, :, :].to(dtype=vae.dtype, device=vae.device),
        vae,
        vae_per_channel_normalize=vae_per_channel_normalize,
    ).float()

    end_init_latents = vae_encode(
        media_items[:, :, -1:,  :, :].to(dtype=vae.dtype, device=vae.device),
        vae,
        vae_per_channel_normalize=vae_per_channel_normalize,
    ).float()

    video_scale_factor = 8
    
    init_len, target_len = (
        start_init_latents.shape[2],
        num_frames // video_scale_factor,
    )
    # if isinstance(vae, CausalVideoAutoencoder):
    #     target_len += 1
    #print(target_len, init_len)
    init_latents = start_init_latents[:, :, :target_len]
    if target_len > init_len:
        repeat_factor = (target_len + init_len - 1) // init_len  # Ceiling division
        start_init_latents = start_init_latents.repeat(1, 1, repeat_factor, 1, 1)[
            :, :, :target_len//3
        ]
        mid_init_latents = mid_init_latents.repeat(1, 1, repeat_factor, 1, 1)[
            :, :, :(target_len//3 + target_len%3)
        ]
        end_init_latents = end_init_latents.repeat(1, 1, repeat_factor, 1, 1)[
            :, :, :target_len//3
        ]
        init_latents = torch.cat([start_init_latents, mid_init_latents, end_init_latents], dim=2)
        
    # Prepare the conditioning mask (1.0 = condition on this token)
    b, n, f, h, w = init_latents.shape
        
    conditioning_mask = torch.zeros([b, 1, f, h, w], device=init_latents.device)
    conditioning_mask[:, :, 0] = 1.0
    conditioning_mask[:, :, f//2] = 1.0
    conditioning_mask[:, :, -1] = 1.0

    # Patchify the init latents and the mask
    #print("before_patch",init_latents.shape)
    conditioning_mask = patchifier.patchify(conditioning_mask).squeeze(-1)
    init_latents = patchifier.patchify(latents=init_latents)
    #print('init_latents', init_latents.shape, 'conditioning_mask', conditioning_mask.shape)
    return init_latents, conditioning_mask

def compute_prompt_embeds(
    text_encoder,
    text_input_ids,
    device=None,
    dtype=None,
    num_videos_per_prompt=1,
):  
    text_encoder.to('cuda')
    batch_size = text_input_ids.size(0) 
    with torch.no_grad(): 
        prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    text_encoder.to('cpu')
    
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
    return prompt_embeds
    
def main():
    args = parse_args()
    logging_dir = '/workspace/debug/'
    print(logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    set_seed(args.seed) 
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    print("accelelrator ", accelerator.is_local_main_process)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

     # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.bfloat16
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
    ## Loading the models
    # Prepare models and scheduler
    tokenizer = T5Tokenizer.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="tokenizer"
    )
    patchifier = SymmetricPatchifier(patch_size=1)

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = torch.bfloat16
    unet = load_unet(Path(model_path) / 'unet')
    scheduler = load_scheduler(Path(model_path) / 'scheduler')

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = T5EncoderModel.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="text_encoder"
    )
        vae = load_vae(Path(model_path) / 'vae')
        # Freeze vae and text_encoder and set transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device)

    unet.train()

    trainable_modules = ['ff.net', 'to_q', 'to_v', 'proj_out',]
    trainable_modules_low_learning_rate = []

    for name, param in unet.named_parameters():
        for trainable_module_name in trainable_modules + trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing() 

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
        
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )

        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    trainable_params_optim = [
        {'params': [], 'lr': args.learning_rate},
        {'params': [], 'lr': args.learning_rate / 2},
    ]
    in_already = []
    for name, param in unet.named_parameters():
        high_lr_flag = False
        if name in in_already:
            continue
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                in_already.append(name)
                high_lr_flag = True
                trainable_params_optim[0]['params'].append(param)
                if accelerator.is_main_process:
                    print(f"Set {name} to lr : {args.learning_rate}")
                break
        if high_lr_flag:
            continue
        for trainable_module_name in trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                in_already.append(name)
                trainable_params_optim[1]['params'].append(param)
                if accelerator.is_main_process:
                    print(f"Set {name} to lr : {args.learning_rate / 2}")
                break
    
    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            # weight_decay=args.adam_weight_decay,
            betas=(0.9, 0.999, 0.9999), 
            eps=(1e-30, 1e-16)
        )
    else:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        # Dataset and Dataloader 
    train_dataset = ImageVideoDataset(data_root=args.data_path, tokenizer=tokenizer, video_dir=args.video_dir) 
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) 
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch 

    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")

    initial_global_step = 0 
    first_epoch = 0 
    global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    video_scale_factor, vae_scale_factor, _ = get_vae_size_scale_factor(
            vae
        )
    for epoch in range(first_epoch, args.num_train_epochs): 
        for step, batch in enumerate(train_dataloader): 
            with torch.no_grad():
                video = batch[0].to(accelerator.device, dtype=vae.dtype)
                video = video[:, :(video.shape[1] - video.shape[1]%8), :, :, :]
                #print("batch_0", video.shape)
                num_frames = video.shape[1]
                height, width = video.shape[-2], video.shape[-1]
                latent_num_frames = num_frames // video_scale_factor
                latent_height = height // vae_scale_factor
                latent_width = width // vae_scale_factor
                target_latents = vae_encode(
                    video.permute(0,2,1,3,4).detach(),
                    vae,
                    False
                )
                target_latents.requires_grad_(False)
                target_latents = patchifier.patchify(latents=target_latents).detach()
                video_latents, conditioning_mask = prepare_conditioning(video, patchifier=patchifier, vae=vae)

                #print("t", target_latents.shape, "v", video_latents.shape)
            
                input_text_ids = batch[1]
                text_encoder.to('cpu')
                prompt_embeds = compute_prompt_embeds(text_encoder, input_text_ids, accelerator.device, weight_dtype,)
            import nvsmi
            # print("Memory for step", step, list(nvsmi.get_gpus())[0])
            models_to_accumulate = [unet]
            with accelerator.accumulate(models_to_accumulate): 
                video_latents = video_latents.to(dtype=weight_dtype)
                # print_stats('video_latents', video_latents)
                # Sample a random timestep for each image
                ## WARN: Doesn't have to be random, use random, single head, single tail, bi-modal etc.
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (args.train_batch_size,), device=video_latents.device
                )
                timesteps = timesteps.long()
                timesteps = timesteps.unsqueeze(-1) * (1 - conditioning_mask)
                #print("cond shape", conditioning_mask.shape, "t shape", timesteps.shape)

                # Sample noise that will be added to the latents
                # No noise will be added to the latents where conditioning mask is 1
                noise = torch.randn_like(video_latents)
                # print_stats('noise', noise)

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                # print("timesteps", timesteps.shape)
                sigmas = timesteps/1000
                print(video_latents.shape)
                noisy_video_latents = (1 - sigmas[:, :, None]) * video_latents + sigmas[:, :, None] * noise
                noisy_model_input = noisy_video_latents
                # print_stats('noisy_model_input', noisy_model_input)

                frame_rate = args.frame_rate
                latent_frame_rate = frame_rate // video_scale_factor

                latent_frame_rates = (
                    torch.ones(
                        noisy_model_input.shape[0], 1, device=noisy_model_input.device
                    )
                    * latent_frame_rate
                )
                
                scale_grid = (
                    (
                        1 / latent_frame_rates,
                        vae_scale_factor,
                        vae_scale_factor,
                    )
                    if unet.use_rope
                    else None
                )
                #print(num_frames, latent_num_frames, height, latent_height, width, latent_width, scale_grid)
                indices_grid = patchifier.get_grid(
                    orig_num_frames=latent_num_frames,
                    orig_height=latent_height,
                    orig_width=latent_width,
                    batch_size=args.train_batch_size,
                    scale_grid=scale_grid,
                    device=accelerator.device,
                )
                # print("Memory for step before unet", step, list(nvsmi.get_gpus())[0])
                # Predict the noise residual
                #print("ip",noisy_model_input.shape,"indices",indices_grid.shape,"encoder",prompt_embeds.shape,"t",timesteps.shape)
                compiled_unet = torch.compile(unet, mode='reduce-overhead')
                model_output = compiled_unet(
                    noisy_model_input.to(unet.dtype),
                    indices_grid,
                    encoder_hidden_states=prompt_embeds.to(unet.dtype),
                    encoder_attention_mask=None,
                    timestep=timesteps/1000,
                    return_dict=False,
                )[0]
                # debug_video_latents(target_latents, latent_height, latent_width, latent_num_frames, vae, patchifier, unet, 'target')
                # debug_video_latents(video_latents, latent_height, latent_width, latent_num_frames, vae, patchifier, unet, 'video')
                # debug_video_latents(noisy_model_input, latent_height, latent_width, latent_num_frames, vae, patchifier, unet, 'noisy_video')
                # print_stats('model_output', model_output)
                # print_stats('target_latents', target_latents)
                ut = target_latents - noise
                loss = F.mse_loss(model_output.float(), ut.float())
                accelerator.backward(loss)
                # print("Memory for step after loss", step, list(nvsmi.get_gpus())[0])
                if accelerator.sync_gradients:
                    params_to_clip = unet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # print("Memory for step after zerograd", step, list(nvsmi.get_gpus())[0])
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step) 

                if accelerator.sync_gradients:
                    if global_step % args.checkpointing_steps == 0: 
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        
def print_stats(variable_name, tensor):
    """Prints statistics of a tensor."""
    if not torch.is_tensor(tensor):
        print(f"{variable_name}: Not a tensor")
        return
    print(f"{variable_name}:")
    print(f"  Mean: {tensor.mean().item():.6f}")
    print(f"  Std: {tensor.std().item():.6f}")
    print(f"  Max: {tensor.max().item():.6f}")
    print(f"  Min: {tensor.min().item():.6f}")
    print(f"  Shape: {tuple(tensor.shape)}")

def debug_video_latents(latents, latent_height, latent_width, latent_num_frames, vae, patchifier,unet, name):
    import imageio
    import numpy as np
    out = patchifier.unpatchify(
        latents=latents,
        output_height=latent_height,
        output_width=latent_width,
        output_num_frames=latent_num_frames,
        out_channels=unet.in_channels // math.prod(patchifier.patch_size),
    )
    image_processor = VaeImageProcessor(vae_scale_factor=32)
    out = vae_decode(
                out,
                vae,
                True,
                False,
            )
    out = image_processor.postprocess(out, output_type='pt')
    for i in range(out.shape[0]):
        # Gathering from B, C, F, H, W to C, F, H, W and then permuting to F, H, W, C
        video_np = out[i].permute(1, 2, 3, 0).cpu().float().numpy()
        # Unnormalizing images to [0, 255] range
        video_np = (video_np * 255).astype(np.uint8)
        fps = 24
        height, width = video_np.shape[1:3]
        # In case a single image is generated
        output_filename = f'/workspace/debug/{name}.mp4'

        # Write video
        with imageio.get_writer(output_filename, fps=fps) as video:
            for frame in video_np:
                video.append_data(frame)


if __name__ == "__main__":
    main()