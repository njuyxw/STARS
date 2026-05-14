"""
The main training script for fine-tuning the Ouro model on a dataset. This script handles loading the model and tokenizer, preparing the dataset, setting up distributed training, and running the training loop with support for gradient accumulation, mixed precision, and checkpointing.
"""

import time
import matplotlib.pyplot as plt
global_start_time = time.time()
import os
import socket
import json
import random
import glob
from typing import TYPE_CHECKING, Any, Optional
import sys
import datetime
import shutil
import numpy as np
from tqdm import tqdm
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets
from contextlib import nullcontext

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

USE_LOCAL_CODE = False
if USE_LOCAL_CODE:
    import litgpt  # noqa


# Check device health immediately after loading torch and standard libraries without loading cuda/hip/dist:
nvml_count = torch.cuda._device_count_amdsmi() if torch.version.hip else torch.cuda._device_count_nvml()
if nvml_count < 1:
    raise ValueError(f"Node failure! Device manager init failed on {socket.gethostname()}")


if TYPE_CHECKING:
    import torch.distributed
    import torch.version
    import torch._dynamo.config


from dataclasses import dataclass, field
from jsonargparse import CLI


end_time = time.time()
if int(os.getenv("SLURM_PROCID", "0")) == 0:
    print(f"{time.ctime()[:-5]}: Time to load libraries: {end_time - global_start_time:.02f} seconds.")


@dataclass
class CLISettings:
    run_name: str = "ouro-sft-reg-randomloop-lastmathtime-400K"
    out_path: str = "outputs"
    # data
    dataset_location: str = "AI-MO/NuminaMath-1.5"
    model_name: str = "/ssd/yangxw/LoopedModel/ouro_experiment/Ouro-1.4B"  # local path to model
    dataset_args: dict[str, Any] = field(default_factory=lambda: dict(q_col="problem", a_col="solution"))

    # other example dataset_args:
    #dataset_args: dict[str, Any] = field(default_factory=lambda: dict(q_col="question", a_col="answer"))
    #dataset_args: dict[str, Any] = field(default_factory=lambda: dict(q_col="instruction", a_col="output"))
    #UNIFIED_PROMPT_COL = "prompt"
    #UNIFIED_RESPONSE_COL = "response"
    #dataset_config: str = "main"

    max_seq_length: int = 2048
    max_samples: Optional[int] = 400_000

    # impl
    micro_batch_size: int = 1
    compile: bool = False
    # log_interval: int = 8

    # training
    max_steps: int = 0
    epochs: int = 1
    batch_size: int = 32
    optim_config: dict[str, Any] = field(
        default_factory=lambda: dict(lr=1e-6, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8)
    )
    scheduler_args: dict[float, Any] = field(default_factory=lambda: dict(warmup=0.1, cooldown=0.1, min_lr_ratio=0.001))  # type: ignore # min_lr = min_lr_ratio * lr
    eval_interval: int = 1000000
    seed: int = 74
    take_loss_over_all_tokens: bool = False  # for chat templated datasets default is to only supervise assistant tokens
    max_grad_norm: float = 1_000_000.0  # i.e. unused unless something is going very wrong
    precision: str = "bf16-true"
    gradient_checkpointing: bool = True

    save_final_checkpoint: bool = True
    loss_window_size: int = 100 
    save_interval: int = 300 
    resume_from_checkpoint: Optional[str] = None # Set to None to disable resume
    def __post_init__(self):
        pass


@dataclass
class Message:
    role: str
    content: str


def is_main_process():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    else:
        return True


def seed_everything(seed):
    import random  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def get_unwrapped_model(state):
    return state["model"].module if state["distributed"] else state["model"]


####################################################################################################
# Main driver functions.
####################################################################################################
DEFAULT_SYS_PROMPT = "You are a helpful assistant that can assist users with reasoning."
def dynamic_padding_collate_fn(batch, tokenizer, max_length=None):
    input_ids_list = [item["input_ids"] for item in batch]
    mask_list = [item["mask"] for item in batch]
    attention_mask_list = [item["attention_mask"] for item in batch]
    
    max_len = max(len(ids) for ids in input_ids_list)
    if max_length is not None:
        max_len = min(max_len, max_length)
    
  
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    padded_input_ids = []
    padded_masks = []
    padded_attention_masks = []
    
    for i, input_ids in enumerate(input_ids_list):
        current_len = len(input_ids)
        if current_len < max_len:
            padding_length = max_len - current_len
            padded_input_ids.append([pad_token_id] * padding_length + input_ids)
            padded_masks.append([0] * padding_length + mask_list[i])
            padded_attention_masks.append([0] * padding_length + attention_mask_list[i])
        else:
            padded_input_ids.append(input_ids[:max_len])
            padded_masks.append(mask_list[i][:max_len])
            padded_attention_masks.append(attention_mask_list[i][:max_len])
    
    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "mask": torch.tensor(padded_masks, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
    }

def startup(cfg: CLISettings):
    """The main setup function for the training script."""
    seed_everything(cfg.seed)

    rank = int(os.getenv("SLURM_PROCID", os.getenv("RANK", "0")))
    local_device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        distributed = True
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=int(os.getenv("SLURM_NTASKS", os.getenv("WORLD_SIZE", -1))),
            device_id=local_device,
            timeout=datetime.timedelta(hours=2),
        )
        world_size = torch.distributed.get_world_size()
        print(f"Comms formed on rank {rank} with device {local_device} out of world size {world_size}.")
    else:
        world_size = 1
        distributed = False
    torch.cuda.set_device(local_device)

    if cfg.precision == "bf16-true":
        torch.set_default_dtype(torch.bfloat16)
        weight_dtype = torch.bfloat16
        autocast_args = {"device_type": "cuda", "enabled": False, "dtype": torch.bfloat16}
    elif cfg.precision == "bf16-mixed":
        torch.set_default_dtype(torch.float32)
        weight_dtype = torch.float32
        autocast_args = {"device_type": "cuda", "enabled": True, "dtype": torch.bfloat16}

    ########## Model and tokenizer ##############
    model_load_path = cfg.model_name
    initial_optimizer_step = 0
    initial_epoch = 0
    if cfg.resume_from_checkpoint:
        model_load_path = cfg.resume_from_checkpoint
        if not os.path.exists(model_load_path):
            raise FileNotFoundError(f"Resume checkpoint directory not found: {model_load_path}")
        if is_main_process():
            print(f"Loading model and tokenizer from resume checkpoint: {model_load_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_load_path,
        trust_remote_code=not USE_LOCAL_CODE,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )
    if cfg.gradient_checkpointing:
        #model.gradient_checkpointing_enable()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # model.config.gradient_checkpointing = True
    tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer.pad_token was not set. Setting it to EOS token: {tokenizer.pad_token}")
    # print("Tokenizer pad token:", tokenizer.pad_token, "eos token:", tokenizer.eos_token)
    # print("Tokenizer pad token id:", tokenizer.pad_token_id, "eos token id:", tokenizer.eos_token_id)
    # token_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    # print(f"Token IDs: {token_ids}")
    
    ##########  Distribute model   ##############
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_device], find_unused_parameters=not cfg.compile, gradient_as_bucket_view=True
        )
    if cfg.compile:
        model = torch.compile(model, fullgraph=False, dynamic=False, mode="max-autotune-no-cudagraphs")
    ##########     Optimizer       ##############
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optim_config)

    def format_and_tokenize_examples_for_mix(examples):
        '''
        This function is specifically for formatting and tokenizing examples for the mixed dataset that uses the unified prompt/response format. It applies the chat template to the prompts and responses, tokenizes them, and creates the appropriate attention masks and loss masks based on the configuration.
        '''
        input_ids_list = []
        mask_list = []
        attention_mask_list = []

        prompts = examples[cfg.UNIFIED_PROMPT_COL]
        responses = examples[cfg.UNIFIED_RESPONSE_COL]
        batch_size = len(prompts)
        for idx in tqdm(range(batch_size), desc="Tokenizing batch"):
            messages = [
                Message(role="system", content=DEFAULT_SYS_PROMPT),
                Message(role="user", content=prompts[idx].strip()),
                Message(role="assistant", content=responses[idx].strip()),]
            chat_encoding = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_assistant_tokens_mask=False,
                    padding=False,  
                    max_length=cfg.max_seq_length + 1,
                    return_tensors=None,  
                    return_dict=True,
                    truncation=True,
                )
            input_ids = chat_encoding["input_ids"]
            attention_mask = chat_encoding["attention_mask"]

            if cfg.take_loss_over_all_tokens:
                assistant_mask = attention_mask
            else:
                assistant_text = messages[-1].content
                assistant_ids = tokenizer(assistant_text, add_special_tokens=False)["input_ids"]
                assistant_mask = [0] * (len(input_ids) - len(assistant_ids)) + [1] * len(assistant_ids)
            input_ids_list.append(input_ids)
            mask_list.append(assistant_mask)
            attention_mask_list.append(attention_mask)
    
        return {
            "input_ids": input_ids_list,
            "mask": mask_list,
            "attention_mask": attention_mask_list,
        }
    def format_and_tokenize_examples(examples):
        '''This function formats and tokenizes examples for datasets that do not use the unified prompt/response format. It constructs the chat messages based on the specified question and answer columns, applies the chat template, tokenizes the input, and creates the appropriate attention masks and loss masks based on the configuration.'''
        input_ids_list = []
        mask_list = []
        attention_mask_list = []
        
        batch_size = len(examples[cfg.dataset_args["q_col"]])
        for idx in tqdm(range(batch_size), desc="Tokenizing batch"):
            if cfg.dataset_args["q_col"] != "text":
                messages = [
                    Message(role="system", content=DEFAULT_SYS_PROMPT),
                    Message(role="user", content=examples[cfg.dataset_args["q_col"]][idx].strip()),
                    Message(role="assistant", content=examples[cfg.dataset_args["a_col"]][idx].strip()),
                ]

                chat_encoding = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_assistant_tokens_mask=False,
                    padding=False, 
                    max_length=cfg.max_seq_length + 1,
                    return_tensors=None,  
                    return_dict=True,
                    truncation=True,
                )

                input_ids = chat_encoding["input_ids"]
                attention_mask = chat_encoding["attention_mask"]

                if cfg.take_loss_over_all_tokens:
                   assistant_mask = attention_mask  
                else:
            
                    assistant_text = messages[-1].content
                    assistant_ids = tokenizer(assistant_text, add_special_tokens=False)["input_ids"]
                    assistant_mask = [0] * (len(input_ids) - len(assistant_ids)) + [1] * len(assistant_ids)

            else:
             
                text = tokenizer.bos_token + examples[cfg.dataset_args["q_col"]][idx].strip()
                chat_encoding = tokenizer(
                    text,
                    padding=False,
                    max_length=cfg.max_seq_length + 1,
                    return_tensors=None,
                    truncation=True,
                )
                input_ids = chat_encoding["input_ids"]
                attention_mask = chat_encoding["attention_mask"]
                assistant_mask = attention_mask  
            
            input_ids_list.append(input_ids)
            mask_list.append(assistant_mask)
            attention_mask_list.append(attention_mask)
        
        return {
            "input_ids": input_ids_list,
            "mask": mask_list,
            "attention_mask": attention_mask_list,
        }

    cfg.token_id_col_name = "input_ids"  # type: ignore
    dataset_save_dir = f"{cfg.out_path}/{cfg.run_name}/dataset"
    if is_main_process():  # only do mapping on rank 0
        try:
            print('loading dataset...')
            #dataset: Dataset = load_dataset(cfg.dataset_location, cfg.dataset_config)["train"]  # type: ignore
            dataset: Dataset = load_dataset(cfg.dataset_location)["train"] 
 
            original_size = len(dataset)
            dataset = dataset.filter(
                 lambda example: example[cfg.dataset_args["q_col"]] is not None and example[cfg.dataset_args["a_col"]] is not None
            )
            cleaned_size = len(dataset)
            if original_size > cleaned_size:
                print(f"Cleaned {original_size - cleaned_size} rows containing null values. Remaining data: {cleaned_size}")
            print(dataset)
        except BaseException:
            dataset: Dataset = load_from_disk(cfg.dataset_location, cfg.dataset_config)  # type: ignore

        if cfg.max_samples is not None:
            dataset = dataset.shuffle(seed=cfg.seed).select(range(cfg.max_samples))

        if os.path.exists(dataset_save_dir):  # delete any old dataset
            shutil.rmtree(dataset_save_dir)
        print('processing dataset...')
        tokenized_dataset = dataset.map(
            format_and_tokenize_examples,
            num_proc=16,
            remove_columns=dataset.column_names,
            batched=True,
            batch_size=1024,
        )
    if distributed:  # load the dataset to other ranks
        if is_main_process():
            tokenized_dataset.save_to_disk(dataset_save_dir)
        torch.distributed.barrier()
        tokenized_dataset = load_from_disk(dataset_save_dir)
        torch.distributed.barrier()

    if rank == 0:
        idx = int(torch.randint(len(tokenized_dataset), (1,)))
        print(f"-----------------------------------Processed Data example idx {idx}:----------------------------")
        print(tokenized_dataset[idx])
        print(tokenizer.decode(tokenized_dataset[idx]["input_ids"], skip_special_tokens=False))
        print("--------------------------------------------------------------------------------------------")
    #tokenized_dataset.set_format("pt")
    from functools import partial
    collate_fn = partial(dynamic_padding_collate_fn, tokenizer=tokenizer, max_length=cfg.max_seq_length + 1)

    if distributed:
        sampler = torch.utils.data.DistributedSampler(
            tokenized_dataset,  # type: ignore
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            seed=cfg.seed,
        )
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,  # type: ignore
            batch_size=cfg.micro_batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,  # type: ignore
            batch_size=cfg.micro_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    ##########     Scheduler       ##############
    accumulation_steps = max(1, cfg.batch_size // cfg.micro_batch_size)
    num_update_steps_per_epoch = math.ceil(len(dataloader) / accumulation_steps)
    max_training_steps = cfg.epochs * num_update_steps_per_epoch
    num_warmup_steps = math.ceil(cfg.scheduler_args["warmup"] * max_training_steps)  # type: ignore
    num_decay_steps = math.ceil(cfg.scheduler_args["cooldown"] * max_training_steps)  # type: ignore
    scheduler = get_scheduler(
        name="warmup_stable_decay",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_training_steps,
        scheduler_specific_kwargs={
            "num_decay_steps": num_decay_steps,
            "min_lr_ratio": cfg.scheduler_args["min_lr_ratio"],  # type: ignore
        },
    )
       
    if cfg.resume_from_checkpoint:
        training_state_path = os.path.join(cfg.resume_from_checkpoint, "training_state.pt")
        if not os.path.exists(training_state_path):
            if is_main_process():
                print(f"Warning: training_state.pt not found in {cfg.resume_from_checkpoint}. "
                      "Model weights loaded, but optimizer/scheduler/progress will NOT be resumed. "
                      "Training will start from scratch with loaded model weights.")
        else:
            if is_main_process():
                print(f"Resuming optimizer and scheduler state from checkpoint: {training_state_path}")
                checkpoint_state = torch.load(training_state_path, map_location=local_device)
                
                optimizer.load_state_dict(checkpoint_state["optimizer"])
                scheduler.load_state_dict(checkpoint_state["scheduler"]) 
                initial_optimizer_step = checkpoint_state["optimizer_step"]
                initial_epoch = checkpoint_state["epoch"]
                print(f"Resumed from epoch {initial_epoch}, optimizer step {initial_optimizer_step}")
            
            if distributed:
                torch.distributed.barrier() 

    state = {
        "model": model,
        "optimizer": optimizer,
        "tokenizer": tokenizer,
        "dataloader": dataloader,
        "distributed": distributed,
        "rank": rank,
        "scheduler": scheduler,
        "autocast_args": autocast_args,
        "initial_optimizer_step": initial_optimizer_step, 
        "initial_epoch": initial_epoch, 
    }

    cfg.world_size = world_size  # type: ignore
    return state, local_device

def sample_random_loops(random_loop_mu, random_loop_sigma, random_loop_min, random_loop_max):

    log_normal_sample = np.random.lognormal(
        mean=random_loop_mu,
        sigma=random_loop_sigma
    )
    num_loops = int(np.round(log_normal_sample))
    num_loops = max(random_loop_min, min(random_loop_max, num_loops))
    return num_loops

def train(state, device, cfg):
    model, optimizer = state["model"], state["optimizer"]
    model.train()

    accumulation_steps = cfg.batch_size // cfg.micro_batch_size
    # optimizer_step = 0

    optimizer_step = state["initial_optimizer_step"]
    initial_epoch = state["initial_epoch"]

    step_time = time.time()
    total_tokens = 0
    total_tokens_with_loss = 0
    tokens_in_step = 0

    metrics_to_agg_data_step = {
        "loss": [],
    }

    train_start_time = time.time()
    step_times = []  
    max_step_times = 50  

    loss_window = []  
    all_losses = []  
    all_avg_losses = []  
    all_reg_losses = []  

    optimizer_steps_history = [] 
   
    total_data_steps_per_epoch = len(state["dataloader"])
    

    full_run_total_optimizer_steps = math.ceil(total_data_steps_per_epoch / accumulation_steps) * cfg.epochs


    if state["rank"] == 0:
        print(f"Total data steps per epoch: {total_data_steps_per_epoch}")
        print(f"Full run total optimizer steps: {full_run_total_optimizer_steps}")
        print(f"Accumulation steps: {accumulation_steps}")
        if optimizer_step > 0:
            print(f"Resuming training: starting from epoch {initial_epoch}, optimizer step {optimizer_step}")

    for epoch in range(initial_epoch, cfg.epochs):

        if state["distributed"]:
            state["dataloader"].sampler.set_epoch(epoch)
        for data_step, inputs in enumerate(state["dataloader"], start=1):
           
            global_data_step = (epoch * total_data_steps_per_epoch) + data_step 

            if optimizer_step > 0 and global_data_step <= (optimizer_step * accumulation_steps):
                if state["rank"] == 0 and data_step % 100 == 0:
                    print(f"Skipping data_step {data_step} in epoch {epoch} as it's already processed (current optimizer_step: {optimizer_step})")
                continue
            # Realize the input and labels tensors.
            input_ids = inputs[cfg.token_id_col_name][:, :].to(dtype=torch.long, device=device, non_blocking=True)
            # Need to take into account the assistant and attention if sequences are being padded
            mask = ~(inputs["mask"].bool() & inputs["attention_mask"].bool())

            labels = torch.where(mask[:, :], -100, inputs[cfg.token_id_col_name][:, :]).to(
                dtype=torch.long, device=device, non_blocking=True
            )

            # print("input_ids:", input_ids)
            # print("mask", mask)
            # print("labels:", labels)

            total_tokens_with_loss += (labels != -100).sum().item()
            tokens_in_step += input_ids.numel()
            is_accumulating = data_step % accumulation_steps != 0

            total_ut_steps = sample_random_loops(
                  random_loop_mu=1.7,
                  random_loop_sigma=0.4,
                  random_loop_min=1,
                  random_loop_max=16
            )
  
            # The actual compute step of  Forward, loss, and backward computation:
            def tightly_scoped_fwd_bwd(model, input_ids, labels, reg_weight=0.1, total_ut_steps=total_ut_steps):
                with model.no_sync() if is_accumulating and state["distributed"] else nullcontext():
                    with torch.autocast(**state["autocast_args"]):
                        if torch.cuda.device_count() > 1:
                            model.module.config.total_ut_steps = total_ut_steps
                        else:
                            model.config.total_ut_steps = total_ut_steps
                            #print(f"Set total_ut_steps to {total_ut_steps}")
                        #print(total_ut_steps)
                        outputs, reg_loss = model(input_ids, labels=labels, reg_weight=reg_weight, total_ut_steps=total_ut_steps)

                    loss= outputs["loss"]
                    if reg_loss is None:
                        total_loss = loss
                    else:
                        total_loss = (1-reg_weight)*loss + reg_weight*reg_loss
                    (total_loss / accumulation_steps).backward()
                    return (total_loss.detach(),reg_loss.detach() if reg_loss is not None else 0)

            loss, reg_loss = tightly_scoped_fwd_bwd(model, input_ids, labels)

            # logging
            metrics_to_agg_data_step["loss"].append(loss.item())

            if not is_accumulating:

         
                loss_value = loss.item()
                loss_window.append(loss_value)
         
                if len(loss_window) > cfg.loss_window_size:
                    loss_window.pop(0)
        
                avg_loss = sum(loss_window) / len(loss_window) if loss_window else loss_value
                
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.max_grad_norm, norm_type=2.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state["scheduler"].step()
                optimizer_step += 1
        
                if cfg.save_interval > 0 and optimizer_step % cfg.save_interval == 0:
                    checkpoint_dir = f"{cfg.out_path}/{cfg.run_name}/checkpoint-{optimizer_step}"
                    if state["rank"] == 0:
                        print(f"\nSaving checkpoint at step {optimizer_step} to {checkpoint_dir}")
                        unwrapped_model = get_unwrapped_model(state)
            
                        unwrapped_model.save_pretrained(checkpoint_dir)
                        state["tokenizer"].save_pretrained(checkpoint_dir)
                        
                        checkpoint_state = {
                            "optimizer": optimizer.state_dict(),
                            "scheduler": state["scheduler"].state_dict(),
                            "optimizer_step": optimizer_step,
                            "epoch": epoch,
                        }
                        torch.save(checkpoint_state, f"{checkpoint_dir}/training_state.pt")

                        checkpoints = sorted(glob.glob(f"{cfg.out_path}/{cfg.run_name}/checkpoint-*"), key=os.path.getmtime)
                        if len(checkpoints) > 2:
                            # Keep cleanup on rank 0 to avoid multi-rank rmtree races.
                            shutil.rmtree(checkpoints[0], ignore_errors=True)

                    if state["distributed"]:
                        # Ensure all ranks stay in lockstep after checkpoint save/cleanup.
                        torch.distributed.barrier()
                if state["rank"] == 0:
                    current_time = time.time()
                    time_interval = (current_time - step_time) / accumulation_steps
                    tok_sec = tokens_in_step * cfg.world_size / (current_time - step_time)
                    
                    step_times.append(time_interval)
                    if len(step_times) > max_step_times:
                        step_times.pop(0)
      
                    avg_step_time = sum(step_times) / len(step_times) if step_times else time_interval
  
                    remaining_optimizer_steps = full_run_total_optimizer_steps - optimizer_step

                    estimated_remaining_seconds = remaining_optimizer_steps * avg_step_time
                    estimated_remaining = datetime.timedelta(seconds=int(estimated_remaining_seconds))
                    
                    elapsed_time = current_time - train_start_time
                    elapsed_timedelta = datetime.timedelta(seconds=int(elapsed_time))
                    
                    progress_pct = (optimizer_step / full_run_total_optimizer_steps * 100) if full_run_total_optimizer_steps > 0 else 0
           
                    all_losses.append(loss.item())
                    all_avg_losses.append(avg_loss)
                    all_reg_losses.append(reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss)
                    optimizer_steps_history.append(optimizer_step) 

                    print(
                        f"GPU: {model.device} | Epoch: {epoch+1}/{cfg.epochs} | Data Step: {data_step:4d}/{total_data_steps_per_epoch} | Updates: {optimizer_step:4d}/{full_run_total_optimizer_steps} ({progress_pct:5.1f}%)"
                        f" | Time/step: {time_interval:2.4f}s | Tok/sec={tok_sec:9.2f}"
                        f" | Loss: {loss:2.4f} | Avg Loss: {avg_loss:2.4f} | Reg Loss: {reg_loss:2.4f} | Grad-Norm {total_norm.item():2.4f}"
                        f" | Elapsed: {elapsed_timedelta} | ETA: {estimated_remaining}"
                    )
                    total_tokens += tokens_in_step * cfg.world_size
                    step_time = time.time()
                    tokens_in_step = 0

            if cfg.max_steps > 0 and global_data_step >= cfg.max_steps:
                break
        

        if cfg.max_steps > 0 and global_data_step >= cfg.max_steps:
            break
        
        unwrapped_model = get_unwrapped_model(state)
        unwrapped_model.save_pretrained(f"{cfg.out_path}/{cfg.run_name}/epoch{epoch}_checkpoint")
        state["tokenizer"].save_pretrained(f"{cfg.out_path}/{cfg.run_name}/epoch{epoch}_checkpoint")

    model.eval()
    
    # optimizer_steps -> optimizer_steps_history
    if state["rank"] == 0 and optimizer_steps_history:
  
        loss_data = {
            "optimizer_steps": optimizer_steps_history,
            "losses": all_losses,
            "avg_losses": all_avg_losses,
            "reg_losses": all_reg_losses,
        }
        os.makedirs(f"{cfg.out_path}/{cfg.run_name}", exist_ok=True)
        loss_json_path = f"{cfg.out_path}/{cfg.run_name}/loss_curve.json"
        with open(loss_json_path, "w") as f:
            json.dump(loss_data, f, indent=2)
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(optimizer_steps_history, all_losses, label="Loss", alpha=0.6, linewidth=1)
        plt.plot(optimizer_steps_history, all_avg_losses, label=f"Avg Loss (window={cfg.loss_window_size})", linewidth=2)
        plt.xlabel("Optimizer Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(optimizer_steps_history, all_reg_losses, label="Reg Loss", color="orange", linewidth=1)
        plt.xlabel("Optimizer Step")
        plt.ylabel("Reg Loss")
        plt.title("Regularization Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        if all_losses and max(all_losses) > min(all_losses):
            norm_losses = [(x - min(all_losses)) / (max(all_losses) - min(all_losses)) for x in all_losses]
        else:
            norm_losses = [0] * len(all_losses) if all_losses else []
        if all_reg_losses and max(all_reg_losses) > min(all_reg_losses):
            norm_reg_losses = [(x - min(all_reg_losses)) / (max(all_reg_losses) - min(all_reg_losses)) for x in all_reg_losses]
        else:
            norm_reg_losses = [0] * len(all_reg_losses) if all_reg_losses else []
    
        if norm_losses:
            plt.plot(optimizer_steps_history, norm_losses, label="Loss (normalized)", alpha=0.6, linewidth=1)
        if norm_reg_losses:
            plt.plot(optimizer_steps_history, norm_reg_losses, label="Reg Loss (normalized)", alpha=0.6, linewidth=1)
        
        plt.xlabel("Optimizer Step")
        plt.ylabel("Normalized Value")
        plt.title("All Metrics (Normalized)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        loss_curve_path = f"{cfg.out_path}/{cfg.run_name}/loss_curve.png"
        plt.savefig(loss_curve_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Loss curve saved to {loss_curve_path}")
        print(f"Loss data saved to {loss_json_path}")
    
    return state


####################################################################################################
# Main control loop
####################################################################################################
def validate(state, step, cfg, task="gsm8k"):
    # eval on-the-fly
    unwrapped_model = get_unwrapped_model(state)
    unwrapped_model.eval()
    results = evaluator.simple_evaluate(
        model=HFLM(
            pretrained=unwrapped_model,
            tokenizer=state["tokenizer"],
            batch_size=16,  # 16: 4:41
        ),
        tasks=[task],
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        system_instruction=DEFAULT_SYS_PROMPT,
        limit=100,
        # batch_size=13,
        num_fewshot=0,
        gen_kwargs={"num_steps": unwrapped_model.config.mean_recurrence},
    )
    print(make_table(results))
    results_by_step = {}
    if results is not None:
        if "groups" in results:
            print(make_table(results, "groups"))
        results_by_step[str(step)] = results["results"][task]

    os.makedirs(f"{cfg.out_path}/{cfg.run_name}", exist_ok=True)
    with open(f"{cfg.out_path}/{cfg.run_name}/eval.json", "a") as f:
        json.dump(results_by_step, f)

    unwrapped_model.train()


def main():
    """Encapsulates main scope away from import calls."""

    # Configuration loader
    cfg: CLISettings = CLI(CLISettings)

    # Print system setup
    if is_main_process():
        print("--------------------------------------------------------------------")
        print(f"------------------ Launching run {cfg.run_name}------------------")
        print("--------------------------------------------------------------------")
        print("--------------------------------------------------------------------")
        print(f"Platform: {sys.platform}, Python: {sys.version.split(' (')[0]}, PyTorch: {torch.__version__}")
        print(f"CPU threads: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.")
        driver = f"HIP/ROCM {torch.version.hip}" if torch.version.hip else f"CUDA: {torch.version.cuda}"
        print(f"GPU : {torch.cuda.get_device_name()}. {driver}.")

    # set flags
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Should be true anyway
    torch._dynamo.config.optimize_ddp = "python_reducer"
    torch._dynamo.config.compiled_autograd = False

    train_time = time.time()

    state, device = startup(cfg)
    state = train(state, device, cfg)
    # validate(state, "final", cfg)

    if cfg.save_final_checkpoint:
        unwrapped_model = get_unwrapped_model(state)
        unwrapped_model.save_pretrained(f"{cfg.out_path}/{cfg.run_name}/final_checkpoint")
        state["tokenizer"].save_pretrained(f"{cfg.out_path}/{cfg.run_name}/final_checkpoint")

    # Now exit
    if is_main_process():
        print("--------------------------------------------------------------------")
        print(f"Training time: {str(datetime.timedelta(seconds=time.time() - train_time))} ")
        max_alloc = f"{torch.cuda.max_memory_allocated(device) / float(1024**3):,.3f} GB"
        max_reserved = f"{torch.cuda.max_memory_reserved(device) / float(1024**3):,.3f} GB"
        print(f"Max. Mem allocated: {max_alloc}. Max. Mem reserved: {max_reserved}.")
        print("--------------------------------------------------------------------")
        dataset_save_dir = f"{cfg.out_path}/{cfg.run_name}/dataset"
        if os.path.exists(dataset_save_dir):
            shutil.rmtree(dataset_save_dir)


def shutdown():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    print(f"---------Total time: {str(datetime.timedelta(seconds=time.time() - global_start_time))} ---------")
    print("-----------------Shutdown complete.--------------------------")


def guarded_main():
    try:
        run_name = main()
        print("--------------------------------------------------------------------")
        print(f"Run {run_name} finished without error.")
    except BaseException:
        print("--------------------------------------------------------------------")
        print("Run finished with errors.")
        raise
    finally:
        shutdown()  # guarantee NCCL deconstruction


if __name__ == "__main__":
    guarded_main()
