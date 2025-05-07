# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os
import pathlib

import torch
import wandb

from slicegpt import data_utils, gpu_utils, hf_utils, utils
from slicegpt.config import config


def ppl_arg_parser(interactive: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model to load",
    )
    parser.add_argument(
        "--sliced-model-path",
        type=str,
        help="Path to load the sliced model and tokenizer from",
        required=True,
    )
    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument(
        "--cal_datasets",
        type=str,
        nargs="+",
        help="Datasets to calculate perplexity on. Can specify multiple datasets.",
        choices=["wikitext2", "ptb", "c4", "alpaca"],
        default=["wikitext2"],
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluating the perplexity.")
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )
    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))
    parser.add_argument('--wandb-project', type=str, default="slicegpt", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="PyTorch device to use. Example values are 'cpu', 'cuda', 'cuda:0'. If not specified it will be defaulted to 'cuda' if available and 'cpu' otherwise.",
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="Sparsity level of the loaded model (in the range [0, 1))"
    )
    parser.add_argument(
        "--round-interval",
        type=int,
        default=8,
        help="Interval for rounding the weights (should match the value used during slicing)",
    )

    return parser.parse_args() if interactive else parser.parse_args('')


def process_args(args):
    for arg, argv in vars(args).items():
        logging.debug(f'{arg} = {argv}')

    if args.device:
        config.device = torch.device(args.device)

    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    else:
        raise argparse.ArgumentTypeError(f"Data type should be one of 'fp16', 'fp32'")


def ppl_main(args: argparse.Namespace) -> None:
    logging.info("Evaluating SliceGPT model perplexity.")
    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")
    logging.info(f"Testing on datasets: {', '.join(args.cal_datasets)}")

    try:
        wandb.init(project=args.wandb_project, config=args, mode='disabled' if args.no_wandb else None)
    except wandb.UsageError as e:
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb')
        wandb.init(project=args.wandb_project, mode='disabled')

    # 加载剪枝后的模型
    model_adapter, tokenizer = hf_utils.load_sliced_model(
        args.model,
        args.sliced_model_path,
        sparsity=args.sparsity,
        round_interval=args.round_interval,
        token=args.hf_token,
    )
    
    model = model_adapter.model

    # 将模型放到指定设备上
    if args.distribute_model:
        # 在多个 GPU 上分布模型
        gpu_utils.distribute_model(model_adapter)
    else:
        model.to(config.device)

    # 计算模型参数数量
    param_count = sum(int(p.nelement()) for p in model.parameters())
    logging.info(f'Model parameters: {param_count:,d}')
    
    # 创建一个字典来存储所有数据集的 PPL 结果
    all_ppls = {}
    
    # 循环处理每个数据集
    for dataset_name in args.cal_datasets:
        logging.info(f"Evaluating perplexity on dataset: {dataset_name}")
        
        # 准备测试数据集
        dataset = data_utils.get_dataset(dataset_name)
        test_dataset = dataset["test"]
        test_loader = data_utils.prepare_test_dataloader(
            dataset=test_dataset, tokenizer=tokenizer, batch_size=args.batch_size
        )

        # 评估模型的 PPL
        dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        logging.info(f'Sliced model perplexity on {dataset_name}: {dataset_ppl:.4f}')
        
        # 记录结果
        all_ppls[dataset_name] = dataset_ppl
        wandb.log({f"sliced_ppl_{dataset_name}": dataset_ppl})
    
    # 输出所有数据集的 PPL 结果汇总
    logging.info("=" * 50)
    logging.info("Perplexity Results Summary:")
    for dataset_name, ppl in all_ppls.items():
        logging.info(f"{dataset_name}: {ppl:.4f}")
    logging.info("=" * 50)


if __name__ == "__main__":
    utils.configure_logging(log_to_console=True, log_to_file=True, level=logging.INFO)
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    args = ppl_arg_parser()
    process_args(args)
    ppl_main(args)
