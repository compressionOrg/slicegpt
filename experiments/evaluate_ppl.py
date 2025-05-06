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
from eval import eval_ppl


def evaluation_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估已剪枝的SliceGPT模型")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="原始模型名称或路径",
    )
    parser.add_argument(
        "--sliced-model-path",
        type=str,
        required=True,
        help="已剪枝模型的路径",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.0,
        help="模型剪枝时使用的稀疏度 (范围 [0, 1))",
    )
    parser.add_argument(
        "--round-interval",
        type=int,
        default=8,
        help="权重取整的间隔（最佳值可能取决于您的硬件）",
    )
    parser.add_argument(
        "--cal-datasets",
        type=str,
        nargs="+",
        help="用于计算困惑度的数据集列表",
        choices=["wikitext2", "ptb", "c4", "alpaca"],
        default=["wikitext2"],
    )
    parser.add_argument(
        "--ppl-eval-batch-size", 
        type=int, 
        default=8, 
        help="评估困惑度时的批量大小"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        help="使用的数据类型", 
        choices=["fp32", "fp16"], 
        default="fp16"
    )
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="使用accelerate将模型分布在多个GPU上进行评估。建议用于30B及以上参数的模型。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch设备。例如 'cpu', 'cuda', 'cuda:0'。如果未指定，默认为'cuda'（如果可用）或'cpu'。",
    )
    parser.add_argument(
        "--hf-token", 
        type=str, 
        default=os.getenv("HF_TOKEN", None),
        help="HuggingFace API令牌"
    )
    parser.add_argument(
        "--wandb-project", 
        type=str, 
        default="slicegpt_evaluation", 
        help="wandb项目名称"
    )
    parser.add_argument(
        "--no-wandb", 
        action="store_true", 
        help="禁用wandb"
    )

    return parser.parse_args()


def process_args(args):
    for arg, argv in vars(args).items():
        logging.debug(f"{arg} = {argv}")

    if not 0 <= args.sparsity < 1:
        raise argparse.ArgumentTypeError(f"稀疏度应在范围 [0, 1) 内")

    if args.device:
        config.device = torch.device(args.device)

    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    else:
        raise argparse.ArgumentTypeError(f"数据类型应为 'fp16' 或 'fp32' 之一")


def main() -> None:
    utils.configure_logging(log_to_console=True, log_to_file=True, level=logging.INFO)
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    args = evaluation_arg_parser()
    process_args(args)

    logging.info("开始评估已剪枝的SliceGPT模型")
    logging.info(f"PyTorch设备: {config.device}")
    logging.info(f"可用CUDA设备数量: {torch.cuda.device_count()}")

    try:
        wandb.init(project=args.wandb_project, config=args, mode="disabled" if args.no_wandb else None)
    except wandb.UsageError as e:
        # 如果用户未登录或进程在非shell环境中运行，wandb.init会抛出错误
        # 例如notebook、IDE、无shell进程等。在这种情况下，我们希望继续而不使用wandb。
        logging.info(f"初始化wandb失败: {e}，继续但不使用wandb")
        wandb.init(project=args.wandb_project, mode="disabled")

    # 加载已剪枝的模型
    logging.info(f"从 {args.sliced_model_path} 加载已剪枝模型")
    model_adapter, tokenizer = hf_utils.load_sliced_model(
        args.model,
        args.sliced_model_path,
        sparsity=args.sparsity,
        round_interval=args.round_interval,
        token=args.hf_token,
    )

    model = model_adapter.model

    # 将模型移至适当的设备
    if args.distribute_model:
        # 将模型分布在可用的GPU上
        logging.info("将模型分布在多个GPU上")
        gpu_utils.distribute_model(model_adapter)
    else:
        logging.info(f"将模型移至设备: {config.device}")
        model.to(config.device)

    # 计算模型参数数量
    param_count = sum(int(p.nelement()) for p in model.parameters())
    logging.info(f"模型参数数量: {param_count:,d}")

    # 对每个数据集进行评估
    results = {}
    for dataset_name in args.cal_datasets:
        logging.info(f"加载评估数据集: {dataset_name}")
        ppl_test = eval_ppl(model, tokenizer, dataset=dataset_name)
        logging.info(f"数据集 {dataset_name} 的困惑度: {ppl_test:.4f}")
        results[dataset_name] = ppl_test

    
    # 输出所有结果的摘要
    logging.info("评估结果摘要:")
    for dataset_name, ppl in results.items():
        logging.info(f"数据集 {dataset_name} 的困惑度: {ppl:.4f}")
    
    logging.info("评估完成")


if __name__ == "__main__":
    main()