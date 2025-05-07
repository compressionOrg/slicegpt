# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os
import pathlib
import time
from typing import Dict, Any, List, Optional, Union

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from slicegpt import data_utils, gpu_utils, hf_utils, utils
from slicegpt.config import config
from data import get_evaluation_dataloader


@torch.no_grad()
def evaluate_perplexity(
    model: torch.nn.Module, 
    dataset: torch.Tensor, 
    limit: int = -1, 
    pad_token_id: Optional[int] = None
) -> float:
    """
    评估模型在给定数据集上的困惑度。
    
    Args:
        model: 要评估的模型
        dataset: 输入ID张量，形状为 [batch, sequence length]
        limit: 评估样本数量限制，-1表示无限制
        pad_token_id: 填充标记ID，用于在损失计算中忽略
        
    Returns:
        困惑度值
    """
    # 同步所有GPU，确保所有操作已完成
    sync_gpus()
    
    start_time = time.time()
    
    model.eval()
    
    # 设置损失函数
    if pad_token_id:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    
    nsamples, seqlen = dataset.size()
    nlls = []
    
    logging.info("评估困惑度...")
    
    # 使用tqdm显示进度
    for i in tqdm(range(nsamples), desc="评估样本"):
        if limit > 0 and i >= limit:
            break
            
        # 准备输入和标签
        input_ids = dataset[i:i+1, :-1].to(model.device)
        labels = dataset[i:i+1, 1:].to(model.device)
        
        # 前向传播
        outputs = model(input_ids=input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        # 计算损失
        nll = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)).float()
        
        # 处理填充标记
        if pad_token_id:
            mask = labels.view(-1) != pad_token_id
            neg_log_likelihood = (nll * mask).sum() / mask.sum()
        else:
            neg_log_likelihood = nll.mean() * seqlen
            
        nlls.append(neg_log_likelihood)
        
        # 释放内存
        del input_ids, labels, logits, nll
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 计算困惑度
    nlls_tensor = torch.stack(nlls)
    ppl = torch.exp(nlls_tensor.mean())
    
    # 再次同步GPU
    sync_gpus()
    
    # 记录评估时间
    elapsed = time.time() - start_time
    logging.info(
        "评估耗时: %s",
        time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
    )
    
    # 记录内存使用情况
    if torch.cuda.is_available():
        logging.info("GPU内存使用: %s MiB\n", torch.cuda.memory_allocated()/1024/1024)
    
    return ppl.item()


def sync_gpus() -> None:
    """同步所有GPU，确保所有操作已完成，这对于正确测量延迟/吞吐量很重要。"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device=i)


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
        "--datasets",
        type=str,
        nargs="+",
        help="用于计算困惑度的数据集列表",
        choices=["wikitext2", "ptb", "c4", "alpaca"],
        default=["wikitext2"],
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="评估困惑度时的批量大小"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="评估样本数量限制，-1表示无限制",
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


def process_args(args: argparse.Namespace) -> None:
    """处理命令行参数并设置全局配置"""
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


def evaluate_datasets(
    model: torch.nn.Module, 
    tokenizer: Any, 
    model_name: str, 
    datasets: List[str], 
    limit: int = -1, 
    batch_size: int = 8,
    pad_token_id: Optional[int] = None
) -> Dict[str, float]:
    """
    评估模型在多个数据集上的困惑度
    
    Args:
        model: 要评估的模型
        tokenizer: 分词器
        model_name: 模型名称，用于缓存
        datasets: 数据集名称列表
        limit: 评估样本数量限制，-1表示无限制
        batch_size: 批处理大小
        pad_token_id: 填充标记ID，用于在损失计算中忽略
        
    Returns:
        包含每个数据集困惑度的字典
    """
    results = {}
    
    for dataset_name in datasets:
        logging.info(f"加载评估数据集: {dataset_name}")
        try:
            # 创建缓存目录（如果不存在）
            os.makedirs("cache", exist_ok=True)
            
            # 尝试从缓存加载数据集
            cache_testloader = f"cache/{dataset_name}_testloader_{model_name.replace('/', '_')}_all.cache"
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader, weights_only=False)
                logging.info(f"从缓存加载数据集: {cache_testloader}")
            else:
                testloader = get_evaluation_dataloader(dataset_name, tokenizer, batch_size=batch_size)
                torch.save(testloader, cache_testloader)
                logging.info(f"数据集已保存到缓存: {cache_testloader}")
            
            # 获取输入ID张量
            testenc = testloader.input_ids
            
            # 评估困惑度
            logging.info(f"开始评估数据集 {dataset_name} 的困惑度...")
            ppl = evaluate_perplexity(model, testenc, limit, pad_token_id)
            logging.info(f"数据集 {dataset_name} 的困惑度: {ppl:.4f}")
            
            # 记录结果
            results[dataset_name] = ppl
            
        except Exception as e:
            logging.error(f"评估数据集 {dataset_name} 时出错: {e}")
    
    return results


def main() -> None:
    """主函数，处理命令行参数并运行评估"""
    utils.configure_logging(log_to_console=True, log_to_file=True, level=logging.INFO)
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    
    # 设置环境变量以避免内存碎片化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
        device = None  # 分布式模型的设备需要动态获取
    else:
        logging.info(f"将模型移至设备: {config.device}")
        model.to(config.device)
        device = config.device

    # 计算模型参数数量
    param_count = sum(int(p.nelement()) for p in model.parameters())
    logging.info(f"模型参数数量: {param_count:,d}")

    # 对每个数据集进行评估
    pad_token_id = getattr(model.config, "pad_token_id", None)
    results = evaluate_datasets(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model,
        datasets=args.datasets,
        limit=args.limit,
        batch_size=args.batch_size,
        pad_token_id=pad_token_id,
    )

    # 输出所有结果的摘要
    logging.info("评估结果摘要:")
    for dataset_name, ppl in results.items():
        logging.info(f"数据集 {dataset_name} 的困惑度: {ppl:.4f}")
        wandb.log({f"{dataset_name}_ppl": ppl})
    
    logging.info("评估完成")


if __name__ == "__main__":
    main()