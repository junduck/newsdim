"""Benchmark Tagger throughput and latency on CPU.

Usage:
  uv run python scripts/benchmark.py
  uv run python scripts/benchmark.py --device cpu --batch-sizes 1,8,32,128
"""

import argparse
import time

import numpy as np

from newsdim import Tagger

SAMPLE_TEXTS = [
    "煤炭板块盘初走强，大有能源涨停，美锦能源冲击涨停，安源煤业、郑州煤电、甘肃能化纷纷上扬。",
    "公司股东范秀莲将持有的本公司股份245万股质押给国金证券股份有限公司，质押起始日2025/03/06。",
    "瑞银证券发表机构观点称，人工智能赋能中国各行业发展的影响会在未来两年到三年体现。",
    "央行宣布降准50个基点，释放长期资金约1万亿元。",
    "分众传媒公告称，公司拟以发行股份及支付现金方式购买张继学等持有的新潮传媒100%股权。",
]


def bench_single(tagger: Tagger, texts: list[str], n_rounds: int = 3) -> dict:
    times = []
    for _ in range(n_rounds):
        t0 = time.perf_counter()
        for text in texts:
            tagger.score(text)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    total = len(texts) * n_rounds
    avg = np.mean(times)
    return {
        "total_calls": total,
        "total_time_s": round(float(avg), 3),
        "avg_latency_ms": round(float(avg / len(texts)) * 1000, 1),
        "throughput_qps": round(float(len(texts) / avg), 1),
    }


def bench_batch(tagger: Tagger, texts: list[str], batch_size: int, n_rounds: int = 3) -> dict:
    times = []
    for _ in range(n_rounds):
        t0 = time.perf_counter()
        tagger.score_batch(texts, batch_size=batch_size)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    avg = np.mean(times)
    return {
        "batch_size": batch_size,
        "total_time_s": round(float(avg), 3),
        "avg_latency_ms": round(float(avg / len(texts)) * 1000, 1),
        "throughput_qps": round(float(len(texts) / avg), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Tagger")
    parser.add_argument("--device", default="cpu", help="Device: cpu, mps, cuda")
    parser.add_argument("--n", type=int, default=100, help="Number of texts per round")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds")
    parser.add_argument("--batch-sizes", default="1,8,32,64,128", help="Comma-separated batch sizes")
    args = parser.parse_args()

    tagger = Tagger(device=args.device)

    texts = (SAMPLE_TEXTS * ((args.n // len(SAMPLE_TEXTS)) + 1))[:args.n]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print(f"device: {args.device}")
    print(f"texts per round: {args.n}")
    print(f"rounds: {args.rounds}")
    print(f"encoder: {tagger.encoder.model_name}")
    print(f"head weight shape: {tagger.head.weight.shape}")
    print()

    # Warmup
    print("warming up...")
    tagger.score("warmup text")
    tagger.score_batch(["warmup"] * 8, batch_size=8)
    print()

    # Single-text benchmark
    print("=== Single text (sequential) ===")
    r = bench_single(tagger, texts, args.rounds)
    print(f"  {r['total_calls']} calls in {r['total_time_s']:.3f}s")
    print(f"  latency: {r['avg_latency_ms']:.1f} ms/query")
    print(f"  throughput: {r['throughput_qps']:.1f} qps")
    print()

    # Batch benchmarks
    print("=== Batch ===")
    print(f"  {'batch':>6} | {'latency':>10} | {'throughput':>10}")
    print(f"  {'':>6} | {'ms/query':>10} | {'qps':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}")
    for bs in batch_sizes:
        r = bench_batch(tagger, texts, batch_size=bs, n_rounds=args.rounds)
        print(f"  {r['batch_size']:>6} | {r['avg_latency_ms']:>10.1f} | {r['throughput_qps']:>10.1f}")


if __name__ == "__main__":
    main()
