#!/usr/bin/env python
"""
计算两个模型在同一批输入上输出的对数几率（logits）之间的 KL 散度，用来衡量模型差异。
推荐的使用方式如下：
1. 先使用精度较高（例如 fp16）的模型运行本程序并指定写文件路径，会把参考模型的 logits 写入压缩文件；
   示例：./llama_kl.py -m <fp16 model> -t <wiki.test.raw> -w <logits.gz>
2. 再使用待比较的量化模型运行本程序，并指定上一步生成的压缩文件作为读入路径；
   示例：./llama_kl.py -m <quantized model> -r <logits.gz>
程序会计算第二个模型相对于第一个模型的 KL 散度，并输出辅助统计信息。
更多可选项请查看 ./llama_kl.py --help。
source:https://gist.github.com/Ttl/0d51f739dc59254b4b2183e259c97d82
"""
import llama_cpp
import numpy as np
import sys
import argparse
import os.path
import struct
import ast
from scipy.special import rel_entr, softmax
import gzip
import pickle
from scipy.stats.mstats import mquantiles_cimj
from scipy.stats import bayes_mvs
from scipy.stats import t as student_t
import random
import time

def kl_div(p, q):
    """计算单个 token 上两个模型 logits 的 KL 散度。"""
    p = softmax(p)
    # After softmax the logits become a proper probability distribution (sum to 1).
    # softmax 将原始 logits 转换成概率分布（所有元素加和为 1）。
    q = softmax(q)
    # rel_entr computes p_i * log(p_i / q_i) for every token i; summing yields D_KL(p || q).
    # Intuitively, this measures how "surprised" the first model would be if the second model's
    # probabilities were the true answers for the same token.
    # rel_entr 逐元素计算 p_i * log(p_i / q_i)，对所有 token 求和即可得到 D_KL(p || q)。
    # 从直观上理解：如果真正的分布来自第二个模型，那么第一个模型会“多吃多少惊”。
    return np.sum(rel_entr(p, q))

def write_header(f, args, ctx, vocab_len, batch):
    """把运行配置写入到输出文件头部，方便之后复现计算环境。"""
    f.write("llama_kl_divergence_v1\n".encode('utf-8'))
    d = vars(args)
    d["n_ctx"] = ctx
    d["n_vocab"] = vocab_len
    d["n_batch"] = batch
    f.write((str(d)+"\n").encode('utf-8'))

def read_header(f):
    """从压缩文件开头读取并恢复运行配置。"""
    header = "llama_kl_divergence_v1\n".encode('utf-8')
    if f.read(len(header)) != header:
        raise ValueError("Invalid header in input logit file")
    args = ast.literal_eval(f.readline().decode('utf-8').strip())
    return args

def write_logits(f, tokens, logits):
    """把 token 序列及对应 logits 序列以二进制形式写入文件。"""
    f.write(struct.pack("<I", len(tokens)))
    f.write(struct.pack("<I", len(logits)))
    f.write(struct.pack("<I", len(logits[0])))
    t = np.array(tokens, dtype=np.uint32).tobytes()
    assert len(t) == 4 * len(tokens)
    f.write(t)
    l = np.array(logits, dtype=np.float32).tobytes()
    assert len(l) == 4 * len(logits) * len(logits[0])
    f.write(l)

def read_logits(f):
    """从文件中读取一段 token 序列及其 logits。到达文件尾时返回 (None, None)。"""
    n_tokens = f.read(4)
    if len(n_tokens) != 4:
        # EOF
        return None, None
    n_tokens = struct.unpack("<I", n_tokens)[0]
    n_logits = struct.unpack("<I",f.read(4))[0]
    n_vocab = struct.unpack("<I",f.read(4))[0]
    tokens = [int(i) for i in np.frombuffer(f.read(n_tokens * 4), dtype=np.uint32)]
    logits = np.frombuffer(f.read(n_logits * n_vocab * 4), dtype=np.float32).reshape(n_logits, n_vocab)
    return tokens, logits

def main(args):
    """主流程：准备模型、读取/生成输入、计算 KL 散度并输出统计结果。"""
    ctx = args.n_ctx

    read_file = None
    if args.read_path is not None:
        print(f"Computing KL-divergence against: {args.read_path}")
        read_file = gzip.open(args.read_path, "rb")
        input_args = read_header(read_file)
        ctx = input_args["n_ctx"]
        # 如果读取参考 logits，就使用参考运行使用的上下文大小，以保证比较公平。

    model = llama_cpp.Llama(model_path=args.model, n_ctx=ctx, n_batch=args.n_batch,
        logits_all=True, n_gpu_layers=args.n_gpu_layers, verbose=args.verbose)
    
    model_name = os.path.split(args.model)[1]
    # 创建 llama.cpp 推理对象，并打开 logits_all 以便拿到每个 token 的原始 logits。

    tokens = None
    if args.text_path and args.read_path is None:
        with open(args.text_path, "r") as f:
            prompt = f.read()
        print(f"Computing logits from text file: {args.text_path}")
        tokens = model.tokenize(prompt.encode('utf-8'))
        bos = model.token_bos()
        # 处理文本开头的特殊 token
        b = 1 if bos is not None else 0
        tokens = [tokens[i:i+ctx-b] for i in range(0, len(tokens), ctx-b)]
        random.seed(123)
        if bos is not None:
            for i in range(len(tokens)):
                tokens[i].insert(0, bos)
        # Improves error estimation during calculation as context correlation to previous
        # context is reduced compared to unshuffled order. Doesn't affect the final result.
        # 将文本分割为多个上下文窗口，并随机打乱顺序，减少相邻窗口之间的相关性，
        # 这样统计量（例如均值、置信区间）的估计更可靠。
        random.shuffle(tokens)

    write_file = None
    if args.write_path is not None:
        write_file = gzip.open(args.write_path, "wb")
        write_header(write_file, args, model.n_ctx(), model.n_vocab(), model.n_batch)
        # 若开启写文件模式，则先写入文件头信息，方便之后在同样设置下重现。

    def next_sample():
        """按需生成下一段要评估的 tokens 及参考 logits。"""
        if read_file is not None:
            while True:
                try:
                    t, logits = read_logits(read_file)
                except EOFError:
                    print("EOF at unexpected location")
                    return
                if t is None:
                    return
                yield logits, t
        elif tokens is not None:
            for t in tokens:
                yield None, t

    # Confidence interval bound
    alpha = 0.01
    # 置信区间的显著性水平，alpha 越小，区间越宽（即越保守）。

    kls = []
    top1 = 0
    top5 = 0
    top10 = 0
    eval_top5 = 0
    eval_top10 = 0
    samples = 0
    written = 0
    written_tokens = 0
    i = 0
    errors = 0

    max_tokens = args.n_tokens
    if max_tokens < 0:
        max_tokens = float('inf')
    try:
        for logits, chunk in next_sample():
            #print(model.detokenize(chunk))
            model.reset()
            output = model.eval(chunk)
            eval_logits = model.eval_logits
            # eval_logits contains raw logits for the current model on each position in the chunk.
            # We will later compare these logits against the reference logits to see how much
            # their predicted probability distributions diverge.
            # eval_logits 保存了当前模型对 chunk 中每个 token 的原始 logits，我们稍后会与参考值比较。
            if np.any(np.isnan(eval_logits)):
                errors += 1
                print("Nan in logits!")
                eval_logits = np.nan_to_num(eval_logits)
            if write_file:
                write_logits(write_file, model.eval_tokens, eval_logits)
                written_tokens += len(model.eval_tokens)
                written += 1
                print(f"[{written}/{len(tokens)}] tokens {written_tokens}")
            if logits is not None:
                # It would probably be better to throw away at least two first tokens
                # in the context window since those are always the same. It doesn't
                # matter that much though unlike in perplexity calculation since
                # we are comparing to reference.
                # This is really slow.
                # Each row in eval_logits/logits is a set of unnormalized log probabilities
                # for a single token. We compute D_KL between the two models for every token,
                # which tells us how many extra "nats" the eval model spends because it
                # disagrees with the reference distribution on that token.
                # 针对参考 logits，每个 token 计算一次 KL 散度，并把结果累加到 kls 列表中。
                # KL 数值越大，说明当前模型与参考模型在该 token 上的分布差距越大。
                new_kls = [kl_div(eval_logits[i], logits[i]) for i in range(len(logits))]
                if np.any(np.isnan(new_kls)):
                    errors += 1
                    print("Nan in computed kls!")
                    new_kls = np.nan_to_num(new_kls)
                kls.extend(new_kls)
                samples += len(logits)
                # This is even slower.
                eval_argmax = np.argmax(eval_logits, axis=-1)
                ref_argmax = np.argmax(logits, axis=-1)
                eval_part5 = np.argpartition(eval_logits, -5, axis=-1)[:,-5:]
                ref_part5 = np.argpartition(logits, -5, axis=-1)[:,-5:]
                eval_part10 = np.argpartition(eval_logits, -10, axis=-1)[:,-10:]
                ref_part10 = np.argpartition(logits, -10, axis=-1)[:,-10:]
                # 计算 top1 / top5 / top10 准确率：统计两个模型对最可能 token 的重合程度。
                # 同时也计算互相在对方 top-n 中的比例，以观察模型排名的一致性。
                top1 += sum([eval_argmax[i] == ref_argmax[i] for i in range(len(logits))])
                top5 += sum([ref_argmax[i] in eval_part5[i] for i in range(len(logits))])
                top10 += sum([ref_argmax[i] in eval_part10[i] for i in range(len(logits))])
                eval_top5 += sum([eval_argmax[i] in ref_part5[i] for i in range(len(logits))])
                eval_top10 += sum([eval_argmax[i] in ref_part10[i] for i in range(len(logits))])
                print(f"[{i}] kl {np.mean(kls):.4g}, top1 {top1 / samples:.4g}", flush=True)
            i += 1
            if samples >= max_tokens:
                print("Token limit reached")
                break
    except KeyboardInterrupt:
        print("Interrupted")

    if write_file:
        write_file.close()
        print(f"Finished writing file: {args.write_path}")
    if read_file:
        read_file.close()
        print(f"Finished reading file: {args.read_path}")

    def bin_conf(p, n, z):
        # Binomial distribution confidence bounds
        # Bayes estimator when p is degenerate
        # 针对二项分布的置信区间计算：在概率为 0 或 1 时使用贝叶斯估计进行平滑处理。
        if p == 0:
            p = 1 / (n + 2)
        if p == 1:
            p = 1 - 1 / (n + 2)
        return z * np.sqrt(p*(1-p)/n)

    if len(kls) > 0:
        # Aggregate the per-token divergences into summary statistics so we can gauge
        # overall similarity between the evaluated model and the reference model.
        # 对每个 token 的 KL 值进行统计汇总，帮助我们从整体上判断模型差异。
        z = student_t.ppf(1 - alpha/2, samples)

        print()
        print("Model:", model_name)
        bpw = 8 * llama_cpp.llama_model_size(model.model) / llama_cpp.llama_model_n_params(model.model)
        print(f"Size: {llama_cpp.llama_model_size(model.model) / 1024**3:.3g} GiB, (BPW {bpw:.2f})")
        print("Tokens:", samples)
        print("KL-divergence:")
        # Confidence interval assuming i.i.d, but that likely isn't true.
        # 简单假设样本独立同分布，计算一个近似的置信区间；实际中可能并不完全成立。
        m_conf = z*np.sqrt(np.mean([k**2 for k in kls])/len(kls))
        m, _, __ = bayes_mvs(kls, 1-alpha)
        print(f"mean: {m[0]:.6g}, [{m[1][0]:.6g} - {m[1][1]:.6g}]")
        q90 = np.quantile(kls, 0.90)
        q95 = np.quantile(kls, 0.95)
        q99 = np.quantile(kls, 0.99)
        q_bounds = mquantiles_cimj(kls, prob=[0.90, 0.95, 0.99])
        print(f"q90: {q90:.4g}, [{q_bounds[0][0]:.4g} - {q_bounds[1][0]:.4g}]")
        print(f"q95: {q95:.4g}, [{q_bounds[0][1]:.4g} - {q_bounds[1][1]:.4g}]")
        print(f"q99: {q99:.4g}, [{q_bounds[0][2]:.4g} - {q_bounds[1][2]:.4g}]")
        print(f"max: {np.max(kls):.4g}")
        print("Reference top token in eval top-n probability:")
        print(f"ref_top1: {top1 / samples:.4g} ± {bin_conf(top1/samples, samples, z):.4g}")
        print(f"ref_top5: {top5 / samples:.4g} ± {bin_conf(top5/samples, samples, z):.4g}")
        print(f"ref_top10: {top10 / samples:4g} ± {bin_conf(top10/samples, samples, z):.4g}")
        print("Eval top token in reference top-n probability:")
        print(f"eval_top5: {eval_top5 / samples:.4g} ± {bin_conf(eval_top5/samples, samples, z):.4g}")
        print(f"eval_top10: {eval_top10 / samples:4g} ± {bin_conf(eval_top10/samples, samples, z):.4g}")
        print(f"errors: {errors}")

        with open(model_name + ".kls.p", 'wb') as f:
            pickle.dump(kls, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='llama.cpp KL-divergence',
        description="Calculate KL-divergence of two models output logits on data set.\n"
        "First call the program with write_path and text_path using fp16 model.\n"
        "This writes logits to file. Then call the program with quantized model with read path\n"
        "KL-divergence to the first run is calculated\n")

    # 通过命令行参数控制输入、输出以及模型设置，新手可以在终端运行 `python demo/kl.py --help` 查看说明。
    parser.add_argument('-m', '--model', help="Model path", required=True)
    parser.add_argument('-t', '--text_path', help="Text dataset path", required=False)
    parser.add_argument('-c', '--n_ctx', help="Context size", default=512, type=int, required=False)
    parser.add_argument('-b', '--n_batch', help="Batch size", default=512, type=int, required=False)
    parser.add_argument('-w', '--write_path', help="Output logits file", required=False)
    parser.add_argument('-r', '--read_path', help="Input logits file", required=False)
    parser.add_argument('-n', '--n_tokens', help="Number of tokens to evaluate. (-1 = whole file)", default=-1, type=int, required=False)
    parser.add_argument('-ngl', '--n-gpu-layers', help="Number of GPU layers", default=0, type=int, required=False)
    parser.add_argument('-v', '--verbose', help="Verbose output", action="store_true")
    args = parser.parse_args()

    if args.read_path is None and args.text_path is None:
        # 如果没有提供文本输入，也没有参考 logits，就无法完成任务，直接提示用户。
        print("Either text dataset or input logit file should be specified")
    if args.write_path is None and args.read_path is None:
        # 既不读取也不写入文件，没有任何输出意义，因此提前终止。
        print("At least one of read_path or write_path needs to be specified")
        sys.exit(1)
    if args.write_path is not None and os.path.exists(args.write_path):
        # 防止覆盖已有文件，初学者应手动删除或换一个输出路径。
        print(f"write_path {args.write_path} already exists")
        sys.exit(1)
    main(args)
