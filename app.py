import streamlit as st
import sys
import os.path
import argparse
import ast
import struct  # 添加缺失的 struct 导入
from scipy.special import rel_entr, softmax
import gzip
import pickle
from scipy.stats.mstats import mquantiles_cimj
from scipy.stats import bayes_mvs
from scipy.stats import t as student_t
import random
import time

# 导入 llama_cpp，如果导入失败则显示错误信息
try:
    import llama_cpp
    import llama_cpp._internals as _llama_internals

    if not getattr(_llama_internals.LlamaModel, "_safe_close_patched", False):
        _original_llama_close = _llama_internals.LlamaModel.close

        def _safe_llama_close(self):
            try:
                _original_llama_close(self)
            except AttributeError:
                sampler = getattr(self, "sampler", None)
                if sampler is not None:
                    custom_samplers = list(getattr(self, "custom_samplers", []))
                    for i, _ in reversed(custom_samplers):
                        llama_cpp.llama_sampler_chain_remove(sampler, i)
                    if hasattr(self, "custom_samplers"):
                        self.custom_samplers.clear()
                exit_stack = getattr(self, "_exit_stack", None)
                if exit_stack is not None:
                    exit_stack.close()

        _llama_internals.LlamaModel.close = _safe_llama_close
        _llama_internals.LlamaModel._safe_close_patched = True
except ImportError:
    st.error("请确保已安装 llama-cpp-python: pip install llama-cpp-python")
    sys.exit(1)

import numpy as np

def kl_div(p, q):
    """计算单个 token 上两个模型 logits 的 KL 散度。"""
    p = softmax(p)
    q = softmax(q)
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

class ModelManager:
    """模型管理器，用于正确处理模型的创建和销毁"""
    def __init__(self, model_path, ctx, batch_size, gpu_layers=0):
        self.model = None
        self.model_path = model_path
        self.ctx = ctx
        self.batch_size = batch_size
        self.gpu_layers = gpu_layers
        
    def __enter__(self):
        status_text = st.empty()
        status_text.text("正在加载模型...")
        
        # 禁用 verbose 以避免过多的输出信息
        self.model = llama_cpp.Llama(
            model_path=self.model_path,
            n_ctx=self.ctx,
            n_batch=self.batch_size,
            logits_all=True,
            n_gpu_layers=self.gpu_layers,
            verbose=False
        )
        
        status_text.text("✅ 模型加载完成")
        return self.model
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.model is not None:
            # 显式关闭模型以避免资源泄漏
            try:
                self.model.close()
            except AttributeError:
                pass
            finally:
                self.model = None

def run_kl_analysis(args):
    """主流程：准备模型、读取/生成输入、计算 KL 散度并输出统计结果。"""
    ctx = args.n_ctx
    progress_bar = st.progress(0)
    status_text = st.empty()

    read_file = None
    if args.read_path is not None:
        status_text.text(f"正在加载参考 logits 文件: {args.read_path}")
        read_file = gzip.open(args.read_path, "rb")
        input_args = read_header(read_file)
        ctx = input_args["n_ctx"]

    # 使用模型管理器来处理模型的创建和销毁
    with ModelManager(args.model, ctx, args.n_batch, args.n_gpu_layers) as model:
        if model is None:
            st.error("❌ 模型加载失败")
            return
            
        model_name = os.path.split(args.model)[1]

        tokens = None
        if args.text_path and args.read_path is None:
            status_text.text(f"正在读取文本文件: {args.text_path}")
            with open(args.text_path, "r") as f:
                prompt = f.read()
            tokens = model.tokenize(prompt.encode('utf-8'))
            bos = model.token_bos()
            b = 1 if bos is not None else 0
            tokens = [tokens[i:i+ctx-b] for i in range(0, len(tokens), ctx-b)]
            random.seed(123)
            if bos is not None:
                for i in range(len(tokens)):
                    tokens[i].insert(0, bos)
            random.shuffle(tokens)

        write_file = None
        if args.write_path is not None:
            write_file = gzip.open(args.write_path, "wb")
            # 注意：这里需要先获取模型信息，所以需要在 with 语句内部调用
            vocab_len = model.n_vocab()
            batch_size = args.n_batch
            write_header(write_file, args, model.n_ctx(), vocab_len, batch_size)

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
                for i, t in enumerate(tokens):
                    progress_bar.progress((i + 1) / len(tokens))
                    status_text.text(f"处理文本片段 {i+1}/{len(tokens)}")
                    yield None, t

        alpha = 0.01
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

        max_tokens = args.n_tokens if args.n_tokens >= 0 else float('inf')
        
        status_text.text("开始计算 KL 散度...")
        try:
            for logits, chunk in next_sample():
                model.reset()
                output = model.eval(chunk)
                eval_logits = model.eval_logits
                
                if np.any(np.isnan(eval_logits)):
                    errors += 1
                    print("Nan in logits!")
                    eval_logits = np.nan_to_num(eval_logits)
                    
                if write_file:
                    # 注意：这里需要确保模型有 eval_tokens 属性
                    if hasattr(model, 'eval_tokens'):
                        write_logits(write_file, model.eval_tokens, eval_logits)
                        written_tokens += len(model.eval_tokens)
                        written += 1
                    
                if logits is not None:
                    new_kls = [kl_div(eval_logits[i], logits[i]) for i in range(len(logits))]
                    if np.any(np.isnan(new_kls)):
                        errors += 1
                        print("Nan in computed kls!")
                        new_kls = np.nan_to_num(new_kls)
                    kls.extend(new_kls)
                    samples += len(logits)
                    
                    eval_argmax = np.argmax(eval_logits, axis=-1)
                    ref_argmax = np.argmax(logits, axis=-1)
                    eval_part5 = np.argpartition(eval_logits, -5, axis=-1)[:,-5:]
                    ref_part5 = np.argpartition(logits, -5, axis=-1)[:,-5:]
                    eval_part10 = np.argpartition(eval_logits, -10, axis=-1)[:,-10:]
                    ref_part10 = np.argpartition(logits, -10, axis=-1)[:,-10:]
                    
                    top1 += sum([eval_argmax[i] == ref_argmax[i] for i in range(len(logits))])
                    top5 += sum([ref_argmax[i] in eval_part5[i] for i in range(len(logits))])
                    top10 += sum([ref_argmax[i] in eval_part10[i] for i in range(len(logits))])
                    eval_top5 += sum([eval_argmax[i] in ref_part5[i] for i in range(len(logits))])
                    eval_top10 += sum([eval_argmax[i] in ref_part10[i] for i in range(len(logits))])
                    
                if samples >= max_tokens:
                    break
                    
        except KeyboardInterrupt:
            status_text.text("计算被用户中断")
        
        if write_file:
            write_file.close()
        if read_file:
            read_file.close()

        def bin_conf(p, n, z):
            if p == 0:
                p = 1 / (n + 2)
            if p == 1:
                p = 1 - 1 / (n + 2)
            return z * np.sqrt(p*(1-p)/n)

        if len(kls) > 0:
            z = student_t.ppf(1 - alpha/2, samples)
            
            st.subheader("📊 计算结果")
            
            with st.expander("模型信息"):
                st.write(f"**模型名称**: {model_name}")
                
                # 安全获取模型大小信息
                try:
                    model_size = llama_cpp.llama_model_size(model.model)
                    params = llama_cpp.llama_n_params(model.model)
                    bpw = 8 * model_size / params if params > 0 else 0
                    st.write(f"**模型大小**: {model_size / 1024**3:.3g} GiB (BPW {bpw:.2f})")
                except:
                    st.write("**模型大小**: 无法获取")
                
                st.write(f"**Token 数量**: {samples}")
            
            with st.expander("🔥 KL 散度统计"):
                m, _, __ = bayes_mvs(kls, 1-alpha)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("均值", f"{m[0]:.6g}")
                    z_interval = z * np.sqrt(np.mean([k**2 for k in kls])/len(kls))
                    st.caption(f"置信区间: [{m[1][0]:.6g} - {m[1][1]:.6g}]")
                
                with col2:
                    q90 = np.quantile(kls, 0.90)
                    q95 = np.quantile(kls, 0.95)
                    st.metric("Q90", f"{q90:.4g}")
                    st.metric("Q95", f"{q95:.4g}")
                
                with col3:
                    q99 = np.quantile(kls, 0.99)
                    max_kl = np.max(kls)
                    st.metric("Q99", f"{q99:.4g}")
                    st.metric("最大值", f"{max_kl:.4g}")

            with st.expander("🎯 准确率统计"):
                z_val = student_t.ppf(1 - alpha/2, samples)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**参考模型 Top Token 在评估模型中的位置**")
                    st.metric(f"Top 1 准确率", f"{top1 / samples:.4g}")
                    st.caption(f"置信区间: ±{bin_conf(top1/samples, samples, z_val):.4g}")
                    
                    st.metric(f"Top 5 准确率", f"{top5 / samples:.4g}")
                    st.caption(f"置信区间: ±{bin_conf(top5/samples, samples, z_val):.4g}")
                    
                    st.metric(f"Top 10 准确率", f"{top10 / samples:.4g}")
                    st.caption(f"置信区间: ±{bin_conf(top10/samples, samples, z_val):.4g}")
                
                with col2:
                    st.write("**评估模型 Top Token 在参考模型中的位置**")
                    st.metric("Top 5 覆盖率", f"{eval_top5 / samples:.4g}")
                    st.caption(f"置信区间: ±{bin_conf(eval_top5/samples, samples, z_val):.4g}")
                    
                    st.metric("Top 10 覆盖率", f"{eval_top10 / samples:.4g}")
                    st.caption(f"置信区间: ±{bin_conf(eval_top10/samples, samples, z_val):.4g}")
                
                if errors > 0:
                    st.warning(f"⚠️ 检测到 {errors} 个错误（NaN 值）")

            with st.expander("💾 数据导出"):
                if st.button("保存 KL 散度数据"):
                    output_file = f"{model_name}.kls.p"
                    with open(output_file, 'wb') as f:
                        pickle.dump(kls, f)
                    st.success(f"KL 散度数据已保存到 {output_file}")
        else:
            st.error("没有可用的 KL 散度计算结果")

# Streamlit 界面设置
st.set_page_config(
    page_title="LLM KL 散度分析工具",
    page_icon="🧮",
    layout="wide"
)

st.title("🧮 LLM KL 散度分析工具")
st.markdown("""
这是一个用于比较两个大型语言模型输出的对数几率（logits）之间差异的工具。
KL 散度可以衡量两个概率分布之间的差异，帮助我们量化不同版本或配置的模型之间的变化程度。

## 使用说明
1. **写入模式**: 首先使用高精度模型运行并保存 logits 到文件
2. **读取模式**: 然后用量化模型运行，读取之前保存的文件来计算 KL 散度

⚠️ **注意**: 如果遇到模型加载错误，请检查 llama_cpp 版本兼容性。
""")

# 侧边栏参数设置
st.sidebar.header("⚙️ 参数设置")

mode = st.sidebar.selectbox(
    "运行模式",
    ["写入模式", "读取模式"],
    help="选择要运行的模式：写入模式生成 logits 文件，读取模式计算 KL 散度"
)

args = argparse.Namespace()

# 基础参数
st.sidebar.subheader("基础设置")
args.model = st.sidebar.text_input(
    "模型路径",
    placeholder="/path/to/model.gguf",
    help="指定 LLM 模型的文件路径 (GGUF 格式)"
)
args.n_ctx = st.sidebar.number_input(
    "上下文大小",
    min_value=1,
    max_value=8192,
    value=512,
    help="模型的最大上下文长度"
)
args.n_batch = st.sidebar.number_input(
    "批处理大小", 
    min_value=1,
    max_value=2048,
    value=512,
    help="推理时的批处理大小"
)

# GPU 层数
st.sidebar.subheader("GPU 加速")
args.n_gpu_layers = st.sidebar.slider(
    "GPU 层数",
    min_value=0,
    max_value=100,
    value=0,
    help="卸载到 GPU 的层数数量（0 表示全部使用 CPU）"
)

# Token 数量限制
st.sidebar.subheader("评估设置")
args.n_tokens = st.sidebar.number_input(
    "Token 限制",
    min_value=-1,
    max_value=100000,
    value=-1,
    help="要评估的 Token 数量（-1 表示全部处理）"
)

# 预设可选参数，确保属性在两种模式下都存在
args.text_path = None
args.read_path = None
args.write_path = None

# 根据模式显示不同的参数
if mode == "写入模式":
    st.sidebar.subheader("📝 写入设置")
    args.text_path = st.sidebar.text_input(
        "文本数据集路径",
        placeholder="/path/to/dataset.txt",
        help="包含输入文本的文件路径"
    )
    args.write_path = st.sidebar.text_input(
        "输出 logits 文件路径",
        placeholder="logits.gz",
        help="保存模型输出的压缩 logits 文件路径"
    )
    args.read_path = None
else:
    st.sidebar.subheader("📖 读取设置")
    args.read_path = st.sidebar.text_input(
        "参考 logits 文件路径",
        placeholder="/path/to/logits.gz",
        help="之前生成的参考模型 logits 文件路径"
    )
    args.write_path = None

args.verbose = False

# 参数验证
if st.sidebar.button("🚀 开始分析", type="primary"):
    if not args.model:
        st.error("❌ 请指定模型路径")
    elif mode == "写入模式" and (not args.text_path or not args.write_path):
        st.error("❌ 写入模式下需要指定文本数据集和输出文件路径")
    elif mode == "读取模式" and not args.read_path:
        st.error("❌ 读取模式下需要指定参考 logits 文件路径")
    else:
        try:
            # 检查输入文件是否存在
            if args.text_path and not os.path.exists(args.text_path):
                raise FileNotFoundError(f"文本文件不存在: {args.text_path}")
            if args.read_path and not os.path.exists(args.read_path):
                raise FileNotFoundError(f"参考 logits 文件不存在: {args.read_path}")
            
            # 检查输出路径是否已存在
            if args.write_path and os.path.exists(args.write_path):
                raise FileExistsError(f"输出文件已存在，请删除或更换路径: {args.write_path}")
                
            st.info("正在准备分析环境...")
            run_kl_analysis(args)
            
        except Exception as e:
            st.error(f"❌ 运行时错误: {str(e)}")
