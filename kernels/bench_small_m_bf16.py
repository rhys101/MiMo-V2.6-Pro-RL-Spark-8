"""Micro-benchmark: small-M BF16 linear (o_proj / lm_head shapes) cuBLAS vs Triton split-N kernel."""
import torch, triton, triton.language as tl

@triton.autotune(configs=[triton.Config({"BN": bn, "BK": bk}, num_warps=w, num_stages=s)
                          for bn in (16, 32, 64) for bk in (128, 256, 512) for w in (4, 8) for s in (3, 4)],
                 key=["M_PAD", "N", "K"])
@triton.jit
def _small_m_linear(x_ptr, w_ptr, y_ptr, M, N, K, M_PAD: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rm = tl.arange(0, M_PAD)
    acc = tl.zeros((M_PAD, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0)
        acc += tl.dot(x, tl.trans(w))
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))

def small_m_linear(x, w):
    M, K = x.shape; N = w.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    _small_m_linear[lambda m: (triton.cdiv(N, m["BN"]),)](x, w, y, M, N, K, M_PAD=max(16, triton.next_power_of_2(M)))
    return y

def bench(fn, ws, reps=5):
    """Time one call per distinct weight (cold L2, like consecutive layers), inside a CUDA graph."""
    for w in ws[:4]: fn(w)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for w in ws: fn(w)
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(reps): g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / (reps * len(ws)) * 1e3  # us

torch.manual_seed(0)
shapes = {"o_proj (TP8)": (6144, 2048, 70), "lm_head (TP8)": (19072, 6144, 4)}
for name, (N, K, L) in shapes.items():
    ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / 50 for _ in range(L)]; w = ws[0]
    for M in (1, 8, 16, 32, 64):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, w); out = small_m_linear(x, w)
        err = ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
        tc, tt = bench(lambda w: torch.nn.functional.linear(x, w), ws), bench(lambda w: small_m_linear(x, w), ws)
        gb = N * K * 2 / 1e9
        print(f"{name:20s} M={M:3d}  cuBLAS {tc:7.1f}us ({gb/tc*1e6:5.0f} GB/s)  triton {tt:7.1f}us ({gb/tt*1e6:5.0f} GB/s)  rel_err {err:.1e}")
