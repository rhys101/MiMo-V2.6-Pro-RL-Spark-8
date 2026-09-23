"""8-rank bf16 all-reduce bandwidth sweep for choosing NCCL protocol/channels (prefill-sized messages)."""
import datetime, json, os, sys, torch, torch.distributed as dist
torch.cuda.set_device(0)
dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=5), device_id=torch.device("cuda:0"))
rank, world = dist.get_rank(), dist.get_world_size()
rows = []
for mb in (1, 4, 16, 50, 100):
    x = torch.ones(mb * 1024 * 1024 // 2, device="cuda", dtype=torch.bfloat16)
    for _ in range(3): dist.all_reduce(x)
    torch.cuda.synchronize(); dist.barrier()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True); n = 10
    s.record()
    for _ in range(n): dist.all_reduce(x)
    e.record(); e.synchronize()
    ms = s.elapsed_time(e) / n
    busbw = 2 * (world - 1) / world * mb * 1.048576e6 / (ms / 1e3) / 1e9
    rows.append({"MB": mb, "ms": round(ms, 3), "busbw_GBps": round(busbw, 1)})
if rank == 0:
    print(json.dumps({"proto": os.environ.get("NCCL_PROTO"), "nch": os.environ.get("NCCL_MAX_NCHANNELS"), "rows": rows}))
dist.destroy_process_group()
