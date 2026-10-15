"""Table 2 latency: ONNX Runtime, CPU 1 thread, batch 1, T=48, 5 interleaved rounds x 2000 runs (median)."""
import time, statistics, torch, onnxruntime as ort, main as M, base_model as B
torch.set_num_threads(1); torch.manual_seed(0); torch.set_flush_denormal(True)
ours = M.MemoryModel(9, pred_len=1); ours.load_state_dict(torch.load("ckpt/best_memory_model.pth", map_location="cpu")); ours.eval()
with torch.no_grad():
    for p in ours.parameters(): p[p.abs() < 1e-30] = 0.0
b33 = B.CNNLSTMModel(33, 3).eval(); b9 = B.CNNLSTMModel(9, 3).eval()
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
so.add_session_config_entry("session.set_denormal_as_zero", "1"); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
def mk(m, args, names, p):
    torch.onnx.export(m, args, p, input_names=names, opset_version=17, dynamo=False)
    s = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]); feed = dict(zip(names, [a.numpy() for a in args]))
    feed = {i.name: feed[i.name] for i in s.get_inputs()}; return lambda: s.run(None, feed)
x = torch.randn(1, 48, 9); h = torch.randint(0, 24, (1, 48)); d = torch.randint(0, 7, (1, 48))
C = {"base33": mk(b33, (torch.randn(1, 48, 33),), ["x"], "b33.onnx"), "base9": mk(b9, (x,), ["x"], "b9.onnx"), "ours": mk(ours, (x, h, d), ["x", "h", "d"], "o.onnx")}
R = {k: [] for k in C}
for _ in range(5):
    for k, f in C.items():
        for _ in range(100): f()
        ts = []
        for _ in range(2000):
            t = time.perf_counter(); f(); ts.append((time.perf_counter() - t) * 1000)
        R[k].append(statistics.median(ts))
for k, v in R.items(): print(k, round(statistics.median(v), 4), [round(t, 4) for t in v])
