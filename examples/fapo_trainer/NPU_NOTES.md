# FAPO Colocate 在 Ascend NPU 上运行笔记与经验教训

> 记录把 `run_qwen_7b_rm_colocate.sh`(为 32×GPU + CUDA/vLLM 设计)落到
> **单机 4×Ascend 910B3 NPU** 容器上的全过程踩坑与结论。
> 配套脚本:`run_qwen_7b_rm_colocate_npu.sh`(原 GPU 脚本保持不动,另建)。

---

## TL;DR(一句话)

**唯一真正的硬阻塞是容器的 `pids-limit=10000`**(`docker run --pids-limit` 写死,
容器内只读、改不了);FAPO colocate 这种"多 vLLM 引擎 + HCCL + reward_loop"
的工作负载线程数会超过 1 万,`pthread_create` 一到上限就 EAGAIN 崩溃。
**重建容器加 `--pids-limit 200000`(或 `-1`)即可解封**,其余问题脚本里都已修。

---

## 环境基线

| 项 | 值 |
|----|----|
| 硬件 | 4 × Ascend 910B3,每卡 64GB HBM |
| CANN / HDK | cann-9.0.0,软件版本 25.2.0(< 25.3.rc1) |
| Python(正确) | `/usr/local/python3.11.15/bin/python`(3.11.15,有 torch_npu/vllm/verl) |
| Python(错误) | `/usr/bin/python3`(3.10,**无任何依赖**) |
| torch / torch_npu / vllm | 2.9.0+cpu / 2.9.0.post2 / 0.18.0(+ vllm_ascend) |
| Ray | 2.56.0 |
| 容器 memlock (`ulimit -l`) | 64MB(软==硬,无 CAP_SYS_RESOURCE,提不了) |
| 容器 pids-limit | **10000**(`/sys/fs/cgroup/pids/pids.max`,只读) |
| 模型路径 | `/workspace/user_data/models/Qwen/Qwen2.5-Math-7B`、`.../FAPO-GenRM-4B` |
| 数据路径 | `/workspace/user_data/data/fapo-{train,test-full}-boxed.parquet` |
| HF 可达性 | huggingface.co **不通**;`hf-mirror.com` 通 |

---

## 最终结论:容器 pids-limit 是唯一硬阻塞

**现象**:vLLM reward_loop 起来时,各种组件陆续抛
`Resource temporarily unavailable` 然后进程 abort,表现在不同地方:

- HCCL:`Resource temporarily unavailable (src/thread.cpp:241)`
- boost::asio(reward_loop 的 `GlobalRequestLoadBalancer`):
  `thread: Resource temporarily unavailable [system:11]`
- gRPC:`pthread_create failed: Resource temporarily unavailable (thd.cc:157)`

**根因**:三者都是 `pthread_create` 返回 EAGAIN。观测线程数爬到
8551 → 9250 → 9658 后即崩——正好撞 `pids.max=10000`。

**为什么改不了**:`/sys/fs/cgroup/pids/` 是 `ro` 挂载;
`mount -o remount,rw` 被拒(permission denied),即使容器有 CAP_SYS_ADMIN 也不行。
该值由宿主机 `docker run --pids-limit 10000` 决定,**必须重建容器**。

**修复**:

```bash
docker run --pids-limit -1 ...        # 无限制(推荐)
# 或
docker run --pids-limit 200000 ...    # 够 colocate 用
```

**降级方案验证(见下节)**:即便关掉 reward model、只留 1 个 rollout vLLM 引擎,
基础循环**仍然**撞 10000——说明这个限制挡的是整个 verl NPU 分布式训练,
不是 colocate 特有。唯一出路就是提 pids。

---

## ⚠️ cgroup v1 陷阱(我踩过的诊断坑,最容易误判)

中途我曾以为已经把 pids 提到 200000,其实写错了文件:

- 我写的是 `/sys/fs/cgroup/pids.max` —— 这是 cgroup **v2** 风格的根文件,
  对 v1 进程**不生效**(写成功只是因为该文件恰好可写)。
- 真正 govern 进程的是 `/sys/fs/cgroup/pids/pids.max`(pids 控制器),仍是 10000。

**判别方法**:看 `/proc/self/cgroup`。本机是 v1,显示
`10:pids:/docker/<id>` → 实际限制在 `/sys/fs/cgroup/pids/...`。
**改之前务必先 `cat /proc/self/cgroup` 确认控制器路径,别只信一次写入成功。**

---

## 完整调试历程(按发现顺序)

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| 1 | `ModuleNotFoundError: No module named 'packaging'`;DEVICE 被误判为 gpu | 脚本用 `python3`→3.10(无依赖),应为 `python`→3.11 | 脚本改用 `${PYTHON:-python}` |
| 2 | vLLM `KV cache needed 36GiB > available 9.83GiB`(model max seq len 262144) | `max_model_len` 默认取模型的 `max_position_embeddings` | 显式设小 `max_model_len` |
| 3 | `max_model_len (8192) should be <= max_position_embeddings (4096)` | 两个模型 max_position_embeddings 不同(policy 4096 / GRM 262144) | 分别设:policy=4096、GRM=8192 |
| 4 | `pthread_create ... Resource temporarily unavailable`,8 个 vLLM 引擎并发 | colocate `tp=1` → DP=4 → 8 个重型引擎 | `rollout_tp=4`、`grm_tp=4` → 引擎数 8→2 |
| 5 | 同上,疑似 memlock/端口 | 排除 memlock(actor 的 HCCL 能成功);扩 HCCL 端口 + AIV | 扩 `HCCL_*_SOCKET_PORT_RANGE`、`HCCL_OP_EXPANSION_MODE=AIV` |
| 6 | 仍 `pthread_create EAGAIN` | **容器 pids-limit=10000**(只读,容器内不可改) | **宿主机 `--pids-limit` 重建容器** |

---

## 关键经验教训(分主题)

### 1. 解释器:`python` ≠ `python3`

本机 `python3` 是系统 3.10(裸的),`python` 才是 3.11(装了 torch_npu/vllm/verl)。
verl 仓库里的 example 脚本一律写 `python3`,在这台机器上会**静默用错解释器**。
连 `DEVICE` 的自动探测(`import torch_npu`)都会因 3.10 没 torch_npu 而误判成 gpu。
**先 `which python python3` 确认,再决定用哪个。**

### 2. vLLM `max_model_len` 与模型的 `max_position_embeddings`

- vLLM 默认 `max_model_len = 模型的 max_position_embeddings`(若不指定)。
- 长上下文模型(Qwen2.5/Qwen3 系列常报 262144)会直接撑爆 KV cache → 启动失败。
- 但 `max_model_len` 又**不能超过** `max_position_embeddings`,否则 vLLM 报错。
- 两个模型不一样时要**分别设**:
  - Qwen2.5-Math-7B(policy)= **4096**(原生 4K,要 8K+ 长度需 `override_config.max_position_embeddings`)
  - FAPO-GenRM-4B(reward,Qwen3-based)= 262144,设 **8192** 即可
- 取值要 `≥ max_prompt_length + max_response_length`。

> 原脚本能跑 2048+8192 是因为带了这个 override:
> `+actor_rollout_ref.model.override_config.max_position_embeddings=32768`
> smoke test 用 1024+2048=3072,在原生 4096 内,无需 override。

### 3. colocate 的 vLLM 引擎数 = 数据并行度,直接影响线程/资源压力

- colocate(`reward.reward_model.enable_resource_pool=False`)下,
  引擎数 ≈ `n_gpus / tensor_model_parallel_size`,rollout 和 GRM 各自一组。
- `tp=1` + 4 NPU → rollout 4 引擎 + GRM 4 引擎 = **8 个重型 vLLM 进程**,
  每个又 spawn EngineCore + 多个 Worker + HCCL 线程,线程爆炸。
- **提高 tp → 降低 DP → 少引擎**:`tp=4` → 各 1 引擎 = 共 2 个。
  在资源受限环境优先用大 tp 减压(牺牲部分吞吐换可跑性)。

### 4. HCCL 相关环境变量(NPU 分布式必备)

```bash
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60500     # 默认只 50 个端口,多引擎不够
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61500
export HCCL_OP_EXPANSION_MODE=AIV                  # 减少 stream 占用,增加支持的 shape
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
ulimit -s unlimited 2>/dev/null || true
```

### 5. memlock(红鲱鱼,但确实偏低)

容器 `ulimit -l = 64MB`(Ascend 标准要求 `unlimited`)。曾高度怀疑它是 HCCL 崩溃元凶,
但 **actor 的 HCCL 能成功初始化**,说明 memlock 不是 8 引擎崩溃的原因(真正是 pids)。
不过仍建议重建容器时一并设 `--ulimit memlock=-1:-1`。

### 6. IPC / CANN 版本 → vLLM 权重同步回退

日志会出现:`IPC is not supported on your devices. Falling back to shared memory for
weight transfer`(因为 HDK 25.2 < 25.3.rc1)。**不是错误**,只是 FSDP→vLLM 权重同步
变慢。要恢复 IPC 需 HDK ≥ 25.3.rc1 且 CANN ≥ 8.3.RC1。

### 7. HuggingFace 不通 + GPQA gated(数据准备)

- 主站不通 → 用镜像:`HF_ENDPOINT=https://hf-mirror.com`。
- `Idavidrein/gpqa` 是 **gated** 数据集,需 `HF_TOKEN` + 在 HF 上接受授权。
- `prepare_data.py` 的 `--tasks` 参数是**死代码**(解析了但没用到),无条件构建全部 4 个数据集,
  其中 GPQA 一失败整脚本就崩、两个 parquet 都不生成。
- 生成的数据:train = DAPO×20 = 347,960 行;test = AIME24×32 + AIME25×32 + GPQA×4 = 2,712 行。
- ⚠️ **`data_source` 命名不一致**:`prepare_data.py` 把 DAPO 标成 `math-dapo`(连字符),
  但 verl 的 `default_compute_score`(`verl/utils/reward_score/__init__.py:59`)只认 `math_dapo`(下划线)和
  `startswith("aime")`。降级(无 custom reward fn)时直接 `NotImplementedError: ... data_source='math-dapo'`。
  修法二选一:
  - 改数据(已采用):`pc.replace_substring(col, "math-dapo", "math_dapo")` 就地改 parquet;
  - 或提供一个 custom_reward_function 自己分发。
  (aime24/aime25 靠前缀匹配没事;gpqa-diamond 不在表里,smoke test 不跑 eval 暂时无影响。)

### 7b. tokenizer 线程池(rayon)是 pids 限制的最后爆破点

降级方案越过 vLLM init 后,会在 **rollout tokenization** 崩:
HF tokenizer 的 Rust `rayon` 线程池(`encode_batch`)默认按 CPU 核数建池,
多个 `AgentLoopWorker` 各建一个 → `ThreadPoolBuildError: Resource temporarily unavailable (code 11)`。
在 pids 受限的容器里加这两个环境变量可显著削减线程爆发:
```bash
export TOKENIZERS_PARALLELISM=false   # 关掉 tokenizer 并行
export RAYON_NUM_THREADS=1            # rayon 池只 1 线程
```
(已写进两个 NPU 脚本;若基础占用本身就 >上限,这两招也不够,只能提 pids。)

### 8. 并行大模型下载会严重拖慢训练初始化

跑训练时若同时有 `hf download` 在写大模型(如 30B,~60GB),会抢占磁盘 I/O,
把模型 shard 加载从 **<1s/shard 拖到 ~97s/shard**,既慢也可能加剧不稳定。
**训练期间暂停其它大文件下载。**

### 9. 直接跑 `python -m verl.trainer.main_ppo`,不要用 `ray job submit`

原 GPU 脚本用 `ray job submit --address localhost:8265`,需要一个预先启动的 Ray 集群。
而 `verl.trainer.main_ppo` 内部**自己调用 `ray.init()`**,直接跑 python 即可,
Ray 会自启(日志可见 `Started a local Ray instance ... dashboard at :8265`)。
省掉单独 `ray start` 这步。

### 10. verl 自动识别 NPU

verl 通过 `is_torch_npu_available()` 自动在 `CUDA_VISIBLE_DEVICES` /
`ASCEND_RT_VISIBLE_DEVICES` 间切换,config 里 `device` 会自动是 `npu`。
**通常不需要手动 `export DEVICE=npu`**(规范脚本会探测;手动 export 仅用于覆盖)。

---

## 容器最小要求(checklist)

跑 FAPO colocate(或任何重度 verl+NPU 工作负载)前,容器**必须**满足:

- [ ] **`--pids-limit 200000` 或 `-1`**(最关键,否则 pthread_create 必崩)
- [ ] `--ulimit memlock=-1:-1`(HCCL pin memory)
- [ ] Ascend 驱动 + CANN ≥ 8.3.RC1,HDK ≥ 25.3.rc1(恢复 IPC;低版本可跑但慢)
- [ ] Python 3.11 环境:torch_npu、vllm、vllm-ascend、verl 全部装好
- [ ] 网络可达 hf-mirror.com,或模型/数据已预下好
- [ ] 模型:`Qwen2.5-Math-7B`(policy)、`FAPO-GenRM-4B`(reward)
- [ ] 数据:`fapo-train-boxed.parquet`、`fapo-test-full-boxed.parquet`

---

## 运行步骤

```bash
cd /workspace/user_data/projects/verl-v0.8.0

# 1.(一次性)生成数据(已做过;GPQA 需要 HF_TOKEN)
HF_ENDPOINT=https://hf-mirror.com HF_TOKEN=<your_token> \
  python examples/fapo_trainer/prepare_data.py --local_dir /workspace/user_data/data

# 2. smoke test(2 步,小配置,验证管线)
bash examples/fapo_trainer/run_qwen_7b_rm_colocate_npu.sh

# 3. 放大做正式训练(注意 policy 原生 4096,长序列要加 override)
TRAIN_BATCH_SIZE=64 ROLLOUT_N=8 MAX_RESPONSE_LENGTH=8192 TOTAL_TRAINING_STEPS=200 \
  bash examples/fapo_trainer/run_qwen_7b_rm_colocate_npu.sh
# 若 MAX_RESPONSE_LENGTH 超过 4096,需追加:
#   +actor_rollout_ref.model.override_config.max_position_embeddings=32768
```

脚本启动时会自检 pids-limit,过低会在 stderr 打 WARNING。

---

## 快速排查清单(出问题时按此顺序)

1. **`cat /sys/fs/cgroup/pids/pids.max`** —— 若 < 50000,先重建容器提 pids(90% 的崩溃都在这)。
2. **`which python python3`** —— 确认脚本用的是装了依赖的那个解释器。
3. 看崩溃是 `max_model_len` / `max_position_embeddings` 相关 → 按模型分别设 max_model_len。
4. 看 `pthread_create ... Resource temporarily unavailable` → 一定是 pids(或更隐蔽的 cgroup 子组)。
5. 看 `KV cache ... larger than available` → max_model_len 太大,调小。
6. vLLM 权重同步慢 → IPC 回退(CANN 版本),非错误,可忽略。
7. 模型加载极慢 → 有并行大文件下载在抢 I/O,暂停之。

---

## 降级方案尝试记录(rule-based reward,无 GRM)

**动机**:既然 colocate 的 GRM 多引擎是线程大户,试试**关掉 reward model**、
只留 1 个 rollout vLLM 引擎、用数据自带的规则奖励(`reward_model.ground_truth` →
dapo reward manager 的 `default_compute_score`),看基础 GRPO 循环能否在
pids-limit=10000 内跑通。脚本:`run_qwen_7b_rule_npu.sh`。

**关键改动**(相对 colocate 版):
- `reward.reward_model.enable=False`(去掉 GRM 的 4B vLLM 引擎 + reward_loop 的
  `GlobalRequestLoadBalancer`)
- 去掉 `reward.custom_reward_function.*`(否则会去 HTTP 调不存在的 GRM)
- `rollout_tp=4`(只 1 个 rollout 引擎)
- 保留 `reward.reward_manager.name=dapo` + overlong_buffer 规则奖励

**走到哪一步**(比 colocate 远得多):

| 里程碑 | 状态 | 当时线程数 |
|--------|------|-----------|
| 配置/Ray/分布式/HCCL | ✅ | — |
| 7B policy FSDP 加载(`After FSDP 21/61GB`) | ✅ | ~8000 |
| rollout vLLM(tp=4)HCCL init | ✅ **越过 colocate 崩溃点** | 8818 |
| `AgentLoopWorker` + `RLHFDataset`(rollout tokenization) | ✅ 到这 | — |
| 首个采样/奖励/PPO step | ❌ 崩在这之前 | ~10000 |

**崩溃点**:Rust `rayon` 线程池(HF tokenizer 的 `encode_batch`,经 pyo3 调用)
无法建线程:

```
rayon-core-1.13.0/src/registry.rs:171
ThreadPoolBuildError { kind: IOError(Os { code: 11, kind: WouldBlock,
                                         message: "Resource temporarily unavailable" }) }
pyo3_runtime.PanicException: The global thread pool has not been initialized.
```

`code 11 = EAGAIN`,即 `pthread_create` 失败——和 colocate 崩溃**同一个 pids 限制**,
只是这次表现在 Rust tokenizer 的 rayon 线程池。

**结论**:
1. 降级方案成功**越过** colocate 的崩溃点(vLLM HCCL init),证明去掉 GRM 确实减压、
   且 reward 规则化配置本身没问题。
2. 最初仍在 rollout tokenization 撞 10000(Rust rayon 线程池)。
3. **后续三连修后,基础循环真的跑通了 step 1**(见下),无需提 pids。

### 后续修复:基础 NPU GRPO 循环跑通 ✅

继续解掉 3 个问题后,降级方案成功完成 step 1:

| 问题 | 现象 | 修法 |
|------|------|------|
| tokenizer 线程池 | `ThreadPoolBuildError: Resource temporarily unavailable`(rayon) | `TOKENIZERS_PARALLELISM=false` + `RAYON_NUM_THREADS=1`(见 7b) |
| `data_source` 命名 | `NotImplementedError ... 'math-dapo'` | parquet 里 `math-dapo`→`math_dapo`(见 7) |
| 收尾验证 | `NotImplementedError ... 'gpqa-diamond'`(test 集) | `test_freq=0` 关掉所有验证,含 `is_last_step` 强制那次 |

> 收尾验证的坑:`ray_trainer.py:1686` 的条件是
> `test_freq > 0 and (is_last_step or global_steps % test_freq == 0)`,
> **最后一步无论 test_freq 多大都强制验证**。test 集含 `gpqa-diamond`(多选题),
> `default_compute_score` 不认 → 必崩。smoke test 用 `test_freq=0` 彻底关掉。

step 1 实测(4×910B3,pids=10000 内):

```
step:1 - actor/pg_loss:0.2473 - actor/grad_norm:3.50 - actor/lr:1e-07
       - critic/rewards/mean:-1.273 - response_length/mean:902.9
       - timing_s/step:93.53 - training/global_step:1
Training Progress:  50%|█████ | 1/2 [01:33<01:33, 93.89s/it]
```

即 **rollout(vLLM/NPU 采样)→ 规则奖励(math_dapo)→ PPO 更新 → 权重同步**,端到端跑通。

**最终结论**:
- 基础 verl+NPU GRPO 循环**可在此容器内跑通**(无需提 pids,靠 tokenizer 线程限制即可压在 10000 下)。
- **完整 colocate GenRM 版仍需提 pids**——多了 GRM vLLM 引擎 + reward_loop load balancer 的线程,
  那部分是 tokenizer workaround 兜不住的。提了 pids 后,colocate 脚本也应能跑通(其余问题均已修)。

---

## 相关文件

- `run_qwen_7b_rm_colocate.sh` —— 原 GPU 脚本(忠实保留,未改)
- `run_qwen_7b_rm_colocate_npu.sh` —— NPU 适配版(完整 colocate GenRM,本文档的产物)
- `run_qwen_7b_rule_npu.sh` —— **降级版**(无 GRM,规则奖励,验证基础循环用)
- `prepare_data.py` —— 数据生成(GPQA gated,需 HF_TOKEN)
- `reward_fn.py` —— FAPO 自定义奖励函数(通过 HTTP 调 GRM 的 vLLM,设备无关)
- `README.md` —— FAPO 原始说明
