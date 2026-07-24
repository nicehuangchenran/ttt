# 动作标签缓存
/mnt/efs/chenran/ttt/infworld/outputs/sekai_action_cache/sekai_actions_0bb11b2191_rank0of1.pt   (46MB)

里面存的是全部 1618 个视频的条目：每个视频的 move/view 标签张量（LongTensor[1800]）+ 探测元信息（真实帧数、原生分辨率、fps、ok 标记）。不含像素数据。

文件名的构成，对应代码里 _ActionCache（prepare_sekai_game_walking.py §3）：

- 0bb11b2191——配置哈希，由 ACTION_LABEL_VERSION + TARGET_FPS + VERIFY_VIDEOS + ActionLabelConfig 全部阈值 算出。改了任何标注参数哈希就变，会自动生成新缓存文件、旧的不会被误用
- rank0of1——单进程运行的分片；将来多卡 DDP 时每个 rank 有自己的文件，避免写冲突

# train.py 训练产物的位置
所有输出都在 outputs/train/<run_name>/（run_name 不指定时是 run_<月日_时分秒>，可用 --run-name / --output-dir 改），结构如下：

```
outputs/train/<run_name>/
├── config.json                        # 本次运行的完整配置快照
├── checkpoints/
│   ├── checkpoint-{step}.ckpt         # 最终模型：完整 DiT state_dict（bf16 ~2.8GB）
│   │                                  #   可直接给 main.py 推理加载；滚动只留最近 3 个
│   └── training_state-{step}.pt       # resume 用：优化器/RNG/训练游标
├── val_videos/step{step}_case12.mp4   # 周期验证生成的视频（每 val_every=50 步）
└── tb/
```