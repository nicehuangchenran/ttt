Validation Loss 的数据视角

1. 使用的数据

- 固定验证集：从训练数据集最前面取 4 个视频（num_val_videos=4）
- 固定 chunk：每个视频只用第 0 个 chunk（81 帧）
- 验证频率：每隔 val_every_n_steps 个训练步做一次验证（例如每 16 步）

2. 输入数据（模型接收什么）

加噪的 latent

- 原始数据：视频 chunk 的 VAE latent x_start（形状：[B, 16, 21, H/8, W/8]）
- 加噪过程：x_t = (1 - t/1000) * x_start + (t/1000) * noise
  - t 是固定的 timestep 值（如 50, 150, 250, ...）
  - noise 是用固定随机种子生成的高斯噪声
  - 例如 t=50 时：x_t = 0.95 * x_start + 0.05 * noise（轻微噪声）
  - 例如 t=950 时：x_t = 0.05 * x_start + 0.95 * noise（几乎全是噪声）

条件信息（和训练完全一致）

- 图像条件 image_cond：视频前面帧的 latent（第 0 个 chunk 就是初始图像）
- 文本 y：文本 prompt 的 UMT5 embedding
- 动作序列：81 帧的 move 和 view action indices

3. 模型输出

模型预测的 velocity field pred（形状和 x_start 相同）

在 flow-matching 框架下，模型学习的是从噪声 noise 到干净数据 x_start 的"流向"。

4. 监督信号（ground truth）

target = x_start - noise
如果启用 use_reversed_velocity=True（默认启用），则：
target = -(x_start - noise) = noise - x_start

这是 rectified flow 的标准目标：模型要学会预测从当前噪声状态 x_t 到目标状态的方向向量。

5. Loss 计算

validation_loss = MSE(pred, target) = 均方误差
对每个像素位置、每个通道计算预测值和目标值的平方差，然后取平均。