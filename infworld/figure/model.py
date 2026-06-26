from infworld.models.dit_model import WanModel
# from torchinfo import summary
import torch



def make_input():
    num="small"
    
    device = "cuda"
    dtype = torch.bfloat16

    B = 1                 # 单样本
    CFG = 2               # classifier-free guidance 把 batch 翻倍
    C_lat = 16            # VAE out_channels
    SEQ = 512             # model_max_length
    DIM = 4096            # text encoder output_dim
    N_FRAMES = 81         # validation_data.num_frames


    if num=="small":
        # 画图专用：结构与 large 一致，只把张量尺寸调小，
        # 让 torchview 的 forward / 抓图 / 建图都快很多
        T = 5                 # 输出 latent 帧数（缩小）
        H_lat, W_lat = 16, 16 # 空间尺寸（缩小）
        T_in = 5              # 历史条件 latent 时间长度（缩小）
    elif num=="large":
        # 真实推理尺寸（summary / 真跑用）
        T = 21                # 每 chunk 输出 latent 帧数 (latent_size[2])
        H_lat, W_lat = 80, 80 # 640x640 像素 / 8
        T_in = 21             # 历史条件的 latent 时间长度（示例值，随历史长度变化）
    else:
        print("check num in make_input()")
        return None

    # x: 噪声 latent，已 CFG 拼接
    x = torch.randn(B * CFG, C_lat, T, H_lat, W_lat, device=device, dtype=dtype)

    # t: timestep，每个样本一个标量
    t = torch.randn(B * CFG, device=device, dtype=dtype)   # 实际取值是 [0, num_timesteps] 的某个 t

    # y: 文本 embedding
    y = torch.randn(B * CFG, 1, SEQ, DIM, device=device, dtype=dtype)
    y_mask = torch.ones(B * CFG, SEQ, dtype=torch.long, device=device)

    # move / view: 动作 id (long)，batch=1（forward 内会 broadcast / image_cond repeat 成 2）
    move = torch.randint(0, 10, (B, N_FRAMES), dtype=torch.long, device=device)
    view = torch.randint(0, 10, (B, N_FRAMES), dtype=torch.long, device=device)

    # image_cond: 历史 latent，CFG 后 batch=2
    image_cond = torch.randn(B * CFG, C_lat, T_in, H_lat, W_lat, device=device, dtype=dtype)
    # 位置参数 (x, t, y) + 关键字参数，避免与 forward 的可选参数错位
    pos = (x, t, y)
    kw = dict(y_mask=y_mask, image_cond=image_cond, move=move, view=view)
    return pos, kw

def make_input2():
    pos, kw = make_input()
    x, t, y = pos

    inputs = {
        "x": x,
        "t": t,
        "y": y,
        "y_mask": kw["y_mask"],
        "image_cond": kw["image_cond"],
        "move": kw["move"],
        "view": kw["view"],
    }

    return inputs


if __name__=="__main__":
    METHOD="draw_graph"

    # 与 configs/infworld_config.yaml 的 model_cfg 保持一致（1.3B 配置），
    # 否则结构 / 通道数与 checkpoint 不符
    model=WanModel(
        model_type="t2v",
        dim=1536,
        in_channels=20,
        ffn_dim=8960,
        freq_dim=256,
        num_heads=12,
        num_layers=30,
    ).cuda().to(torch.bfloat16)

    if METHOD=="summary":
        # summary() 返回 ModelStatistics，str() 就是终端里那张表；
        # 跑一次，复用结果写文件，避免重复 forward
        pos, kw = make_input()
        stats = summary(
            model,
            input_data=pos,
            **kw,
            depth=4,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            # row_settings 让每层显示完整层级路径，便于定位
            row_settings=["var_names", "depth"],
            verbose=1,  # 1=打印到终端；写文件用下面的 str(stats)
        )

        # 1) torchinfo 宽表（参数量 / 输入输出形状）
        with open("figure/model_summary.txt", "w") as f:
            f.write(str(stats))

        print("\nsaved -> figure/model_summary.txt")
    elif METHOD=="draw_graph":
        from torchview import draw_graph
        inputs=make_input2()
        graph = draw_graph(
            model,
            input_data=inputs,  # 按你的输入改
            depth=2,
            # expand_nested=True,             # 展开子模块
            graph_name="WanModel"
        )

        graph.visual_graph.render(filename="figure/wan_model_graph_small", format="svg",cleanup=True) # cleanup删除wan_model_graph2中间产物
        print("完成")

    elif METHOD=="tensor_board":
        from torch.utils.tensorboard import SummaryWriter

        # 说明：TensorBoard 的 GRAPHS 面板依赖 add_graph -> torch.jit.trace，
        # 而本模型用了 flash-attn 的 varlen kernel + auto_grad_checkpoint，
        # trace 无法穿透（会报 "untraceable code" / Constant 值不一致），所以这里
        # 不画算子级计算图，而是把模型的模块层级结构写进 TEXT 面板供浏览。

        log_dir = "figure/tb_model_graph"
        writer = SummaryWriter(log_dir=log_dir)

        # repr(model) 即 print(model) 的内容；外面包 ``` 让 TB 用等宽字体渲染
        writer.add_text("WanModel/structure", "```\n" + repr(model) + "\n```")

        # 顺带记录每个子模块的参数量，便于在 TEXT 面板里看各部分规模
        lines = ["| module | #params |", "| --- | --- |"]
        for name, mod in model.named_modules():
            n = sum(p.numel() for p in mod.parameters(recurse=False))
            if n > 0:
                lines.append(f"| {name or '(root)'} | {n:,} |")
        total = sum(p.numel() for p in model.parameters())
        lines.append(f"| **total** | **{total:,}** |")
        writer.add_text("WanModel/param_count", "\n".join(lines))

        writer.close()

        print(f"saved -> {log_dir}  (查看: tensorboard --logdir {log_dir} ，看 TEXT 面板)")

    else:
        print("检查 METHOD 变量")
            