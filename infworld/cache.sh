CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
    scripts/main.py \
    --dataset-dir dataset/sekai-game-walking-854_480_30fps --begin-idx 1 --num 8  \
    --output-dir videos/sekai-game-walking-256-steps30-chunk20 \
    --bucket-config-name ASPECT_RATIO_256 --shift 3 --steps 30 --max-chunks 20

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
    scripts/main.py \
    --dataset-dir dataset/sekai-game-walking-854_480_30fps --begin-idx -1 --num 10  \
    --output-dir videos/sekai-game-walking-256-steps30-chunk20 \
    --steps 1 --max-chunks 20
# 256分辨率下一个 step 大概 0.5s, 默认的 bucket name 是ASPECT_RATIO_627_F64, 这个分辨率下一个 step 2.5s 左右


CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2  --local-ranks-filter 0 \
    scripts/main.py \
        --dataset-dir dataset/demo \
        --output-dir videos/demo-627-steps20-chunks3 \
        --steps 20 --max-chunks 3 && \
        /mnt/efs/chenran/claude_notify.sh 'aws' '视频生成了'


CUDA_VISIBLE_DEVICES=4 python scripts/main.py \
    --dataset-dir dataset/demo \
    --output-dir videos/demo-256-steps20-chunks20 \
    --steps 20 --max-chunks 20 \
    --bucket-config-name ASPECT_RATIO_256 --shift 3

# 上传生成的视频到 oss
ossutil cp -r  --update /mnt/efs/chenran/ttt/infworld/videos/   oss://wbench/ttt/infworld/videos/  
# 上传cases dataset 到 oss
ossutil cp -r  --update  -j 32  /mnt/efs/chenran/ttt/infworld/dataset/sekai-game-walking-352_192_30fps/ \
 oss://wbench/ttt/infworld/dataset/sekai-game-walking-352_192_30fps/ --dry-run


# 发通知
/mnt/efs/chenran/claude_notify.sh 'aws' '视频生成了'