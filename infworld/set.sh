#!/bin/bash
export CLAUDE_CONFIG_DIR=/mnt/efs/chenran/claude-config
export PATH=/mnt/efs/chenran/claude-code/bin:$PATH

# 初始化 conda
source /mnt/efs/chenran/miniconda3/etc/profile.d/conda.sh

# git 身份
git config --global user.name "nicehuangchenran"
git config --global user.email "3163328424@qq.com"
export GIT_SSH_COMMAND='ssh -i /mnt/efs/chenran/.ssh/github_ed25519 -o IdentitiesOnly=yes'
git config --global url."git@github.com:".insteadOf "https://github.com/"

# 我安装的命令
export PATH="/mnt/efs/chenran/bin:$PATH"

# ossutil配置
export OSSUTIL_CONFIG_FILE="/mnt/efs/chenran/.ossutil/config"
export OSSUTIL_PROFILE="wbench"

# 自动选择最新的 VS Code IPC socket
if [ -d "/run/user/1000" ]; then
    LATEST_SOCK=$(ls -t /run/user/1000/vscode-ipc-*.sock 2>/dev/null | head -1)
    if [ -n "$LATEST_SOCK" ]; then
        export VSCODE_IPC_HOOK_CLI="$LATEST_SOCK"
    fi
fi

