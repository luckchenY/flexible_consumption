#!/usr/bin/env bash
set -euo pipefail

# 自动同步脚本 - 每30分钟自动提交并推送到GitHub
cd /home/chenyang2/flexible_consumption

# 检查是否有变更（工作区或暂存区）
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "$(date): 检测到变更，开始自动提交..."
    
    # 添加所有变更
    git add -A
    
    # 创建带时间戳的提交信息
    COMMIT_MSG="chore(auto): periodic sync $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    
    # 提交变更
    git commit -m "$COMMIT_MSG"
    
    # 推送到远程仓库
    git push origin main
    
    echo "$(date): 自动同步完成"
else
    echo "$(date): 无变更，跳过同步"
fi
