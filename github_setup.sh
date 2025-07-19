#!/bin/bash

# CircuitNet项目GitHub设置脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== CircuitNet GitHub 设置向导 ===${NC}"
echo ""

# 检查当前Git状态
echo -e "${BLUE}1. 检查当前Git状态...${NC}"
git status --short

echo ""
echo -e "${BLUE}2. 当前分支和提交历史:${NC}"
git log --oneline -5

echo ""
echo -e "${YELLOW}请按照以下步骤操作:${NC}"
echo ""

echo -e "${GREEN}步骤1: 在GitHub上创建新仓库${NC}"
echo "1. 访问 https://github.com/new"
echo "2. 仓库名称建议: CircuitNet-ML"
echo "3. 描述: Deep Learning Models for Circuit Design - Congestion Prediction"
echo "4. 选择 Public 或 Private"
echo "5. 不要初始化 README, .gitignore 或 license (我们已经有了)"
echo "6. 点击 'Create repository'"
echo ""

echo -e "${GREEN}步骤2: 获取仓库URL${NC}"
echo "创建后，GitHub会显示类似这样的URL:"
echo "https://github.com/你的用户名/CircuitNet-ML.git"
echo ""

read -p "请输入你的GitHub仓库URL (例如: https://github.com/username/CircuitNet-ML.git): " repo_url

if [[ -z "$repo_url" ]]; then
    echo -e "${RED}错误: 仓库URL不能为空${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}3. 配置远程仓库...${NC}"
git remote add origin "$repo_url"

echo -e "${GREEN}远程仓库已配置: $repo_url${NC}"

echo ""
echo -e "${BLUE}4. 验证远程仓库配置...${NC}"
git remote -v

echo ""
echo -e "${BLUE}5. 推送到GitHub...${NC}"

# 推送master分支
echo "推送master分支..."
git push -u origin master

# 推送develop分支
echo "推送develop分支..."
git push -u origin develop

echo ""
echo -e "${GREEN}✅ 成功推送到GitHub!${NC}"
echo ""
echo -e "${BLUE}接下来你可以:${NC}"
echo "- 访问你的GitHub仓库查看代码"
echo "- 使用 'git push' 推送后续更改"
echo "- 使用 './git_workflow.sh' 进行日常开发"
echo ""

echo -e "${YELLOW}常用GitHub操作:${NC}"
echo "git push origin develop          # 推送develop分支"
echo "git push origin feature/分支名   # 推送功能分支"
echo "git pull origin develop          # 拉取最新代码"
echo ""

echo -e "${BLUE}项目README建议:${NC}"
echo "建议在GitHub上添加一个详细的README.md文件，包含:"
echo "- 项目介绍和目标"
echo "- 模型架构说明"
echo "- 安装和使用指南"
echo "- 数据集要求"
echo "- 训练和测试说明"
