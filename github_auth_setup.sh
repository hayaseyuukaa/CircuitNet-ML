#!/bin/bash

# GitHub身份验证设置脚本

echo "=== GitHub身份验证设置 ==="
echo ""

echo "方案1: 使用Personal Access Token (推荐)"
echo "1. 访问: https://github.com/settings/tokens"
echo "2. 点击 'Generate new token' → 'Generate new token (classic)'"
echo "3. 选择权限: 勾选 'repo' (完整仓库访问权限)"
echo "4. 生成并复制token"
echo ""

read -p "请输入你的GitHub用户名: " username
read -s -p "请输入你的Personal Access Token: " token
echo ""

# 更新远程URL以包含认证信息
git remote set-url origin https://$username:$token@github.com/hayaseyuukaa/CircuitNet-ML.git

echo "身份验证配置完成！"
echo ""

# 尝试推送
echo "推送master分支..."
git push -u origin master

echo ""
echo "推送develop分支..."
git push -u origin develop

echo ""
echo "✅ 推送完成！"
echo "你可以访问 https://github.com/hayaseyuukaa/CircuitNet-ML 查看你的项目"
