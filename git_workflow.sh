#!/bin/bash

# CircuitNet项目Git工作流脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示帮助信息
show_help() {
    echo -e "${BLUE}CircuitNet Git 工作流管理脚本${NC}"
    echo ""
    echo "用法: ./git_workflow.sh [命令]"
    echo ""
    echo "命令:"
    echo "  status      - 显示当前状态"
    echo "  clean       - 清理工作区"
    echo "  feature     - 创建新功能分支"
    echo "  commit      - 智能提交"
    echo "  sync        - 同步分支"
    echo "  release     - 准备发布"
    echo "  backup      - 备份当前工作"
    echo "  help        - 显示此帮助"
}

# 显示当前状态
show_status() {
    echo -e "${BLUE}=== Git 状态 ===${NC}"
    git status --short
    echo ""
    echo -e "${BLUE}=== 当前分支 ===${NC}"
    git branch --show-current
    echo ""
    echo -e "${BLUE}=== 最近提交 ===${NC}"
    git log --oneline -5
}

# 清理工作区
clean_workspace() {
    echo -e "${YELLOW}清理工作区...${NC}"
    
    # 显示将要删除的文件
    echo "将要删除的未跟踪文件:"
    git clean -n
    
    read -p "确认删除这些文件? (y/N): " confirm
    if [[ $confirm == [yY] ]]; then
        git clean -f
        echo -e "${GREEN}工作区已清理${NC}"
    else
        echo -e "${YELLOW}取消清理${NC}"
    fi
}

# 创建功能分支
create_feature() {
    read -p "输入功能分支名称 (例如: model-improvement): " branch_name
    if [[ -z "$branch_name" ]]; then
        echo -e "${RED}分支名称不能为空${NC}"
        return 1
    fi
    
    # 确保在develop分支
    git checkout develop
    git pull origin develop
    
    # 创建新分支
    git checkout -b "feature/$branch_name"
    echo -e "${GREEN}已创建并切换到分支: feature/$branch_name${NC}"
}

# 智能提交
smart_commit() {
    # 检查是否有更改
    if git diff --quiet && git diff --cached --quiet; then
        echo -e "${YELLOW}没有检测到更改${NC}"
        return 0
    fi
    
    # 显示更改
    echo -e "${BLUE}=== 当前更改 ===${NC}"
    git status --short
    
    # 选择要提交的文件
    echo ""
    read -p "添加所有更改? (Y/n): " add_all
    if [[ $add_all != [nN] ]]; then
        git add .
    else
        echo "请手动添加要提交的文件: git add <文件>"
        return 0
    fi
    
    # 使用模板提交
    git commit
}

# 同步分支
sync_branch() {
    current_branch=$(git branch --show-current)
    echo -e "${BLUE}同步分支: $current_branch${NC}"
    
    # 如果是feature分支，先同步develop
    if [[ $current_branch == feature/* ]]; then
        git checkout develop
        git pull origin develop
        git checkout $current_branch
        git rebase develop
    else
        git pull origin $current_branch
    fi
    
    echo -e "${GREEN}分支同步完成${NC}"
}

# 准备发布
prepare_release() {
    echo -e "${BLUE}准备发布...${NC}"
    
    # 切换到develop分支
    git checkout develop
    git pull origin develop
    
    # 创建release分支
    read -p "输入版本号 (例如: v1.0.0): " version
    if [[ -z "$version" ]]; then
        echo -e "${RED}版本号不能为空${NC}"
        return 1
    fi
    
    git checkout -b "release/$version"
    echo -e "${GREEN}已创建发布分支: release/$version${NC}"
    echo "请在此分支进行最终测试和文档更新"
}

# 备份当前工作
backup_work() {
    timestamp=$(date +"%Y%m%d_%H%M%S")
    backup_branch="backup/work_$timestamp"
    
    git add .
    git commit -m "backup: 工作备份 $timestamp" || true
    git branch $backup_branch
    
    echo -e "${GREEN}工作已备份到分支: $backup_branch${NC}"
}

# 主函数
main() {
    case "$1" in
        "status")
            show_status
            ;;
        "clean")
            clean_workspace
            ;;
        "feature")
            create_feature
            ;;
        "commit")
            smart_commit
            ;;
        "sync")
            sync_branch
            ;;
        "release")
            prepare_release
            ;;
        "backup")
            backup_work
            ;;
        "help"|"")
            show_help
            ;;
        *)
            echo -e "${RED}未知命令: $1${NC}"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
