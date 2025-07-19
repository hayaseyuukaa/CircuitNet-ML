# CircuitNet项目Git维护指南

## 项目结构和分支策略

### 分支模型
- `master`: 稳定的生产版本
- `develop`: 开发主分支，集成所有功能
- `feature/*`: 功能开发分支
- `release/*`: 发布准备分支
- `hotfix/*`: 紧急修复分支
- `backup/*`: 工作备份分支

### 目录结构
```
CircuitNet6/
├── .gitignore              # Git忽略文件配置
├── COMMIT_TEMPLATE.md      # 提交信息模板
├── git_workflow.sh         # Git工作流脚本
├── GIT_GUIDE.md           # 本指南
├── CircuitNet-main/       # 主要代码目录
│   ├── run_*.sh          # 各模型训练脚本
│   ├── configs.py        # 配置文件
│   └── ...
└── work_dir/              # 训练输出目录(被忽略)
```

## 日常Git操作

### 1. 查看项目状态
```bash
# 使用脚本
./git_workflow.sh status

# 或手动执行
git status
git log --oneline -5
```

### 2. 开始新功能开发
```bash
# 使用脚本创建功能分支
./git_workflow.sh feature

# 或手动创建
git checkout develop
git pull origin develop
git checkout -b feature/新功能名称
```

### 3. 提交代码
```bash
# 使用脚本智能提交
./git_workflow.sh commit

# 或手动提交
git add .
git commit  # 会自动使用模板
```

### 4. 同步代码
```bash
# 使用脚本同步
./git_workflow.sh sync

# 或手动同步
git pull origin develop
```

## 提交信息规范

### 提交类型
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 重构代码
- `test`: 添加或修改测试
- `chore`: 构建过程或辅助工具的变动
- `perf`: 性能优化
- `model`: 模型相关更改

### 提交示例
```
feat(模型): 添加ibUNet模型实现

- 实现了ibUNet架构用于拥塞预测
- 添加了相应的训练和测试脚本
- 集成到现有的训练流水线中

测试: 在N28数据集上验证通过
```

## 机器学习项目特殊考虑

### 1. 大文件管理
- 数据集文件不提交到git
- 模型权重文件不提交到git
- 使用`.gitignore`忽略大文件

### 2. 实验管理
```bash
# 备份当前实验
./git_workflow.sh backup

# 创建实验分支
git checkout -b experiment/模型优化_20240719
```

### 3. 模型版本管理
```bash
# 为重要模型创建标签
git tag -a v1.0-ibunet -m "ibUNet模型第一个稳定版本"
git push origin v1.0-ibunet
```

## 常用命令速查

### 基础操作
```bash
git status                    # 查看状态
git add .                     # 添加所有更改
git commit                    # 提交(使用模板)
git push origin 分支名         # 推送到远程
```

### 分支操作
```bash
git branch                    # 查看本地分支
git branch -r                 # 查看远程分支
git checkout 分支名            # 切换分支
git merge 分支名               # 合并分支
```

### 历史查看
```bash
git log --oneline -10         # 查看最近10次提交
git log --graph --oneline     # 图形化显示提交历史
git show 提交ID               # 查看具体提交
```

### 撤销操作
```bash
git checkout -- 文件名        # 撤销工作区更改
git reset HEAD 文件名         # 撤销暂存区更改
git reset --soft HEAD~1      # 撤销最后一次提交(保留更改)
```

## 最佳实践

1. **频繁提交**: 每完成一个小功能就提交
2. **清晰的提交信息**: 使用提供的模板
3. **分支隔离**: 不同功能使用不同分支
4. **定期同步**: 经常从develop分支拉取最新代码
5. **备份重要工作**: 使用backup功能保护重要实验
6. **标签管理**: 为重要版本打标签

## 故障排除

### 合并冲突
```bash
# 查看冲突文件
git status

# 手动解决冲突后
git add 冲突文件
git commit
```

### 误操作恢复
```bash
# 查看操作历史
git reflog

# 恢复到指定状态
git reset --hard HEAD@{n}
```

### 清理工作区
```bash
# 使用脚本安全清理
./git_workflow.sh clean

# 或手动清理
git clean -n  # 预览要删除的文件
git clean -f  # 删除未跟踪文件
```
