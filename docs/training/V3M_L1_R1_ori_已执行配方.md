# V3M · L1-R1_ori 已执行训练配方

本说明对应已训练完成的“GT总加粗1px修复版、V3M、L1+R1_ori、500K”。这里只同步训练代码和配置，不包含数据、随机初始化文件或训练权重。配置快照中的服务器路径仅用于追溯，不会自动连接服务器。

## 与计划模板的区别

`configs/WH_SCRATCH/` 是另一份计划模板；本次实际配方单独放在 `configs/V3M_EXECUTED/`，不覆盖原模板。

| 项目 | 本次已执行配方 |
|---|---|
| 模型 | BasicTextLineR4StudentY4UVV3M，通道4/8/16/32 |
| 画布与输入 | RGB 256×256，模型内部使用6通道Y4UV |
| 数据 | 总加粗1px mask修复后的VOSR+S2.2 GT；V3预处理LQ；5284个训练父样本，排除Mojito |
| 训练 | 共享随机初始化，seed=20260828；batch=4；500000 iter |
| 优化器 | Adam，初始LR=1e-4，betas=(0.9,0.99)，EMA关闭 |
| 学习率 | 200K/300K/400K/450K起分别5e-5/2.5e-5/1e-5/5e-6；不启用warmup |
| 基础L1 | native Y4UV，通道权重1/1/1/1/0.25/0.25；前景、背景权重均为1 |
| R1_ori | Missing=1，False=0.25；ring_radius=3，margin=0.03，temperature=0.01，min_polarity_contrast=0.03 |
| 辅助loss启用 | aux_start_iter=2000，aux_ramp_iters=5000；不做粗细归一化 |
| 关闭项 | GT退化混合、GT-LQ对比度对齐、CP、SDF、guard、背景低频和独立UV辅助loss |
| 采样 | S22Dataset，有效文字patch；train/val已有离线bank与配套line-mask索引 |
| 保存与验证 | 每5000 iter保存，每25000 iter验证 |

验证集沿用历史验证bank，并非重新生成的同版本修复GT。`plus1_500k.executed.yml` 是原始配置字节快照，含历史兼容字段和未启用的退化参数；不能仅根据字段存在就认定该分支已启用。

## 文件和入口

- `configs/V3M_EXECUTED/plus1_500k.executed.yml`：原执行配置，只作追溯。
- `configs/V3M_EXECUTED/V3M_L1_R1_ori_500k.template.yml`：相同配方的路径可绑定模板。
- `configs/V3M_EXECUTED/bindings.example.json`：需要填写的路径。
- `tools/train_v3m_r1_ori_portable.py`：跨目录可用的新入口，注册既有 `FilteredPairedModel`、`S22Dataset` 和 `LatestVOSR500K`；不改loss算法。
- `docs/v3m_r1_ori_source_manifest.json`：实际配置、参考checkpoint以及训练依赖源码SHA256。

## 使用

本配方的历史bank依赖POSIX文件锁，正式运行环境为Linux。先按仓库既有环境说明准备依赖，使用本仓库修改过的BasicSR。不要直接启动历史服务器调度器。Windows可以做语法及独立模块回归检查，但不能直接执行这条历史训练依赖链。

```sh
# 若尚无共同随机初始化，生成一次；已存在时不要覆盖。
python tools/sg_wh_training_runtime.py init --directory initializations --seed 20260828

# 复制并填写bindings.example.json；INIT_ROOT指向上一步目录。
python tools/sg_wh_training_runtime.py bind --template configs/V3M_EXECUTED/V3M_L1_R1_ori_500k.template.yml --bindings bindings.V3M.json --output bound/V3M_L1_R1_ori.yml

# 仅校验配置、注册、网络构建与学习率表，不读取数据、不训练。
python tools/train_v3m_r1_ori_portable.py --check-only -opt bound/V3M_L1_R1_ori.yml

# 在已验证CUDA环境与真实bank后显式启动。
python tools/train_v3m_r1_ori_portable.py -opt bound/V3M_L1_R1_ori.yml
```

本入口是配对训练入口，不是 `sg_wh_training_runtime.py train` 的替代通用训练器。两者注册的数据类型和配方不同。散装图片目录也不能直接代替历史bank；需要训练分片、有效区域、文字mask及line-mask侧车。设置 `lq_preprocess.enabled=false` 是因为输入已离线完成V3预处理，避免重复处理。

## 验证边界

本次同步执行CPU注册、网络构建、学习率边界、配置绑定及既有相关回归检查；不启动训练，不将CPU检查表述为新环境的CUDA训练验收。历史训练已完成的事实与本次代码交付验证分开记录。
