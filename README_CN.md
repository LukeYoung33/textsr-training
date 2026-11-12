# SG / WH 训练文件包使用说明

已执行的 **V3M·L1-R1_ori·总加粗1px GT·500K** 配方已单独同步，见 [实际训练配置和启动说明](docs/training/V3M_L1_R1_ori_已执行配方.md)。它使用配对训练、原始R1、分段学习率及S22Dataset，区别于下面的SG/WH计划模板；数据与权重不随代码上传。

本包包含当前训练代码、配置模板和辅助工具，不包含数据。正式训练前，仍需绑定数据与权重路径，并在目标 GPU 上完成验证。

## 一、目录与用途

| 目录或文件 | 用途 |
|---|---|
| basicsr/、realesrgan/ | 训练库、模型结构、损失、数据读取代码与自定义算子源码 |
| tools/ | 训练依赖工具，以及便携启动、路径绑定和初始化工具 |
| configs/SG_PRE/ | 5 份共享 L1 预训练模板 |
| configs/SG_FT/ | 10 份 L1 / L1-R1_ori 微调模板 |
| configs/WH_SCRATCH/ | 10 份新版数据从头训练模板 |
| configs/WH_KD/ | 8 份特征蒸馏组件配置，尚非完整训练入口 |
| options/、tests/ | 四档教师结构配置片段与相关 CPU 回归测试 |
| docs/training/ | 中英文训练计划、结构与蒸馏说明 |
| requirements_runtime.txt | 运行依赖列表 |
| environment_reference.json | 本地环境版本参考，不是目标 GPU 环境锁定文件 |
| bindings.example.json | 数据、初始化、教师权重和输出路径的绑定模板 |
| STAGES.json | 33 个计划阶段与就绪状态 |
| MANIFEST.json | 文件 SHA256 校验清单 |
| VALIDATION.json | 交付副本的验证结果与尚未验证的范围 |

## 二、环境准备

建议使用独立 Python 环境。先根据目标 GPU/CUDA 安装匹配的 torch 和 torchvision，再安装 requirements_runtime.txt 中的其他依赖。本包不会自动安装或下载。

原 requirements.txt 仅供参考，其中通过 pip 安装的 basicsr 不能覆盖包内修改过的 basicsr 源码。请在解压后的文件包根目录执行命令，无须执行 setup.py 或可编辑安装。

本地验证使用 CPU PyTorch，不能替代目标机器的 CUDA 验证。

## 三、检查环境并生成共同随机初始化

```sh
python tools/sg_wh_training_runtime.py check
python tools/sg_wh_training_runtime.py init --directory initializations --seed 20260828
```

初始化工具生成以下文件：

- V3M.pth
- V3L16.pth
- V3L32.pth
- V3L64.pth
- V3L128.pth

这些是随机初始权重，不是预训练模型。只生成一次，再将同一份文件复制到 SG 和 WH。已有同名文件时，工具拒绝覆盖。

## 四、绑定数据与权重路径

复制 bindings.example.json，分别填写 SG 和 WH 的实际路径：

| 字段 | 含义 |
|---|---|
| PRETRAIN_TRAIN_MANIFEST | 预训练数据的训练集 manifest |
| PRETRAIN_VAL_MANIFEST | 预训练数据的验证集 manifest |
| NEW_TRAIN_MANIFEST | 新版数据的训练集 manifest |
| NEW_VAL_MANIFEST | 新版数据的验证集 manifest |
| INIT_ROOT | 共同随机初始权重目录 |
| SG_PRETRAIN_ROOT | SG 选出的预训练权重目录 |
| WH_TEACHER_ROOT | WH 选出的教师权重目录 |
| RUN_ROOT | 实验输出父目录 |

不同阶段只要求填写实际引用的字段。SG_PRETRAIN_ROOT 中的权重文件名与初始化文件名一致；WH_TEACHER_ROOT 中的教师权重例如 V3L16_L1.pth。

下面以 WH 的 V3M L1 从头训练为例：

```sh
python tools/sg_wh_training_runtime.py bind --template configs/WH_SCRATCH/V3M_L1.template.yml --bindings bindings.WH.json --output bound/V3M_L1.yml
python tools/sg_wh_training_runtime.py check --config bound/V3M_L1.yml
```

真实数据和 CUDA 验证通过后，再显式启动：

```sh
python tools/sg_wh_training_runtime.py train -opt bound/V3M_L1.yml
```

便携入口注册现有 Block256RegionModel 和新增教师模型，然后调用现有 BasicSR 训练流程。模板使用计划建议参数，仍需评审和实际验证。L1-R1_ori 对应原始 Missing / False 损失，系数为 1 / 0.25，不进行笔画粗细归一化。

tools/ 中同时保留了历史依赖脚本，其中可能有旧机器路径和旧任务参数。不要直接启动历史远程控制器；本轮模板使用上述便携入口。

## 五、数据格式约定

本包不含图片、数组分片或真实样本清单。监督训练模板使用现有 OfflineAugmentedRegionDataset：

- 每份 manifest 的父目录作为对应数据根目录。
- 根目录需包含 COMPLETE.json，以及 manifest 引用的索引、分片和元数据。
- 训练和验证可以来自不同根目录。
- 完成标记、build_id 和画布尺寸必须匹配。
- 分片为 uint8[N,256,256,8]，依次保存 BGR LQ、BGR GT、文字掩码和有效区域掩码；读取器转换为 RGB 张量。

格式参考 tools/build_offline_augmented_dataset.py。散装图片目录不能直接替代离线数据包。若单独整理的数据使用其他格式，需明确接入合适的读取器，不能仅修改路径就启动。

## 六、蒸馏当前可用范围

八层特征损失、通道适配器和单设备更新引擎位于 realesrgan/models/v3_feature_distillation.py，相关 CPU 测试已通过。

**configs/WH_KD/ 下的配置仅描述蒸馏组件，不是完整训练配置。** 完整数据、采样、续训流程、基础损失绑定（尤其 R1 与掩码）、实际 CUDA，以及需要使用的混合精度和多卡接入尚待完成。启动工具会拒绝将这些组件配置直接用于监督训练。

没有已训练教师权重随包交付。特征权重 1 是未调优默认值，应按照中文版计划先进行 20K 权重试跑，再重新初始化开展正式实验。

## 七、验证与完整训练计划

```sh
python tools/profile_v3l_teachers.py
python -m pytest -q tests/test_v3l_feature_distillation.py tests/test_v3m_textboost.py --basetemp=.artifacts/pytest-kit
```

完整分工、蒸馏方法和预算见包内 `docs/training/sg_wh_training_plan_20261106_CN.md`。

本包不含数据集、训练图片、真实数据 manifest、训练结果、已训练模型、缓存、Git/SVN 对象、安装好的环境或编译后二进制。可选 OCR/感知损失的外部权重不包含，当前模板关闭这些目标。

VALIDATION.json 明确区分交付副本检查与目标 CUDA 就绪状态。evaluation/ 下的既有 CPU 验证记录不能视为目标机器验收。解压不会自动训练或访问网络。
