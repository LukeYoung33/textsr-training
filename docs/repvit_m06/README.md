# RepViT-M0.6 训练与部署代码

对应最初 VOSR-GT 总加粗 1px 修复版 5284 对、500K iterations 的实验，不是后续 Titans246 重训版本。只提交代码、配置和数值/性能记录；数据集、初始化、checkpoint、ONNX/TFLite 和设备输入不入库。

## 结构与配方

原始结构在 `tools/idea010_repvit_arch.py`，共享解码模块来自 `tools/idea009_evit_arch.py`。输入输出为 PyTorch NCHW `[N,6,128,128]` Y4UV，对应 RGB256 画布；编码器宽度 40/80/160/320，17 blocks，7 SE，五级 skip 解码及输入残差。训练使用精确 GELU，部署副本改为 tanh 近似。

两组使用共同随机初始化 seed=20260828、batch=4、Adam lr=1e-4、betas=(0.9,0.99)、500K iterations。学习率在 200K/300K/400K/450K 降到 5e-5/2.5e-5/1e-5/5e-6。R1_ori 权重 missing=1、false=0.25，2000 步开始、5000 步渐入；L1-only 关闭两项 R1。基础 native L1 的通道权重为 1/1/1/1/0.25/0.25。无在线退化或对比度对齐。

`configs/REPVIT_M06_EXECUTED/ori.executed.yml` 是实际 R1 配置原始字节快照，供追溯；两个 template 保留训练配方并替换路径，使用既有 FilteredPairedModel，省略原服务器监督器的停止标志扩展。L1 模板依据原控制器中已执行的 loss 消融规则生成。历史验证沿用 583 个父样本，不代表同版本修复 GT 的新验证集。

## 启动

Linux 正式训练复用本仓库的 S22Dataset、FilteredPairedModel 和 LatestVOSR500K；历史数据依赖包含 POSIX 文件锁。先安装仓库既有依赖，不能用 pip 的 BasicSR 覆盖本库。配置需要完整的既有离线 bank、有效区域/文字 mask 和 line-mask 侧车，散装图片不能直接替代。

```sh
# 两组共用一次生成的初始化；拒绝覆盖文件。
python tools/train_repvit_m06_portable.py --initialize initializations/RepViT_M06.pth
# 复制 configs/REPVIT_M06_EXECUTED/bindings.example.json 并填入真实路径。
python tools/sg_wh_training_runtime.py bind --template configs/REPVIT_M06_EXECUTED/ori.template.yml --bindings bindings.repvit.json --output bound/repvit_ori.yml
python tools/sg_wh_training_runtime.py bind --template configs/REPVIT_M06_EXECUTED/l1.template.yml --bindings bindings.repvit.json --output bound/repvit_l1.yml
python tools/train_repvit_m06_portable.py --check-only -opt bound/repvit_ori.yml
python tools/train_repvit_m06_portable.py -opt bound/repvit_ori.yml
# 第二组使用另一 GPU / 分布式端口，替换对应配置。
python tools/train_repvit_m06_portable.py -opt bound/repvit_l1.yml
python -m pytest tests/test_idea010_repvit.py -q
```

Windows 的 check-only 仅验证配置与独立架构前向，不宣称完成 Linux 训练依赖注册。新环境仍需 CUDA 和数据 smoke 验证；本次同步不启动训练。

## 部署修复与历史脚本

`tools/idea010_repvit_tanh_arch.py` 提供 `RepViTM06TanhNativeRestore`，保留原模型的参数键和形状，只将18处 GELU 切换为 tanh 近似；可 strict 加载原始权重。它不是本轮已执行训练结构，不能替换历史配方后仍宣称原实验复现。稳定 Tanh 与 SE 的 GPU 修复仍需要下述转换步骤。

`historical/` 保存真实执行过的训练监督器、导出、转换及审计脚本，以及配置和部署绑定。它们含原服务器路径及报告依赖，是历史记录，不能在新机器直接运行。便携训练应使用上面的新入口，不启动历史服务器控制器。

历史转换的执行链为 export_model.py → convert_fixed.py → audit_model.py：

1. 严格加载 R1_ori 500K checkpoint，SHA256 `fe21c9ad603a50b0782668631fcc50b1d6fc8c317bd28645176a3b9018b62165`；融合 14 RepDW / 81 BN，保留 87 卷积。
2. 18 GELU 改为 tanh 近似，消除 FlexErf。
3. 转换期将 tanh 降低为 `2*sigmoid(2*x)-1`，避免低精度指数溢出。
4. 7 SE 的空间 gate 通过重排、nearest 物化到 feature 同形，再相乘恢复 shape，避免 GPU 混用 batch0 gate。
5. 保留 native batch16；TFLite FP32 NHWC I/O `[16,128,128,6]`，权重 FP16，不增加最终 PACK。

转换依赖外部 `onnx2tflite-benchmark` 工具包的 `common`、`precheck_fix`、`gpu_batch_compat`、`audit_native_batch` 及其配置绑定的 onnx2tflite 转换器。历史环境 PyTorch1.12/TF2.10/ONNX1.12；ONNX 导出环境另用 PyTorch2.12/ONNX1.21。外部工具源码不在本项目授权范围内，本仓库不包含该转换器；需另行准备并修改历史路径、输入和产物目录。历史脚本还依赖评估绑定与真实测试输入，不能仅靠这些源码重现原有设备测量。

已部署文件 SHA256 `bcfc5efd12b7e5bc3a2d4239b08349a8452d431044c937eb72236dc73031df50`，大小 4,622,992 字节。TB323FU / Adreno840 / TFLite2.10 实测 583/583 节点进入单 GPU 分区，无 CPU 回退；低精度 Invoke 7.334ms/b16，含输入输出复制11.491ms。完整自包含历史报告见 [deployment_report.html](deployment_report.html)。这些是历史结果，不是此次同步在新环境重新测量，也不等于最终主观画质验收。
