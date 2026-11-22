# NumPy 图像编码器实验

深圳大学 Python 程序设计 Project 1：使用 NumPy 完成课程指定 OpenCLIP ViT-B-32 图像编码器的前向计算，不训练模型。

输入为预处理后的 float32 数组 `[B,3,224,224]`，输出为 `[B,512]`，每张图片对应一个单位长度的特征向量。

## 文件结构

| 文件 | 用途 |
| --- | --- |
| image_encoder.py | 学生完成的六组 TODO：切块、嵌入、多头注意力、MLP 与残差、CLS 输出 |
| helpers.py | 课程提供的 LayerNorm、GELU、注意力系数与单位向量辅助函数 |
| preprocess_image.py | 课程提供的缩放、裁剪和像素归一化 |
| runtime.py / run.py | 课程提供的资源加载与命令行运行入口 |
| check.py | 课程提供的分阶段数值检查器 |
| config.json | 固定模型的配置 |
| requirements.txt | 课程提供的依赖范围 |
| requirements-tested.txt | 本次验证环境中的依赖版本 |
| check_results.txt | 完整检查输出（UTF-8） |
| assets/, weights/ | 完整测试图片、参考数组、模型参数及其清单 |
| lecture-demos/, practice/ | 课堂演示与练习材料 |
| transformer_steps.py | 小尺寸 Transformer 算术演示 |

## 课程材料与复现

本仓库包含完整的学生项目目录：课程代码与说明、课堂演示、练习材料、十张测试图片、模型参数和参考数组。克隆仓库后不需要再覆盖课程包文件。

```powershell
git lfs install
git clone https://github.com/hangbose/szu-python-project1-image-encoder.git
cd szu-python-project1-image-encoder
```

模型权重 `weights/image_encoder.npz` 通过 Git LFS 保存。克隆时需要安装并启用 Git LFS，才能取到真实权重文件。仓库中的参数清单和 SHA-256 指纹由 `check.py --stage assets` 核验。

项目关键材料包括：

```text
weights/image_encoder.npz
weights/manifest.json
assets/images/（十张课程图片）
assets/images.json
assets/manifest.json
assets/reference_outputs.npz
```

参数形状和配置必须匹配，不应单独修改配置来更换模型。

## 环境与运行

已验证环境：Windows、Python 3.13.14、NumPy 2.5.4、Pillow 12.3.0、ImageIO 2.38.1。原课程 README 要求 Python 3.9 或更新版本；这里只报告实际验证的版本。

在项目根目录执行：

```powershell
python -m pip install -r requirements.txt
python check.py --stage all
python run.py --output output_tokens.npz
```

若需安装本次验证的准确依赖版本，可用 `python -m pip install -r requirements-tested.txt` 替代第一条。依赖安装可能需要联网；具备依赖与课程资源后，计算无需联网、GPU、PyTorch 或 OpenCLIP 软件包。

处理自己的图片：

```powershell
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

图片路径需要替换为实际路径。输出文件包含 `ids` 和 `output_tokens`，顺序与输入一致。默认处理课程包的十张图片。

## 验证结果

完整记录见 `check_results.txt`。所有阶段通过；以下为实际运行的绝对误差。

| 阶段 | 最大绝对误差 | 平均绝对误差 |
| --- | ---: | ---: |
| preprocess | 0 | 0 |
| patches | 0 | 0 |
| tokens | 4.005e-05 | 1.289e-06 |
| block0 | 7.063e-06 | 3.250e-07 |
| last_tokens | 4.768e-05 | 2.696e-06 |
| output | 5.439e-07 | 8.311e-08 |
| batch | 4.619e-07 | 7.632e-08 |

检查器逐元素检查 `abs(actual-reference) <= atol + rtol*abs(reference)`，并核对形状、float32 与数值有限性。preprocess 的 atol 为 2e-6，output/batch 为 2e-5，其余为 2e-4；rtol 均为 2e-4。最终输出还检查单位长度。

batch 使用索引 `[1,0,1]` 检查输入顺序改变和重复图片。通过这些检查说明符合所提供的测试条件，不代表完成任意图片的效果评估。参考数组只供检查器比较，不作为编码器输入。

## 实现与调试记录

多头组织先将 `[B,N,768]` 拆为 `[B,N,12,64]`，再转置为 `[B,12,N,64]`；合并时反向调整轴顺序。每个头仍处理全部 token，拆分的是特征轴。

实现 MLP 时曾遗漏权重转置，使 `[B,N,768]` 与 `[3072,768]` 无法相乘。根据课程权重 `[out,in]` 的布局，修正两层权重方向后通过检查。最终输出投影已经是 `[768,512]`，无需转置。

## 贡献与辅助说明

学生在课程框架中完成六组 TODO 并执行数值验证；预处理、辅助函数、加载脚本、检查器、配置及依赖范围来自课程包。

ChatGPT 提供概念讲解、代码审阅、维度错误定位和 LaTeX 报告文字辅助；后续协助转换日志编码、整理 README 和仓库、删除 return 后不可执行的旧占位报错。Git 记录从整理提交材料时开始，不代表此前逐步开发的历史，不补造历史提交。

课程提供文件保留原有归属，本仓库不另行声明它们的开源许可证。
