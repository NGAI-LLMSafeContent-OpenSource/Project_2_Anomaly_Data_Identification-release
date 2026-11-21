# FilterGuard 异常数据识别

面向LLM训练与评测数据，提供关键词、神经元激活与token异常检测及统一风险分类。

发布单位：NGAI-LLMSafeContent。发布分支：`release/1.0`；更新日期：2026-11-21；自研代码许可：Apache-2.0，第三方保留原许可。

## 功能

- 关键词CLI/HTTP、可选外部词库
- 神经元、ONION/token基线、L1/L2统一风险分类
- 关键词/神经元/FineWeb级联、吞吐评测与数据准备工具

## 安装与快速开始

Python ≥3.10。关键词运行使用轻量依赖；模型、数据和可视化扩展按pyproject.toml安装。

从公开仓下载同一发布分支：

```bash
git clone --branch release/1.0 https://github.com/NGAI-LLMSafeContent-OpenSource/Project_2_Anomaly_Data_Identification-release.git
cd Project_2_Anomaly_Data_Identification-release
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell改用 .venv\Scripts\Activate.ps1
```

以下命令均以仓库根目录为起点，注释中的外部条件需先准备：

```bash
python -m pip install -e .
filterguard predict --methods keyword --text "A neutral document about gardening."
filterguard serve --methods keyword --host 127.0.0.1 --port 8000
# 使用模型方法前安装扩展
python -m pip install -e ".[ml,data,viz]"
PYTHONPATH=src python src/neuron_risk_multiclass_abnormal.py --help
PYTHONPATH=src python scripts/benchmark_throughput.py --help
```

## 使用与输出

健康接口为 `/health`，关键词预测为 `POST /predict/keyword`，请求体形如 `{"text":"A neutral document."}`。模型校准数据使用JSONL或Hugging Face磁盘数据集，字段和标签见原仓CLI帮助。各基线入口保留在 `src/`，数据准备与评测工具在 `scripts/`。

## 配置与外部依赖

原始基准数据与大型模型需自行获取。第三方词库按其许可单独准备，可通过 `scripts/download_sensitive_wordlists.py` 获取；没有把许可未确认的词库内容改标为Apache。神经元方法存在已复现的版本兼容问题，见KNOWN_ISSUES.md。

真实凭据只在自己的运行环境或平台Secrets中配置，不提交.env、用户上传、模型权重或运行结果。原仓文件中的内网服务地址经发布负责人确认按原样保留，使用者应在本地配置自己的可达服务。

## 验证与已知问题

完成对应依赖准备后运行：

```bash
python tools/release_check.py
```

检查包含发布文件完整性、原仓文件一致性与声明的最小测试。它不会自动下载模型或申请API，也不代表全部真实业务/准确率验收。保留的原仓错误、未运行项及外部依赖见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)。

## 发布版本与文件清单

本版本以原开发仓主分支提交 `e2cc59e1ac70f6ae9859938b5ea65f6102ea1d5b` 为源。功能实现与对应原文件保持一致；README、许可说明和发布辅助材料单独登记。当前功能的必要源码、配置、界面与运行资源保留，明确弃用的原型、历史结果和开发缓存不捆绑。

- [逐文件发布清单](RELEASE_FILES.md)
- [原仓文件与Git对象对应关系](SOURCE_ALIGNMENT.json)
- [完整文件SHA-256及权限](RELEASE_MANIFEST.json)，校验表为SHA256SUMS

校验值按Git存储字节生成；若原.gitattributes规定检出换行转换，请用Git对象或发布检查脚本校验，避免将LF/CRLF误认为源码变化。各公开平台release/1.0使用同一个提交，平台生成的运行分支不属于此发布清单。

## 反馈、贡献与许可证

请通过 [公开仓Issues](https://github.com/NGAI-LLMSafeContent-OpenSource/Project_2_Anomaly_Data_Identification-release/issues) 提交复现步骤、环境和日志摘要，勿附真实凭据或用户数据。代码问题由课题维护者在原开发仓处理，发布负责人同步审核后的版本；不要仅修一个平台造成分歧。

自研代码采用[Apache-2.0](LICENSE)。模型、数据集、外部服务和第三方代码遵循各自许可与使用条件，见[第三方说明](THIRD_PARTY_NOTICES.md)。
