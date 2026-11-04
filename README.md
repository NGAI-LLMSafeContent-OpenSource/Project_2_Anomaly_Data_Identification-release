# FilterGuard 异常数据识别工具

发布方：NGAI-LLMSafeContent；版本：1.0；日期：2026-11-04；Apache-2.0。
源仓库 NGAI-LLMSafeContent/Project_2_Anomaly_Data_Identification，固定提交 e2cc59e1ac70f6ae9859938b5ea65f6102ea1d5b。

公开 filterguard 模块，提供关键词筛选 CLI/HTTP，以及神经元、token 异常检测代码。不包含原始数据、外部敏感词库、模型权重、第三方子模块和实验产物。

## 前置条件与最小运行

Python >=3.12。关键词模式仅使用标准库，无需 pip 安装第三方依赖。从仓库根目录执行：

```sh
python tools/release_smoke.py
python tools/filterguard_cli.py predict --methods keyword --text "release-example-marker"
python tools/filterguard_cli.py serve --methods keyword --host 127.0.0.1 --port 8000
```

GET http://127.0.0.1:8000/health；POST /predict/keyword，JSON 请求体 `{"text":"ordinary educational text"}`。随包规则仅匹配人工构造标记 release-example-marker，用于验证流程，不是正式敏感词库，不可用作实际内容风险判断。自行修改 configs/keyword_rules_base.json 后再用于业务。

## 可选模型路径

神经元与 token 模型需自行安装 requirements-ml.txt 中的可选依赖，并准备有权使用的 Qwen 模型和校准数据。`python tools/filterguard_cli.py predict --help` 可查看模型名称、设备与本地数据配置。使用自己的合法数据路径，避免自动下载未获授权的数据。该发布没有包含模型服务或权重。

本次自动验证只覆盖关键词 CLI、真实 HTTP 预测与外部词库清单加载/缺失拒绝，测试使用新编写的中性标记。神经元、token 模型和 GPU 推理未验证，原仓的重模型与外部数据测试未计入通过数。

第三方见 THIRD_PARTY_NOTICES.md，发布文件与 SHA 校验见 RELEASE_MANIFEST.json、SHA256SUMS。
