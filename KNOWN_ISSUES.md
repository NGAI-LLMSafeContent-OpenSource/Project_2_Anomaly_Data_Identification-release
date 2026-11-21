# 已知问题与验收边界

对应原仓提交：`e2cc59e1ac70f6ae9859938b5ea65f6102ea1d5b`。本分支不修正业务源码；以下问题不作为“已修复”发布。

## 外部运行条件

原始基准数据与大型模型需自行获取。第三方词库按其许可单独准备，可通过 `scripts/download_sensitive_wordlists.py` 获取；没有把许可未确认的词库内容改标为Apache。神经元方法存在已复现的版本兼容问题，见KNOWN_ISSUES.md。

发布检查仅覆盖release_checks.json中声明的文件一致性、规则/协议、基础HTTP或构建测试；真实模型、训练、GPU、受限数据、原始知识库和跨课题服务未配置的路径不计作真实端到端通过。

## 原仓状态与保留问题

- 在本轮独立环境的Transformers 4.57.6与微型Qwen3上，原仓和发布版的神经元路径均报 `create_causal_mask() got an unexpected keyword argument inputs_embeds`。对应 `src/filterguard/qwen3_truncated.py`，源码原样保留；由维护者提供兼容环境或在原仓修复。

## 问题处理

请将错误与复现条件提交至公开仓Issues，由原仓维护者处理。发布分支仅同步审核后的源码和说明，其他平台不单独修改实现。
