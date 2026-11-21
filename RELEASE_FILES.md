# 发布文件清单

原仓：`NGAI-LLMSafeContent/Project_2_Anomaly_Data_Identification`；源提交：`e2cc59e1ac70f6ae9859938b5ea65f6102ea1d5b`；分支：`release/1.0`。

本清单逐文件列出发布内容。原有非README实现/配置/资源保持Git对象字节和模式一致；README与发布新增材料单独登记。RELEASE_MANIFEST.json和SHA256SUMS由发布器生成，不形成自引用哈希。

## 来自原仓、原样保留的文件

- `.gitignore`
- `.gitmodules`
- `configs/README.md`
- `configs/dsojevic_profanity_en.txt`
- `configs/dsojevic_profanity_list_LICENSE.txt`
- `configs/keyword_external_wordlists.json`
- `configs/keyword_rules_base.json`
- `published_skills/claude-code/generate-multi-ethnic-taboo-data/SKILL.md`
- `published_skills/claude-code/generate-multi-ethnic-taboo-data/agents/openai.yaml`
- `published_skills/claude-code/generate-multi-ethnic-taboo-data/references/seed_rules.jsonl`
- `published_skills/claude-code/generate-multi-ethnic-taboo-data/scripts/check_difficulty_ladder.py`
- `published_skills/claude-code/generate-multi-ethnic-taboo-data/scripts/select_seed_rules.py`
- `published_skills/claude-code/generate-multi-ethnic-taboo-data/scripts/validate_samples.py`
- `published_skills/codex/generate-multi-ethnic-taboo-data/SKILL.md`
- `published_skills/codex/generate-multi-ethnic-taboo-data/agents/openai.yaml`
- `published_skills/codex/generate-multi-ethnic-taboo-data/references/seed_rules.jsonl`
- `published_skills/codex/generate-multi-ethnic-taboo-data/scripts/check_difficulty_ladder.py`
- `published_skills/codex/generate-multi-ethnic-taboo-data/scripts/select_seed_rules.py`
- `published_skills/codex/generate-multi-ethnic-taboo-data/scripts/validate_samples.py`
- `pyproject.toml`
- `scripts/benchmark_throughput.py`
- `scripts/build_seed_fineweb_mixed_dataset.py`
- `scripts/download_hf_benchmarks.py`
- `scripts/download_sensitive_wordlists.py`
- `scripts/download_third_datasets.sh`
- `scripts/eval_llama_guard4_official_binary.py`
- `scripts/eval_llama_guards_binary.py`
- `scripts/eval_ngram_neuron_cascade.py`
- `scripts/eval_oss_safeguard_binary.py`
- `scripts/eval_oss_safeguard_vllm_api.py`
- `scripts/export_misclassified_cases.py`
- `scripts/generate_dataset_v0_1.py`
- `scripts/generate_dataset_v2.py`
- `scripts/generate_neutral_minimal_pairs.py`
- `scripts/generate_story_style_samples.py`
- `scripts/mem_scripts.sh`
- `scripts/mine_ngram_keyword_rules.py`
- `scripts/plot_dataset_pie.py`
- `scripts/prepare_multi_ethnic_benchmark.py`
- `scripts/run_multi_ethnic_jobs.py`
- `scripts/score_partial_per_sample.py`
- `scripts/shuffle_dataset_v0_1_to_v0_1_1.py`
- `scripts/translate_neutral_pairs.py`
- `src/filterguard/__init__.py`
- `src/filterguard/cascade.py`
- `src/filterguard/cli.py`
- `src/filterguard/data.py`
- `src/filterguard/fineweb.py`
- `src/filterguard/keyword.py`
- `src/filterguard/metric.py`
- `src/filterguard/multi_ethnic.py`
- `src/filterguard/neuron.py`
- `src/filterguard/qwen3_5_truncated.py`
- `src/filterguard/qwen3_truncated.py`
- `src/filterguard/risk.py`
- `src/filterguard/server.py`
- `src/filterguard/service.py`
- `src/filterguard/token_anomaly.py`
- `src/filterguard/torchutil.py`
- `src/filterguard/util.py`
- `src/fineweb_edu_baseline_abnormal.py`
- `src/hybrid_keyword_fineweb_neuron_abnormal.py`
- `src/hybrid_keyword_neuron_abnormal.py`
- `src/keyword_baseline_abnormal.py`
- `src/metric.py`
- `src/neuron_baseline_abnormal.py`
- `src/neuron_layerwise_earlyexit_abnormal.py`
- `src/neuron_risk_multiclass_abnormal.py`
- `src/token_anomaly_baseline_abnormal.py`
- `tests/test_cli_api.py`
- `tests/test_keyword_wordlist_manifest.py`
- `tests/test_risk_multiclass.py`
- `tests/test_token_anomaly_workflow.py`

## README及发布新增材料

- `KNOWN_ISSUES.md`
- `LICENSE`
- `README.md`
- `RELEASE_FILES.md`
- `SOURCE_ALIGNMENT.json`
- `THIRD_PARTY_NOTICES.md`
- `release_checks.json`
- `requirements-ml.txt`
- `requirements.txt`
- `tools/release_check.py`

## 生成的校验文件

- `RELEASE_MANIFEST.json`
- `SHA256SUMS`

## 排除范围

明确弃用的历史原型、旧版本快照、缓存、数据库、用户配置和预跑产物未发布。模型、原始数据/知识库和外部服务按README准备，不以缩减/改写同路径数据文件冒充完整交付。源码原有错误保留，见KNOWN_ISSUES.md。
