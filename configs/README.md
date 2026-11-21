# configs

This folder stores configuration files used by baselines/scripts.

## Keyword baseline

- `keyword_rules_base.json`: Base keyword-scoring rules (hand-curated).

## External wordlists (optional)

You can add additional wordlists and point the baseline CLI to them.
Downloaded third-party lists and their metadata report are stored under
`external_sensitive_wordlists/` in this config directory.

Recommended source used in experiments:
- `@dsojevic/profanity-list` (MIT license): https://github.com/dsojevic/profanity-list
  - Wordlist: `dsojevic_profanity_en.txt`
  - License: `dsojevic_profanity_list_LICENSE.txt`
