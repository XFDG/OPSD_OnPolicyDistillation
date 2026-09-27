# OPSD input data provenance

All four inputs below were downloaded on the Beijing CPU pod on 2026-09-24. The
Hugging Face dataset revisions are pinned; the local names match the paths
expected by `src/data/prepare_grpo_data.py`. Downloads and temporary files were
kept on the shared GPFS volume under `/volume/pt-train/users/zhaoye/OPSD-runtime`.

| Local path (relative to `data/`) | Source file and pinned revision | SHA-256 | Rows | Columns |
| --- | --- | --- | ---: | --- |
| `DAPO-Math-17k-dedup/distinct-prompts-with-rewards.parquet` | [YouJiacheng/DAPO-Math-17k-dedup](https://huggingface.co/datasets/YouJiacheng/DAPO-Math-17k-dedup/resolve/6e26a33abdabd3e6aaa1d742326b790758f7dbc5/distinct-prompts-with-rewards.parquet), revision `6e26a33abdabd3e6aaa1d742326b790758f7dbc5` | `c65f47247b9b13cb62321462243c2d1fe8d4ede2d800644a7dda61fea76d4286` | 17,398 | `prompt`, `reward_model` |
| `AIME_2024/aime_2024_problems.parquet` | [HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024/resolve/2fe88a2f1091d5048c0f36abc874fb997b3dd99a/data/train-00000-of-00001.parquet), revision `2fe88a2f1091d5048c0f36abc874fb997b3dd99a` | `26139847601a5037c237d5928b195e7260ca8074cf4f264b794af42847f79ccf` | 30 | `id`, `problem`, `solution`, `answer`, `url`, `year` |
| `AIME_2025/train.jsonl` | [math-ai/aime25](https://huggingface.co/datasets/math-ai/aime25/resolve/563bb8404243c5f09de6ec262f2db674fe5bce9b/test.jsonl), revision `563bb8404243c5f09de6ec262f2db674fe5bce9b` | `b4e273c02d3e7fe1b74b59eae768fc8230bfb0f79539890cb56f4361caac0331` | 30 | `problem`, `answer`, `id` |
| `MATH-500/test.jsonl` | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl), revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` | `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132` | 500 | `problem`, `solution`, `answer`, `subject`, `level`, `unique_id` |

The AIME 2024 source filename is `data/train-00000-of-00001.parquet`; it was
downloaded unchanged and placed at the filename required by OPSD. Another
public file called `aime_2024_problems.parquet` from `Maxwell-Jia/AIME_2024`
has capitalized `Problem` and `Solution` columns and is incompatible with the
current preparation script. The chosen HuggingFaceH4 file has the lowercase
columns that script expects.

**AIME 2024 answer handling:** The upstream OPSD `extract_boxed_answer`
function extracts the first simple `\boxed{...}` from `solution`. Compared with
the dataset's explicit `answer` column, that produced different strings for 5
of 30 rows (zero-based indices 0, 17, 19, 20, 28); one solution has no
`\boxed{...}` at all. The local data preparation scripts now prefer
`str(example["answer"])` as `reward_model.ground_truth`, preserving the answer
key and avoiding a full solution as the ground truth. The processed Parquet
files below were generated with this local fix.

Raw input checks passed: all four files are nonempty, the Parquet files are
readable, and every JSONL record has a nonempty `problem` and nonnull `answer`.

## Prepared outputs

On 2026-09-24, `prepare_grpo_data.py` ran with `--train-ratio 0.8 --seed 42`,
and `process_eval_data.py` ran with `--instruction_variant boxed`. Both used
the GPFS resident `aiperf-venv` Python and GPFS paths for temporary and Hugging
Face cache files. The commands completed successfully on the Beijing CPU pod.

| Output path (relative to `data/`) | Rows | SHA-256 |
| --- | ---: | --- |
| `grpo_processed/train.parquet` | 13,918 | `c77673dab5631621d6f71963d8dc22b5284c4eb6483b024c298e173d73a6d0b6` |
| `grpo_processed/val_dapo.parquet` | 3,480 | `130c8b759981a20513f3abec5dcd729219e8d3d64efbd389e76369392380375d` |
| `grpo_processed/val_aime24.parquet` | 30 | `eea4f77c6ef06e1174b46318aefd519f983eae27412d32ac55b4f76bcb5a110a` |
| `grpo_processed/val_aime25.parquet` | 30 | `eab04abf439f31f2c5258729861c6acf842aea3d7ea42499919a98fa8a686799` |
| `grpo_processed/val_math500.parquet` | 500 | `076e3f6d4e75c54bfeb7d50847c6043ada06ed187d92aac10fdd26bdd4af1c03` |

The three `eval_processed/boxed/val_*.parquet` files have the same respective
row counts and SHA-256 values as the GRPO AIME 2024, AIME 2025, and MATH-500
validation files. All processed files have the five verl data columns and a
nonnull `reward_model.ground_truth`. The DAPO train and validation splits are
disjoint and cover all 17,398 original rows. AIME 2024 ground truths match the
raw `answer` column for all 30 rows in both processed output locations.
