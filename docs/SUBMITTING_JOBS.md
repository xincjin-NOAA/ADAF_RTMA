# Submitting Training Jobs from YAML

Training runs can be defined in YAML and submitted with one command, instead of editing the sbatch launchers by hand. This is a port of the adaf repo's `submit_train_adaf*.sh` and `submit_experiments_from_yaml.sh`. Here both are built on a single tool, [tools/submit_experiments.py](../tools/submit_experiments.py), with two thin wrappers:

| Command | Use it for |
|---|---|
| `./submit_train.sh <config.yaml> [key=value ...]` | One run from a flat config, e.g. [configs/train_example.yaml](../configs/train_example.yaml) or [configs/train_ocelot3_example.yaml](../configs/train_ocelot3_example.yaml) |
| `./submit_experiments_from_yaml.sh [file\|-] [filter]` | Many runs from `defaults:` + `experiments:`, e.g. [experiment_configs.yaml](../experiment_configs.yaml) |

Flags accepted by both commands:

| Flag | Effect |
|---|---|
| `--dry-run` | Write the job scripts but don't submit them |
| `--debug` | Set the time limit to 00:30:00 |
| `--version V` | Put outputs under `<run_dir>/V` |
| `--force` | Allow overwriting an existing `ckpt.tar` |

`submit_experiments_from_yaml.sh` also takes `--set key=value`, which overrides an option for every selected experiment and can be repeated. `submit_train.sh` takes the same overrides as bare `key=value` arguments.

```bash
./submit_experiments_from_yaml.sh - smoke --dry-run      # preview the smoke tests
./submit_experiments_from_yaml.sh - goes_base            # submit one experiment
./submit_train.sh configs/train_example.yaml max_epochs=1 time=00:20:00
```

Run these on a login node. They need `python3` with `ruamel.yaml` (available in the ADAF environment) or PyYAML; set `PYTHON=/path/to/python` to choose the interpreter.

## What a submission does

For each selected experiment, the tool:

1. Merges the options: `defaults`, then the experiment's own keys, then `--set` or `key=value` overrides.
2. **Validates** every training option against the keys of the experiment's `config_filepath`. A typo such as `max_epoch` is rejected, with a suggestion, before anything is submitted.
3. Creates `run_dir` (default `training_runs/<name>`) and writes `job_<name>.sh` into it.
4. Submits the job with `sbatch`, unless `--dry-run` is set.

Everything for a run lives in `run_dir`:

- the SLURM logs `log_<jobid>.out` and `log_<jobid>.err`;
- the job script;
- `ckpt.tar` and `best_ckpt.tar`.

If `ckpt.tar` already exists there, the tool refuses to submit unless `resume: true`, `--version` or `--force` is given.

The job script uses the same launch recipe as `train_launcher_sbatch_ges_goes.sh`:

- one `torchrun` per node, with `gpus_per_node` ranks each and a c10d rendezvous, so `nodes > 1` is real multi-node DDP;
- `ptxas` and the Triton/Inductor caches staged on node-local `/tmp` for `torch.compile`, then removed when the job exits.

`ptxas` is taken from the directory of the Python interpreter that `env_setup` activates, so there are no hard-coded environment paths.

## Option reference

Any key that isn't a submit option is a **training option**, passed as `--key value`.

### Submit options

| Key | Default | Meaning |
|---|---|---|
| `name` | file stem | Run name (single-run files only; in multi-experiment files the experiment key is the name) |
| `description` | `""` | Shown at submit time and in the job script |
| `script` | `train_ges_goes.py` | Training entry point |
| `account` / `partition` / `qos` | `gpu-emc-ai` / `u1-h100` / `gpu` | SLURM account, partition and QOS; `qos: null` omits the QOS |
| `nodes` | 1 | Nodes; each runs one `torchrun` |
| `gpus_per_node` | 2 | Ranks per node; also sets `--gres=gpu:N` |
| `cpus_per_task` | 24 | CPUs per node, shared by the ranks and their data workers |
| `mem` | `"0"` | `--mem`; `"0"` means the whole node's memory |
| `time` | `04:00:00` | Time limit; `--debug` sets 00:30:00 |
| `extra_sbatch` | `[]` | Extra sbatch options, e.g. `["--exclusive"]` |
| `env_setup` | `[]` | Shell lines run first, such as `module load` and `conda activate` |
| `python` | `python` | Interpreter after `env_setup`; `ptxas` comes from its bin directory |
| `env_vars` | `{}` | Variables exported in the job |
| `stage_ptxas` | `true` | Stage `ptxas` and the compile caches on `/tmp` |
| `run_dir` | `training_runs/{name}` | Output directory; `{name}` is replaced by the experiment name |
| `resume` | `false` | Sets `resuming: True` and resumes from `resume_checkpoint_path`, which defaults to `<run_dir>/ckpt.tar` |

### Training options

- **Which keys are valid.** Every key in the `config_filepath` YAML: [params_lowres_ges_goes.yaml](../config/params_lowres_ges_goes.yaml), or [params_lowres_ocelot3.yaml](../config/params_lowres_ocelot3.yaml), which adds the `ocelot3_*` keys. [experiment_configs.yaml](../experiment_configs.yaml) lists all of them with their defaults, and marks the ones `train_ges_goes.py` declares but never reads as *(no-op)*.
- **Checkpoint paths.** `checkpoint_path` and `best_checkpoint_path` default to files in `run_dir`.
- **Value formats.** Write values as normal YAML:
  - booleans as `true` / `false`;
  - lists as `[4, 4]`;
  - `None` stays `None`.

## Command-line flags of the training script

`utils/misc_functions.set_user_params` now **generates** one flag per key of the config file, typed from its YAML value:

| YAML value | Flag behaviour |
|---|---|
| bool | Accepts `true` / `false` |
| list | Space-separated values |
| int or float | Parsed as that number type |
| `None` | Type inferred from the value given |

This means:

- every config key can be set on the command line, including the 25 (such as `arch`, `body_dim` and `stem_dims`) that previously had no flag;
- new config keys need no code change;
- `--compile_model False` now really means `False`. Before this change, boolean flags were parsed as strings, so `"False"` was truthy.

`python train_ges_goes.py --config_filepath <cfg> --help` lists every flag.

## Resuming

**Same run directory.** `resume: true` picks up `<run_dir>/ckpt.tar` and keeps writing to the same directory. [experiment_configs.yaml](../experiment_configs.yaml) has a `goes_base_resume` example.

**Different run directory.** Set `resume_checkpoint_path` to the checkpoint you want and give the new run its own `run_dir`.

**`max_epochs`.** Training continues from the checkpoint's epoch, so `max_epochs` must be larger than that epoch.

## Relation to the old launchers

`train_launcher_sbatch_ges_goes.sh` and `train_resume_launcher_ges_goes_sbatch.sh` still work unchanged. `goes_base` and `goes_base_resume` reproduce their settings, but through the YAML tooling. There is no submit tooling for inference yet, because `inference.py` has no command-line interface (it is a script with hard-coded paths).
