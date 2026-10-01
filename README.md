# kenv

**Disposable Kaggle kernels for your terminal.** &nbsp; *By EpicRaven* &nbsp; · &nbsp; Version 1.6.0

kenv starts a Kaggle kernel on demand, gives you a Jupyter URL for your local notebook (VS Code, Cursor), runs your scripts on Kaggle's CPU or GPU, and pulls the results back into your own project folder. When you are done, the kernel is **always deleted** (on `exit`, Ctrl-C, closing the terminal, or `kill`).

`kenv.py` is a single file that needs only the Python standard library. The optional dashboard (`kenv ui`) ships as a separate prebuilt folder.

---

## Table of contents

1. [Why kenv](#why-kenv)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
   - [Linux](#linux)
   - [macOS](#macos)
   - [Windows](#windows)
5. [Kaggle credentials](#kaggle-credentials)
6. [Verify the setup](#verify-the-setup)
7. [Quick start](#quick-start)
8. [Feature walkthroughs](#feature-walkthroughs)
9. [Command reference](#command-reference)
10. [Using kenv from code](#using-kenv-from-code)
11. [The dashboard (`kenv ui`)](#the-dashboard-kenv-ui)
12. [Configuration](#configuration)
13. [Updating and uninstalling](#updating-and-uninstalling)
14. [Troubleshooting](#troubleshooting)
15. [Notes and limits](#notes-and-limits)

---

## Why kenv

Kaggle gives free CPU and GPU time, but using it from a laptop normally means uploading notebooks by hand, copying outputs back, and remembering to shut things down. kenv removes that friction:

- **Free accelerators from your own editor.** Work in your local folder and run on Kaggle's hardware.
- **Nothing left behind.** Kernels are private and are deleted automatically, even if the terminal closes.
- **Project-aware.** The folder you run `kenv init` in is the project. Relative paths behave the same locally and on the kernel.
- **Reproducible.** Every version snapshots code, packages, datasets and metrics, so you can diff, roll back, export and re-create an experiment.
- **Safe.** Built-in secret scanning, and only secret *names* are ever stored, never values.

---

## Features

| Area | What you get |
| --- | --- |
| **Sessions** | Start a disposable kernel with `kenv init`; get the Kaggle link, Jupyter URL and attach ID; attach more terminals with `kenv -id`. |
| **Project sync** | The project is mirrored to `/kaggle/working/<project-folder-name>`. Changed files come back on `run`, `save`, `stop` and exit. `.kenvignore` (gitignore syntax) excludes big folders. |
| **Datasets** | `kenv data push <folder>` uploads a folder once as a private Kaggle dataset, attached to every later session at `/kaggle/input/<name>`. |
| **Remote execution** | `kenv run script.py` syncs, runs on the kernel, streams output and pulls new files back. `kenv exec` runs any shell command on the kernel. |
| **Accelerators** | Choose or switch between CPU, GPU T4 x2 and GPU L4 (`kenv init --gpu`, `kenv gpu`). Availability depends on your Kaggle account. |
| **Monitoring** | `kenv status` shows CPU, RAM, disk and GPU usage of the kernel. |
| **Versioning** | `commit`, `diff`, `rollback`, `branch`, `tag` and `metric`, covering code, packages, datasets and results. |
| **Logs** | Live-streamed, timestamped run and error logs per version, with `--tail`, `--grep` and `--since`. |
| **Core file** | A readable, diffable `core.toml` per version: kernel, accelerator, Python, exact package versions, datasets, outputs (size and SHA256), peak resources and run history. |
| **Portability** | `export` and `import` zip a version with a SHA256 manifest; `rebuild` reinstalls its packages; `convert` reads and writes `requirements.txt`, `environment.yml` and `kernel-metadata.json`. |
| **Safety and quality** | `scan` for keys and tokens (with an optional git pre-commit hook), `deps` for missing or conflicting packages, `doctor` for health checks, `quota` for an estimate of weekly GPU hours. |
| **In-code API** | `import kenv` on the kernel: `kenv.cli(...)`, `kenv.metric(...)`, `kenv.log(...)`, `kenv.timed(...)`. |
| **Shim** | `from kenv_shim import kenv` keeps your code working where kenv does not exist (CI, GitHub, a teammate's laptop). |
| **Clip / unclip** | Comment out every kenv statement before sharing code, and restore them afterwards. |
| **Dashboard** | `kenv ui` opens a local, read-only browser dashboard: versions, timeline, metrics, resource charts, I/O map, dependency lock, logs and diffs. |
| **Lazy local access** *(experimental)* | `kenv init --lazy-local` lets the kernel read heavy files from your machine on demand through a tunnel, instead of uploading them. |

---

## Requirements

| Requirement | Details |
| --- | --- |
| **Python** | 3.8 or newer. |
| **Kaggle CLI** | `pip install -U kaggle` (a recent version; it must support `kaggle kernels delete`). |
| **Kaggle account** | With an API token (see [Kaggle credentials](#kaggle-credentials)). Kaggle may ask you to verify your phone number before GPU or internet access is enabled on your account. |
| **bash** | Linux and macOS only (used for the kenv shell). Windows uses `cmd.exe`. |
| **Internet access** | kenv reaches your kernel's URLs through a relay (`https://ntfy.sh` by default). |
| **Node.js 18+** | *Optional.* Only needed if you want to build the dashboard yourself. |
| **VS Code / Cursor + Jupyter extension** | *Optional.* To use a session as a notebook kernel. |

Get the project files (the folder must contain `kenv.py` and `kenv_phase5_ui/`):

```bash
git clone https://github.com/EpicRaven/kenv.git
cd kenv
```

Run every install command below from inside that folder.

---

## Installation

kenv is installed by copying `kenv.py` and the dashboard folder to a permanent location and adding a small launcher named `kenv` to your `PATH`.

> **Important:** the launchers below set `KENV_UI_DIR`. kenv looks for the dashboard in a folder named `kenv_ui` *next to* `kenv.py`, but this project ships it as `kenv_phase5_ui/kenv_ui/`. Without `KENV_UI_DIR`, `kenv ui` reports that the dashboard files were not found.

### Linux

**1. Install Python and the Kaggle CLI**

```bash
sudo apt install python3 python3-pip
pip install -U kaggle
```

On newer distributions (Ubuntu 23.04+, Debian 12+, Fedora) pip may refuse with an `externally-managed-environment` error. Use `pipx` instead:

```bash
sudo apt install pipx && pipx ensurepath
pipx install kaggle
```

**2. Install kenv**

```bash
mkdir -p ~/.local/share/kenv ~/.local/bin
cp -r kenv.py kenv_phase5_ui ~/.local/share/kenv/
chmod +x ~/.local/share/kenv/kenv.py

cat > ~/.local/bin/kenv << 'EOF'
#!/usr/bin/env bash
export KENV_UI_DIR="${KENV_UI_DIR:-$HOME/.local/share/kenv/kenv_phase5_ui/kenv_ui}"
exec python3 "$HOME/.local/share/kenv/kenv.py" "$@"
EOF
chmod +x ~/.local/bin/kenv

grep -q '.local/bin' ~/.zshrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Using bash instead of zsh? Replace `~/.zshrc` with `~/.bashrc` in the last two lines.

**3.** Continue with [Kaggle credentials](#kaggle-credentials).

### macOS

**1. Install Python and the Kaggle CLI**

```bash
brew install python
pip3 install -U kaggle
```

Homebrew's Python may reject `pip3 install` with an `externally-managed-environment` error. Use `pipx` instead:

```bash
brew install pipx && pipx ensurepath
pipx install kaggle
```

**2. Install kenv** (macOS uses zsh by default, so `~/.zshrc` is correct)

```bash
mkdir -p ~/.local/share/kenv ~/.local/bin
cp -r kenv.py kenv_phase5_ui ~/.local/share/kenv/
chmod +x ~/.local/share/kenv/kenv.py

cat > ~/.local/bin/kenv << 'EOF'
#!/usr/bin/env bash
export KENV_UI_DIR="${KENV_UI_DIR:-$HOME/.local/share/kenv/kenv_phase5_ui/kenv_ui}"
exec python3 "$HOME/.local/share/kenv/kenv.py" "$@"
EOF
chmod +x ~/.local/bin/kenv

grep -q '.local/bin' ~/.zshrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Using bash? Replace `~/.zshrc` with `~/.bash_profile`.

**3.** Continue with [Kaggle credentials](#kaggle-credentials).

### Windows

Use **PowerShell**.

**1. Install Python and the Kaggle CLI**

```powershell
winget install Python.Python.3.12
py -m pip install -U kaggle
```

Close and reopen PowerShell after installing Python so that `py` is on your `PATH`.

**2. Install kenv**

```powershell
$dir = "$env:LOCALAPPDATA\kenv"
New-Item -ItemType Directory -Force $dir | Out-Null
Copy-Item kenv.py $dir -Force
Copy-Item kenv_phase5_ui $dir -Recurse -Force

@'
@echo off
if not defined KENV_UI_DIR set "KENV_UI_DIR=%LOCALAPPDATA%\kenv\kenv_phase5_ui\kenv_ui"
py "%LOCALAPPDATA%\kenv\kenv.py" %*
'@ | Set-Content "$dir\kenv.cmd" -Encoding ascii

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (($userPath -split ";") -notcontains $dir) {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$dir", "User")
}
```

Close and reopen PowerShell so the new `PATH` takes effect.

**3.** Continue with [Kaggle credentials](#kaggle-credentials).

> **Tip:** on Windows the kenv shell opens inside `cmd.exe`, with the prompt `(kenv:<name>) ...`.

---

## Kaggle credentials

1. Sign in at [kaggle.com](https://www.kaggle.com), then open **Settings → API → Create New Token**. This downloads `kaggle.json`.
2. Place it where the Kaggle CLI expects it:

| System | Location |
| --- | --- |
| Linux / macOS | `~/.kaggle/kaggle.json` |
| Windows | `C:\Users\<you>\.kaggle\kaggle.json` |

```bash
# Linux / macOS
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

```powershell
# Windows (PowerShell)
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle" | Out-Null
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\"
```

Alternatively, use environment variables:

```bash
# Linux / macOS
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

```bat
:: Windows (cmd)
setx KAGGLE_USERNAME your_username
setx KAGGLE_KEY your_api_key
```

kenv also respects `KAGGLE_CONFIG_DIR` if your `kaggle.json` lives elsewhere. Never commit `kaggle.json` or share it; it is your account's API key.

---

## Verify the setup

```bash
kenv --version
kenv --cred
```

`kenv --cred` checks your OS, Python, the Kaggle CLI, bash (Linux/macOS), relay reachability and your Kaggle credentials, and prints the fix for anything missing. A healthy run ends with:

```
[ok]  Kaggle accepted your credentials. You are ready: run `kenv init`
```

---

## Quick start

```bash
cd my-project          # any folder: this becomes the kenv project
kenv init              # start a session and open the kenv shell
```

Inside the kenv shell:

```bash
kenv --url             # Kaggle link, Jupyter URL and attach ID
kenv run train.py      # run a script on Kaggle; new files come back automatically
kenv status            # CPU / RAM / disk / GPU usage
exit                   # pulls your changes back and deletes the kernel
```

To use the session from VS Code or Cursor, copy the Jupyter URL from `kenv --url` and choose it as an existing Jupyter server when selecting a notebook kernel.

To run a one-off script without starting a session first:

```bash
kenv run train.py      # starts a temporary session, saves to ./kenv_output, deletes the kernel
```

---

## Feature walkthroughs

Each walkthrough shows the commands, what they do, and what to expect. Output blocks marked **Real output** were captured by running kenv 1.6.0 on a sample project (`price-model`, with versions `v1`–`v4`). Commands that need a live Kaggle kernel are shown without output, since it depends on your account and session.

A sample project used throughout:

```
price-model/
├── train.py
├── utils.py
├── eval.py
├── data/            <- big folder, listed in .kenvignore
└── .kenv/           <- created by kenv
```

### 1. Sessions: start, inspect, attach, stop

A **session** is one disposable running kernel. A **version** is the persistent project state that survives every session.

```bash
cd price-model
kenv init                      # first time: creates .kenv/ and a version; later: resumes the last-used version
kenv -n "lr-sweep"             # same, but the session is called "lr-sweep"
kenv init --gpu                # same, but you choose the accelerator at startup
```

Inside the kenv shell your prompt changes to `(kenv:<name>) ...`. Everything typed there can use `kenv` commands directly.

```bash
kenv --url                     # Kaggle link, Jupyter URL and attach ID
```

The **attach ID** looks like `name#secret`. Use it from a second terminal to join the same session:

```bash
kenv -id swift-raven#<secret>  # attach this terminal to the running session
```

Ways a session ends (the kernel is deleted every time):

| How it ends | What happens |
| --- | --- |
| `exit` in the kenv shell | Changed files are pulled back, kernel deleted. |
| `kenv stop` | Pulls changes back, then deletes the kernel right now. |
| Ctrl-C or closing the terminal | Cleanup still runs (Windows console close is handled too). |
| Your machine vanishes | The kernel stops itself after `--idle` minutes (default 20). |
| `kill -9` or power loss | Run `kenv --sweep` later to delete leftover `kenv-*` kernels. |

Tune timing at startup:

```bash
kenv init --idle 45 --startup 1200   # stop after 45 idle min; wait up to 1200 s for the kernel to come online
```

### 2. Running code on the kernel

**Run a script and get the results back:**

```bash
kenv run train.py                      # sync, run on Kaggle, stream output, pull new/changed files back
kenv run --out results train.py        # kenv options go BEFORE the script
kenv run train.py --epochs 20 --lr 0.03   # everything after the script goes to the script
```

If no session is active, `kenv run` starts a temporary one, saves outputs to `./kenv_output`, and deletes the kernel when finished. This is the quickest way to try one script on a GPU.

**Run any shell command on the kernel:**

```bash
kenv exec nvidia-smi
kenv exec pip list
kenv exec python --version
```

**Look at the kernel:**

```bash
kenv status        # CPU / RAM / disk / GPU usage
kenv ls            # files in the project folder on the kernel
kenv ls data            # files in a subfolder of the project on the kernel
```

**Switch hardware without restarting:**

```bash
kenv gpu t4        # GPU T4 x2 (Kaggle's default GPU)
kenv gpu l4        # GPU L4
kenv gpu none      # CPU only
kenv gpu           # prompts you to choose
```

Which accelerators you actually get depends on your Kaggle account.

**Use the session as a notebook kernel.** Run `kenv --url`, copy the Jupyter URL, and in VS Code or Cursor choose *Select Kernel → Existing Jupyter Server* and paste it. Relative paths in your notebook behave like your local folder.

### 3. Project sync and `.kenvignore`

The project folder is mirrored to `/kaggle/working/<project-folder-name>` on the kernel (so `price-model/` becomes `/kaggle/working/price-model`). Notebooks and scripts start there.

```bash
kenv sync                        # pull the kernel's new/changed files, then push your local edits
kenv save                        # pull new/changed files back
kenv save models/best.pt         # pull a specific path
kenv save outputs --to results/  # pull into a local folder
kenv put data/small.csv          # upload a file
kenv put assets --dest inputs/   # upload a folder to a remote directory
```

Changed or new kernel files come back automatically on `kenv run`, `kenv save`, `kenv stop` and a normal exit.

**`.kenvignore`** uses gitignore syntax. `kenv init` creates one for you:

**Real output** (`.kenvignore` as generated):

```
# .kenvignore - what kenv does NOT sync between this folder and the Kaggle kernel (.gitignore syntax).
# Always ignored (built in): .git .venv venv node_modules __pycache__ .ipynb_checkpoints .kenv
#
# Ignored paths are not uploaded to the kernel and are not pulled back automatically.
# Big inputs: upload once as a private Kaggle dataset ->  kenv data push <folder>
# Big outputs: fetch them on purpose ->  kenv save <path>
#
# data/
# models/
# *.pt
# *.ckpt

# kenv exports
*.kenv.zip
```

Uncomment the lines for your big folders. A typical setup:

```
data/
models/
*.pt
*.ckpt
```

**Edit conflicts.** If you change a file locally while the kernel changed the same file, **your file wins** and the kernel's copy is saved next to it as `<name>.kenv-remote<ext>` (for example `train.kenv-remote.py`). Nothing is lost.

### 4. Datasets: upload big data once

Every session uploads the non-ignored project files to a fresh kernel, so large data should not live in the project. Push it once as a private Kaggle dataset instead:

```bash
kenv data push data/            # upload the folder as a private dataset
kenv data list                  # show attached datasets
```

It is attached to every later session and appears on the kernel under `/kaggle/input/<name>`:

```python
import pandas as pd
df = pd.read_csv("/kaggle/input/naija-houses/houses.csv")
```

Attached datasets are recorded in each commit, in `core.toml`, and shown by `kenv diff`.

### 5. Versioning: commit, diff, rollback, branch, tag, metric

A commit snapshots your tracked files (`.kenvignore` applies), the package lock (`deps.lock`), attached datasets and the metrics recorded since the last commit into the **next numbered version**.

**Record results, then commit:**

```bash
kenv metric rmse 32.7                  # from the terminal
kenv metric                            # list recorded metrics waiting for the next commit
```

```python
kenv.metric("rmse", 32.7)              # or from code on the kernel
```

**Real output:**

```
$ kenv metric
  rmse = 32.7

$ kenv commit -m 'try new features'
[kenv] Committed v5  "try new features"   (branch tuning2, parent v3, id kv:YsSFhwuUjdy44KdbxIQBhgyQa9Ln8Hx5)
  code    : 2 file(s); 1 new, 1 changed, 2 removed since v3 - 0 shared with the previous snapshot, 2 copied
  packages: 5 in deps.lock
  metrics : rmse=32.7
```

**List versions:**

```
$ kenv versions
Project: /home/claude/demo/price-model      (* = active)
    v1   -                kv:T2skhZ3h...  2 session(s)   39m 00s used  last 5d ago  "baseline"
    v2   -                kv:DjZtQFPZ...  2 session(s)   56m 00s used  last 2d ago  [champion]  "lower learning rate"
  * v3   -                kv:I6FNk6ZO...  2 session(s)    1h 17m used  last 15h ago  "xgboost experiment"
    v4   scratch          kv:xCJb59ZK...  0 session(s)        0s used  never used
```

**Inspect one version:**

```
$ kenv v3
  Version : v3   (active)
  ID      : kv:I6FNk6ZOh34NZzKHGKPpEa2o1YNPVAWe
  Created : 2026-10-01T10:51:44Z
  Commit  : "xgboost experiment"   branch tuning, parent v2, 3 file(s)
  Metrics : r2=0.83, rmse=31.4
  Sessions: 2   total 1h 17m
    2026-09-30T10:51:44Z   52m 00s  user   [swift-raven]
    2026-09-30T18:51:44Z   25m 00s  idle   [calm-otter]
```

The `idle` and `user` labels tell you *how* each session ended.

**Compare two versions.** `kenv diff` compares code, libraries, datasets and metrics in one report:

```
$ kenv diff v2 v3
Diff  v2  ->  v3

Code: 1 added, 2 changed, 0 removed
  A eval.py  (+1 -0)
  M train.py  (+2 -2)
  M utils.py  (+1 -1)

Libraries: 1 added, 2 upgraded/downgraded, 0 removed
  + xgboost==2.1.1
  ~ lightgbm 4.3.0 -> 4.5.0
  ~ numpy 1.26.4 -> 2.0.2

Inputs (datasets): identical

Metrics: 2 differ
  r2: 0.8 -> 0.83   (+0.03)
  rmse: 33.8 -> 31.4   (-2.4)
```

Add `-p` / `--patch` to include the text diff of each changed file:

```
--- v2/train.py
+++ v3/train.py
@@ -1,4 +1,4 @@
 import pandas as pd
-LR = 0.05
-EPOCHS = 12
+LR = 0.03
+EPOCHS = 20
 print('training on naija-houses')
```

`kenv diff v2` compares v2 against your working files; `kenv diff` compares the active version against your working files.

**Roll back.** Restores a version's code, datasets and packages. Your uncommitted changes are saved to a safety version first.

```
$ kenv rollback v3
[kenv] Rolled back to v3: 3 file(s) restored, 1 removed, branch tuning
[kenv] Packages are rebuilt when a session starts in v3 (`kenv init`).
[kenv] Only code, dependencies and config come back - a kernel is disposable, so its memory and files cannot be restored.
```

**Branches** record a parent and let you try an idea without losing your main line:

```
$ kenv branch
Branches (* = current). Code is not touched when you switch: use `kenv rollback <tip>` to restore a branch's files.
  * main              tip v2          2 commit(s)
    tuning            tip v3          1 commit(s)  from v2

$ kenv branch tuning2
[kenv] branch 'tuning2' created from v3 and switched to. Commits made now belong to it.
```

```bash
kenv branch switch main     # change branch (files are not touched; use rollback <tip> to restore them)
kenv branch rm tuning2      # remove a branch
```

**Tags** are names for versions and work anywhere a version does:

```
$ kenv tag v3 best-rmse
[kenv] tag 'best-rmse' -> v3

$ kenv tag
  best-rmse          -> v3                        "xgboost experiment"
  champion           -> v2                        "lower learning rate"
```

```bash
kenv diff champion best-rmse    # compare by tag
kenv activate best-rmse         # switch by tag
kenv rollback champion          # restore by tag
kenv tag --delete best-rmse     # remove a tag
```

**Switch and rename versions:**

```
$ kenv activate champion
[kenv] Active version: v2   (kv:DjZtQFPZLDRHqgdpHYCeiuk7Y2GAqR8M)

$ kenv v4 -r experiments
[kenv] v4 (scratch) is now v4 (experiments)
```

You can address a version by number (`v2`), name (`model-champ`), tag, id (`kenv activate -id kv:<id>`), or point at another project's `.kenv` folder (`kenv activate ../other/.kenv`).

### 6. Logs

Logs stream live from the kernel while a session runs. They are plain text with UTC timestamps, stored per version:

| File | Contents |
| --- | --- |
| `.kenv/vN/logs/run.log` | Agent and Jupyter messages, output of `kenv run` / `kenv exec`, metrics, `kenv.log(...)` lines. |
| `.kenv/vN/logs/errors.log` | The error lines and tracebacks from the same stream. |

```bash
kenv logs                       # active version's log
kenv logs v3                    # a specific version
kenv logs v3 --errors           # errors only
kenv logs --tail                # follow live (Ctrl-C to stop)
kenv logs --grep "epoch 1[0-9]" # regex, case-insensitive
kenv logs --since 10m           # relative: 30s, 10m, 2h, 1d
kenv logs --since "2026-09-30 12:00"   # or absolute
kenv logs --since "12:30 today" # time of day, today
```

**Real output** (`kenv logs v3 --since 1d`):

```
2026-09-30T10:52:44.116Z [stdout] epoch 3/12 loss=0.3333
2026-09-30T10:53:14.116Z [stdout] epoch 4/12 loss=0.2500
2026-09-30T10:53:44.116Z [stdout] epoch 5/12 loss=0.2000
```

Text printed by a notebook cell stays in the notebook. To capture it, use `kenv.log("...")` or run the code as a script with `kenv run`.

### 7. The core file (`core.toml`)

Every version has one readable, diffable file at `.kenv/vN/core.toml`:

```
$ kenv core v3
  Core file : /home/claude/demo/price-model/.kenv/v3/core.toml
  Version   : v3   (kv:I6FNk6ZOh34NZzKHGKPpEa2o1YNPVAWe)
  Kernel    : adaeze/kenv-demo   GPU   python 3.11.13
  Packages  : 5 locked, 1 imported by your code
  Inputs    : 2 attached dataset(s), 2 mounted
  Outputs   : 2 file(s) with sizes and SHA256
  Resources : RAM peak 9.6 GB, VRAM peak 8.4 GB, CPU peak 99%, disk 6.0 GB, 52 min
  Secrets   : HF_TOKEN, WANDB_API_KEY   (names only)
  Runs      : 7
    r0005  metric rmse = 32.7
    r0006  timed  fit                ok      10m 24s  exit 0
    r0007  script train.py           ok      1m 01s  exit 0
```

It records the kernel, accelerator, Python, exact package versions plus the imports found in your code, datasets, output files (size and SHA256), peak RAM, VRAM, CPU and disk, every run, and the **names** of the secrets your code reads, never their values.

### 8. Measuring code: `time_start`, `time_end`, `timed`

Measure time, RAM, VRAM, disk and CPU around any block. On the kernel:

```python
import kenv

kenv.time_start("load")
df = load_data()
kenv.time_end()
kenv.time_output()               # prints the difference; also stored in the version's run history

with kenv.timed("fit"):          # the same three calls in one block
    model.fit(X, y)
```

The same measurements work from the terminal: `kenv time_start load`, `kenv time_end`, `kenv time_output`. Every measurement shows up in `kenv core` under *Runs* (for example `r0006 timed fit ok 10m 24s exit 0`) and in the dashboard.

### 9. Packages and dependencies

```bash
kenv deps                        # imports in your code that are missing from the lock
kenv deps fix                    # add them
kenv deps add requests==2.32.3   # declare a package by hand (carried by every commit)
kenv deps conflicts              # run `pip check` on the kernel and report conflicts
kenv rebuild                     # install locked and declared packages on the running kernel, then pip check
kenv rebuild --dry-run           # only show what would be installed
kenv rebuild --all               # every differing package
```

**Real output:**

```
$ kenv deps add requests==2.32.3
[kenv] declared in v3: requests==2.32.3
  Commits carry them forward and `kenv rebuild` installs them on the kernel.

$ kenv deps
[kenv] every import in your code is covered by v3's dependency lock
```

`kenv rebuild` also warns when Kaggle's Python or Docker image differs from what the version recorded.

### 10. Converting to and from other formats

```bash
kenv convert --to requirements.txt                    # active version
kenv convert --to requirements.txt v3 --file reqs.txt # a chosen version, custom output file
kenv convert --to environment.yml
kenv convert --to kernel-metadata.json
kenv convert --from requirements.txt                  # read it into the active version
```

**Real output:**

```
$ kenv convert --to requirements.txt v3 --file reqs.txt
[kenv] wrote reqs.txt  (from v3)

$ cat reqs.txt
lightgbm==4.5.0
numpy==2.0.2
pandas==2.2.3
scikit-learn==1.5.2
xgboost==2.1.1

$ kenv convert --from requirements.txt     # file contained: numpy==1.26.4 and pandas>=2.0
[kenv] v3: 1 pinned package(s) merged into the dependency lock
  1 package(s) without a pinned version were recorded as declared packages: pandas
  `kenv rebuild` installs them on the kernel; pin them (name==version) for exact reproducibility.
```

Pinned packages (`name==version`) go into the lock; unpinned ones are recorded as declared packages. Add `--all` to convert every version and `--force` to overwrite an existing output file.

### 11. Export and import: move an experiment between machines or accounts

```bash
kenv export v3 --out v3.kenv.zip      # default version: the active one; add --force to overwrite
```

The zip holds the core file, code snapshot, `deps.lock`, config, logs, metadata and a SHA256 manifest. Secrets are scanned for first and logs are redacted, so **no secret values and no kernel reference of yours** are included.

**Real output:**

```
$ kenv export v3 --out v3.zip
[kenv] exported v3: 3 file(s), 4.5 KB -> v3.zip
  Secret NAMES the code uses (add them in Kaggle > Add-ons > Secrets on the new account): HF_TOKEN, WANDB_API_KEY
```

On the receiving side:

```bash
mkdir restored && cd restored
kenv import ../v3.zip --init              # recreate the version here and start a session now
kenv import v3.zip --to ~/projects/restored --name restored   # or into another folder, with a name
```

**Real output:**

```
$ kenv import /tmp/v3.zip --to /tmp/imported --name restored
[kenv] imported as v1 (restored): 3 file(s) written into the project, 3 in the snapshot
  Kaggle datasets kept: adaeze/naija-houses, chinedu/lagos-rents
  Add these secrets in your Kaggle account (names only were exported): HF_TOKEN, WANDB_API_KEY
  Recorded on the original kernel: Python 3.11.13, image kaggle-gpu-2025-09. The first session compares them with what Kaggle provides now and warns on a mismatch.
  Next: `kenv init` starts a kernel under YOUR credentials and installs the locked packages (run it inside /tmp/imported).
```

Import verifies every hash, refuses unsafe paths, archives old logs with their original timestamps, and drops datasets your account cannot read.

### 12. Secret scanning

`kenv scan` looks for known key formats and high-entropy strings.

**Real output** (a file containing a fake GitHub token):

```
$ kenv scan leak.py
[kenv] Possible secrets found:
  leak.py
    line 1: GitHub token  (ghp_...bB)  fingerprint 0f758b047f69
    line 1: Generic assignment  (API_...B")  fingerprint 6847217edbe6
  Move the value out of the file (Kaggle Secrets + kaggle_secrets, or an environment variable). A false alarm: add
  `# kenv:allow` to that line, or run `kenv scan allow` (it stores the 12-character fingerprint, never the text).
```

Commits are blocked while a secret is present. The next command was run in the same project:

```
$ kenv commit -m 'try new features'
[kenv] Possible secrets found:
  ...
[kenv] Commit stopped: 2 possible secret(s) in 1 file(s). Nothing was written. Kenv's scan is a safety net, not a guarantee - always review what you share.
```

Ways to resolve a finding:

```bash
# 1. Best: move the value out of the code
#    use Kaggle Secrets (kaggle_secrets) or an environment variable

# 2. False alarm on one line: add a comment
x = "not-really-a-key"  # kenv:allow

# 3. False alarm in general: store only a 12-character fingerprint
kenv scan allow leak.py
kenv scan allow 0f758b047f69
```

Scan only what is staged for git, and install a pre-commit hook:

```bash
kenv scan --staged
kenv scan hook          # blocks git commits that contain secret-looking strings
kenv scan unhook
```

**Real output:**

```
$ kenv scan hook
[kenv] installed .git/hooks/pre-commit: commits with secret-looking strings are blocked (bypass once: git commit --no-verify)
```

### 13. Health checks (`kenv doctor`)

```bash
kenv doctor            # everything
kenv doctor env        # credentials, Kaggle CLI, relay
kenv doctor files      # core files, snapshots, I/O hashes, tags, branches, last_active
kenv doctor files --fix   # repair dangling tags, last_active, dangling branches and leftover temp files
```

Each problem is printed with a `fix:` line, and the command exits with status 1 when anything is wrong (useful in scripts):

```
Environment
  [ok]  Python 3.12.3
  [problem]  Kaggle CLI not found
              fix: pip install -U kaggle
  [problem]  No Kaggle credentials found
              fix: run `kenv --cred` (or set KAGGLE_USERNAME and KAGGLE_KEY)
```

### 14. GPU quota estimate

```bash
kenv quota                       # weekly GPU hours used and left
kenv quota set limit 30          # your own weekly limit (hours)
kenv quota set warn 80,95        # warn at 80% and 95%
kenv quota reset
```

**Real output:**

```
$ kenv quota
GPU this week (approximate, rolling 7 days, from kenv's own logs): 1.6 h of 30 h used, about 28.4 h left (5%)
  #...................   warnings at 80%, 95%
  This is an ESTIMATE: kenv counts the GPU sessions it logged in the projects it knows about. Kaggle has no
  official API for your quota, other GPU use (notebooks in the browser) is not included, and the week may
  reset on a different day. Set your own limit: kenv quota set limit 30
```

### 15. Keeping kenv lines out of shared code

You have three ways to keep `import kenv` and `kenv.metric(...)` calls from breaking code that runs somewhere without kenv.

**Option A: the shim (simplest).** `kenv shim` writes `kenv_shim.py` into the project. Then import it instead of `kenv`:

```python
from kenv_shim import kenv

kenv.metric("rmse", 31.4)   # works with kenv; a no-op returning None in CI, GitHub or a teammate's laptop
```

**Option B: clip and unclip.** Opt a file in by putting `kenv.clip()` right after `import kenv`:

```python
import kenv
kenv.clip()
import pandas as pd

kenv.time_start("load")
df = pd.read_csv("data/houses.csv")
kenv.time_end()
kenv.time_output()
kenv.metric("rmse", 31.4)
```

**Real output:**

```
$ kenv clip
  train.py: 6 kenv statement(s), lines 1-2, 5, 7-9   [opted in]
[kenv] state: active   (`kenv unclip` comments these lines out; a session start brings them back)

$ kenv unclip --dry-run
  would comment out: train.py  (6 statement(s), lines 1-2, 5, 7-9)
[kenv] dry run: 1 file(s) would change

$ kenv unclip
  commented out: train.py  (6 statement(s), lines 1-2, 5, 7-9)
[kenv] 1 file(s) saved. A new kenv session (or `kenv unclip --undo`) brings the lines back.
```

After `unclip`, `train.py` looks like this:

```python
#kenv:clip# import kenv
#kenv:clip# kenv.clip()
import pandas as pd

#kenv:clip# kenv.time_start("load")
df = pd.read_csv("data/houses.csv")
#kenv:clip# kenv.time_end()
#kenv:clip# kenv.time_output()
#kenv:clip# kenv.metric("rmse", 31.4)
```

```
$ kenv unclip --undo
[kenv] restored 6 kenv line(s) in 1 file(s)
```

Notes: multi-line calls and notebook cells are handled. A statement kenv cannot comment out safely (`x = kenv.cli(...)`, `with kenv.timed():`, two statements on one line) stops the whole file and is reported; fix the line, or use `-f` to comment out the rest. An interrupted run can be repaired with `kenv unclip --recover`.

**Option C: automatic at commit time.** Install a git pre-commit hook so commits get a copy of your files with kenv lines commented out, while your working files stay untouched:

```bash
kenv unclip hook        # bypass once with: git commit --no-verify
kenv unclip --staged    # do the same by hand for what is staged
kenv unclip unhook
```

### 16. Running kenv commands from code

On the kernel (notebook cells, scripts, `kenv exec`), `import kenv` works with no `pip install`:

```python
import kenv

kenv.cli("kenv v2 -r champ")          # any terminal command
kenv.cli(["kenv", "sync"])            # list form
kenv.cli("kenv status", timeout=60, capture=True)
kenv.cli('kenv commit -m "after epoch 10"')   # commit, diff, rollback, branch, tag, logs also work

kenv.status()                         # same as `kenv status`
kenv.log("epoch 3 done")              # line in the session log
kenv.log("oops", "error")             # also lands in errors.log
kenv.metric("auc", 0.93)              # recorded for `kenv diff`
kenv.kaggle_root()                    # "/kaggle/working"
kenv.kaggle_root(chdir=True)          # switch there
```

Commands that need your files run **on your machine**, through a queue answered by the window that started the session, so keep that window open. Limits:

- `kenv.cli()` refuses anything that starts, stops or attaches a session (`init`, `stop`, `gpu`, `exec`, `run`, `-id` ...), `logs --tail`, and any path outside the project folder.
- `import`, `quota set/reset`, `scan allow/hook` are blocked from code.
- With no listening kenv window it raises an error at once.

Allowed from code: `doctor`, `export`, `convert`, `quota`, `deps`, `scan`, `rebuild`, `commit`, `diff`, `rollback`, `branch`, `tag`, `logs`, `ui`.

### 17. The dashboard

```bash
kenv ui                       # open the dashboard in your browser
kenv ui -uri kv:<id>          # one version only (also a vN, a name or a tag)
kenv ui --port 8765 --no-open # fixed port, print the link only
kenv ui --idle 30             # stop after 30 idle minutes
```

**Real output:**

```
$ kenv ui --no-open --port 8765 --idle 1
[kenv] Dashboard for price-model  (read-only, this machine only)
  http://127.0.0.1:8765/?t=<random-token>
  The link carries a private access token; anyone without it is rejected. Ctrl-C stops the server.
```

Pages: versions, branches and tags; the session timeline; run history and metrics; RAM, VRAM, CPU, disk and runtime charts; the I/O map (sizes and hashes); the dependency lock; logs with search; diffs between versions; the GPU quota estimate.

With `-uri`, kenv makes that version the active one when no session is running.

If kenv cannot find the dashboard files you will see:

```
[kenv] The dashboard files (the prebuilt React + Ant Design bundle) were not found. ...
kenv looked in:
    ~/.local/share/kenv/kenv_ui
    ~/.kenv/ui
```

That is fixed by the `KENV_UI_DIR` line already in the launchers from the [Installation](#installation) section.

### 18. Lazy local access (experimental)

For projects with very large local data, let the kernel read files **from your machine on demand** instead of uploading them:

```bash
kenv init --lazy-local                       # tunnel defaults to cloudflared
kenv init --lazy-local --tunnel ngrok        # ngrok needs an account and auth token; free plan has bandwidth caps
```

On the kernel:

```python
import kenv

kenv.prefetch("data/")                    # warm the cache before training
kenv.lazy_path("img/a.png")               # a real path for C-level loaders
print(kenv.lazy_status())                 # hits / misses / bytes
```

How it behaves:

- Only small code and config files are synced the normal way. The kernel sees `os.getcwd()` as your local project path.
- A read-only file server on your machine, limited to the project folder, sits behind a random token and a tunnel. It rejects `../` paths, symlinks that leave the folder, `.kenv`, `.git` and key or `.env` files, and it stops with the session.
- Writes never go to your machine; they land in the kernel's project folder and come back with the normal sync.
- C-level file access (some image, audio and video loaders, memory maps) bypasses the patch; use `kenv.lazy_path(...)`.
- Cache misses cross your home internet connection, so loops that re-read big files every epoch are slow until the cache is warm.
- If `cloudflared` is not installed, kenv downloads it from `github.com/cloudflare/cloudflared` (Linux, macOS and Windows builds are supported).

### 19. End-to-end examples

**A. Try one script on a GPU, nothing to clean up**

```bash
cd my-project
kenv run train.py          # temporary session; results land in ./kenv_output; kernel deleted
```

**B. A normal day of experiments**

```bash
cd price-model
kenv init --gpu                      # pick a T4
kenv --url                           # paste the Jupyter URL into VS Code, work in your notebook
kenv run train.py --lr 0.03          # kenv options before the script; script args after
kenv metric rmse 31.4                # (or kenv.metric(...) inside train.py)
kenv commit -m "lr 0.03, 20 epochs"  # becomes the next numbered version
kenv tag v3 best-rmse
exit                                 # outputs pulled back, kernel deleted
kenv diff champion best-rmse         # compare against the previous best
```

**C. Reproduce an experiment on another machine or account**

```bash
# machine 1
kenv export best-rmse --out best.kenv.zip

# machine 2
mkdir best && cd best
kenv --cred                              # confirm Kaggle credentials
kenv import ../best.kenv.zip --init      # hashes verified, packages reinstalled on a fresh kernel
```

**D. Share code safely**

```bash
kenv scan --staged          # nothing secret-looking in what you are about to commit
kenv scan hook              # block future commits that contain secrets
kenv unclip hook            # strip kenv lines from commits automatically
git commit -am "model code"
```

**E. Recover after a crash**

```bash
kenv --sweep                # delete leftover kenv-* kernels
kenv doctor --fix           # repair dangling tags, branches and temp files
kenv logs --errors          # see what went wrong
```

---

## Command reference

Commands are ordered from basic to advanced.

### Basic

| Command | What it does |
| --- | --- |
| `kenv --help` / `-h` | Show every command. |
| `kenv --version` | Show the version. |
| `kenv --cred` | Check OS, packages and Kaggle credentials, and show how to fix them. |
| `kenv init` | Start a session in this folder and open a kenv shell. Creates `.kenv/` the first time; later runs resume the last-used version. |
| `kenv -n "<name>"` | Start a session called `<name>` (also `kenv init -n "<name>"`). |
| `kenv --url` | Show the Kaggle link, the Jupyter URL and the attach ID. |
| `kenv status` | CPU, RAM, disk and GPU usage of the kernel. |
| `kenv run <script.py> [args]` | Sync, run a script on the kernel, stream output and pull new files back (`--out <dir>`). |
| `kenv exec <command>` | Run a shell command on the kernel, e.g. `kenv exec nvidia-smi`. |
| `kenv ls [path]` | List files on the kernel. |
| `kenv stop` | Pull your changes back, then delete the kernel now. |
| `kenv --sweep` | Delete leftover `kenv-*` kernels (after `kill -9` or a power loss). |

### Intermediate

| Command | What it does |
| --- | --- |
| `kenv init --gpu` | Start a session and choose an accelerator at startup. |
| `kenv gpu [t4\|l4\|none]` | Switch the accelerator (prompts if omitted). |
| `kenv -id <attach-id>` | Attach this terminal to a running session from another window. |
| `kenv sync` | Pull the kernel's new or changed files, then push your local edits. |
| `kenv save [paths]` | Download paths (or all new/changed files) from the kernel (`--to <dir>`). |
| `kenv put <files/folders>` | Upload to the kernel (`--dest <remote/dir>`). |
| `kenv data push <folder>` | Upload a folder once as a private Kaggle dataset; attached to later sessions at `/kaggle/input/<name>`. |
| `kenv data list` | List attached datasets. |
| `kenv versions` | List versions: number, name, id (`kv:...`), sessions, time used. |
| `kenv new [name]` | Create the next version and make it active. |
| `kenv activate <v2 \| name \| -id kv:... \| path/.kenv>` | Switch the active version. |
| `kenv v2 -r <new-name>` | Rename a version. `kenv v2` shows its id, sessions and how each ended. |
| `kenv logs [version]` | Print the log (`--errors`, `--tail`, `--grep <re>`, `--since 10m`). |
| `kenv core [version]` | Summary of the version's core file. |
| `kenv quota` | Estimated weekly GPU hours used and left (approximate). |
| `kenv ui` | Open the local dashboard (see [below](#the-dashboard-kenv-ui)). |
| `kenv shim` | Write `kenv_shim.py` so kenv lines are harmless outside a session. |

**Ignoring files.** Put patterns in a `.kenvignore` file (gitignore syntax) at the project root to keep big data and model folders out of the sync.

**Two kinds of ids.** `kenv -id <attach-id>` attaches to a running session (`name#secret`). `kenv activate -id kv:<id>` selects a project version. Version ids always start with `kv:`.

### Advanced

| Command | What it does |
| --- | --- |
| `kenv commit -m "msg"` | Snapshot tracked files, `deps.lock`, attached datasets and recorded metrics into the next numbered version. |
| `kenv diff [v2] [v3]` | Compare code, libraries, datasets and metrics (`-p` / `--patch` adds text diffs). |
| `kenv rollback <version>` | Restore a version's code, datasets and packages. Uncommitted changes are saved to a safety version first. |
| `kenv branch [name]` / `branch switch <name>` / `branch rm <name>` | List, create/switch, change and remove branches. |
| `kenv tag [version tag]` / `tag --delete <tag>` | List, attach and delete tags. A tag works anywhere a version does. |
| `kenv metric <name> <value>` | Record a result for `kenv diff`. |
| `kenv export [version]` | Zip a version with a SHA256 manifest (`--out <file.zip>`, `--force`). No secret values are included. |
| `kenv import <zip> [--to <dir>] [--init] [--name <n>]` | Verify hashes and recreate a version in this folder. |
| `kenv rebuild [version]` | Install locked and declared packages on the running kernel and run `pip check` (`--dry-run`, `--all`). |
| `kenv convert --to <fmt> / --from <file>` | Convert to or from `requirements.txt`, `environment.yml` and `kernel-metadata.json`. |
| `kenv deps` / `deps fix` / `deps add <pkg[==ver]>` / `deps conflicts` | Find imports missing from the lock, fix them, declare packages, and check for conflicts. |
| `kenv quota set limit <hours>` / `quota set warn 80,95` / `quota reset` | Configure GPU-hour limits and warnings. |
| `kenv scan [paths] [--staged]` | Look for keys and tokens (known formats plus high-entropy strings). |
| `kenv scan allow [paths \| <fingerprint>]` | Allow findings (only a 12-character fingerprint is stored). |
| `kenv scan hook` / `unhook` | Install or remove a git pre-commit hook that runs `scan --staged`. |
| `kenv doctor [env \| files] [--fix]` | Health checks for credentials, CLI, relay, core files, snapshots, tags and branches. `--fix` repairs what it safely can. |
| `kenv clip [files]` / `unclip [files]` | List or comment out kenv statements (`--dry-run`, `--undo`, `--staged`, `--recover`). |
| `kenv unclip hook` / `unhook` | Install or remove a pre-commit hook that comments out kenv lines in staged files. |
| `kenv init --lazy-local [--tunnel cloudflared\|ngrok]` | *Experimental.* Let the kernel read heavy files from your machine on demand. |
| `kenv init --idle <min> --startup <sec>` | Tune the idle shutdown (default 20 min) and kernel startup wait (default 900 s). |

---

## Using kenv from code

On the kernel, every session installs `import kenv`; no `pip` needed.

```python
import kenv

kenv.metric("auc", 0.93)          # record a result for `kenv diff`
kenv.log("epoch 3 done")          # line in the session log
kenv.log("oops", "error")         # also lands in errors.log

with kenv.timed("fit"):           # time, RAM, VRAM, disk, CPU
    model.fit(X, y)

kenv.status()                     # same as `kenv status`
kenv.cli("kenv v2 -r champ")      # run a terminal command from code
kenv.cli('kenv commit -m "msg"')  # commit, diff, rollback, branch, tag, logs also work
kenv.kaggle_root()                # "/kaggle/working"
```

Commands that need your files run on **your** machine through a queue answered by the window that started the session, so keep that window open.

To keep the same code working where kenv does not exist (CI, GitHub, a teammate's laptop):

```python
from kenv_shim import kenv        # every call is a no-op returning None there
```

Run `kenv shim` to write `kenv_shim.py` into the project.

---

## The dashboard (`kenv ui`)

`kenv ui` opens a read-only dashboard in your browser with: versions, branches and tags; the session timeline; run history and metrics; RAM, VRAM, CPU, disk and runtime charts; the I/O map (sizes and hashes); the dependency lock; logs with search; diffs between versions; and the GPU quota estimate.

```bash
kenv ui                      # open the dashboard
kenv ui -uri kv:<id>         # one version only (also vN, a name or a tag)
kenv ui --port 8765          # fixed port
kenv ui --no-open            # print the link only
kenv ui --idle 30            # stop after 30 idle minutes
```

- Binds to `127.0.0.1` only, answers `GET` only, and sits behind a random token in the link.
- Reads from `.kenv/` alone and shows secret **names**, never values.
- From code, `kenv.cli("kenv ui")` starts it in the background (stops after 60 idle minutes) and keeps the link out of the output.

**Where kenv looks for the dashboard files** (first match with `index.html` and `app.js` wins):

1. `$KENV_UI_DIR`
2. `kenv_ui/` next to `kenv.py`
3. `~/.kenv/ui`

The launchers in [Installation](#installation) set `KENV_UI_DIR` to the bundled `kenv_phase5_ui/kenv_ui` folder.

**Rebuilding the dashboard yourself** (needs Node.js 18+; `kenv.py` itself stays stdlib-only):

```bash
cd kenv_phase5_ui/kenv_ui_src
npm install
npm run build        # writes ../kenv_ui (index.html, app.js, app.css)
```

---

## Configuration

| Variable | Purpose |
| --- | --- |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Kaggle credentials (alternative to `kaggle.json`). |
| `KAGGLE_CONFIG_DIR` | Folder containing `kaggle.json` (default `~/.kaggle`). |
| `KENV_RELAY` | Relay server used to find your kernel's URLs (default `https://ntfy.sh`). Point it at your own ntfy server if needed. |
| `KENV_UI_DIR` | Folder holding the dashboard files. |
| `NO_COLOR` | Disable coloured output. |

Files kenv creates:

| Path | Contents |
| --- | --- |
| `<project>/.kenv/` | Versions (`vN/`), `core.toml`, logs, session times, sync hashes, attached datasets. |
| `<project>/.kenvignore` | Paths excluded from sync. |
| `<project>/kenv_shim.py` | Written automatically; keeps `from kenv_shim import kenv` safe. |
| `~/.kenv/sessions` | Per-machine session state. |

---

## Updating and uninstalling

**Update:** pull the latest project files, then repeat the copy step of your platform's install (the launcher does not need to be recreated).

```bash
cd kenv && git pull
```

**Uninstall (Linux / macOS):**

```bash
rm -rf ~/.local/share/kenv ~/.local/bin/kenv ~/.kenv
```

Then remove the `PATH` line from `~/.zshrc` (or `~/.bashrc`) if you no longer need it.

**Uninstall (Windows, PowerShell):**

```powershell
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\kenv", "$env:USERPROFILE\.kenv"
```

Then remove `%LOCALAPPDATA%\kenv` from your user `PATH` (System Properties → Environment Variables).

Project folders keep their own `.kenv/`; delete it per project if you want it gone.

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `kenv: command not found` | Open a new terminal, or `source ~/.zshrc`. Check that `~/.local/bin` is on `PATH` (`echo $PATH`). On Windows, reopen PowerShell. |
| `kaggle CLI` missing or "too old" | `pip install -U kaggle` (or `pipx upgrade kaggle`). kenv needs `kaggle kernels delete`. |
| `externally-managed-environment` on `pip install` | Use `pipx install kaggle`, or a virtual environment. |
| "Kaggle rejected them" | Create a fresh token (Kaggle → Settings → API → Create New Token) and replace `kaggle.json`. |
| `relay NOT reachable` | Check your connection or firewall, or set `KENV_RELAY` to your own ntfy server. |
| `kenv ui` says the dashboard files were not found | Make sure the launcher sets `KENV_UI_DIR` to `.../kenv_phase5_ui/kenv_ui`, or build the dashboard. |
| "This folder is not a kenv project" | Run `kenv init` in your project folder (or a folder below it). |
| Leftover kernels after a crash | `kenv --sweep`. |
| GPU not available | Availability depends on your Kaggle account and quota. Try `kenv gpu t4`, `kenv gpu l4`, or check phone verification on Kaggle. |
| Something odd with versions, tags or branches | `kenv doctor files --fix`. |
| Windows: `py` not recognised | Reinstall Python and tick "Add Python to PATH", or use `python` in `kenv.cmd` instead of `py`. |

---

## Notes and limits

- **Secret scanning is a safety net, not a guarantee.** It can miss a secret and can flag harmless strings (add `# kenv:allow` on the line, or use `kenv scan allow`). Always review what you share.
- **The GPU quota is an estimate** from kenv's own logs; Kaggle has no official API for it.
- **Kernels are disposable.** Files on the kernel disappear with it. Project files sync back on `run`, `save`, `stop` and a normal exit. A kernel that dies on its own cannot be pulled from.
- **Every session uploads the non-ignored project files** to a fresh kernel. Keep big folders in `.kenvignore` or use `kenv data push`.
- **Conflicting edits:** if you change a file locally while the kernel changed it too, your file wins and the kernel's copy is saved as `<name>.kenv-remote<ext>`.
- **kenv options go before the script** in `kenv run`; everything after the script is passed to it.
- **Idle shutdown:** the kernel stops itself after `--idle` minutes (default 20) if your machine vanishes.
- **`kenv.cli()` from code** refuses anything that starts, stops or attaches a session, `logs --tail`, and any path outside the project folder.
- **Lazy local access is experimental** and off by default. It serves a read-only view of the project folder through a tunnel, behind a random token; it rejects `../` paths, outward-pointing symlinks, `.kenv`, `.git` and key or `.env` files. Reads that miss the cache cross your home internet connection.

---

*kenv · By EpicRaven*
