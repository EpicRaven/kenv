# kenv

**Disposable Kaggle kernels for your terminal.** By EpicRaven.

`kenv` starts a private Kaggle kernel on demand, gives you a Jupyter URL for your local notebook (VS Code / Cursor), lets you run scripts on Kaggle's hardware, pulls the results back to your own machine, and then **always deletes the kernel afterwards** (on `exit`, Ctrl-C, closing the terminal, or `kill`).

It is a single Python file with no third-party Python dependencies of its own. It only needs the official Kaggle CLI.

---

## Table of Contents

1. [Why kenv matters](#why-kenv-matters)
2. [How it works](#how-it-works)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Set up Kaggle credentials](#set-up-kaggle-credentials)
6. [Basic commands](#basic-commands)
7. [Intermediate commands](#intermediate-commands)
8. [Advanced commands and options](#advanced-commands-and-options)
9. [Common workflows](#common-workflows)
10. [Cleanup guarantees and limits](#cleanup-guarantees-and-limits)
11. [Security notes](#security-notes)
12. [Troubleshooting](#troubleshooting)
13. [Updating and uninstalling](#updating-and-uninstalling)

---

## Why kenv matters

- **Free compute from your own terminal.** Kaggle offers CPU and GPU kernels, but normally you have to work inside the Kaggle website. kenv brings that hardware to your local workflow.
- **Use your own editor.** Every session exposes a Jupyter URL. In VS Code or Cursor, choose *Select Kernel → Existing Jupyter Server* and paste it. Your notebook runs in your editor while the code executes on Kaggle.
- **Nothing left behind.** Kernels are disposable. When your session ends for any reason, the kernel is deleted. If your machine dies mid-session, the kernel stops itself after an idle timeout, and `kenv --sweep` removes leftovers.
- **One-shot jobs.** `kenv run train.py` executes a script on Kaggle, streams the output live, and saves every new or changed file to `./kenv_output`. No manual uploading, no clicking through a UI.
- **Simple and portable.** One file, Python 3.8+, works on Linux, macOS and Windows.

---

## How it works

1. `kenv init` pushes a private kernel to your Kaggle account.
2. A small agent on the kernel starts a Jupyter server and opens two temporary Cloudflare quick tunnels (one for the agent, one for Jupyter).
3. The agent publishes the tunnel URLs to a relay topic (ntfy.sh by default) so your terminal can find them.
4. Your terminal talks to the agent to run commands, transfer files and read resource stats.
5. When you exit, kenv deletes the kernel and forgets the session.

---

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.8+ | Check with `python3 --version` (`py --version` on Windows) |
| Kaggle CLI | `pip install -U kaggle` (must be recent enough to have `kaggle kernels delete`) |
| Kaggle account and API credentials | See [Set up Kaggle credentials](#set-up-kaggle-credentials) |
| Kaggle internet access | The kernel needs internet for its tunnels. Kaggle typically requires a phone-verified account for this |
| Git | To clone the repository |
| `bash` (Linux/macOS only) | Used for the interactive kenv shell |
| Internet access to the relay | Defaults to `https://ntfy.sh` |

Optional: VS Code or Cursor with the Jupyter extension, to use a session as a notebook kernel.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/EpicRaven/kenv.git
cd kenv
```

Then follow the steps for your operating system.

### Linux

```bash
# from inside the cloned kenv folder
chmod +x kenv.py
mkdir -p ~/.local/bin
cp kenv.py ~/.local/bin/kenv
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Install the Kaggle CLI if you do not have it yet:

```bash
sudo apt install python3 python3-pip     # Debian/Ubuntu; use your distro's package manager otherwise
pip install -U kaggle
```

> **Using bash instead of zsh?** Replace `~/.zshrc` with `~/.bashrc` in the last two lines.
>
> **"externally-managed-environment" error?** Newer distributions block system-wide `pip` installs. Use `pipx install kaggle` (after `sudo apt install pipx`), or install into a virtual environment. kenv only needs the `kaggle` command to be on your `PATH`.

### macOS

```bash
# from inside the cloned kenv folder
chmod +x kenv.py
mkdir -p ~/.local/bin
cp kenv.py ~/.local/bin/kenv
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

macOS uses zsh by default, so `~/.zshrc` is correct. If you switched to bash, use `~/.bash_profile` instead.

Install Python and the Kaggle CLI if needed:

```bash
brew install python
pip3 install -U kaggle
```

> **"externally-managed-environment" error with Homebrew Python?** Use `brew install pipx` then `pipx install kaggle`.

### Windows

Windows does not use the shebang line, so kenv is installed with a small `kenv.cmd` launcher. Run these in **PowerShell** from inside the cloned `kenv` folder:

```powershell
# 1. Create a personal bin folder and copy the script into it
$bin = "$HOME\.local\bin"
New-Item -ItemType Directory -Force -Path $bin | Out-Null
Copy-Item .\kenv.py "$bin\kenv.py" -Force

# 2. Create the launcher so you can type `kenv` anywhere
Set-Content -Path "$bin\kenv.cmd" -Value '@py "%~dp0kenv.py" %*' -Encoding Ascii

# 3. Add the folder to your user PATH (once)
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$bin*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$bin", "User")
}
```

**Close and reopen your terminal**, then continue.

Install Python and the Kaggle CLI if needed:

```powershell
winget install Python.Python.3.12
py -m pip install -U kaggle
```

> **`kaggle` not recognised?** Add Python's `Scripts` folder to your `PATH`, or install with `pipx install kaggle`. kenv looks for the `kaggle` command on your `PATH`.
>
> **Prefer WSL?** Inside a WSL terminal, follow the [Linux](#linux) steps instead.
>
> **Not installing globally?** You can always run it directly: `py kenv.py init`

### Verify the install

```bash
kenv --version
kenv --cred
```

`kenv --cred` checks your operating system, Python, the Kaggle CLI, relay reachability and your Kaggle credentials, and prints the exact fix for anything that is missing.

---

## Set up Kaggle credentials

1. On Kaggle, go to **Settings → Create New Token**. This downloads `kaggle.json`.
2. Place the file where kenv can find it:

| System | Location |
|---|---|
| Linux / macOS | `~/.kaggle/kaggle.json` |
| Windows | `C:\Users\<you>\.kaggle\kaggle.json` |

On Linux and macOS, restrict its permissions:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Alternative: environment variables**

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

Run `kenv --cred` again. When it reports that Kaggle accepted your credentials, you are ready.

---

## Basic commands

These are all you need to start a session, move files, and shut it down. Every session starts with `kenv init`, which opens a kenv shell (your prompt is prefixed with `(kenv:<name>)`). The commands below are typed inside that shell.

| Command | What it does |
|---|---|
| `kenv --help` (or `-h`) | Show every command |
| `kenv --version` | Show the installed version |
| `kenv --cred` | Check your setup and Kaggle credentials |
| `kenv init` | Start a session with a random name and open the kenv shell |
| `kenv status` | Show CPU, RAM and disk usage of the kernel |
| `kenv ls [path]` | List files on the kernel (default `/kaggle/working`) |
| `kenv put <files/folders>` | Upload files or folders from your machine to the kernel |
| `kenv save <paths>` | Download files from the kernel to your current folder |
| `kenv stop` | Delete the session's kernel immediately |
| `exit` | Leave the kenv shell; the kernel is deleted |

**A first session**

```bash
kenv init                  # start a session
kenv status                # check the kernel
kenv put data/             # upload a folder
kenv ls                    # confirm it arrived
kenv save results/         # download results to your machine
exit                       # end the session; the kernel is deleted
```

Paths given to `kenv save` and `kenv ls` are relative to `/kaggle/working` on the kernel. `kenv save .` grabs everything.

---

## Intermediate commands

Once the basics are comfortable, these commands cover running code, naming sessions, using notebooks, and controlling where files go.

### Run a script on the kernel

```bash
kenv run <script.py> [args]
```

kenv uploads the script, runs it on the kernel, streams the output live, and saves every new or changed file to `./kenv_output`. It supports `.py` and `.sh` files.

- Everything **after** the script name is passed to your script.
- kenv's own options must come **before** the script.

```bash
kenv run --out results train.py --epochs 3     # correct: --out is kenv's, --epochs is the script's
kenv run train.py --out results                # --out goes to your script, not to kenv
```

If you have no active session, `kenv run` starts a temporary one, runs the script, saves the results, and deletes the kernel.

### Run a shell command on the kernel

```bash
kenv exec nvidia-smi
kenv exec "pip install torch"
```

### Name your session

```bash
kenv -n "my-experiment"        # same as: kenv init -n "my-experiment"
```

Names need at least 3 letters or digits.

### Use a session from a notebook

```bash
kenv --url
```

This prints the Kaggle link, the **Jupyter URL** and the **Attach ID**. In VS Code or Cursor, choose *Select Kernel → Existing Jupyter Server* and paste the Jupyter URL.

### Open a second terminal on the same session

```bash
# in the first terminal
kenv --url                                   # note the Attach ID, e.g. kenv://swift-raven-42#<secret>

# in another terminal
kenv -id kenv://swift-raven-42#<secret>
```

### Choose where files go

| Option | Used with | Meaning |
|---|---|---|
| `--dest <dir>` | `kenv put` | Destination folder on the kernel |
| `--to <dir>` | `kenv save` | Destination folder on your machine |
| `--out <dir>` | `kenv run` | Local folder for the files the script produced (default `./kenv_output`) |

---

## Advanced commands and options

These deal with accelerators, timing, recovery from crashes, and configuration.

### Accelerators (GPU)

Kaggle decides the exact hardware, and your account's weekly GPU quota applies.

```bash
kenv init --gpu           # pick an accelerator when the session starts
kenv gpu                  # switch accelerator inside a session (prompts you)
kenv gpu t4               # switch directly: t4, l4 or none
```

| Choice | Hardware |
|---|---|
| `none` | CPU only |
| `t4` | GPU T4 x2 (Kaggle's default GPU) |
| `l4` | GPU L4 |

`kenv status` also shows GPU usage once an accelerator is attached. `--gpu` works with `kenv run` too, for one-shot GPU jobs:

```bash
kenv run --gpu train.py --epochs 10
```

### Timing options

| Option | Meaning |
|---|---|
| `--idle <min>` | Stop the kernel after this many idle minutes (default 20) |
| `--startup <sec>` | How long to wait for the kernel to come online (default 900) |

### Cleanup after a crash

```bash
kenv --sweep
```

Deletes leftover `kenv-*` kernels after a `kill -9` or a power loss, and forgets local sessions whose terminal is gone.

### Environment variables

| Variable | Meaning |
|---|---|
| `KENV_RELAY` | Use your own ntfy server instead of `https://ntfy.sh` |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Kaggle credentials, as an alternative to `kaggle.json` |
| `KAGGLE_CONFIG_DIR` | Custom folder containing `kaggle.json` |
| `NO_COLOR` | Disable coloured output |

---

## Common workflows

**Use a Kaggle GPU from a notebook in VS Code or Cursor**

1. Run `kenv init --gpu` and wait for the session to come online.
2. Copy the Jupyter URL printed by kenv (`kenv --url` shows it again).
3. In VS Code or Cursor: *Select Kernel → Existing Jupyter Server →* paste the URL.
4. Work as normal. Type `exit` in the kenv terminal when finished.

**Run a script once and collect the results**

```bash
kenv run train.py --epochs 10
# outputs land in ./kenv_output
```

**Clean up after a crash**

```bash
kenv --sweep
```

---

## Cleanup guarantees and limits

- The kernel is deleted on `exit`, Ctrl-C, closing the terminal, or `kill`.
- If your machine vanishes, the kernel stops itself after `--idle` minutes (default 20).
- Every kernel also has a hard maximum lifetime of **8 hours**.
- Files on the kernel disappear with it. Use `kenv save` or `kenv run` for anything you want to keep.
- Uploads are capped at roughly 95 MB per request because of the tunnel limits.

---

## Security notes

- Treat the **Jupyter URL** and the **Attach ID** like passwords. Anyone who has them can control your session.
- Kernels are private and are created under your own Kaggle account.
- The relay (ntfy.sh by default) only carries the session's tunnel URLs, under a topic derived from a hash of the session name and a random secret. For full control, run your own ntfy server and set `KENV_RELAY`.
- On the kernel, kenv downloads `cloudflared` from Cloudflare's GitHub releases to create the tunnels.
- Session state is stored locally in `~/.kenv/sessions`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Kaggle CLI not found` | `pip install -U kaggle`, and make sure `kaggle` is on your `PATH` |
| `This kaggle CLI has no kernels delete` | Upgrade: `pip install -U kaggle` |
| `No Kaggle credentials found` | Run `kenv --cred` and follow the instructions |
| `Cannot reach the relay` | Check your connection, or point `KENV_RELAY` at your own ntfy server |
| Kernel stops before coming online | Make sure internet is enabled for your Kaggle account (phone verification) |
| `kenv: command not found` (Linux/macOS) | Confirm `~/.local/bin` is in `PATH`: `echo $PATH`, then re-run `source ~/.zshrc` |
| `kenv` not recognised (Windows) | Reopen the terminal after the PATH change, or run `py kenv.py` directly |
| Leftover kernels on Kaggle | `kenv --sweep` |
| "You are already inside a kenv session" | Type `exit` first |

---

## Updating and uninstalling

**Update**

```bash
cd kenv
git pull
cp kenv.py ~/.local/bin/kenv            # Linux / macOS
```

```powershell
git pull                                 # Windows (PowerShell)
Copy-Item .\kenv.py "$HOME\.local\bin\kenv.py" -Force
```

**Uninstall**

```bash
rm ~/.local/bin/kenv                     # Linux / macOS
rm -rf ~/.kenv                           # optional: remove local session state
```

```powershell
Remove-Item "$HOME\.local\bin\kenv.py", "$HOME\.local\bin\kenv.cmd"   # Windows
Remove-Item -Recurse -Force "$HOME\.kenv"                              # optional
```

Then remove the `PATH` line you added to `~/.zshrc` (or the folder from your Windows user `PATH`) if you no longer need it.

---

**kenv v1.0.0** · By EpicRaven
