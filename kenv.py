#!/usr/bin/env python3
"""
kenv - disposable Kaggle kernels for your terminal.                     By EpicRaven

Start a session, get a Jupyter URL for your local notebook, run scripts on Kaggle's
CPU/GPU, pull files back to your own folder. The kernel is ALWAYS deleted afterwards
(exit, Ctrl-C, closing the terminal, kill). Run `kenv --help` for every command.

Project-aware: the folder you run `kenv init` in is the project. kenv keeps a .kenv/ folder
there (versions, session times, sync hashes, attached Kaggle datasets, one core.toml per version) and
mirrors the project to the kernel, so relative paths in notebooks and scripts behave like your local folder.
On the kernel `import kenv` works: kenv.cli("kenv ...") runs any command, kenv.time_start/time_end/time_output
measure code.

Needs only: Python 3.8+, `pip install -U kaggle`, and Kaggle credentials (`kenv --cred`).
"""
import argparse
import ast
import base64
import calendar
import contextlib
import datetime
import difflib
import fnmatch
import hashlib
import hmac
import io
import json
import math
import os
import platform
import posixpath
import queue
import random
import re
import secrets
import shlex
import shutil
import signal
import stat
import string
import subprocess
import sys
import tempfile
import threading
import time
import tokenize
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VERSION = "1.6.0"
RELAY = os.environ.get("KENV_RELAY", "https://ntfy.sh").rstrip("/")
STATE_DIR = Path.home() / ".kenv" / "sessions"
UA = "Mozilla/5.0 (kenv)"
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git", ".venv", "venv", "node_modules"}
RETRY_CODES = {502, 503, 504, 521, 522, 523, 524, 530}
MAX_UPLOAD = 95 * 1024 * 1024  # quick tunnels cap request bodies at ~100 MB

# key, label, Kaggle accelerator id. Kaggle decides the real hardware; availability depends on your account.
GPUS = [
    ("none", "None (CPU only)", None),
    ("t4", "GPU T4 x2 (Kaggle's default GPU)", "NvidiaTeslaT4"),
    ("l4", "GPU L4", "NvidiaL4"),
]

CONNECT_RE = re.compile(r"^(?:kenv://)?([a-z0-9][a-z0-9-]*)#([0-9a-f]{32})$")
VALUE_OPTS = {"-n", "--name", "-id", "--id", "--out", "--to", "--dest", "--idle", "--startup", "-r", "--rename",
              "-m", "--message", "--since", "--grep", "--to-fmt", "--from", "--limit", "--warn", "--file", "--tunnel", "-uri", "--uri", "--port"}


class KenvError(Exception):
    pass


# ----------------------------------------------------------------------------- output helpers

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(text, code):
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def say(msg=""):
    try:
        print(msg, flush=True)
    except OSError:  # terminal already gone (SIGHUP) - cleanup must still continue
        pass


def die(msg, code=1):
    print(msg, file=sys.stderr)
    sys.exit(code)


def _glyphs():
    K = [" _  __ ", "| |/ / ", "| ' /  ", "| . \\  ", "|_|\\_\\ "]
    E = [" _____ ", "| ____|", "|  _|  ", "| |___ ", "|_____|"]
    N = [" _   _ ", "| \\ | |", "|  \\| |", "| |\\  |", "|_| \\_|"]
    V = ["__     __", "\\ \\   / /", " \\ \\ / / ", "  \\ V /  ", "   \\_/   "]
    return [" ".join(g[i] for g in (K, E, N, V)) for i in range(5)]


def banner():
    say()
    say(c("\n".join("  " + ln for ln in _glyphs()), "36;1"))
    say(c("  By EpicRaven", "33;1") + c(f"   v{VERSION} - disposable Kaggle kernels", "2"))
    say()


def box(title, lines):
    width = max(len(title), *(len(x) for x in lines)) + 4
    say(c("+" + "-" * width + "+", "33"))
    say(c("|", "33") + "  " + c(title.ljust(width - 2), "33;1") + c("|", "33"))
    say(c("|" + " " * width + "|", "33"))
    for x in lines:
        say(c("|", "33") + "  " + x.ljust(width - 2) + c("|", "33"))
    say(c("+" + "-" * width + "+", "33"))


def gb(n):
    return f"{n / 1024 ** 3:.1f} GB"


def bar(pct, width=20):
    pct = max(0, min(100, pct))
    full = int(round(width * pct / 100))
    return "#" * full + "." * (width - full)


# ----------------------------------------------------------------------------- Kaggle helpers

def cli(*args, stdin=None, timeout=None):
    return subprocess.run(["kaggle", *args], capture_output=True, text=True, input=stdin, timeout=timeout)


def cli_delete(ref):
    """Delete a kernel. Tries `-y`, then falls back to answering the prompt."""
    for extra, stdin in ((["-y"], None), ([], "y\n")):
        try:
            if cli("kernels", "delete", ref, *extra, stdin=stdin).returncode == 0:
                return True
        except Exception:
            pass
    return False


def cred_file():
    return Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))) / "kaggle.json"


def find_creds():
    """-> (username, where) or None"""
    u, k = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if u and k:
        return u, "environment variables"
    f = cred_file()
    if f.exists():
        try:
            d = json.loads(f.read_text())
            if d.get("username") and d.get("key"):
                return d["username"], str(f)
        except Exception:
            pass
    return None


def get_username():
    cr = find_creds()
    if not cr:
        raise KenvError("No Kaggle credentials found. Run: kenv --cred")
    return cr[0]


def require_cli():
    if not shutil.which("kaggle"):
        raise KenvError("Kaggle CLI not found. Install it: pip install -U kaggle   (see: kenv --cred)")
    if cli("kernels", "delete", "--help").returncode != 0:  # fail fast: cleanup depends on it
        raise KenvError("This kaggle CLI has no `kernels delete`. Upgrade: pip install -U kaggle")


def require_relay():
    try:
        with http(f"{RELAY}/v1/health", timeout=8) as r:
            r.read()
    except Exception as e:
        raise KenvError(f"Cannot reach the relay at {RELAY} ({e}).\n"
                        "kenv uses it so your terminal can find the kernel's URLs. "
                        "Check your connection, or point KENV_RELAY at your own ntfy server.")


# ----------------------------------------------------------------------------- networking

def http(url, data=None, headers=None, timeout=30, method=None):
    h = {"User-Agent": UA}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def topic_of(name, secret):
    return "kenv-" + hashlib.sha256(f"{name}:{secret}".encode()).hexdigest()[:32]


def connect_str(name, secret):
    return f"kenv://{name}#{secret}"


class Endpoint:
    """Talks to the agent running on the kernel. URLs are looked up through the relay topic."""

    def __init__(self, name, secret):
        self.name, self.secret = name, secret
        self.topic = topic_of(name, secret)
        self.info = None

    def latest(self):
        """Newest message the kernel's agent published (highest generation), or None."""
        try:
            with http(f"{RELAY}/{self.topic}/json?poll=1&since=all", timeout=20) as r:
                lines = r.read().decode("utf-8", "replace").splitlines()
        except Exception:
            return None
        best = None
        for ln in lines:
            try:
                ev = json.loads(ln)
                if ev.get("event") != "message":
                    continue
                m = json.loads(ev["message"])
                if "agent_url" not in m:
                    continue
            except Exception:
                continue
            if best is None or m.get("gen", 0) >= best.get("gen", 0):
                best = m
        return best

    def resolve(self, force=False):
        if self.info and not force:
            return self.info
        m = self.latest()
        if not m:
            raise KenvError("No running session found for that ID (the kernel may still be starting, "
                            "or it already ended).")
        self.info = m
        return m

    def call(self, path, payload=None, raw=None, query="", timeout=60, tries=4):
        last = None
        for i in range(tries):
            info = self.resolve(force=i > 0)
            url = info["agent_url"].rstrip("/") + path + query
            data, hdr = raw, {"Authorization": f"Bearer {self.secret}"}
            if payload is not None:
                data = json.dumps(payload).encode()
                hdr["Content-Type"] = "application/json"
            try:
                return http(url, data=data, headers=hdr, timeout=timeout,
                            method="POST" if data is not None else "GET")
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise KenvError("The session refused this ID (wrong secret?).")
                if e.code not in RETRY_CODES:
                    raise KenvError(f"Session error {e.code}: {e.read()[:200]!r}")
                last = e
            except (urllib.error.URLError, OSError) as e:  # DNS for a fresh tunnel can lag a few seconds
                last = e
            time.sleep(3)
        raise KenvError(f"Cannot reach the session ({last}). It may have ended or idled out. Check: kenv --url")

    def json_call(self, path, payload=None, raw=None, query="", timeout=60, tries=4):
        with self.call(path, payload=payload, raw=raw, query=query, timeout=timeout, tries=tries) as r:
            return json.loads(r.read() or b"{}")


# ----------------------------------------------------------------------------- local session state

def state_path(name):
    return STATE_DIR / f"{name}.json"


def load_state(name):
    try:
        return json.loads(state_path(name).read_text())
    except Exception:
        return None


def save_state(st):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    p = state_path(st["name"])
    p.write_text(json.dumps(st))
    try:
        p.chmod(0o600)
    except OSError:
        pass


def remove_state(name):
    for p in (state_path(name), STATE_DIR / f"{name}.sync"):
        try:
            p.unlink()
        except OSError:
            pass


def list_states():
    out = []
    if STATE_DIR.exists():
        for f in sorted(STATE_DIR.glob("*.json")):
            try:
                out.append(json.loads(f.read_text()))
            except Exception:
                pass
    return out


def pid_alive(pid):
    if not pid:
        return False
    if os.name == "nt":  # os.kill(pid, 0) would TERMINATE the process on Windows
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sanitize(name):
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:30].strip("-")
    if len(s) < 3:
        raise KenvError("Session names need at least 3 letters/digits.")
    return s


_ADJ = ["swift", "brave", "calm", "quiet", "lucky", "witty", "bold", "eager", "fuzzy", "gentle",
        "jolly", "mellow", "nimble", "proud", "rapid", "sunny"]
_NOUN = ["seal", "raven", "otter", "falcon", "lynx", "panda", "heron", "gecko", "bison", "koala",
         "wombat", "marten", "osprey", "tapir", "manta", "newt"]


def random_name():
    return f"{random.choice(_NOUN)}-{random.choice(_ADJ)}-{random.randint(10, 99)}"


def parse_target(s):
    """Attach ID (kenv://name#secret) or the name of a session started on this machine."""
    s = s.strip()
    if s.lower().startswith("kv:"):
        raise KenvError("That is a VERSION id (kv:...), not an attach ID. Switch versions with: kenv activate -id " + s)
    m = CONNECT_RE.match(s)
    if m:
        return m.group(1), m.group(2)
    try:
        st = load_state(sanitize(s))
    except KenvError:
        st = None
    if st:
        return st["name"], st["secret"]
    raise KenvError("That is not a valid attach ID. It looks like kenv://<name>#<secret> - "
                    "run `kenv --url` in the window that started the session.")


def find_target():
    s = os.environ.get("KENV_SESSION")
    if s:
        return parse_target(s)
    live = [st for st in list_states() if pid_alive(st.get("pid"))]
    if len(live) == 1:
        return live[0]["name"], live[0]["secret"]
    return None


def need_ep():
    t = find_target()
    if not t:
        live = [st for st in list_states() if pid_alive(st.get("pid"))]
        hint = ""
        if len(live) > 1:
            hint = "\nSeveral sessions are running here: " + ", ".join(s["name"] for s in live) + \
                   "\nPick one with: kenv -id <name>"
        raise KenvError("No active kenv session in this terminal. Start one with `kenv init`, "
                        "or attach with `kenv -id <id>`." + hint)
    return Endpoint(*t)


def accel_label(g):
    for _, label, gid in GPUS:
        if (gid or "none") == g:
            return label
    return g or "None (CPU only)"


def print_info(ep):
    m = ep.info or ep.resolve()
    say(f"  Session     : {c(ep.name, '1')}")
    say(f"  Accelerator : {accel_label(m.get('gpu', 'none'))}")
    say(f"  Kaggle      : https://www.kaggle.com/code/{m['ref']}")
    say(f"  Jupyter     : {m['jupyter_url']}")
    say(f"  Attach ID   : {connect_str(ep.name, ep.secret)}")
    say()
    say(c("  Notebook : VS Code / Cursor -> Select Kernel -> Existing Jupyter Server -> paste the Jupyter URL.", "2"))
    say(c("  Terminal : open another window and run  kenv -id <Attach ID>", "2"))
    say(c("  Treat the Jupyter URL and Attach ID like passwords.", "2"))


# ----------------------------------------------------------------------------- project folder (.kenv)
#
# <project>/.kenvignore        what is NOT synced (gitignore syntax)
# <project>/.kenv/last_active  the version the last session used
# <project>/.kenv/sync.json    local file hashes (hash cache + baseline for incremental sync)
# <project>/.kenv/datasets.json  Kaggle datasets that are attached to every new kernel
# <project>/.kenv/vN/sessions.json  one entry per session: start, end, duration, how it ended
# <project>/.kenv/vN/meta.json     version id (kv:...) and optional name
# Only names and hashes live here - never keys, tokens or attach secrets.

PROJECT_DIR = ".kenv"
IGNORE_FILE = ".kenvignore"
DEFAULT_IGNORES = [".git", ".venv", "venv", "node_modules", "__pycache__", ".ipynb_checkpoints", ".kenv"]
BATCH_LIMIT = 40 * 1024 * 1024   # a zip batch is sent once it reaches this size ...
BIG_FILE = 32 * 1024 * 1024      # ... and files above this are sent raw in chunks, so a request stays < MAX_UPLOAD
CHUNK = 64 * 1024 * 1024
SYNC_WARN = 1024 ** 3            # ask before the first upload of a project bigger than this
PULL_BATCH = 100 * 1024 * 1024
PULL_MAX_FILE = 500 * 1024 * 1024
VERSION_RE = re.compile(r"^v(\d+)$", re.I)
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")
BRANCHES_FILE, TAGS_FILE, LOG_DIR = "branches.json", "tags.json", "logs"

IGNORE_TEMPLATE = """\
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
"""


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def write_json(path, obj, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=indent))
    os.replace(tmp, path)


def iso(ts=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if ts is None else ts))


def from_iso(s):
    try:
        return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))
    except Exception:
        return None


def ago(ts):
    s = max(0, int(time.time() - ts))
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h ago"
    return f"{s // 86400}d ago"


def dur(secs):
    h, r = divmod(int(secs or 0), 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


# ---- .kenvignore (gitignore syntax, stdlib only)

def _glob_to_regex(pat):
    out, i, n = [], 0, len(pat)
    while i < n:
        ch = pat[i]
        if ch == "*":
            if pat[i:i + 2] == "**":
                j = i + 2
                if pat[j:j + 1] == "/":  # "**/" = zero or more directories
                    out.append("(?:.*/)?")
                    i = j + 1
                    continue
                out.append(".*")
                i = j
                continue
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        elif ch == "[":
            j = pat.find("]", i + 2)
            if j == -1:
                out.append("\\[")
            else:
                body = pat[i + 1:j]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                i = j
        else:
            out.append(re.escape(ch))
        i += 1
    return "".join(out)


class IgnoreRules:
    """Last matching rule wins; an ignored directory is never entered (same as git)."""

    def __init__(self, lines=()):
        self.rules = []
        for raw in list(DEFAULT_IGNORES) + list(lines):
            self._add(raw)

    def _add(self, raw):
        ln = raw.rstrip("\r\n")
        if not ln.strip() or ln.startswith("#"):
            return
        ln = ln.rstrip()
        neg = ln.startswith("!")
        if neg:
            ln = ln[1:]
        elif ln.startswith("\\"):  # "\#file" and "\!file"
            ln = ln[1:]
        dir_only = ln.endswith("/")
        ln = ln.rstrip("/")
        anchored = "/" in ln
        ln = ln.lstrip("/")
        if not ln:
            return
        try:
            rx = re.compile(("^" if anchored else "^(?:.*/)?") + _glob_to_regex(ln) + "$")
        except re.error:
            return
        self.rules.append((rx, neg, dir_only))

    @classmethod
    def load(cls, root):
        try:
            return cls((Path(root) / IGNORE_FILE).read_text(errors="replace").splitlines())
        except OSError:
            return cls()

    def ignored(self, rel, is_dir):
        res = False
        for rx, neg, dir_only in self.rules:
            if dir_only and not is_dir:
                continue
            if rx.match(rel):
                res = not neg
        return res

    def path_ignored(self, rel):
        """A file path (posix, relative to the project): ignored itself or inside an ignored folder."""
        parts = rel.split("/")
        for i in range(1, len(parts)):
            if self.ignored("/".join(parts[:i]), True):
                return True
        return self.ignored(rel, False)


# ---- hashing and scanning the local project

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def file_entry(p):
    st = os.stat(p)
    return {"sha256": sha256_file(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def scan_local(root, rules, cache=None):
    """-> {relpath: {sha256, size, mtime_ns}} for every non-ignored file. `cache` avoids re-hashing
    files whose size and mtime did not change (files touched in the last 2 s are always re-hashed)."""
    cache, out, now = cache or {}, {}, time.time_ns()
    root = str(root)
    for cur, dirs, files in os.walk(root):
        rel_dir = os.path.relpath(cur, root).replace(os.sep, "/")
        rel_dir = "" if rel_dir == "." else rel_dir
        keep = []
        for d in dirs:
            rel = f"{rel_dir}/{d}" if rel_dir else d
            if d == PROJECT_DIR or os.path.islink(os.path.join(cur, d)) or rules.ignored(rel, True):
                continue
            keep.append(d)
        dirs[:] = keep
        for f in files:
            rel = f"{rel_dir}/{f}" if rel_dir else f
            fp = os.path.join(cur, f)
            if os.path.islink(fp) or rules.ignored(rel, False):
                continue
            try:
                st = os.stat(fp)
                old = cache.get(rel)
                if (old and old.get("size") == st.st_size and old.get("mtime_ns") == st.st_mtime_ns
                        and now - st.st_mtime_ns > 2_000_000_000):
                    out[rel] = old
                else:
                    out[rel] = {"sha256": sha256_file(fp), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
            except OSError:
                pass  # vanished or unreadable
    return out


def diff_local(old, new):
    added = [r for r in new if r not in old]
    removed = [r for r in old if r not in new]
    modified = [r for r in new if r in old and old[r].get("sha256") != new[r]["sha256"]]
    return added, modified, removed


def alt_name(rel):
    d, f = posixpath.split(rel)
    stem, ext = posixpath.splitext(f)
    return posixpath.join(d, f"{stem}.kenv-remote{ext}")


# ---- the project itself

def new_version_id():
    return "kv:" + "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))


class Project:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.kdir = self.root / PROJECT_DIR

    @property
    def kname(self):
        """Folder name on the kernel (/kaggle/working/<kname>)."""
        return re.sub(r"[^A-Za-z0-9._-]+", "-", self.root.name).strip("-.") or "project"

    def guard(self):
        if self.root == Path.home().resolve() or len(self.root.parts) <= 1:
            raise KenvError("This is your home (or the root) folder, not a project. `cd` into a project folder first.\n"
                            "kenv keeps a .kenv folder in the project and mirrors the folder to the kernel.")

    def ensure(self):
        """Create .kenv and its files if missing. -> True when .kenv did not exist before."""
        self.guard()
        created = not self.kdir.is_dir()
        self.kdir.mkdir(exist_ok=True)
        if not (self.kdir / "sync.json").exists():
            write_json(self.kdir / "sync.json", {"version": 1, "updated": iso(), "files": {}})
        if not (self.kdir / "datasets.json").exists():
            write_json(self.kdir / "datasets.json", {"datasets": []})
        if not (self.root / IGNORE_FILE).exists():
            (self.root / IGNORE_FILE).write_text(IGNORE_TEMPLATE)
        write_shim(self.root)  # kenv_shim.py: `from kenv_shim import kenv` never breaks outside a session
        return created

    def rules(self):
        return IgnoreRules.load(self.root)

    def live_session(self):
        for st in list_states():
            if st.get("project") == str(self.root) and pid_alive(st.get("pid")):
                return st
        return None

    # -- versions
    def version_names(self):
        nums = []
        if self.kdir.is_dir():
            for d in self.kdir.iterdir():
                m = VERSION_RE.match(d.name)
                if m and d.is_dir():
                    nums.append(int(m.group(1)))
        return [f"v{n}" for n in sorted(nums)]

    def meta(self, v):
        p = self.kdir / v / "meta.json"
        m = read_json(p, None)
        if not isinstance(m, dict) or not m.get("id"):  # also heals a hand-made vN folder
            m = {"id": new_version_id(), "name": (m or {}).get("name"), "created": iso()}
            write_json(p, m)
        return m

    def label(self, v):
        n = self.meta(v).get("name")
        return f"{v} ({n})" if n else v

    def new_version(self, name=None):
        if name is not None:
            self._check_name(name)
        nums = [int(v[1:]) for v in self.version_names()]
        v = f"v{(max(nums) + 1) if nums else 1}"
        (self.kdir / v).mkdir(parents=True)
        write_json(self.kdir / v / "meta.json", {"id": new_version_id(), "name": name, "created": iso()})
        write_json(self.kdir / v / "sessions.json", {"sessions": []})
        return v

    def _check_name(self, name, me=None):
        if not NAME_RE.match(name) or VERSION_RE.match(name):
            raise KenvError("Version names use letters, digits, '.', '_' or '-' (max 40 characters) "
                            "and cannot look like v1, v2 ...")
        for o in self.version_names():
            if o != me and (self.meta(o).get("name") or "").lower() == name.lower():
                raise KenvError(f"Another version is already called '{name}'.")
        if name.lower() in {t.lower() for t in self.tags()}:
            raise KenvError(f"'{name}' is already a tag; tags and version names share one namespace.")

    def rename(self, v, name):
        self._check_name(name, me=v)
        m = self.meta(v)
        m["name"] = name
        write_json(self.kdir / v / "meta.json", m)

    def find_version(self, target):
        """'v2' | version name | 'kv:...' id  ->  'vN' (KenvError when nothing matches)."""
        t, vs = target.strip(), self.version_names()
        if t.lower().startswith("kv:"):
            for v in vs:
                if self.meta(v)["id"] == t:
                    return v
            raise KenvError(f"No version in this project has the id {t}. List them: kenv versions")
        m = VERSION_RE.match(t)
        if m:
            v = f"v{int(m.group(1))}"
            if v in vs:
                return v
            raise KenvError(f"There is no {v} in this project. List them: kenv versions")
        for k, x in self.tags().items():
            if k.lower() == t.lower() and x in vs:
                return x
        for v in vs:
            if (self.meta(v).get("name") or "").lower() == t.lower():
                return v
        raise KenvError(f"No version, name or tag called '{t}' in this project. List them: kenv versions / kenv tag")

    def set_active(self, v):
        (self.kdir / "last_active").write_text(v + "\n")

    def last_active(self):
        vs = self.version_names()
        if not vs:
            return None
        try:
            return self.find_version((self.kdir / "last_active").read_text().strip())
        except (OSError, KenvError):
            return max(vs, key=lambda x: (self.last_ts(x) or 0, int(x[1:])))  # newest session wins

    # -- tags, branches and snapshots (Phase 3)
    def tags(self):
        d = read_json(self.kdir / TAGS_FILE, {})
        return {k: v for k, v in d.items() if isinstance(v, str)} if isinstance(d, dict) else {}

    def save_tags(self, d):
        write_json(self.kdir / TAGS_FILE, dict(sorted(d.items())))

    def tags_of(self, v):
        return sorted(k for k, x in self.tags().items() if x == v)

    def branches(self):
        d = read_json(self.kdir / BRANCHES_FILE, {})
        return dict(d) if isinstance(d, dict) else {}

    def save_branches(self, d):
        write_json(self.kdir / BRANCHES_FILE, dict(sorted(d.items())))

    def current_branch(self):
        try:
            b = (self.kdir / "branch").read_text().strip()
        except OSError:
            b = ""
        return b if b in self.branches() else "main"

    def set_branch(self, b):
        (self.kdir / "branch").write_text(b + "\n")

    def branch_versions(self, b):
        """Committed versions made on branch `b`, oldest first (the automatic safety copies before a rollback are not counted)."""
        return [v for v in self.version_names()
                if self.meta(v).get("committed") and not self.meta(v).get("auto")
                and (self.meta(v).get("branch") or "main") == b]

    def has_snapshot(self, v):
        return (self.kdir / v / "code.json").is_file()

    def load_snapshot(self, v):
        d = read_json(self.kdir / v / "code.json", {})
        f = d.get("files") if isinstance(d, dict) else None
        return f if isinstance(f, dict) else {}

    def set_meta(self, v, **kw):
        m = self.meta(v)
        for k, x in kw.items():
            if x is None:
                m.pop(k, None)
            else:
                m[k] = x
        write_json(self.kdir / v / "meta.json", m)

    # -- sessions
    def sessions(self, v):
        return read_json(self.kdir / v / "sessions.json", {}).get("sessions", [])

    def _save_sessions(self, v, sess):
        write_json(self.kdir / v / "sessions.json", {"sessions": sess})

    def last_ts(self, v):
        best = None
        for e in self.sessions(v):
            t = from_iso(e.get("end")) or from_iso(e.get("last_seen")) or from_iso(e.get("start"))
            if t and (best is None or t > best):
                best = t
        return best

    def total_time(self, v):
        return sum(e.get("duration_s") or 0 for e in self.sessions(v))

    def open_session(self, v, sid, session, idle_min, gpu):
        sess = self.sessions(v)
        sess.append({"sid": sid, "session": session, "start": iso(), "end": None, "duration_s": None,
                     "ended_by": None, "last_seen": iso(), "idle_min": idle_min, "gpu": gpu, "pid": os.getpid()})
        self._save_sessions(v, sess)
        self.set_active(v)

    def touch_session(self, v, sid):
        sess = self.sessions(v)
        for e in sess:
            if e.get("sid") == sid and not e.get("end"):
                e["last_seen"] = iso()
                self._save_sessions(v, sess)
                return

    def session_entry(self, v, sid):
        for e in self.sessions(v):
            if e.get("sid") == sid:
                return e
        return None

    def close_session(self, v, sid, ended_by, end_ts):
        """Write the end time once; a session that is already closed is left alone."""
        sess = self.sessions(v)
        for e in sess:
            if e.get("sid") == sid and not e.get("end"):
                start = from_iso(e.get("start")) or end_ts
                e.update(end=iso(end_ts), duration_s=int(max(0, end_ts - start)), ended_by=ended_by)
                self._save_sessions(v, sess)
                return True
        return False

    def repair_crashed(self):
        """Entries without an end (kill -9, power loss) -> 'unexpected', end estimated from the idle timeout:
        the kernel stops itself idle_min minutes after the last heartbeat we recorded."""
        fixed = []
        for v in self.version_names():
            sess, dirty = self.sessions(v), False
            for e in sess:
                if e.get("end"):
                    continue
                st = load_state(e["session"]) if e.get("session") else None
                if st and st.get("sid") == e.get("sid") and pid_alive(st.get("pid")):
                    continue  # genuinely still running in another terminal
                start = from_iso(e.get("start")) or time.time()
                seen = max(from_iso(e.get("last_seen")) or start, start)
                end = min(time.time(), seen + int(e.get("idle_min") or 20) * 60)
                e.update(end=iso(end), duration_s=int(max(0, end - start)), ended_by="unexpected", estimated=True)
                dirty = True
                fixed.append((v, e))
            if dirty:
                self._save_sessions(v, sess)
        return fixed

    # -- sync + datasets files
    def load_sync(self):
        d = read_json(self.kdir / "sync.json", {})
        if not isinstance(d.get("files"), dict):
            d["files"] = {}
        return d

    def save_sync(self, files):
        write_json(self.kdir / "sync.json", {"version": 1, "updated": iso(), "files": files}, indent=None)

    def datasets(self):
        return read_json(self.kdir / "datasets.json", {}).get("datasets", [])

    def add_dataset(self, ref, folder):
        ds = [d for d in self.datasets() if d.get("ref") != ref]
        ds.append({"ref": ref, "folder": folder, "updated": iso()})
        write_json(self.kdir / "datasets.json", {"datasets": ds})


def current_project(create=False):
    """The project of this shell (kenv session) or the nearest folder above the cwd that has a .kenv."""
    env = os.environ.get("KENV_SESSION")
    if env:
        try:
            st = load_state(parse_target(env)[0])
            if st and st.get("project") and (Path(st["project"]) / PROJECT_DIR).is_dir():
                return Project(st["project"])
        except KenvError:
            pass
    home, d = Path.home().resolve(), Path.cwd().resolve()
    for p in [d, *d.parents]:
        if p == home:
            break  # ~/.kenv is kenv's own folder, not a project
        if (p / PROJECT_DIR).is_dir():
            return Project(p)
    if create:
        return Project(d)
    raise KenvError("This folder is not a kenv project (no .kenv here or above). Run `kenv init` in your project folder.")


def project_ctx(ep):
    """(Project, session state) when the session was started from a local project on this machine."""
    st = load_state(ep.name)
    if st and st.get("project") and (Path(st["project"]) / PROJECT_DIR).is_dir():
        return Project(st["project"]), st
    return None


# ---- per-session sync state: what the kernel holds. Kept out of the *.json state so list_states() stays light.

def ksync_path(name):
    return STATE_DIR / f"{name}.sync"


def load_ksync(name):
    d = read_json(ksync_path(name), {})
    d.setdefault("pushed", {})  # rel -> sha256 of the copy we uploaded (or pulled) last
    d.setdefault("kbase", {})   # rel -> [size, mtime_ns] on the kernel when we last synced that file
    return d


def save_ksync(name, d):
    write_json(ksync_path(name), d, indent=None)


def kernel_alive(ep, timeout=10):
    try:
        ep.json_call("/ping", timeout=timeout, tries=1)
        return True
    except Exception:
        return False


class ZipBatcher:
    """Sends many small files as a few zip requests, each far below the ~95 MB tunnel limit."""

    def __init__(self, ep):
        self.ep, self.sent = ep, []
        self._new()

    def _new(self):
        self.buf = io.BytesIO()
        self.z = zipfile.ZipFile(self.buf, "w", zipfile.ZIP_DEFLATED, strict_timestamps=False)
        self.names = []

    def add(self, fp, rel):
        self.z.write(fp, rel)
        self.names.append(rel)
        if self.buf.tell() >= BATCH_LIMIT:
            self.flush()

    def flush(self):
        if not self.names:
            return
        self.z.close()
        data = self.buf.getvalue()
        if len(data) > MAX_UPLOAD:
            raise KenvError("Internal error: an upload batch exceeded the tunnel limit.")
        self.ep.json_call("/put", raw=data, query="?dest=", timeout=900)
        self.sent += self.names
        self._new()


def upload_big(ep, fp, rel):
    off = 0
    with open(fp, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            ep.json_call("/putchunk", raw=chunk, query=f"?path={urllib.parse.quote(rel)}&offset={off}", timeout=1800)
            off += len(chunk)


def push_sync(ep, proj, cache=None, quiet=False):
    """Upload local files the kernel does not have yet (or has an older copy of). The local files are the
    source of truth: this never touches them. -> number of files uploaded."""
    ks = load_ksync(ep.name)
    old = proj.load_sync()["files"]
    local = scan_local(proj.root, proj.rules(), cache if cache is not None else old)
    lazy = lazy_on(ep)  # Phase 6: only small code/config files are uploaded; the rest is read on demand
    todo = sorted(r for r, v in local.items() if ks["pushed"].get(r) != v["sha256"] and (not lazy or lazy_eager(r, v["size"])))
    if not todo:
        proj.save_sync(local)
        if not quiet:
            say("[kenv] the kernel already has your current project files" + (" (lazy mode: the rest is read on demand)" if lazy else ""))
        return 0
    total = sum(local[r]["size"] for r in todo)
    say(f"[kenv] syncing {len(todo)} file(s), {human(total)} -> /kaggle/working/{proj.kname}")
    zb, done = ZipBatcher(ep), []
    for rel in todo:
        fp = proj.root / rel
        try:
            if local[rel]["size"] > BIG_FILE:
                say(f"[kenv]   large file {rel} ({human(local[rel]['size'])}) - sending in chunks")
                upload_big(ep, fp, rel)
                done.append(rel)
            else:
                zb.add(fp, rel)
        except OSError as e:
            say(f"[kenv]   skipped {rel}: {e}")
    zb.flush()
    done += zb.sent
    snap = ep.json_call("/snapshot", payload={}, timeout=120)
    ks = load_ksync(ep.name)  # re-read: another shell of this session may have synced meanwhile
    for r in done:
        ks["pushed"][r] = local[r]["sha256"]
        if r in snap:
            ks["kbase"][r] = snap[r]
    save_ksync(ep.name, ks)
    proj.save_sync(local)
    return len(done)


def pull_back(ep, proj, dest=None, quiet=False):
    """Download files that are new or changed on the kernel since the last sync into the project.
    Ignored paths stay on the kernel. If you also edited the same file locally, your copy is kept and the
    kernel's is saved next to it as <name>.kenv-remote<ext>. -> list of local paths written."""
    ks = load_ksync(ep.name)
    rules = proj.rules()
    snap = ep.json_call("/snapshot", payload={}, timeout=120)
    changed = sorted(r for r, v in snap.items() if ks["kbase"].get(r) != v and not rules.path_ignored(r))
    if not changed:
        if not quiet:
            say("[kenv] no new or changed files on the kernel")
        return []
    target = Path(dest) if dest else proj.root
    to_root = dest is None
    baseline, conflicts, skipped = proj.load_sync()["files"], {}, []
    if to_root:
        for r in changed:
            p = proj.root / r
            if p.is_file():
                ref = ks["pushed"].get(r) or (baseline.get(r) or {}).get("sha256")
                if ref is None or sha256_file(p) != ref:  # edited locally (or never synced): do not overwrite
                    conflicts[r] = alt_name(r)
    written, batch, size = [], [], 0

    def fetch():
        with ep.call("/zip", payload={"paths": batch}, timeout=900) as r:
            data = r.read()
        written.extend(extract_zip(data, target, redirect=(lambda n: conflicts.get(n, n)) if to_root else None))

    for r in changed:
        sz = snap[r][0]
        if sz > PULL_MAX_FILE:
            skipped.append(r)
            continue
        if batch and size + sz > PULL_BATCH:
            fetch()
            batch, size = [], 0
        batch.append(r)
        size += sz
    if batch:
        fetch()
    ks, hashed = load_ksync(ep.name), {}
    for r in changed:
        ks["kbase"][r] = snap[r]  # acknowledged, so it is not pulled again and again
        p = proj.root / r
        if to_root and r not in conflicts and r not in skipped and p.is_file():
            e = file_entry(p)
            ks["pushed"][r] = e["sha256"]
            baseline[r] = e
            hashed[r] = e
    save_ksync(ep.name, ks)
    if to_root:
        proj.save_sync(baseline)
    for n in written[:20]:
        say(f"[kenv] pulled {n}")
    if len(written) > 20:
        say(f"[kenv] ... and {len(written) - 20} more")
    say(f"[kenv] {len(written)} file(s) saved to {target.resolve()}")
    for r, alt in conflicts.items():
        say(c(f"[kenv] you also changed {r} locally - kept yours; the kernel's copy is {alt}", "33"))
    for r in skipped:
        say(c(f"[kenv] {r} is over {human(PULL_MAX_FILE)} - fetch it on purpose with: kenv save {r}", "33"))
    if to_root and hashed:
        record_outputs(ep, proj, hashed)  # the core file's I/O map
    return written


def mark_synced(ep, proj, names):
    """After an explicit `kenv save <paths>`: those files match the kernel now."""
    snap = ep.json_call("/snapshot", payload={}, timeout=120)
    ks, baseline, hashed = load_ksync(ep.name), proj.load_sync()["files"], {}
    for n in names:
        p = proj.root / n
        if n in snap and p.is_file():
            ks["kbase"][n] = snap[n]
            e = file_entry(p)
            ks["pushed"][n] = e["sha256"]
            baseline[n] = e
            hashed[n] = e
    save_ksync(ep.name, ks)
    proj.save_sync(baseline)
    record_outputs(ep, proj, hashed)


def classify_end(exc, reason, ep, last_seen, idle_min, end_ts):
    """How did the session end? clean exit | ctrl-c | terminal closed | idle timeout | kernel killed | unexpected"""
    if reason:
        return reason
    if isinstance(exc, KeyboardInterrupt):
        return "ctrl-c"
    if isinstance(exc, SystemExit):
        code = exc.code
        if isinstance(code, int) and code > 128:
            return "terminal closed"  # SIGHUP / SIGTERM
        return "clean exit" if code in (0, None) else "unexpected"
    if exc is not None:
        return "unexpected"
    if kernel_alive(ep):
        return "clean exit"
    silent = end_ts - (last_seen or end_ts)
    return "idle timeout" if silent >= idle_min * 60 - 90 else "kernel killed"


# ----------------------------------------------------------------------------- the agent (runs ON the kernel)

AGENT_SRC = r'''
import base64, collections, hmac, io, json, os, platform, re, shutil, subprocess, sys, threading, time, urllib.request, zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

CFG = json.loads(base64.b64decode("__CFG__"))
SECRET, TOPIC, RELAY = CFG["secret"], CFG["topic"], CFG["relay"]
WORK = CFG.get("work", "/kaggle/working")
PROJECT = CFG.get("project")  # local project folder name; the kernel mirrors it under WORK
BASE = os.path.join(WORK, PROJECT) if PROJECT else WORK  # cwd for notebooks, scripts, exec; base of relative paths
AGENT_PORT, JUP_PORT = CFG.get("agent_port", 8899), CFG.get("jup_port", 8898)
SKIP = {"__pycache__", ".ipynb_checkpoints", ".git"}
START = time.time()
STATE = {"last": time.time(), "busy": 0, "stop": False}
LOCK = threading.Lock()
PROCS = []


LOGS, LOGSEQ, LOGCV = collections.deque(maxlen=20000), [0], threading.Condition()   # Phase 3: kernel log ring buffer
ACCESS_RE = re.compile(r"\]\s+[23]\d\d\s+(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s")


def add_log(src, text):
    now = time.time()
    with LOGCV:
        for ln in (str(text).splitlines() or [""]):
            LOGSEQ[0] += 1
            LOGS.append((LOGSEQ[0], now, src, ln[:4000]))
        LOGCV.notify_all()


def feed_log(src, data):
    """Turn raw output bytes into log lines; returns the unfinished last line."""
    parts = data.split(b"\n")
    for ln in parts[:-1]:
        t = [x for x in ln.decode("utf-8", "replace").split("\r") if x]  # a progress bar keeps only its last state
        add_log(src, t[-1] if t else "")
    last = parts[-1]
    if len(last) > 65536:
        add_log(src, last.decode("utf-8", "replace"))
        last = b""
    return last


def get_logs(req):
    since, limit = int(req.get("since", 0)), max(1, min(int(req.get("limit", 2000)), 5000))
    if req.get("epoch") not in (None, int(START)):
        since = 0  # the local side saw an older kernel: start over
    end = time.time() + min(float(req.get("wait", 0)), 25.0)
    with LOGCV:
        while LOGSEQ[0] <= since and time.time() < end:
            LOGCV.wait(timeout=max(0.05, min(1.0, end - time.time())))
        rows = [list(r) for r in LOGS if r[0] > since][:limit]
        first = LOGS[0][0] if LOGS else 1
    return {"epoch": int(START), "seq": LOGSEQ[0], "first": first, "lines": rows}


def log(*a):
    print("[kenv-agent]", *a, flush=True)
    add_log("agent", " ".join(str(x) for x in a))


def touch():
    STATE["last"] = time.time()


def resolve(p):
    p = p if os.path.isabs(p) else os.path.join(BASE, p)
    return os.path.normpath(p)


def arcname(ap):
    try:
        if os.path.commonpath([ap, BASE]) == BASE:
            return os.path.relpath(ap, BASE).replace(os.sep, "/")
    except ValueError:
        pass
    return ap.lstrip("/")


def snapshot():
    out = {}
    for root, dirs, files in os.walk(BASE):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for f in files:
            p = os.path.join(root, f)
            try:
                s = os.stat(p)
                out[os.path.relpath(p, BASE).replace(os.sep, "/")] = [s.st_size, s.st_mtime_ns]
            except OSError:
                pass
    return out


def cgroup_mem():
    pairs = (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
             ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"))
    for tot, used in pairs:
        try:
            t = open(tot).read().strip()
            u = int(open(used).read().strip())
            if t != "max" and int(t) < (1 << 50):
                return int(t), u
        except Exception:
            pass
    return None


def meminfo():
    m = {}
    for ln in open("/proc/meminfo"):
        k, v = ln.split(":")
        m[k] = int(v.split()[0]) * 1024
    return m["MemTotal"], m["MemTotal"] - m.get("MemAvailable", m["MemFree"])


def cpu_pct():
    def rd():
        v = list(map(int, open("/proc/stat").readline().split()[1:]))
        return sum(v), v[3] + (v[4] if len(v) > 4 else 0)
    t1, i1 = rd()
    time.sleep(0.4)
    t2, i2 = rd()
    return round(100.0 * (1 - (i2 - i1) / max(t2 - t1, 1)), 1)


def gpus():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    rows = []
    for ln in out.strip().splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) >= 5:
            try:
                rows.append({"name": p[0], "used": float(p[1]), "total": float(p[2]),
                             "util": float(p[3]), "temp": float(p[4])})
            except ValueError:
                pass
    return rows


def stats():
    mem = cgroup_mem() or meminfo()
    try:
        cores = len(os.sched_getaffinity(0))
    except Exception:
        cores = os.cpu_count() or 1
    disks = []
    for path in (WORK, "/tmp"):
        try:
            d = shutil.disk_usage(path)
            disks.append({"path": path, "total": d.total, "used": d.used})
        except Exception:
            pass
    return {"uptime": int(time.time() - START), "gen": CFG["gen"],
            "cpu": {"pct": cpu_pct(), "cores": cores, "load": list(os.getloadavg())},
            "mem": {"total": mem[0], "used": mem[1]}, "disk": disks, "gpus": gpus()}


# ---- Phase 2: the importable `kenv` module, the command queue and time tracking

PKG_DIR = "/tmp/kenv_pkg"
LAZY_CONF = PKG_DIR + "/lazy.json"
CV = threading.Condition()
JOBS, QORDER, JSEQ = {}, [], [0]
POLL = {"until": 0.0}          # the local kenv counts as "listening" until this time
EVENTS, EVSEQ = [], [0]        # finished measurements, handed to the local side (it writes them into .kenv)
TRACK = {}                     # label -> {"start": snapshot, "end": snapshot, "diff": result}
SAMPLES = collections.deque(maxlen=6000)   # (t, ram, vram, cpu, disk) from the background sampler
PEAK = {"ram": 0, "vram": 0, "cpu": 0.0, "disk": 0}
CORE_CACHE = {"t": 0.0, "lock": None}


def iso_utc(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def install_module():
    """Make `import kenv` work in notebook cells, scripts and `kenv exec`: the module file goes on PYTHONPATH
    (inherited by Jupyter and its kernels) and the agent's address is passed through the environment."""
    try:
        src = base64.b64decode(CFG["module"]).decode() if CFG.get("module") else ""
        if not src:
            return
        os.makedirs(PKG_DIR, exist_ok=True)
        with open(os.path.join(PKG_DIR, "kenv.py"), "w") as f:
            f.write(src)
        old = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = PKG_DIR + (os.pathsep + old if old else "")
        os.environ.update({"KENV_AGENT_PORT": str(AGENT_PORT), "KENV_AGENT_TOKEN": SECRET, "KENV_WORK": WORK,
                           "KENV_BASE": BASE, "KENV_PKG": PKG_DIR})
        lazy = bool(CFG.get("lazy") and CFG.get("lazy_src"))
        if lazy:  # Phase 6: lazy local access - the patch module, the switch file and the config path
            with open(os.path.join(PKG_DIR, "kenv_lazy.py"), "w") as f:
                f.write(base64.b64decode(CFG["lazy_src"]).decode())
            open(os.path.join(PKG_DIR, "lazy.enable"), "w").close()
            os.environ["KENV_LAZY_CONF"] = LAZY_CONF
        try:  # belt and braces for kernels started with a clean environment
            import site
            for sp in site.getsitepackages():
                try:
                    with open(os.path.join(sp, "kenv_pkg.pth"), "w") as f:
                        f.write(PKG_DIR + "\n")
                        if lazy:  # every new Python process on the kernel gets the file patch
                            f.write("import sys; exec(\"try:\\n import kenv_lazy; kenv_lazy.autoinstall()\\nexcept Exception: pass\")\n")
                    break
                except OSError:
                    continue
        except Exception:
            pass
    except Exception as e:
        log("could not install the kenv module:", e)


def sampler():
    has_gpu = CFG.get("gpu", "none") != "none"
    while not STATE["stop"]:
        try:
            mem = cgroup_mem() or meminfo()
            cpu = cpu_pct()
            vram = int(sum(g["used"] for g in gpus()) * 1048576) if has_gpu else 0
            disk = shutil.disk_usage(WORK).used
            SAMPLES.append((time.time(), mem[1], vram, cpu, disk))
            PEAK["ram"] = max(PEAK["ram"], mem[1])
            PEAK["vram"] = max(PEAK["vram"], vram)
            PEAK["cpu"] = max(PEAK["cpu"], cpu)
            PEAK["disk"] = max(PEAK["disk"], disk)
        except Exception:
            pass
        time.sleep(3.5)


def snap_now():
    mem = cgroup_mem() or meminfo()
    g = gpus()
    try:
        disk = shutil.disk_usage(WORK).used
    except Exception:
        disk = 0
    return {"t": time.time(), "ram": mem[1], "ram_total": mem[0],
            "vram": int(sum(x["used"] for x in g) * 1048576), "vram_total": int(sum(x["total"] for x in g) * 1048576),
            "disk": disk, "cpu": cpu_pct()}


def fmt_bytes(n):
    n = float(n)
    if abs(n) >= 1024 ** 3:
        return "%.2f GB" % (n / 1024 ** 3)
    if abs(n) >= 1024 ** 2:
        return "%.1f MB" % (n / 1024 ** 2)
    return "%.0f KB" % (n / 1024)


def fmt_delta(n):
    return ("+" if n >= 0 else "-") + fmt_bytes(abs(n))


def fmt_secs(s):
    if s < 60:
        return "%.2f s" % s
    m, sec = divmod(int(round(s)), 60)
    h, m = divmod(m, 60)
    return "%dh %02dm %02ds" % (h, m, sec) if h else "%dm %02ds" % (m, sec)


def track_diff(label, a, b, status, error):
    win = [s for s in list(SAMPLES) if a["t"] <= s[0] <= b["t"]]
    return {"label": label, "started": iso_utc(a["t"]), "elapsed": b["t"] - a["t"],
            "ram_start": a["ram"], "ram_end": b["ram"], "ram_peak": max([s[1] for s in win] + [a["ram"], b["ram"]]),
            "vram_start": a["vram"], "vram_end": b["vram"], "vram_peak": max([s[2] for s in win] + [a["vram"], b["vram"]]),
            "vram_total": b["vram_total"], "disk_start": a["disk"], "disk_end": b["disk"],
            "cpu_start": a["cpu"], "cpu_end": b["cpu"],
            "cpu_avg": sum([s[3] for s in win] + [a["cpu"], b["cpu"]]) / float(len(win) + 2),
            "status": status, "error": error}


def fmt_diff(d):
    head = "[kenv] timing '%s'" % d["label"]
    if d.get("status") == "failed":
        head += "   (FAILED: %s)" % (d.get("error") or "error")
    rows = ["  Elapsed : %s" % fmt_secs(d["elapsed"]),
            "  RAM     : %s -> %s  (%s)   peak %s" % (fmt_bytes(d["ram_start"]), fmt_bytes(d["ram_end"]),
                                                     fmt_delta(d["ram_end"] - d["ram_start"]), fmt_bytes(d["ram_peak"]))]
    if d.get("vram_total"):
        rows.append("  VRAM    : %s -> %s  (%s)   peak %s" % (fmt_bytes(d["vram_start"]), fmt_bytes(d["vram_end"]),
                                                         fmt_delta(d["vram_end"] - d["vram_start"]), fmt_bytes(d["vram_peak"])))
    rows.append("  Disk    : %s -> %s  (%s)" % (fmt_bytes(d["disk_start"]), fmt_bytes(d["disk_end"]),
                                              fmt_delta(d["disk_end"] - d["disk_start"])))
    rows.append("  CPU     : %.0f%% -> %.0f%%   avg %.0f%%" % (d["cpu_start"], d["cpu_end"], d["cpu_avg"]))
    return "\n".join([head] + rows)


def do_track(req):
    op, label = req.get("op"), (req.get("label") or "default")
    if op == "start":
        s = snap_now()
        s["t"] = time.time()  # the clock starts after the snapshot, so measuring costs the user nothing
        TRACK[label] = {"start": s}
        return {"text": "[kenv] timer '%s' started" % label}
    tr = TRACK.get(label) or {}
    if op == "end":
        if "start" not in tr:
            return {"error": "No timer '%s' is running. Call kenv.time_start() first." % label}
        if "end" in tr:
            return {"error": "Timer '%s' is already stopped. Read it with kenv.time_output(), or start a new one." % label}
        now = time.time()
        e = snap_now()
        e["t"] = now
        d = track_diff(label, tr["start"], e, "failed" if req.get("status") == "failed" else "ok",
                       str(req.get("error") or "")[:200])
        tr["end"], tr["diff"] = e, d
        with CV:
            EVSEQ[0] += 1
            EVENTS.append(dict(d, seq=EVSEQ[0], epoch=int(START)))
            del EVENTS[:-500]
            CV.notify_all()
        return {"text": "[kenv] timer '%s' stopped after %s" % (label, fmt_secs(d["elapsed"])), "result": d}
    if op == "output":
        if "diff" not in tr:
            return {"error": "Nothing measured for '%s' yet: kenv.time_start(), your code, kenv.time_end(), then kenv.time_output()." % label}
        return {"text": fmt_diff(tr["diff"]), "result": tr["diff"]}
    return {"error": "unknown timer operation %r" % (op,)}


METRIC_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./ -]{0,59}$")


def do_metric(req):
    name = str(req.get("name") or "").strip()
    if not METRIC_NAME.match(name):
        return {"error": "a metric name uses letters, digits, space . _ / - (max 60 characters), got %r" % name}
    val = req.get("value")
    try:
        if isinstance(val, bool):
            raise ValueError
        val = val if isinstance(val, int) else float(val)
        if val != val or val in (float("inf"), float("-inf")):
            raise ValueError
    except (TypeError, ValueError):
        return {"error": "a metric value must be a finite number, got %r" % (req.get("value"),)}
    with CV:
        EVSEQ[0] += 1
        EVENTS.append({"kind": "metric", "name": name, "value": val, "started": iso_utc(time.time()),
                       "seq": EVSEQ[0], "epoch": int(START)})
        del EVENTS[:-500]
        CV.notify_all()
    add_log("metric", "%s = %s" % (name, val))
    return {"text": "[kenv] metric %s = %s" % (name, val)}


def image_tag():
    for k, v in sorted(os.environ.items()):
        ku = k.upper()
        if ku.startswith("KAGGLE") and ("IMAGE" in ku or "DOCKER" in ku) and v:
            return v
    return "unknown"


def dist_lock():
    if CORE_CACHE["lock"] is not None and time.time() - CORE_CACHE["t"] < 30:
        return CORE_CACHE["lock"]
    import importlib.metadata as md
    lock = {}
    for d in md.distributions():
        try:
            name = d.metadata["Name"]
        except Exception:
            continue
        if name:
            lock.setdefault(name, d.version)
    CORE_CACHE["lock"] = dict(sorted(lock.items(), key=lambda kv: kv[0].lower()))
    CORE_CACHE["t"] = time.time()
    return CORE_CACHE["lock"]


def resolve_modules(mods, lock):
    """import name -> 'distribution==version' for third-party modules (standard library modules are skipped)."""
    import importlib.metadata as md
    import importlib.util
    import sysconfig
    stdlib = sysconfig.get_paths().get("stdlib") or ""
    try:
        pd = md.packages_distributions()
    except Exception:
        pd = {}
    out = {}
    for m in mods:
        if m in sys.builtin_module_names or m in getattr(sys, "stdlib_module_names", ()):
            continue
        dists = sorted(set(pd.get(m) or []))
        if dists:
            out[m] = "%s==%s" % (dists[0], lock.get(dists[0], "?"))
            continue
        try:
            spec = importlib.util.find_spec(m)
        except Exception:
            spec = None
        if spec is None:
            out[m] = "not installed"
            continue
        origin = getattr(spec, "origin", None) or ""
        if origin in ("built-in", "frozen") or (stdlib and origin.startswith(stdlib) and "site-packages" not in origin):
            continue
        out[m] = "unknown distribution"
    return dict(sorted(out.items()))


def coreinfo(mods):
    lock = dist_lock()
    try:
        mounted = sorted(os.listdir("/kaggle/input"))
    except OSError:
        mounted = []
    return {"python": platform.python_version(), "image": image_tag(), "lock": lock,
            "detected": resolve_modules(mods, lock), "mounted": mounted,
            "gpu_models": sorted({g["name"] for g in gpus()}),
            "peaks": {"ram": PEAK["ram"], "vram": int(PEAK["vram"]), "cpu": PEAK["cpu"], "disk": PEAK["disk"]},
            "runtime_s": int(time.time() - START), "gen": CFG["gen"]}


# -- command queue: code on the kernel asks; the local kenv (which can reach your machine) answers

def q_submit(argv, timeout):
    with CV:
        for j in [j for j in QORDER if JOBS[j]["state"] in ("done", "cancelled") and time.time() - JOBS[j]["t"] > 600]:
            QORDER.remove(j)
            JOBS.pop(j, None)
        JSEQ[0] += 1
        jid = "j%d" % JSEQ[0]
        JOBS[jid] = {"id": jid, "argv": [str(x) for x in argv], "timeout": timeout, "state": "pending", "t": time.time()}
        QORDER.append(jid)
        CV.notify_all()
    return jid


def q_poll(wait, ack):
    end = time.time() + wait
    with CV:
        while True:
            POLL["until"] = time.time() + 25  # listening: the local side is (re)polling
            job = next((JOBS[j] for j in QORDER if JOBS[j]["state"] == "pending"), None)
            evs = [e for e in EVENTS if e["seq"] > ack]
            if job or evs or time.time() >= end:
                break
            CV.wait(timeout=max(0.05, min(1.0, end - time.time())))
        out = {"epoch": int(START), "seq": EVSEQ[0], "events": evs, "job": None}
        if job:
            job["state"] = "claimed"
            out["job"] = {"id": job["id"], "argv": job["argv"], "timeout": job["timeout"]}
    return out


def q_result(jid, rc, out):
    with CV:
        j = JOBS.get(jid)
        if j and j["state"] == "claimed":  # a cancelled (timed-out) job's late answer is dropped
            j.update(state="done", rc=rc, out=out[-20000:], t=time.time())
            CV.notify_all()


def q_wait(jid, wait):
    end = time.time() + wait
    with CV:
        while True:
            j = JOBS.get(jid)
            if j is None:
                return {"done": True, "rc": 1, "out": "unknown or expired job"}
            if j["state"] in ("done", "cancelled") or time.time() >= end:
                break
            CV.wait(timeout=max(0.05, min(1.0, end - time.time())))
        res = {"done": j["state"] == "done", "claimed": j["state"] == "claimed",
               "polling": time.time() < POLL["until"], "rc": j.get("rc"), "out": j.get("out")}
        if res["done"]:
            QORDER.remove(jid)
            JOBS.pop(jid, None)
        return res


def q_cancel(jid):
    with CV:
        j = JOBS.get(jid)
        if j and j["state"] in ("pending", "claimed"):
            j.update(state="cancelled", t=time.time())
            CV.notify_all()


def do_lazy_config(req):
    """Phase 6: the local kenv tells us where its read-only file server is. We check we can reach it, then write the
    config the patched Python processes read. Nothing is written when the tunnel cannot be reached."""
    if not CFG.get("lazy"):
        return {"ok": False, "error": "lazy access was not enabled for this session"}
    url, token = str(req.get("url", "")).rstrip("/"), str(req.get("token", ""))
    if not re.match(r"^https://[A-Za-z0-9.-]+(:[0-9]+)?$", url) or not re.match(r"^[0-9a-f]{32,128}$", token):
        return {"ok": False, "error": "bad address or token"}
    conf = {"url": url, "token": token, "root": str(req.get("root", "")), "style": "nt" if req.get("style") == "nt" else "posix",
            "base": BASE, "cache": "/tmp/kenv_lazy_cache"}
    if not conf["root"]:
        return {"ok": False, "error": "no project folder given"}
    err = ""
    for _ in range(25):  # a fresh tunnel's DNS can take a little while
        try:
            r = urllib.request.Request(url + "/ping", headers={"Authorization": "Bearer " + token, "User-Agent": "kenv-lazy",
                                                                "ngrok-skip-browser-warning": "1"})
            urllib.request.urlopen(r, timeout=10).read()
            err = ""
            break
        except Exception as e:
            err = repr(e)[:160]
            time.sleep(2)
    if err:
        return {"ok": True, "reachable": False, "error": err}
    tmp = LAZY_CONF + ".tmp"
    with open(tmp, "w") as f:
        json.dump(conf, f)
    os.chmod(tmp, 0o600)
    os.replace(tmp, LAZY_CONF)
    log("lazy local access configured")
    return {"ok": True, "reachable": True}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _auth(self):
        tok = self.headers.get("Authorization", "")
        if tok.startswith("Bearer "):
            tok = tok[7:]
        if not hmac.compare_digest(tok.encode(), SECRET.encode()):
            self._send(401, b'{"error":"unauthorized"}')
            return False
        touch()
        return True

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def do_GET(self):
        if not self._auth():
            return
        path = self.path.split("?")[0]
        if path == "/ping":
            self._send(200, b'{"ok":true}')
        elif path == "/stats":
            self._send(200, json.dumps(stats()).encode())
        elif path == "/cli/status":
            self._send(200, json.dumps({"polling": time.time() < POLL["until"]}).encode())
        else:
            self._send(404, b"{}")

    def do_POST(self):
        if not self._auth():
            return
        path = self.path.split("?")[0]
        try:
            if path == "/exec":
                self.do_exec()
            elif path == "/zip":
                self.do_zip()
            elif path == "/put":
                self.do_put()
            elif path == "/putchunk":
                self.do_putchunk()
            elif path == "/snapshot":
                self._send(200, json.dumps(snapshot()).encode())
            elif path == "/track":
                self._send(200, json.dumps(do_track(json.loads(self._body() or b"{}"))).encode())
            elif path == "/logs":
                self._send(200, json.dumps(get_logs(json.loads(self._body() or b"{}"))).encode())
            elif path == "/metric":
                self._send(200, json.dumps(do_metric(json.loads(self._body() or b"{}"))).encode())
            elif path == "/log":
                req = json.loads(self._body() or b"{}")
                add_log("error" if req.get("level") in ("error", "critical") else "user", str(req.get("msg", ""))[:4000])
                self._send(200, b'{"ok":true}')
            elif path == "/coreinfo":
                self._send(200, json.dumps(coreinfo(json.loads(self._body() or b"{}").get("modules") or [])).encode())
            elif path.startswith("/cli/"):
                self.do_cli(path)
            elif path == "/lazy/config":
                self._send(200, json.dumps(do_lazy_config(json.loads(self._body() or b"{}"))).encode())
            elif path == "/shutdown":
                STATE["stop"] = True
                self._send(200, b'{"ok":true}')
            else:
                self._send(404, b"{}")
        except OSError:
            pass
        except Exception as e:
            try:
                self._send(500, json.dumps({"error": repr(e)}).encode())
            except Exception:
                pass

    def do_cli(self, path):
        req = json.loads(self._body() or b"{}")
        if path == "/cli/submit":
            out = {"id": q_submit(req.get("argv") or [], float(req.get("timeout") or 120))}
        elif path == "/cli/poll":
            out = q_poll(min(float(req.get("wait", 20)), 30.0), int(req.get("ack", 0)))
        elif path == "/cli/result":
            q_result(req["id"], int(req.get("rc", 1)), str(req.get("out", "")))
            out = {"ok": True}
        elif path == "/cli/wait":
            out = q_wait(req["id"], min(float(req.get("wait", 20)), 30.0))
        elif path == "/cli/cancel":
            q_cancel(req["id"])
            out = {"ok": True}
        elif path == "/cli/events":
            since = int(req.get("since", 0))
            out = {"epoch": int(START), "events": [e for e in EVENTS if e["seq"] > since]}
        else:
            self._send(404, b"{}")
            return
        self._send(200, json.dumps(out).encode())

    def do_exec(self):
        req = json.loads(self._body() or b"{}")
        cwd = resolve(req["cwd"]) if req.get("cwd") else BASE
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with LOCK:
            STATE["busy"] += 1
        rc, p, pend = -1, None, b""
        add_log("exec", "$ " + str(req["cmd"])[:500])
        try:
            p = subprocess.Popen(["bash", "-c", req["cmd"]], cwd=cwd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 env=dict(os.environ, PYTHONUNBUFFERED="1"))
            if req.get("timeout"):
                threading.Timer(float(req["timeout"]), p.kill).start()
            fd = p.stdout.fileno()
            while True:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                pend = feed_log("exec", pend + chunk)
                self.wfile.write(chunk)
                self.wfile.flush()
            rc = p.wait()
            if pend:
                add_log("exec", pend.decode("utf-8", "replace"))
            add_log("exec", "[exit %d]" % rc)
        except OSError:  # client went away
            if p:
                p.kill()
        except Exception as e:
            try:
                self.wfile.write(("[kenv-agent] %r\n" % (e,)).encode())
            except OSError:
                pass
        finally:
            with LOCK:
                STATE["busy"] -= 1
        try:
            self.wfile.write(("\n@@KENV_EXIT:%d\n" % rc).encode())
            self.wfile.flush()
        except OSError:
            pass

    def do_zip(self):
        req = json.loads(self._body() or b"{}")
        buf, missing = io.BytesIO(), []

        def add(z, ap):
            try:
                z.write(ap, arcname(ap))
            except OSError:
                pass

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in req.get("paths", []):
                ap = resolve(p)
                if os.path.isfile(ap):
                    add(z, ap)
                elif os.path.isdir(ap):
                    for root, dirs, files in os.walk(ap):
                        dirs[:] = [d for d in dirs if d not in SKIP]
                        for f in files:
                            add(z, os.path.join(root, f))
                else:
                    missing.append(p)
        self._send(200, buf.getvalue(), "application/zip", {"X-Kenv-Missing": json.dumps(missing)})

    def do_put(self):
        q = parse_qs(urlparse(self.path).query)
        dest = resolve(q.get("dest", [""])[0])
        os.makedirs(dest, exist_ok=True)
        root = os.path.realpath(dest)
        data = self._body()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = z.namelist()
            for m in names:
                t = os.path.realpath(os.path.join(dest, m))
                if t != root and not t.startswith(root + os.sep):
                    raise ValueError("unsafe path in upload: " + m)
            z.extractall(dest)
        self._send(200, json.dumps({"files": len(names), "dest": dest}).encode())

    def do_putchunk(self):
        # one piece of a file too big for a single request: ?path=<rel to the project>&offset=<bytes so far>
        q = parse_qs(urlparse(self.path).query)
        rel, off = q.get("path", [""])[0], int(q.get("offset", ["0"])[0])
        t, root = os.path.realpath(resolve(rel)), os.path.realpath(BASE)
        if not rel or not t.startswith(root + os.sep):
            raise ValueError("unsafe path in upload: " + rel)
        if off > 0 and not os.path.exists(t):
            raise ValueError("chunk received before the start of the file")
        data = self._body()
        try:
            os.makedirs(os.path.dirname(t), exist_ok=True)
            with open(t, "r+b" if off > 0 else "wb") as f:
                f.seek(off)
                f.write(data)
        except OSError as e:  # do_POST swallows OSError (client gone); a disk error must reach the client
            raise ValueError(repr(e))
        self._send(200, json.dumps({"ok": True, "bytes": len(data)}).encode())


CF = "/tmp/cloudflared"


def get_cloudflared():
    if os.path.exists(CF):
        return
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    req = urllib.request.Request(url, headers={"User-Agent": "kenv"})
    with urllib.request.urlopen(req, timeout=180) as r, open(CF, "wb") as f:
        shutil.copyfileobj(r, f)
    os.chmod(CF, 0o755)


def tunnel(port):
    for attempt in range(3):
        p = subprocess.Popen([CF, "tunnel", "--url", "http://127.0.0.1:%d" % port, "--no-autoupdate",
                              "--protocol", "http2"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        PROCS.append(p)
        found = []

        def pump(p=p, found=found):
            for ln in p.stdout:  # keep draining so the pipe never fills
                m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", ln)
                if m and not found:
                    found.append(m.group(0))

        threading.Thread(target=pump, daemon=True).start()
        t0 = time.time()
        while time.time() - t0 < 60 and not found and p.poll() is None:
            time.sleep(0.5)
        if found:
            return found[0]
        p.kill()
    raise RuntimeError("could not open a tunnel for port %d" % port)


def start_jupyter():
    try:
        import jupyter_server  # noqa
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jupyter_server"], check=False)
    args = ["--no-browser", "--ip=127.0.0.1", "--port=%d" % JUP_PORT, "--allow-root",
            "--ServerApp.token=" + SECRET, "--ServerApp.allow_origin=*",
            "--ServerApp.allow_remote_access=True", "--ServerApp.disable_check_xsrf=True",
            "--ServerApp.trust_xheaders=True", "--ServerApp.root_dir=" + BASE]
    code = "from jupyter_server.serverapp import main; main()"
    jp = subprocess.Popen([sys.executable, "-c", code] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1)
    PROCS.append(jp)

    def pump():
        try:
            with open("/tmp/jupyter.log", "w") as f:
                for ln in jp.stdout:  # keep draining so the pipe never fills
                    f.write(ln)
                    f.flush()
                    if not ACCESS_RE.search(ln):  # successful requests are noise
                        add_log("jupyter", ln.rstrip("\n"))
        except Exception:
            pass

    threading.Thread(target=pump, daemon=True).start()


def publish(msg):
    body = json.dumps(msg).encode()
    for _ in range(5):
        try:
            req = urllib.request.Request(RELAY + "/" + TOPIC, data=body, headers={"User-Agent": "kenv"})
            urllib.request.urlopen(req, timeout=20).read()
            return True
        except Exception as e:
            log("relay publish failed:", e)
            time.sleep(3)
    return False


def main():
    log("boot", CFG["ref"], "gen", CFG["gen"])
    os.makedirs(BASE, exist_ok=True)
    os.chdir(BASE)  # Jupyter kernels and every child process start in the project folder
    install_module()  # before Jupyter starts, so its kernels inherit PYTHONPATH and the agent address
    threading.Thread(target=sampler, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", AGENT_PORT), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    get_cloudflared()
    start_jupyter()
    agent_url = tunnel(AGENT_PORT)
    jup_url = tunnel(JUP_PORT)
    publish({"gen": CFG["gen"], "ref": CFG["ref"], "gpu": CFG["gpu"], "idle_min": CFG["idle_min"],
             "agent_url": agent_url, "jupyter_url": jup_url + "/?token=" + SECRET, "t": int(time.time())})
    log("online")
    touch()
    while True:
        time.sleep(5)
        if STATE["stop"]:
            log("shutdown requested")
            break
        if STATE["busy"] == 0 and time.time() - STATE["last"] > CFG["idle_min"] * 60:
            log("idle timeout - stopping")
            break
        if time.time() - START > CFG["max_hours"] * 3600:
            log("max lifetime reached")
            break
    time.sleep(1)
    for p in PROCS:
        try:
            p.kill()
        except Exception:
            pass


if __name__ == "__main__":
    main()
'''


KENV_MODULE_SRC = r'''"""kenv - the kernel side of a kenv session. The kenv agent installs this file, so `import kenv` works in
notebook cells, scripts and `kenv exec` on the kernel.

    import kenv
    kenv.time_start()
    model.fit(X, y)
    kenv.time_end()
    kenv.time_output()                  # elapsed time and the change in RAM / VRAM / disk / CPU
    kenv.cli("kenv v2 -r model-champ")  # any kenv command; commands that touch your files run on YOUR machine
    kenv.cli("kenv.time_start")         # the dotted form works too
    kenv.metric("auc", 0.93)            # a result `kenv diff` can compare; stored with the version
    kenv.log("epoch 3 done")            # a line in .kenv/<version>/logs/run.log   (kenv.log("...", "error") -> errors.log too)

Commands that need your local files or your local .kenv folder cannot run here (the kernel cannot reach your
machine), so they go through a queue: the kenv window that started this session runs them and sends the
result back. If that window is not listening, kenv.cli raises an error instead of hanging.
"""
import json
import os
import shlex
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

__kenv_kernel__ = True  # lets `from kenv_shim import kenv` tell this module from any other module called kenv
__all__ = ["cli", "time_start", "time_end", "time_output", "timed", "status", "kaggle_root", "metric", "log", "KenvError",
           "clip", "lazy_path", "prefetch", "lazy_status"]

_KERNEL_CMDS = ("time_start", "time_end", "time_output", "status", "metric")
_NO_POLLER = ("No local kenv is listening for commands from code. They are run by the terminal that started this "
              "session (`kenv init`), so keep that window open. A window attached with `kenv -id` does not run them.")
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # the agent is on localhost: never via a proxy


class KenvError(RuntimeError):
    def __init__(self, msg, rc=None):
        RuntimeError.__init__(self, msg)
        self.rc = rc


def _call(path, payload=None, timeout=30):
    port, tok = os.environ.get("KENV_AGENT_PORT"), os.environ.get("KENV_AGENT_TOKEN")
    if not port or not tok:
        raise KenvError("kenv only works inside a kenv session (start one with `kenv init`).")
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request("http://127.0.0.1:%s%s" % (port, path), data=data,
                                 headers={"Authorization": "Bearer " + tok, "Content-Type": "application/json"})
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise KenvError("the kenv agent answered with an error (%s)" % e.code)
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise KenvError("cannot reach the kenv agent on this kernel (%s)" % e)


# ---- time and status tracking (runs here, on the kernel)

def _track(op, label="default", **extra):
    r = _call("/track", dict(extra, op=op, label=label or "default"), timeout=60)
    if r.get("error"):
        raise KenvError(r["error"])
    return r["text"], r.get("result")


def time_start(label="default"):
    """Record a snapshot (time, RAM, VRAM, disk, CPU). Pair it with time_end()."""
    print(_track("start", label)[0])


def time_end(label="default"):
    """Record the second snapshot. Returns the measurement as a dict."""
    text, res = _track("end", label)
    print(text)
    return res


def time_output(label="default"):
    """Print the difference between the two snapshots. Returns the measurement as a dict."""
    text, res = _track("output", label)
    print(text)
    return res


@contextmanager
def timed(label="default"):
    """with kenv.timed("fit"): ...   start, end and output in one block (a failing block is recorded as failed)."""
    time_start(label)
    try:
        yield
    except BaseException as e:
        try:
            print(_track("end", label, status="failed", error="%s: %s" % (type(e).__name__, str(e)[:160]))[0])
            print(_track("output", label)[0])
        except KenvError:
            pass
        raise
    time_end(label)
    time_output(label)


def _num(s):
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return s
    try:
        return int(s)
    except (TypeError, ValueError):
        try:
            return float(s)
        except (TypeError, ValueError):
            raise KenvError("a metric value must be a number, got %r" % (s,))


def _metric_call(name, value):
    r = _call("/metric", {"name": name, "value": value}, timeout=30)
    if r.get("error"):
        raise KenvError(r["error"])
    return r["text"]


def metric(name, value):
    """Record a result (kenv.metric("auc", 0.93)). It is stored in the version's run history and
    `kenv diff v2 v3` compares it. Returns the value."""
    print(_metric_call(name, _num(value)))
    return value


def log(message, level="info"):
    """Write a line to the session log (.kenv/<version>/logs/run.log); level "error" also lands in errors.log."""
    _call("/log", {"msg": str(message), "level": level}, timeout=30)


def _gb(n):
    return "%.1f GB" % (n / 1024.0 ** 3)


def _bar(pct, width=20):
    pct = max(0, min(100, pct))
    full = int(round(width * pct / 100.0))
    return "#" * full + "." * (width - full)


def _status_text():
    s = _call("/stats", timeout=30)
    cpu, mem = s["cpu"], s["mem"]
    mp = 100.0 * mem["used"] / max(mem["total"], 1)
    rows = ["  Kernel  : up %d min" % (s["uptime"] // 60),
            "  CPU     : %s %.0f%%   %d cores   load %s" % (_bar(cpu["pct"]), cpu["pct"], cpu["cores"],
                                                          " ".join("%.2f" % x for x in cpu["load"])),
            "  RAM     : %s %.0f%%   %s / %s" % (_bar(mp), mp, _gb(mem["used"]), _gb(mem["total"]))]
    for d in s["disk"]:
        dp = 100.0 * d["used"] / max(d["total"], 1)
        rows.append("  Disk    : %s %.0f%%   %s / %s   %s" % (_bar(dp), dp, _gb(d["used"]), _gb(d["total"]), d["path"]))
    for i, g in enumerate(s["gpus"]):
        gp = 100.0 * g["used"] / max(g["total"], 1)
        rows.append("  GPU %d   : %s %.0f%%   %s   %.0f/%.0f MiB   util %.0f%%   %.0f C"
                    % (i, _bar(gp), gp, g["name"], g["used"], g["total"], g["util"], g["temp"]))
    if not s["gpus"]:
        rows.append("  GPU     : none")
    return "\n".join(rows)


def status():
    """CPU / RAM / disk / GPU usage of this kernel."""
    print(_status_text())


def kaggle_root(chdir=False):
    """Kaggle's real root (/kaggle/working). Files written under your project folder sync back to your machine
    on their own; use this for explicit saves to Kaggle's own output folder. kaggle_root(chdir=True) switches there."""
    root = os.environ.get("KENV_WORK") or "/kaggle/working"
    if chdir:
        os.chdir(root)
    return root


# ---- Phase 6: kenv.clip() and lazy local access

def clip():
    """Put `kenv.clip()` at the top of a file that has kenv lines: `kenv unclip` (in your terminal) comments them out before
    production / GitHub, and the next kenv session brings them back. Here it records which lines are kenv commands, in
    .kenv/clip on your machine (only when a kenv window is listening; it never fails and never blocks)."""
    try:
        import sys
        f = sys._getframe(1).f_globals.get("__file__")
        base = os.environ.get("KENV_BASE") or ""
        if f and base:
            f, base = os.path.abspath(f), os.path.abspath(base)
            if f.startswith(base + os.sep) and _call("/cli/status", timeout=5).get("polling"):
                _queue(["clip", os.path.relpath(f, base).replace(os.sep, "/")], 30)
    except Exception:
        pass


def _lazy_mod():
    try:
        import kenv_lazy
        return kenv_lazy
    except ImportError:
        raise KenvError("lazy local access is not on in this session. Start it with: kenv init --lazy-local")


def lazy_path(path):
    """A real file path on the kernel for a file of your local project (fetched now, then cached). For libraries that
    read files in C code (some image/audio/video loaders), which the on-demand patch cannot see."""
    return _lazy_mod().real_path(path)


def prefetch(path=".", workers=4):
    """Download a file or a whole folder of your local project into the kernel's cache now (instead of during training)."""
    n, b = _lazy_mod().prefetch(path, workers)
    print("kenv: fetched %d file(s), %.1f MB" % (n, b / 1048576.0))
    return n, b


def lazy_status():
    """Cache hits / misses / bytes of lazy local access."""
    return _lazy_mod().status()


# ---- kenv.cli: every terminal command, from code

def _argv(command):
    if isinstance(command, (list, tuple)):
        argv = [str(x) for x in command]
    else:
        try:
            argv = shlex.split(str(command))
        except ValueError as e:
            raise KenvError("cannot parse the command: %s" % e)
    if argv and argv[0].lower() == "kenv":
        argv = argv[1:]
    elif argv and argv[0].lower().startswith("kenv."):
        argv = [argv[0][5:]] + argv[1:]
    if not argv or not argv[0]:
        raise KenvError('empty command. Example: kenv.cli("kenv status")')
    return argv


def _cancel(jid):
    try:
        _call("/cli/cancel", {"id": jid}, timeout=10)
    except KenvError:
        pass


def _queue(argv, timeout):
    if not _call("/cli/status", timeout=15).get("polling"):
        raise KenvError(_NO_POLLER)
    jid = _call("/cli/submit", {"argv": argv, "timeout": timeout}, timeout=15)["id"]
    deadline = time.time() + timeout
    while True:
        left = deadline - time.time()
        if left <= 0:
            _cancel(jid)
            raise KenvError("`kenv %s` did not finish within %s s (timeout). Raise it with kenv.cli(..., timeout=600)."
                            % (" ".join(argv), timeout))
        r = _call("/cli/wait", {"id": jid, "wait": min(20.0, left)}, timeout=min(20.0, left) + 20)
        if r.get("done"):
            return int(r.get("rc") or 0), r.get("out") or ""
        if not r.get("claimed") and not r.get("polling"):  # nobody picked it up and nobody is listening any more
            _cancel(jid)
            raise KenvError(_NO_POLLER)


def cli(command, timeout=120, capture=False):
    """Run a kenv command:  kenv.cli("kenv v2 -r model-champ"),  kenv.cli("kenv.time_start"),  kenv.cli(["kenv", "sync"]).
    Prints the output (capture=True returns it as text instead). A failing command raises kenv.KenvError."""
    argv = _argv(command)
    name = argv[0].lower().replace("-", "_")
    if name in _KERNEL_CMDS:  # needs nothing from your machine: runs right here
        if name == "status":
            text = _status_text()
        elif name == "metric":
            if len(argv) < 3:
                raise KenvError("usage: kenv metric <name> <value>")
            text = _metric_call(argv[1], _num(argv[2]))
        else:
            text = _track(name[5:], argv[1] if len(argv) > 1 else "default")[0]
        out = text
    else:
        rc, out = _queue(argv, timeout)
        if rc != 0:
            raise KenvError((out.strip() or "the command failed") + "   [exit %d]" % rc, rc)
    if capture:
        return out
    if out.strip():
        print(out.rstrip("\n"))
'''



# kernel side of lazy local access (Phase 6): kept as its own file, installed by the agent next to kenv.py
LAZY_MODULE_SRC = r'''"""kenv_lazy - kernel side of lazy local access (EXPERIMENTAL, started with `kenv init --lazy-local`).

Paths under your local project folder are fetched on demand from a read-only file server on YOUR machine,
cached on the kernel, and then behave like ordinary files. The kenv agent installs this file; it only patches
Python processes that are notebook kernels or scripts started with `kenv run`.

What is patched: open / io.open / os.open, os.stat / lstat / access, os.listdir / scandir (so os.walk and glob work),
os.getcwd / chdir (getcwd reports your LOCAL path), mkdir / remove / unlink / rmdir / rename / replace, os.path.isabs.
Writes never go to your machine: they land in the kernel's project folder, which kenv syncs back as usual.

Limits:
  * C-level file access (some image, audio and video loaders, memory maps of remote files) bypasses the patch.
    Use kenv.lazy_path("data/img.png") to get a real kernel path for such libraries.
  * A cache miss crosses your home internet connection. Training loops that re-read big files every epoch are slow
    until the cache is warm: kenv.prefetch("data/") warms it up front.
  * The server only serves files inside the project folder; it is read-only and stops with the session.
"""
import builtins
import errno
import http.client
import io
import json
import os
import posixpath
import re
import shutil
import stat as _statm
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CONF = os.environ.get("KENV_LAZY_CONF") or "/tmp/kenv_pkg/lazy.json"
TTL = float(os.environ.get("KENV_LAZY_TTL") or 30.0)   # seconds a remote stat / listing is trusted
CHUNK = 4 * 1024 * 1024

_REAL = dict(open=builtins.open, osopen=os.open, stat=os.stat, lstat=os.lstat, access=os.access,
             listdir=os.listdir, scandir=os.scandir, getcwd=os.getcwd, chdir=os.chdir, mkdir=os.mkdir,
             remove=os.remove, unlink=os.unlink, rmdir=os.rmdir, rename=os.rename, replace=os.replace,
             makedirs=os.makedirs, isabs=posixpath.isabs)   # the original functions, saved before patching
_INSTALLED = []
_TL = threading.local()         # .busy: we are inside our own code - never patch ourselves
_CFG = {"t": 0.0, "mt": None, "d": None}
_STATS = {"hits": 0, "misses": 0, "bytes": 0, "files": 0}
_STAT_C, _LIST_C, _FLOCKS = {}, {}, {}
_GLOCK = threading.Lock()
_WARNED = set()


# ----------------------------------------------------------------------------- configuration

def _cfg():
    """The address / token the local kenv posted to the agent, or None while lazy access is not configured."""
    now = time.time()
    if now - _CFG["t"] < 2.0:
        return _CFG["d"]
    _CFG["t"] = now
    try:
        mt = _REAL["stat"](CONF).st_mtime_ns
        if mt != _CFG["mt"]:
            with _REAL["open"](CONF, "r", encoding="utf-8") as f:
                d = json.load(f)
            root = str(d["root"])
            nt = d.get("style") == "nt"
            d["nt"] = nt
            d["rootn"] = root.replace("\\", "/").rstrip("/") if nt else posixpath.normpath(root)
            d["base"] = posixpath.normpath(str(d["base"]))
            d["cache"] = str(d.get("cache") or "/tmp/kenv_lazy_cache")
            _CFG.update(mt=mt, d=d)
    except OSError:
        _CFG.update(mt=None, d=None)
    except (ValueError, KeyError):
        pass
    return _CFG["d"]


class _Busy(object):
    def __enter__(self):
        self.old = getattr(_TL, "busy", False)
        _TL.busy = True

    def __exit__(self, *a):
        _TL.busy = self.old


def _busy():
    return getattr(_TL, "busy", False)


# ----------------------------------------------------------------------------- path mapping

def _strip(q, root, ci):
    """q under root -> the part after root ('' for root itself), else None."""
    qq = posixpath.normpath(q) if not q.startswith("//") else q
    a, b = (qq.lower(), root.lower()) if ci else (qq, root)
    if a == b:
        return ""
    if a.startswith(b.rstrip("/") + "/"):
        return qq[len(root.rstrip("/")) + 1:]
    return None


def _rel(p):
    """A path as the program wrote it -> path relative to the project folder, or None when it is not ours."""
    c = _cfg()
    if c is None:
        return None
    try:
        p = os.fspath(p)
    except TypeError:
        return None
    if isinstance(p, bytes):
        try:
            p = p.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not p or "\0" in p:
        return None
    if c["nt"]:
        q = p.replace("\\", "/")
        if re.match(r"^[A-Za-z]:", q) or q.startswith("//"):
            return _strip(q, c["rootn"], True)
    elif p.startswith("/"):
        r = _strip(p, c["rootn"], False)
        if r is not None:
            return r
    if p.startswith("/"):
        return _strip(p, c["base"], False)
    try:
        cwd = _REAL["getcwd"]()
    except OSError:
        return None
    return _strip(posixpath.join(cwd, p), c["base"], False)


def _kpath(rel):
    base = _cfg()["base"]
    return posixpath.join(base, rel) if rel else base


def _cpath(rel):
    return posixpath.join(_cfg()["cache"], rel)


# ----------------------------------------------------------------------------- talking to the local file server

class _Remote(Exception):
    def __init__(self, code, msg=""):
        Exception.__init__(self, "%s %s" % (code, msg))
        self.code, self.msg = code, msg


def _open_remote(path, params, headers=None, timeout=60, tries=3):
    c = _cfg()
    if c is None:
        raise OSError(errno.EIO, "kenv lazy: not configured")
    url = c["url"].rstrip("/") + path + "?" + urllib.parse.urlencode(params)
    hdr = {"Authorization": "Bearer " + c["token"], "User-Agent": "kenv-lazy", "ngrok-skip-browser-warning": "1"}
    hdr.update(headers or {})
    last = None
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403, 404, 416):
                raise _Remote(e.code, e.reason)
            last = e
        except (urllib.error.URLError, OSError) as e:
            last = e
        time.sleep(1.5 * (i + 1))
    raise OSError(errno.EIO, "kenv lazy: cannot reach your machine (%s). Is the kenv window still open?" % (last,))


def _remote_json(path, rel):
    with _Busy():
        with _open_remote(path, {"p": rel}) as r:
            return json.loads(r.read().decode("utf-8"))


def _rstat(rel, force=False):
    """{'type': 'file'|'dir'|'none', size, mtime_ns} for a project path, cached for TTL seconds."""
    now = time.time()
    e = _STAT_C.get(rel)
    if e and not force and now - e[0] < TTL:
        return e[1]
    try:
        d = _remote_json("/stat", rel)
    except _Remote as x:
        if x.code == 404:
            d = {"type": "none"}
        elif x.code in (401, 403):
            raise PermissionError(errno.EACCES, "kenv lazy: your machine refused this path (%s)" % rel)
        else:
            raise OSError(errno.EINVAL, "kenv lazy: bad path (%s)" % rel)
    if len(_STAT_C) > 50000:
        _STAT_C.clear()
    _STAT_C[rel] = (now, d)
    return d


def _rlist(rel):
    """[(name, type, size, mtime_ns)] of a remote directory, or None when it is not a directory there."""
    now = time.time()
    e = _LIST_C.get(rel)
    if e and now - e[0] < TTL:
        return e[1]
    try:
        d = _remote_json("/list", rel)
        ents = [tuple(x) for x in d.get("entries", [])]
    except _Remote as x:
        if x.code == 404:
            ents = None
        elif x.code in (401, 403):
            raise PermissionError(errno.EACCES, "kenv lazy: your machine refused this path (%s)" % rel)
        else:
            raise OSError(errno.EINVAL, "kenv lazy: bad path (%s)" % rel)
    if len(_LIST_C) > 5000:
        _LIST_C.clear()
    _LIST_C[rel] = (now, ents)
    return ents


# ----------------------------------------------------------------------------- the cache

def _warn(msg):
    try:
        sys.__stderr__.write("[kenv lazy] " + msg + "\n")
        sys.__stderr__.flush()
    except Exception:
        pass


def _human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return "%d %s" % (n, u) if u == "B" else "%.1f %s" % (n, u)
        n /= 1024.0


def _real_stat(p):
    try:
        return _REAL["stat"](p)
    except OSError:
        return None


def _read_meta(cp):
    try:
        with _REAL["open"](cp + ".kmeta", "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _flock(rel):
    with _GLOCK:
        return _FLOCKS.setdefault(rel, threading.Lock())


def _rm(p):
    try:
        _REAL["remove"](p)
    except OSError:
        pass


def _ensure_cached(rel, d):
    """Make sure the cache holds the current copy of a remote file -> its kernel path."""
    cp = _cpath(rel)
    with _flock(rel):
        with _Busy():
            meta = _read_meta(cp)
            st = _real_stat(cp)
            if (meta and st and meta.get("size") == d.get("size") and meta.get("mtime_ns") == d.get("mtime_ns")
                    and st.st_size == d.get("size")):
                _STATS["hits"] += 1
                return cp
            _STATS["misses"] += 1
            size = int(d.get("size") or 0)
            cdir = os.path.dirname(cp)
            _REAL["makedirs"](cdir, exist_ok=True)
            try:
                free = shutil.disk_usage(cdir).free
            except OSError:
                free = None
            if free is not None and size + 256 * 1024 * 1024 > free:
                raise OSError(errno.ENOSPC, "kenv lazy: not enough disk on the kernel for %s (%s needed, %s free)"
                              % (rel, _human(size), _human(free)))
            if size > 64 * 1024 * 1024:
                _warn("fetching %s (%s) from your machine ..." % (rel, _human(size)))
            part = "%s.part-%d-%d" % (cp, os.getpid(), threading.get_ident())
            _rm(part)
            info, last = None, None
            for attempt in range(5):
                ps = _real_stat(part)
                have = ps.st_size if ps else 0
                if info is not None and have == info["size"]:
                    break
                hdr = {"Range": "bytes=%d-" % have} if have else {}
                try:
                    r = _open_remote("/file", {"p": rel}, headers=hdr, timeout=120)
                except _Remote as x:
                    if x.code == 416:  # our partial copy does not fit any more: start over
                        _rm(part)
                        info = None
                        continue
                    if x.code == 404:
                        raise FileNotFoundError(errno.ENOENT, "No such file or directory", rel)
                    raise PermissionError(errno.EACCES, "kenv lazy: your machine refused %s" % rel)
                try:
                    with r:
                        h = r.headers
                        cur = {"size": int(h.get("X-Kenv-Size") or size), "mtime_ns": int(h.get("X-Kenv-Mtime-Ns") or 0)}
                        if info is not None and cur != info:  # the local file changed while we were downloading
                            _rm(part)
                            info = None
                            continue
                        info = cur
                        resumed = have > 0 and getattr(r, "status", 200) == 206
                        with _REAL["open"](part, "ab" if resumed else "wb") as f:
                            shutil.copyfileobj(r, f, CHUNK)
                except (OSError, http.client.HTTPException) as e:  # connection dropped: resume from what we have
                    last = e
                    time.sleep(1.0)
            ps = _real_stat(part)
            if info is None or not ps or ps.st_size != info["size"]:
                _rm(part)
                raise OSError(errno.EIO, "kenv lazy: could not download %s completely (%s)" % (rel, last))
            _REAL["replace"](part, cp)
            with _REAL["open"](cp + ".kmeta", "w") as f:
                json.dump(info, f)
            _STATS["bytes"] += info["size"]
            _STATS["files"] += 1
            _STAT_C[rel] = (time.time(), {"type": "file", "size": info["size"], "mtime_ns": info["mtime_ns"]})
            return cp


def _locate(rel, fetch=True):
    """-> (kind, path, info): 'k' a file/dir in the kernel's project folder, 'r' a remote file (path = cached copy
    when fetch is True), 'd' a remote directory, (None, None, None) when nothing is there."""
    kp = _kpath(rel)
    st = _real_stat(kp)
    if st is not None:
        return "k", kp, st
    d = _rstat(rel)
    if d.get("type") == "file":
        return "r", (_ensure_cached(rel, d) if fetch else None), d
    if d.get("type") == "dir":
        return "d", None, d
    return None, None, None


# ----------------------------------------------------------------------------- patched functions

def _p_open(file, mode="r", *args, **kw):
    if _busy() or isinstance(file, int):
        return _REAL["open"](file, mode, *args, **kw)
    rel = _rel(file)
    if rel is None:
        return _REAL["open"](file, mode, *args, **kw)
    m = mode if isinstance(mode, str) else "r"
    kp = _kpath(rel)
    if any(ch in m for ch in "wxa+"):
        if ("a" in m or "+" in m) and "w" not in m and "x" not in m and _real_stat(kp) is None:
            d = _rstat(rel)
            if d.get("type") == "file":  # appending to / updating a file that only exists on your machine
                src = _ensure_cached(rel, d)
                with _Busy():
                    _REAL["makedirs"](os.path.dirname(kp), exist_ok=True)
                    shutil.copyfile(src, kp)
        with _Busy():
            _REAL["makedirs"](os.path.dirname(kp), exist_ok=True)
        return _REAL["open"](kp, mode, *args, **kw)
    kind, path, _ = _locate(rel)
    if kind == "d":
        raise IsADirectoryError(errno.EISDIR, "Is a directory", str(file))
    if kind is None:
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", str(file))
    return _REAL["open"](path, mode, *args, **kw)


def _p_osopen(path, flags, *args, **kw):
    if _busy() or isinstance(path, int):
        return _REAL["osopen"](path, flags, *args, **kw)
    rel = _rel(path)
    if rel is None:
        return _REAL["osopen"](path, flags, *args, **kw)
    writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC)
    if writing:
        kp = _kpath(rel)
        with _Busy():
            _REAL["makedirs"](os.path.dirname(kp), exist_ok=True)
        return _REAL["osopen"](kp, flags, *args, **kw)
    kind, p, _ = _locate(rel)
    if kind is None or kind == "d":
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", str(path))
    return _REAL["osopen"](p, flags, *args, **kw)


def _synth(rel, d):
    t = int(d.get("mtime_ns") or 0) // 1000000000
    if d.get("type") == "dir":
        mode, size = _statm.S_IFDIR | 0o755, 4096
    else:
        mode, size = _statm.S_IFREG | 0o444, int(d.get("size") or 0)
    return os.stat_result((mode, hash(rel) & 0x7FFFFFFF, 0, 1, 0, 0, size, t, t, t))


def _stat_impl(path, real_name, *a, **kw):
    real = _REAL[real_name]
    if _busy() or isinstance(path, int) or kw.get("dir_fd") is not None:
        return real(path, *a, **kw)
    rel = _rel(path)
    if rel is None:
        return real(path, *a, **kw)
    try:
        return real(_kpath(rel), *a, **kw)
    except FileNotFoundError:
        pass
    d = _rstat(rel)
    if d.get("type") in ("file", "dir"):
        return _synth(rel, d)
    raise FileNotFoundError(errno.ENOENT, "No such file or directory", str(path))


def _p_stat(path, *a, **kw):
    return _stat_impl(path, "stat", *a, **kw)


def _p_lstat(path, *a, **kw):
    return _stat_impl(path, "lstat", *a, **kw)


def _p_access(path, mode, *a, **kw):
    if _busy() or isinstance(path, int):
        return _REAL["access"](path, mode, *a, **kw)
    rel = _rel(path)
    if rel is None:
        return _REAL["access"](path, mode, *a, **kw)
    kp = _kpath(rel)
    if _real_stat(kp) is not None:
        return _REAL["access"](kp, mode, *a, **kw)
    try:
        d = _rstat(rel)
    except OSError:
        return False
    if d.get("type") not in ("file", "dir"):
        return False
    return not (mode & (os.W_OK | os.X_OK))


def _names(rel, path_for_error):
    """Merged directory content: the kernel's folder + the remote one -> {name: (type, size, mtime_ns, kernel_path|None)}."""
    out, found = {}, False
    kp = _kpath(rel)
    st = _real_stat(kp)
    if st is not None:
        if not _statm.S_ISDIR(st.st_mode):
            raise NotADirectoryError(errno.ENOTDIR, "Not a directory", str(path_for_error))
        found = True
        with _Busy():
            for e in _REAL["scandir"](kp):
                try:
                    is_dir = e.is_dir()
                    es = e.stat()
                    out[e.name] = ("dir" if is_dir else "file", es.st_size, es.st_mtime_ns, e.path)
                except OSError:
                    out[e.name] = ("file", 0, 0, e.path)
    try:
        ents = _rlist(rel)
    except OSError:
        if not found:
            raise
        ents = None
    if ents is not None:
        found = True
        for name, typ, size, mt in ents:
            out.setdefault(name, (typ, size, mt, None))
    if not found:
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", str(path_for_error))
    return out


def _p_listdir(path=None):
    if _busy() or isinstance(path, int):
        return _REAL["listdir"](path) if path is not None else _REAL["listdir"]()
    rel = _rel("." if path is None else path)
    if rel is None:
        return _REAL["listdir"](path) if path is not None else _REAL["listdir"]()
    names = sorted(_names(rel, path))
    if isinstance(path, bytes):
        return [n.encode("utf-8") for n in names]
    return names


class _Entry(object):
    def __init__(self, name, path, typ, size, mtime_ns, kp):
        self.name, self.path, self._t, self._size, self._mt, self._kp = name, path, typ, size, mtime_ns, kp

    def is_dir(self, follow_symlinks=True):
        return self._t == "dir"

    def is_file(self, follow_symlinks=True):
        return self._t == "file"

    def is_symlink(self):
        return False

    def stat(self, follow_symlinks=True):
        if self._kp:
            return _REAL["stat"](self._kp)
        return _synth(self.path, {"type": self._t, "size": self._size, "mtime_ns": self._mt})

    def inode(self):
        return hash(self.path) & 0x7FFFFFFF

    def __fspath__(self):
        return self.path

    def __repr__(self):
        return "<DirEntry %r>" % (self.name,)


class _Scan(object):
    def __init__(self, entries):
        self._it = iter(entries)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def close(self):
        self._it = iter(())


def _p_scandir(path=None):
    if _busy() or isinstance(path, int):
        return _REAL["scandir"](path) if path is not None else _REAL["scandir"]()
    rel = _rel("." if path is None else path)
    if rel is None:
        return _REAL["scandir"](path) if path is not None else _REAL["scandir"]()
    shown = "." if path is None else os.fspath(path)
    if isinstance(shown, bytes):
        shown = shown.decode("utf-8", "replace")
    names = _names(rel, path)
    ents = []
    for name in sorted(names):
        typ, size, mt, kp = names[name]
        ents.append(_Entry(name, posixpath.join(shown, name), typ, size, mt, kp))
    return _Scan(ents)


def _p_getcwd():
    real = _REAL["getcwd"]()
    c = _cfg()
    if c is None or _busy():
        return real
    r = _strip(real, c["base"], False)
    if r is None:
        return real
    root = str(c["root"])
    if not r:
        return root
    sep = "\\" if c["nt"] else "/"
    return root.rstrip("\\/") + sep + r.replace("/", sep)


def _p_getcwdb():
    return os.fsencode(_p_getcwd())


def _p_chdir(path):
    if _busy() or isinstance(path, int):
        return _REAL["chdir"](path)
    rel = _rel(path)
    if rel is None:
        return _REAL["chdir"](path)
    kp = _kpath(rel)
    st = _real_stat(kp)
    if st is None:
        d = _rstat(rel)
        if d.get("type") == "dir":
            with _Busy():
                _REAL["makedirs"](kp, exist_ok=True)
        elif d.get("type") == "file":
            raise NotADirectoryError(errno.ENOTDIR, "Not a directory", str(path))
        else:
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", str(path))
    return _REAL["chdir"](kp)


def _wrap_path(name):
    real_name = name

    def f(path, *a, **kw):
        real = _REAL[real_name]
        if _busy() or isinstance(path, int):
            return real(path, *a, **kw)
        rel = _rel(path)
        if rel is None:
            return real(path, *a, **kw)
        return real(_kpath(rel), *a, **kw)
    f.__name__ = name
    return f


def _wrap_two(name):
    def f(src, dst, *a, **kw):
        real = _REAL[name]
        if _busy() or isinstance(src, int) or isinstance(dst, int):
            return real(src, dst, *a, **kw)
        rs, rd = _rel(src), _rel(dst)
        return real(_kpath(rs) if rs is not None else src, _kpath(rd) if rd is not None else dst, *a, **kw)
    f.__name__ = name
    return f


def _p_isabs(s):
    if _REAL["isabs"](s):
        return True
    c = _cfg()
    if c is not None and c["nt"]:
        try:
            q = os.fspath(s)
        except TypeError:
            return False
        if isinstance(q, str):
            return bool(re.match(r"^[A-Za-z]:[\\/]", q))
    return False


# ----------------------------------------------------------------------------- public helpers (also exposed as kenv.*)

def real_path(path):
    """A real kernel path for a project path (fetches the file first). Use it for libraries that open files in C code."""
    rel = _rel(path)
    if rel is None:
        return os.fspath(path)
    kind, p, _ = _locate(rel)
    if kind in ("k", "r") and p:
        return p
    if kind is None:
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", str(path))
    return _kpath(rel)


def prefetch(path=".", workers=4):
    """Download a file, or every file under a folder, into the cache now. -> (files, bytes) fetched."""
    rel = _rel(path)
    if rel is None:
        raise ValueError("%r is not inside the project folder" % (path,))
    todo = []

    def walk(r):
        ents = _rlist(r)
        if ents is None:
            d = _rstat(r)
            if d.get("type") == "file":
                todo.append((r, d))
            return
        for name, typ, size, mt in ents:
            sub = (r + "/" + name) if r else name
            if typ == "dir":
                walk(sub)
            else:
                todo.append((sub, {"type": "file", "size": size, "mtime_ns": mt}))

    walk(rel)
    before = (_STATS["files"], _STATS["bytes"])
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        for f in [ex.submit(_ensure_cached, r, d) for r, d in todo if _real_stat(_kpath(r)) is None]:
            f.result()
    return _STATS["files"] - before[0], _STATS["bytes"] - before[1]


def status():
    c = _cfg()
    used = 0
    if c:
        for root, _, files in os.walk(c["cache"]):
            for n in files:
                try:
                    used += os.path.getsize(os.path.join(root, n))
                except OSError:
                    pass
    return {"configured": bool(c), "local_root": (c or {}).get("root"), "cache_dir": (c or {}).get("cache"),
            "cache_bytes": used, **_STATS}


# ----------------------------------------------------------------------------- install

def install():
    """Patch this process (idempotent)."""
    if _INSTALLED:
        return
    _INSTALLED.append(True)
    builtins.open = _p_open
    io.open = _p_open
    os.open = _p_osopen
    os.stat = _p_stat
    os.lstat = _p_lstat
    os.access = _p_access
    os.listdir = _p_listdir
    os.scandir = _p_scandir
    os.getcwd = _p_getcwd
    os.getcwdb = _p_getcwdb
    os.chdir = _p_chdir
    for n in ("mkdir", "remove", "unlink", "rmdir"):
        setattr(os, n, _wrap_path(n))
    for n in ("rename", "replace"):
        setattr(os, n, _wrap_two(n))
    posixpath.isabs = _p_isabs
    try:  # pathlib of Python <= 3.11 keeps its own references to the os functions
        import pathlib
        acc = getattr(pathlib, "_NormalAccessor", None)
        if acc is not None:
            for nm, fn in (("stat", _p_stat), ("lstat", _p_lstat), ("open", _p_open), ("listdir", _p_listdir),
                           ("scandir", _p_scandir)):
                if hasattr(acc, nm):
                    setattr(acc, nm, staticmethod(fn))
    except Exception:
        pass


def autoinstall():
    """Called from the kenv .pth file in every new Python process of a lazy session: patch when the switch file exists."""
    try:
        if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lazy.enable")):
            install()
    except Exception:
        pass
'''


# ----------------------------------------------------------------------------- launching kernels

def ref_for(user, name, gen):
    return f"{user}/kenv-{name}" + (f"-g{gen}" if gen > 1 else "")


def push_kernel(ref, cfg, gpu_id, datasets=None):
    slug = ref.split("/", 1)[1]
    d = Path(tempfile.mkdtemp(prefix="kenv_"))
    try:
        src = AGENT_SRC.replace("__CFG__", base64.b64encode(json.dumps(cfg).encode()).decode())
        (d / "run.py").write_text(src)
        meta = {"id": ref, "title": slug, "code_file": "run.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": bool(gpu_id),
                "enable_tpu": False, "enable_internet": True, "dataset_sources": [],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
        args = ["kernels", "push", "-p", str(d)]
        if gpu_id:
            meta["machine_shape"] = gpu_id
            args += ["--accelerator", gpu_id]

        def attempt(sources):
            meta["dataset_sources"] = list(sources)
            (d / "kernel-metadata.json").write_text(json.dumps(meta))
            p = cli(*args)
            out = p.stdout + p.stderr
            if p.returncode != 0 and gpu_id and "--accelerator" in out:  # older CLI: metadata alone is enough
                p = cli("kernels", "push", "-p", str(d))
                out = p.stdout + p.stderr
            return p, out

        p, out = attempt(datasets or [])
        failed = p.returncode != 0 or "error" in out.lower()
        if failed and datasets and "dataset" in out.lower():  # e.g. a dataset was deleted or is still processing
            say("[kenv] Could not attach the project's Kaggle dataset(s) - starting without them:\n  "
                + out.strip()[:300].replace("\n", "\n  "))
            p, out = attempt([])
            failed = p.returncode != 0 or "error" in out.lower()
        if failed:
            raise KenvError(f"kaggle kernels push failed:\n{out.strip()}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def show_kernel_log(ref):
    tmp = Path(tempfile.mkdtemp(prefix="kenv_log_"))
    try:
        cli("kernels", "output", ref, "-p", str(tmp), timeout=60)
        for f in tmp.glob("*.log"):
            say("---- kernel log (tail) ----")
            say(f.read_text(errors="replace")[-1500:])
    except Exception:
        pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def wait_ready(ep, ref, gen, timeout, poll=6):
    t0, n = time.time(), 0
    while True:
        m = ep.latest()
        if m and m.get("gen", 0) >= gen:
            ep.info = m
            try:
                ep.json_call("/ping", timeout=15)  # a fresh tunnel's DNS can lag a little
                return m
            except KenvError:
                pass
        n += 1
        if n % 2 == 0:
            p = cli("kernels", "status", ref)
            out = (p.stdout + p.stderr).lower()
            if p.returncode == 0 and any(w in out for w in ("error", "cancel", "complete")):
                show_kernel_log(ref)
                raise KenvError(f"The kernel stopped before it came online: {out.strip()}\n"
                                "Is internet enabled for your Kaggle account (phone-verified)?")
        if time.time() - t0 > timeout:
            raise KenvError(f"Kernel did not come online within {timeout // 60} minutes.")
        if n % 5 == 0:
            say(f"[kenv] still starting ... {int(time.time() - t0)}s")
        time.sleep(poll)


def launch(user, name, secret, gen, gpu_id, idle_min, ep, startup, kproj=None, datasets=None, lazy=False):
    ref = ref_for(user, name, gen)
    cfg = {"secret": secret, "topic": ep.topic, "relay": RELAY, "gen": gen, "ref": ref,
           "gpu": gpu_id or "none", "idle_min": idle_min, "max_hours": 8, "work": "/kaggle/working",
           "module": base64.b64encode(KENV_MODULE_SRC.encode()).decode()}
    if kproj:
        cfg["project"] = kproj
    if lazy:  # Phase 6: the kernel gets the file-patch module and waits for the address of your file server
        cfg["lazy"] = True
        cfg["lazy_src"] = base64.b64encode(LAZY_MODULE_SRC.encode()).decode()
    st = load_state(name)
    if st and ref not in st["refs"]:  # recorded BEFORE pushing so an interrupt mid-push still gets cleaned up
        st["refs"].append(ref)
        save_state(st)
    say(f"[kenv] Pushing {ref} ...")
    push_kernel(ref, cfg, gpu_id, datasets)
    say("[kenv] Waiting for the kernel to come online (queue + boot is usually 1-3 min) ...")
    return wait_ready(ep, ref, gen, startup)


def delete_refs(refs, ep=None):
    for ref in dict.fromkeys(refs):
        ok = cli_delete(ref)
        if not ok and ep is not None:  # maybe it refuses while running: ask the agent to stop, retry
            try:
                ep.json_call("/shutdown", payload={}, timeout=8)
                time.sleep(4)
            except Exception:
                pass
            ok = cli_delete(ref)
        if not ok:
            say(f"[kenv] Could not auto-delete {ref}. Remove it: https://www.kaggle.com/code/{ref}  (or run: kenv --sweep)")


def install_handlers(cleanup):
    def _sig(signum, frame):
        raise SystemExit(128 + signum)  # becomes a normal exception, so the context manager runs

    for n in ("SIGHUP", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, n):
            signal.signal(getattr(signal, n), _sig)
    if os.name == "nt":  # closing a Windows console window does not raise a Python signal
        try:
            import ctypes
            proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)

            def handler(ctrl):
                if ctrl in (2, 5, 6):  # CLOSE, LOGOFF, SHUTDOWN
                    cleanup()
                    return 1
                return 0

            install_handlers._keep = proto(handler)
            ctypes.windll.kernel32.SetConsoleCtrlHandler(install_handlers._keep, True)
        except Exception:
            pass


class Owner:
    """
    with Owner("name") as o:
        o.start() ...
    On exit (normal, error, Ctrl-C, terminal close, kill) the kernel and local state are removed.
    """

    def __init__(self, name, gpu_id=None, idle_min=20, startup=900, extra=None, kproj=None, datasets=None, lazy=False):
        self.user = get_username()
        self.name, self.secret = name, secrets.token_hex(16)
        self.gpu_id, self.idle_min, self.startup = gpu_id, idle_min, startup
        self.extra, self.kproj, self.datasets, self.lazy = extra or {}, kproj, datasets or [], lazy
        self.on_end = None  # on_end(exc, reason, end_ts): called first thing in cleanup, before the kernel is deleted
        self.ep = Endpoint(name, self.secret)
        self._done = False

    def __enter__(self):
        require_cli()
        require_relay()
        old = load_state(self.name)
        if old and pid_alive(old.get("pid")):
            raise KenvError(f"A session called '{self.name}' is already running on this machine.")
        save_state({**self.extra, "name": self.name, "secret": self.secret, "user": self.user, "refs": [],
                    "pid": os.getpid(), "created": int(time.time())})
        install_handlers(lambda: self.cleanup(reason="terminal closed"))
        return self

    def start(self):
        launch(self.user, self.name, self.secret, 1, self.gpu_id, self.idle_min, self.ep, self.startup,
               self.kproj, self.datasets, lazy=self.lazy)

    def cleanup(self, exc=None, reason=None):
        if self._done:
            return
        self._done = True
        end_ts = time.time()
        for n in ("SIGINT", "SIGHUP", "SIGTERM", "SIGBREAK"):  # nothing may abort cleanup
            if hasattr(signal, n):
                try:
                    signal.signal(getattr(signal, n), signal.SIG_IGN)
                except (ValueError, OSError):
                    pass
        if self.on_end:  # the end time is written before the (slow) kernel deletion
            try:
                self.on_end(exc, reason, end_ts)
            except Exception:
                pass
        st = load_state(self.name)
        if st:  # no state file means `kenv stop` already deleted everything
            delete_refs(st.get("refs", []), self.ep if st.get("refs") else None)
            remove_state(self.name)

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            say("\n[kenv] Stopping. Cleaning up Kaggle resources (please wait) ...")
        self.cleanup(exc=exc)
        return False  # never swallow exceptions


# ----------------------------------------------------------------------------- interactive helpers

def confirm(q):
    try:
        return input(f"{q} [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        say()
        return False


def pick_gpu(choice=None):
    """-> accelerator id (None = CPU). Prompts unless `choice` (t4 / l4 / none) is given."""
    if choice:
        for key, _, gid in GPUS:
            if choice.lower() in (key, (gid or "").lower()):
                return gid
        raise KenvError("Unknown accelerator '%s'. Options: %s" % (choice, ", ".join(k for k, _, _ in GPUS)))
    say("Pick an accelerator:")
    for i, (_, label, _) in enumerate(GPUS, 1):
        say(f"  {i}) {label}")
    say(c("  Kaggle decides the exact hardware and your account's weekly GPU quota applies.", "2"))
    while True:
        try:
            raw = input(f"Choice [1-{len(GPUS)}, default 1]: ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            say()
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(GPUS):
            return GPUS[int(raw) - 1][2]
        say("  please type one of the numbers above")


def open_shell(name, secret, owner, ep=None, on_beat=None):
    """Subshell like `conda activate`: `kenv ...` commands work inside it."""
    tmp = Path(tempfile.mkdtemp(prefix="kenv_"))
    script = str(Path(__file__).resolve())
    env = {**os.environ, "KENV_SESSION": connect_str(name, secret)}
    stop = threading.Event()
    if ep is not None:
        def beat():
            while not stop.wait(60):
                try:
                    ep.json_call("/ping", timeout=20, tries=1)
                    if on_beat:
                        on_beat()
                except Exception:
                    pass

        threading.Thread(target=beat, daemon=True).start()
    if os.name == "nt":
        (tmp / "kenv.bat").write_text(f'@"{sys.executable}" "{script}" %*\r\n')
        env["PATH"] = str(tmp) + os.pathsep + env.get("PATH", "")
        env["PROMPT"] = f"(kenv:{name}) $P$G"
        cmd = [os.environ.get("COMSPEC", "cmd.exe")]
    else:
        bindir = tmp / "bin"
        bindir.mkdir()
        w = bindir / "kenv"
        w.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        w.chmod(0o755)
        rc = tmp / "bashrc"
        rc.write_text(f'[ -f ~/.bashrc ] && source ~/.bashrc\nexport PATH="{bindir}:$PATH"\n'
                      f'PS1="(kenv:{name}) $PS1"\n')
        cmd = ["bash", "--rcfile", str(rc), "-i"]
    prev = signal.signal(signal.SIGINT, lambda *a: None)  # Ctrl-C belongs to the subshell
    try:
        subprocess.run(cmd, env=env)
    finally:
        signal.signal(signal.SIGINT, prev)
        stop.set()
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------- file transfer

def zip_local(paths):
    buf, n = io.BytesIO(), 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for raw in paths:
            p = Path(raw).expanduser()
            if p.is_file():
                z.write(p, p.name)
                n += 1
            elif p.is_dir():
                base = p.resolve().parent
                for root, dirs, files in os.walk(p):
                    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                    for f in files:
                        fp = (Path(root) / f).resolve()
                        z.write(fp, fp.relative_to(base).as_posix())
                        n += 1
            else:
                raise KenvError(f"Not found: {p}")
    return buf.getvalue(), n


def extract_zip(data, dest, redirect=None):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root, names = dest.resolve(), []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for m in z.infolist():
            if m.is_dir():
                continue
            name = redirect(m.filename) if redirect else m.filename
            t = (root / name).resolve()
            if t != root and root not in t.parents:  # zip-slip guard
                continue
            t.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as s, open(t, "wb") as d:
                shutil.copyfileobj(s, d)
            names.append(name)
    return names


def pull(ep, paths, out):
    with ep.call("/zip", payload={"paths": paths}, timeout=900) as r:
        missing = json.loads(r.headers.get("X-Kenv-Missing") or "[]")
        data = r.read()
    for m in missing:
        say(f"[kenv] not found on the kernel: {m}")
    return extract_zip(data, out)


def stream_exec(ep, cmd, cwd=None):
    """Run a shell command on the kernel, print output live, return its exit code."""
    stream_exec.tail = ""  # last ~4 KB of output, for the run history
    payload = {"cmd": cmd}
    if cwd:
        payload["cwd"] = cwd
    r = ep.call("/exec", payload=payload, timeout=1800)
    out, buf, tail = sys.stdout.buffer, b"", b""
    try:
        while True:
            chunk = r.read1(4096)
            if not chunk:
                break
            buf += chunk
            tail = (tail + chunk)[-4096:]
            if len(buf) > 64:  # hold back the tail: it carries the exit marker
                out.write(buf[:-64])
                out.flush()
                buf = buf[-64:]
    finally:
        r.close()
    stream_exec.tail = tail.decode("utf-8", "replace")
    m = re.search(rb"\n@@KENV_EXIT:(-?\d+)\s*$", buf)
    out.write(buf[:m.start()] if m else buf)
    out.flush()
    if not m:
        say("\n[kenv] connection to the kernel was lost before the command finished")
        return 1
    return int(m.group(1))


# ----------------------------------------------------------------------------- the .kenv core file (TOML)
#
# <project>/.kenv/vN/core.toml   the version's core file: manifest, dependency lock, I/O map, resource
#                                snapshot, run history, secret NAMES. Sessions stay in sessions.json next to it.
# Human-readable and diffable: fixed section order, one fact per line, sorted tables, and `updated` only
# changes when something else in the file did. Names and hashes only - never secret values.

CORE_FILE = "core.toml"
CORE_FORMAT = 1
CORE_ORDER = ["format", "manifest", "dependencies", "io", "resources", "secrets", "sessions", "metrics", "runs"]
CORE_HEADER = ["kenv core file - written by kenv, safe to read and to commit.",
               "Names and hashes only; secret values are never stored here."]
MAX_RUNS = 1000
BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def toml_str(s):
    out = []
    for ch in str(s):
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif o < 0x20 or o == 0x7F:
            out.append("\\u%04x" % o)
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def toml_key(k):
    k = str(k)
    return k if BARE_KEY.match(k) else toml_str(k)


def toml_val(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(round(v, 6)) if v == v and abs(v) != float("inf") else "0.0"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(toml_val(x) for x in v) + "]"
    return toml_str(v)


def toml_dumps(doc, header=()):
    """Tables and scalars only (that is all the core file needs). Insertion order is the file order."""
    lines = ["# " + h for h in header]

    def emit(path, table):
        scalars = [(k, v) for k, v in table.items() if not isinstance(v, dict)]
        if path and scalars:
            if lines:
                lines.append("")
            lines.append("[" + ".".join(toml_key(p) for p in path) + "]")
        for k, v in scalars:
            lines.append(f"{toml_key(k)} = {toml_val(v)}")
        for k, v in table.items():
            if isinstance(v, dict):
                emit(path + [k], v)

    emit([], doc)
    return "\n".join(lines) + "\n"


_BARE_RUN = re.compile(r"[A-Za-z0-9_-]+")


def _toml_key(s, i):
    if s[i] == '"':
        return json.JSONDecoder().raw_decode(s, i)
    m = _BARE_RUN.match(s, i)
    if not m:
        raise ValueError("bad key")
    return m.group(0), m.end()


def _toml_fallback(text):
    """Reads what toml_dumps writes (Python < 3.11 has no tomllib)."""
    root = cur = {}
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s[0] == "#":
            continue
        if s[0] == "[":
            i, path = 1, []
            while True:
                while s[i] == " ":
                    i += 1
                k, i = _toml_key(s, i)
                path.append(k)
                while s[i] == " ":
                    i += 1
                if s[i] == ".":
                    i += 1
                    continue
                break
            cur = root
            for k in path:
                cur = cur.setdefault(k, {})
            continue
        k, i = _toml_key(s, 0)
        rest = s[i:].strip()
        if not rest.startswith("="):
            raise ValueError("expected '='")
        cur[k] = json.loads(rest[1:].strip())
    return root


def toml_loads(text):
    try:
        import tomllib
    except ImportError:
        return _toml_fallback(text)
    return tomllib.loads(text)


@contextlib.contextmanager
def file_lock(path, timeout=8.0):
    """A tiny lock file so the session's background thread and a `kenv run` in the shell never write at once."""
    lock, t0, held = str(path) + ".lock", time.time(), False
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            held = True
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > 30:  # left behind by a killed process
                    os.unlink(lock)
                    continue
            except OSError:
                pass
            if time.time() - t0 > timeout:
                raise KenvError("The core file is busy (another kenv process is writing it). Try again.")
            time.sleep(0.05)
        except OSError:
            break  # cannot lock here; write without
    try:
        yield
    finally:
        if held:
            try:
                os.unlink(lock)
            except OSError:
                pass


def canon(doc):
    out = {"format": CORE_FORMAT}
    for k in CORE_ORDER[1:]:
        if k in doc:
            out[k] = doc[k]
    for k in doc:
        out.setdefault(k, doc[k])
    return out


class CoreStore:
    """Reads and writes .kenv/<version>/core.toml."""

    def __init__(self, proj, v):
        self.proj, self.v = proj, v
        self.path = proj.kdir / v / CORE_FILE

    def load(self):
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as e:
            raise KenvError(f"Cannot read {self.path}: {e}")
        try:
            d = toml_loads(text)
        except Exception as e:
            raise KenvError(f"{self.path} is not valid TOML ({e}). Fix it, or delete it and kenv rebuilds it.")
        return d if isinstance(d, dict) else {}

    @staticmethod
    def _render(doc):
        body = json.loads(json.dumps(doc))
        (body.get("manifest") or {}).pop("updated", None)
        return toml_dumps(canon(body), CORE_HEADER)

    def update(self, fn):
        """load -> fn(doc) -> write. Nothing is written (and `updated` stays) unless something changed."""
        with file_lock(self.path):
            old = self.load()
            new = json.loads(json.dumps(old))
            fn(new)
            if self.path.exists() and self._render(old) == self._render(new):
                return False
            m = new.setdefault("manifest", {})
            m.pop("updated", None)
            m["updated"] = iso()
            data = redact_secrets(toml_dumps(canon(new), CORE_HEADER)).encode("utf-8")  # bytes: no CRLF noise in diffs on Windows
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, self.path)
            return True


# ---- what goes into the core file

def scrub(s):
    """Error text ends up in a file people commit: blank out anything that looks like a credential."""
    s = re.sub(r"kenv://\S+", "kenv://<redacted>", str(s))
    s = re.sub(r"(?i)\bbearer\s+\S+", "Bearer <redacted>", s)
    s = re.sub(r"(?i)\b(token|secret|password|passwd|api[_-]?key|authorization)\b(\s*[:=]\s*)\S+", r"\1\2<redacted>", s)
    return re.sub(r"[A-Za-z0-9_\-]{32,}", "<redacted>", s)


def error_summary(tail):
    lines = [x.strip() for x in str(tail).splitlines() if x.strip()]
    if not lines:
        return "exited with a non-zero status"
    for x in reversed(lines):
        if re.search(r"(Error|Exception|Killed|Out of memory|OOM)", x):
            return scrub(x)[:200]
    return scrub(lines[-1])[:200]


SECRET_CALL = re.compile(r"""get_secret\(\s*["']([^"'\n]{1,80})["']""")
ENV_READ = re.compile(r"""(?:os\.environ\s*\[\s*|os\.environ\.get\(\s*|os\.getenv\(\s*)["']([A-Za-z_][A-Za-z0-9_]{0,60})["']""")
SECRETISH = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)", re.I)


def iter_source_files(proj, limit=3000):
    rules, root, n = proj.rules(), str(proj.root), 0
    for cur, dirs, files in os.walk(root):
        rel_dir = os.path.relpath(cur, root).replace(os.sep, "/")
        rel_dir = "" if rel_dir == "." else rel_dir
        dirs[:] = [d for d in dirs if d != PROJECT_DIR and not os.path.islink(os.path.join(cur, d))
                   and not rules.ignored(f"{rel_dir}/{d}" if rel_dir else d, True)]
        for f in files:
            if not f.endswith((".py", ".ipynb")) or (not rel_dir and f == SHIM_FILE):
                continue  # kenv_shim.py is kenv's own file, not part of your code
            rel = f"{rel_dir}/{f}" if rel_dir else f
            fp = os.path.join(cur, f)
            try:
                if rules.ignored(rel, False) or os.path.getsize(fp) > 2 * 1024 * 1024:
                    continue
            except OSError:
                continue
            yield rel, fp
            n += 1
            if n >= limit:
                return


def code_cells(fp):
    try:
        text = Path(fp).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    cells = [text]
    if fp.endswith(".ipynb"):
        try:
            nb = json.loads(text)
        except ValueError:
            return []
        cells = []
        for cell in nb.get("cells", []):
            if isinstance(cell, dict) and cell.get("cell_type") == "code":
                src = cell.get("source", "")
                cells.append("".join(src) if isinstance(src, list) else str(src))
    return ["\n".join("" if ln.lstrip().startswith(("%", "!")) else ln for ln in c_.splitlines()) for c_ in cells]


def imports_of(code):
    mods = set()
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        for m in re.finditer(r"^\s*(?:from\s+([A-Za-z_]\w*)|import\s+([A-Za-z_]\w*))", code, re.M):
            mods.add(m.group(1) or m.group(2))
        return mods
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def scan_code(proj):
    """-> (imported top-level modules that are not the project's own, secret NAMES the code asks for)."""
    mods, names, local = set(), set(), set()
    for rel, fp in iter_source_files(proj):
        local.add(Path(rel).stem)
        local.add(rel.split("/")[0])
        for code in code_cells(fp):
            mods |= imports_of(code)
            names.update(SECRET_CALL.findall(code))
            names.update(n for n in ENV_READ.findall(code) if SECRETISH.search(n))
    local |= {"kenv", "kenv_shim"}  # provided by the session itself, not a dependency
    return sorted(m for m in mods if m not in local), sorted(names)


def core_manifest(proj, v, old, ref, gpu_id, info):
    old = old or {}
    m = {"version": v, "id": proj.meta(v)["id"]}
    if proj.meta(v).get("name"):
        m["name"] = proj.meta(v)["name"]
    m.update(kernel=ref, accelerator="GPU" if gpu_id else "CPU", accelerator_id=gpu_id or "none")
    models = ", ".join(info.get("gpu_models") or []) if info else old.get("gpu_model", "")
    if gpu_id and models:
        m["gpu_model"] = models
    m["python"] = info["python"] if info else old.get("python", "unknown")
    m["docker_image"] = info["image"] if info else old.get("docker_image", "unknown")  # recorded only: Kaggle controls it
    m["created"] = old.get("created") or iso()
    m["updated"] = old.get("updated") or iso()
    return m


def core_resources(old, info, sid):
    if not info:
        return old or {}
    p = info.get("peaks") or {}
    return {"session": sid, "runtime_min": int(info.get("runtime_s", 0) // 60),
            "ram_peak_gb": round(p.get("ram", 0) / 1024 ** 3, 1), "vram_peak_gb": round(p.get("vram", 0) / 1024 ** 3, 1),
            "cpu_peak_pct": int(p.get("cpu", 0)), "disk_peak_gb": round(p.get("disk", 0) / 1024 ** 3, 1)}


def core_add_runs(doc, runs):
    tbl = dict(doc.get("runs") or {})
    seen = {r.get("evid") for r in tbl.values() if isinstance(r, dict) and r.get("evid")}
    nxt = max([int(k[1:]) for k in tbl if re.fullmatch(r"r\d+", k)] or [0]) + 1
    for r in runs:
        if r.get("evid") and r["evid"] in seen:
            continue
        tbl[f"r{nxt:04d}"] = {k: v for k, v in r.items() if v is not None}
        seen.add(r.get("evid"))
        nxt += 1
    keys = sorted((k for k in tbl if re.fullmatch(r"r\d+", k)), key=lambda k: int(k[1:]))[-MAX_RUNS:]
    doc["runs"] = {k: tbl[k] for k in keys}


def run_from_event(ev, sid):
    if ev.get("kind") == "metric":  # kenv.metric(...) on the kernel
        return {"kind": "metric", "label": ev.get("name"), "name": ev.get("name"), "value": ev.get("value"),
                "session": sid, "started": ev.get("started"), "status": "ok",
                "evid": f"{ev.get('epoch')}-{ev.get('seq')}"}
    ok = ev.get("status") != "failed"
    r = {"kind": "timed", "label": ev.get("label", "default"), "session": sid, "started": ev.get("started"),
         "duration_s": round(ev.get("elapsed", 0), 2), "status": "ok" if ok else "failed", "exit_code": 0 if ok else 1}
    if not ok:
        r["error"] = scrub(ev.get("error") or "error")[:200]
    mb = 1024 ** 2
    r["ram_delta_mb"] = round((ev["ram_end"] - ev["ram_start"]) / mb, 1)
    r["ram_peak_gb"] = round(ev["ram_peak"] / 1024 ** 3, 2)
    if ev.get("vram_total"):
        r["vram_delta_mb"] = round((ev["vram_end"] - ev["vram_start"]) / mb, 1)
        r["vram_peak_gb"] = round(ev["vram_peak"] / 1024 ** 3, 2)
    r["disk_delta_mb"] = round((ev["disk_end"] - ev["disk_start"]) / mb, 1)
    r["cpu_avg_pct"] = round(ev.get("cpu_avg", 0))
    r["evid"] = f"{ev.get('epoch')}-{ev.get('seq')}"
    return r


class SessionCore:
    """Keeps .kenv/<version>/core.toml current for one running session (all of it happens on the local side)."""

    def __init__(self, ep, proj, ver, sid):
        self.ep, self.proj, self.ver, self.sid = ep, proj, ver, sid
        self.store = CoreStore(proj, ver)
        self._lock, self._last = threading.Lock(), time.time()

    def refresh(self, quiet=True):
        with self._lock:
            try:
                imports, names = scan_code(self.proj)
                info, events = None, []
                try:
                    info = self.ep.json_call("/coreinfo", payload={"modules": imports}, timeout=60, tries=2)
                    events = self.ep.json_call("/cli/events", payload={"since": 0}, timeout=20, tries=1).get("events", [])
                except KenvError:
                    pass  # the kernel is busy or gone: keep what the file already says
                ref = (self.ep.info or {}).get("ref") or "unknown"
                gpu = (self.ep.info or {}).get("gpu", "none")
                gpu_id = None if gpu in ("none", None) else gpu
                proj, v, sid = self.proj, self.ver, self.sid

                def fn(doc):
                    doc["manifest"] = core_manifest(proj, v, doc.get("manifest"), ref, gpu_id, info)
                    dep = doc.get("dependencies") or {}
                    doc["dependencies"] = {"detected": dict(info["detected"]) if info else dep.get("detected", {}),
                                           "lock": dict(info["lock"]) if info else dep.get("lock", {})}
                    io = doc.get("io") or {}
                    old_in = io.get("inputs") or {}
                    doc["io"] = {"inputs": {"datasets": sorted(d["ref"] for d in proj.datasets() if d.get("ref")),
                                            "mounted": sorted(info["mounted"]) if info else old_in.get("mounted", [])},
                                 "outputs": io.get("outputs", {})}
                    doc["resources"] = core_resources(doc.get("resources"), info, sid)
                    doc["secrets"] = {"names": names}  # names only
                    sess = proj.sessions(v)
                    doc["sessions"] = {"file": "sessions.json", "count": len(sess), "total_s": proj.total_time(v)}
                    core_add_runs(doc, [run_from_event(e, sid) for e in events])

                self.store.update(fn)
                self._last = time.time()
            except Exception:  # bookkeeping must never break a session
                if not quiet:
                    raise

    def add_events(self, events):
        runs = [run_from_event(e, self.sid) for e in events]
        self.store.update(lambda doc: core_add_runs(doc, runs))

    def tick(self):
        if time.time() - self._last > 600:  # a slow refresh from the heartbeat, so a crash loses little
            self.refresh()


def core_for_session(ep, proj):
    st = load_state(ep.name) or {}
    v = st.get("version")
    return (CoreStore(proj, v), st) if v and (proj.kdir / v).is_dir() else (None, st)


def record_outputs(ep, proj, entries):
    """entries: {relative path: {size, sha256}} of files that came back from the kernel."""
    store, _ = core_for_session(ep, proj)
    if not store or not entries:
        return

    def fn(doc):
        io = doc.get("io") or {}
        outs = dict(io.get("outputs") or {})
        for rel, e in entries.items():
            outs[rel.replace("\\", "/")] = {"size": e["size"], "sha256": e["sha256"]}
        outs = {k: v for k, v in sorted(outs.items()) if (proj.root / k).is_file()}
        doc["io"] = {"inputs": io.get("inputs", {}), "outputs": outs}

    try:
        store.update(fn)
    except Exception:  # the I/O map is bookkeeping: never fail a pull because of it
        pass


def record_script_run(proj, st, script, t0, rc, tail):
    v = (st or {}).get("version")
    if not v or not (proj.kdir / v).is_dir():
        return
    ok = rc == 0
    run = {"kind": "script", "label": Path(script).name, "session": st.get("sid"), "started": iso(t0),
           "duration_s": round(time.time() - t0, 1), "status": "ok" if ok else "failed", "exit_code": rc}
    if not ok:
        run["error"] = error_summary(tail)
    try:
        CoreStore(proj, v).update(lambda doc: core_add_runs(doc, [run]))
    except Exception:
        pass


# ----------------------------------------------------------------------------- kenv.cli() from the kernel: the queue worker
#
# Code on the kernel cannot reach this machine, so it drops a request in the agent's queue. This worker
# (a background thread of the process that owns the session) long-polls the queue, runs the command as a
# normal `kenv ...` subprocess in the project folder, and posts the output back.

POLL_WAIT = 20
BLOCKED_NAMES = {"init", "stop", "cred", "sweep", "url", "help", "gpu", "exec", "run", "attach"}
BLOCKED_FLAGS = {"--cred", "--sweep", "--url", "--gpu", "-n", "--name"}
QUEUE_ALLOWED = {"activate", "versions", "new", "sync", "save", "put", "ls", "data", "core", "shim",
                 # later phases: they already route through the queue (an older kenv answers "unknown command")
                 "commit", "diff", "rollback", "branch", "tag", "metric", "log", "logs", "export", "import",
                 "doctor", "convert", "quota", "deps", "rebuild", "secret-scan", "scan", "clip", "ui"}


def _inside(root, p):
    try:
        Path(root, os.path.expanduser(p)).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


def check_queue_command(argv, root):
    """(ok, reason). The kernel may only ask for project-scoped kenv commands: nothing that starts, stops or
    attaches sessions, and no file outside the project folder - a notebook (or a package it installs)
    must not be able to write to or read from the rest of your disk."""
    if not argv or argv[0].startswith("-"):
        return False, "the command name must come first, e.g. kenv.cli(\"kenv versions\")"
    name = argv[0].lower()
    flags = [t.split("=")[0] for t in argv[1:] if t.startswith("-")]
    if name in BLOCKED_NAMES or any(f in BLOCKED_FLAGS for f in flags):
        return False, f"'{argv[0]}' cannot be run from code on the kernel; type it in your kenv terminal"
    if any(f in ("-id", "--id") for f in flags) and name != "activate":
        return False, "attaching to a session cannot be done from code"
    if name in ("log", "logs") and "--tail" in flags:
        return False, "--tail follows the log until you press Ctrl-C; type it in your kenv terminal"
    is_ref = bool(VERSION_RE.match(argv[0])) or name.startswith("kv:") or any(f in ("-r", "--rename") for f in flags)
    if name not in QUEUE_ALLOWED and not is_ref:
        return False, f"'{argv[0]}' is not a command that can run from code"
    pos, i = [], 1
    while i < len(argv):
        t = argv[i]
        if t in VALUE_OPTS:
            if t in ("--to", "--out", "--from", "--file") and i + 1 < len(argv) and not _inside(root, argv[i + 1]):
                return False, "paths must be inside the project"
            i += 2
            continue
        if t.startswith(("--to=", "--out=", "--from=", "--file=")) and not _inside(root, t.split("=", 1)[1]):
            return False, "paths must be inside the project"
        if not t.startswith("-"):
            pos.append(t)
        i += 1
    if name == "quota" and pos[:1] and pos[0].lower() in ("set", "reset"):
        return False, "the quota limit is a setting of your machine; change it in your kenv terminal"
    if name == "import":
        return False, "import writes into your project from a zip; run it in your kenv terminal"
    if name == "scan" and pos[:1] and pos[0].lower() in ("allow", "hook", "unhook"):
        return False, "allowlist and git-hook changes must be made in your kenv terminal"
    check = pos if name in ("put", "import", "clip") else (pos[1:] if name == "data" and pos[:1] == ["push"] else
                                       [p for p in pos if re.search(r"[\\/~]|^\.", p)] if name == "activate" else [])
    if any(not _inside(root, p) for p in check):
        return False, "local paths must be inside the project folder"
    return True, ""


class QueueWorker:
    def __init__(self, ep, proj, core):
        self.ep, self.proj, self.core = ep, proj, core
        self.stop_ev, self.jobs = threading.Event(), queue.Queue()
        self.ack, self.epoch = 0, None
        self.threads = [threading.Thread(target=self._poll_loop, daemon=True),
                        threading.Thread(target=self._exec_loop, daemon=True)]

    def start(self):
        for t in self.threads:
            t.start()

    def stop(self):
        self.stop_ev.set()

    def _poll_loop(self):
        fails = 0
        while not self.stop_ev.is_set():
            try:
                r = self.ep.json_call("/cli/poll", payload={"wait": POLL_WAIT, "ack": self.ack},
                                      timeout=POLL_WAIT + 25, tries=1)
                fails = 0
            except Exception:
                fails += 1
                if fails % 3 == 0:  # the tunnel address changes when the accelerator is switched
                    try:
                        self.ep.resolve(force=True)
                    except Exception:
                        pass
                self.stop_ev.wait(min(20, 2 * fails))
                continue
            if r.get("epoch") != self.epoch:  # a new kernel: its counters start over
                self.epoch, self.ack = r.get("epoch"), 0
            evs = r.get("events") or []
            if evs:
                self.ack = max(self.ack, max(e.get("seq", 0) for e in evs))
                try:
                    self.core.add_events(evs)
                except Exception:
                    pass
            if r.get("job"):
                self.jobs.put(r["job"])

    def _exec_loop(self):
        while not self.stop_ev.is_set():
            try:
                job = self.jobs.get(timeout=1)
            except queue.Empty:
                continue
            rc, out = self.execute(job)
            for _ in range(3):
                try:
                    self.ep.json_call("/cli/result", payload={"id": job["id"], "rc": rc, "out": out}, timeout=30, tries=2)
                    break
                except Exception:
                    time.sleep(2)

    def execute(self, job):
        argv = [str(x) for x in job.get("argv") or []]
        ok, why = check_queue_command(argv, self.proj.root)
        if not ok:
            return 2, f"[kenv] {why}"
        env = {**os.environ, "KENV_SESSION": connect_str(self.ep.name, self.ep.secret), "NO_COLOR": "1",
               "PYTHONIOENCODING": "utf-8", "KENV_FROM_CODE": "1"}
        try:
            p = subprocess.run([sys.executable, str(Path(__file__).resolve()), *argv], cwd=str(self.proj.root), env=env,
                               stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=float(job.get("timeout") or 120))
            return p.returncode, (p.stdout + p.stderr)[-20000:]
        except subprocess.TimeoutExpired:
            return 124, f"[kenv] timed out after {job.get('timeout')} s"
        except OSError as e:
            return 1, f"[kenv] could not run the command: {e}"


# ----------------------------------------------------------------------------- kenv_shim: safe kenv lines in code

SHIM_FILE = "kenv_shim.py"
SHIM_SRC = '''"""kenv_shim - keeps `kenv` lines in your code harmless outside a kenv session (CI, GitHub, a teammate's laptop).

    from kenv_shim import kenv

    kenv.time_start()
    model.fit(X, y)
    kenv.time_end()
    kenv.time_output()              # inside a session: prints the change in time, RAM, VRAM, disk, CPU
    kenv.cli("kenv v2 -r champ")    # inside a session: runs the command; elsewhere: does nothing

Inside a kenv session this is the real kernel-side module. Anywhere else every call does nothing and returns None
(so kenv.time_output() is None there), and `with kenv.timed():` just runs its body. The kenv lines stay visible
in your code; that is the trade for never breaking a run that has no kenv. `kenv init` puts this file in your
project root; copy it next to any other code you want to protect.
"""
import os
from contextlib import nullcontext


def _real():
    try:
        import kenv as m
    except ImportError:
        m = None
    if m is not None and getattr(m, "__kenv_kernel__", False):
        return m
    pkg = os.environ.get("KENV_PKG")  # set on kenv kernels; guards against another module named kenv shadowing it
    path = os.path.join(pkg, "kenv.py") if pkg else ""
    if path and os.path.isfile(path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("kenv_kernel", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return None


class _NoOp(object):
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: None

    def timed(self, *a, **k):
        return nullcontext()

    def __repr__(self):
        return "<kenv: not inside a kenv session - calls do nothing>"


kenv = _real() or _NoOp()
'''


def write_shim(root, force=False):
    p = Path(root) / SHIM_FILE
    if p.exists() and not force:
        return False
    p.write_text(SHIM_SRC, encoding="utf-8")
    return True


# ----------------------------------------------------------------------------- Phase 3: versioning, metrics and logs
#
#   .kenv/branches.json   branch -> parent version          .kenv/tags.json   tag -> version        .kenv/branch  current branch
#   .kenv/vN/meta.json    id, name, message, parent, branch, committed ...
#   .kenv/vN/code/        the tracked project files at commit time (+ code.json: path -> sha256, size)
#   .kenv/vN/deps.lock    pip-freeze style lock       config.json  datasets, accelerator, python, secret NAMES
#   .kenv/vN/logs/        run.log, errors.log         core.toml    gets a [metrics] table when the version is committed
# A version that was never committed (v1 made by `kenv init`) has no snapshot; sessions keep running in the version they
# started in, and `kenv commit` always makes the NEXT numbered version and makes it the one `kenv init` resumes.

SNAP_MAX_FILE = 25 * 1024 * 1024      # bigger files are listed (with their hash) but not copied into a snapshot
LOCK_LINE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;#]+)")
METRIC_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./ -]{0,59}$")


def norm_pkg(n):
    return re.sub(r"[-_.]+", "-", str(n)).lower()


def read_lock(path):
    out = {}
    try:
        for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            m = LOCK_LINE.match(ln)
            if m:
                out[m.group(1)] = m.group(2)
    except OSError:
        pass
    return out


def write_lock(path, lock):
    body = "".join(f"{k}=={v}\n" for k, v in sorted(lock.items(), key=lambda kv: kv[0].lower()) if v and v != "?")
    Path(path).write_bytes((body or "# no dependency data was available for this commit\n").encode("utf-8"))


def parse_number(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        pass
    try:
        v = float(s)
    except (TypeError, ValueError):
        raise KenvError(f"A metric value must be a number, got '{s}'.")
    if v != v or v in (float("inf"), float("-inf")):
        raise KenvError("A metric value must be a finite number.")
    return v


def tracked_files(proj):
    """Non-ignored project files (what a snapshot and a sync contain). kenv_shim.py is kenv's own file."""
    files = scan_local(proj.root, proj.rules(), proj.load_sync()["files"])
    files.pop(SHIM_FILE, None)
    return files


def project_datasets(proj):
    return sorted(d["ref"] for d in proj.datasets() if d.get("ref"))


def working_version(proj):
    """(version the running session works in | the active version, its live session state or None)."""
    live = proj.live_session()
    if live and live.get("version") and (proj.kdir / live["version"]).is_dir():
        return live["version"], live
    return proj.last_active(), None


def working_lock(proj, w, live):
    """-> (lock {distribution: version} | None when nothing is known yet, coreinfo | None)."""
    if live:
        try:
            info = Endpoint(live["name"], live["secret"]).json_call(
                "/coreinfo", payload={"modules": scan_code(proj)[0]}, timeout=60, tries=2)
            return dict(info["lock"]), info
        except Exception:
            pass  # the kernel is busy or gone: fall back to what the core file says
    lock = (CoreStore(proj, w).load().get("dependencies") or {}).get("lock") if w else None
    return (dict(lock) if lock else None), None


def pending_metrics(proj, w):
    """Metrics recorded in the working version since its last commit consumed them -> ({name: latest value}, highest run no.)."""
    doc = CoreStore(proj, w).load() if w else {}
    runs = doc.get("runs") or {}
    upto = int(proj.meta(w).get("metrics_upto") or 0) if w else 0
    latest, top = {}, upto
    for k in sorted((k for k in runs if re.fullmatch(r"r\d+", k)), key=lambda k: int(k[1:])):
        n, r = int(k[1:]), runs[k]
        top = max(top, n)
        if n > upto and isinstance(r, dict) and r.get("kind") == "metric" and r.get("name") is not None:
            latest[r["name"]] = r.get("value")
    return latest, top


def metrics_of(proj, v):
    """The metrics of a version: the table frozen at its commit; for a never-committed version, the latest recorded values."""
    doc = CoreStore(proj, v).load()
    t = doc.get("metrics")
    if proj.meta(v).get("committed") and isinstance(t, dict):
        return dict(t)
    out = {}
    runs = doc.get("runs") or {}
    for k in sorted((k for k in runs if re.fullmatch(r"r\d+", k)), key=lambda k: int(k[1:])):
        r = runs[k]
        if isinstance(r, dict) and r.get("kind") == "metric" and r.get("name") is not None:
            out[r["name"]] = r.get("value")
    return out


def record_metric(proj, v, name, value, sid=None):
    if not METRIC_NAME.match(name):
        raise KenvError("A metric name uses letters, digits, space . _ / - (max 60 characters).")
    run = {"kind": "metric", "label": name, "name": name, "value": value, "session": sid or "local",
           "started": iso(), "status": "ok"}
    CoreStore(proj, v).update(lambda doc: core_add_runs(doc, [run]))


def do_commit(proj, message, auto=False):
    """Snapshot code + dependencies + config into the next numbered version. -> summary dict, or None when
    nothing changed since the active version's snapshot."""
    proj.guard()
    head = proj.last_active()
    w, live = working_version(proj)
    files = tracked_files(proj)
    if not auto:
        secret_gate(proj, sorted(files), "Commit")  # keys must not land in a snapshot under .kenv
    lock, info = working_lock(proj, w, live)
    datasets = project_datasets(proj)
    metrics, upto = pending_metrics(proj, w)
    prev = proj.load_snapshot(head) if head and proj.has_snapshot(head) else None
    changed = None
    if prev is not None:
        changed = diff_local(prev, files)
        if not auto:
            cfg = read_json(proj.kdir / head / "config.json", {})
            old = {norm_pkg(k): x for k, x in read_lock(proj.kdir / head / "deps.lock").items()}
            same_lock = lock is None or {norm_pkg(k): x for k, x in lock.items()} == old
            if not (any(changed) or metrics or not same_lock or datasets != cfg.get("datasets", datasets)):
                return None
    src_doc = CoreStore(proj, w).load() if w else {}
    v = proj.new_version()
    vd = proj.kdir / v
    try:
        prev_code = proj.kdir / head / "code" if prev is not None else None
        entries, linked, copied, skipped = {}, 0, 0, []
        for rel, e in sorted(files.items()):
            if e["size"] > SNAP_MAX_FILE:
                entries[rel] = {"sha256": e["sha256"], "size": e["size"], "skipped": True}
                skipped.append(rel)
                continue
            dst, p, done = vd / "code" / rel, (prev or {}).get(rel), False
            dst.parent.mkdir(parents=True, exist_ok=True)
            if p and p.get("sha256") == e["sha256"] and not p.get("skipped") and (prev_code / rel).is_file():
                try:
                    os.link(prev_code / rel, dst)  # unchanged since the last snapshot: share the bytes
                    entries[rel] = {"sha256": e["sha256"], "size": e["size"]}
                    linked += 1
                    done = True
                except (OSError, NotImplementedError):
                    pass
            if not done:
                try:
                    shutil.copyfile(proj.root / rel, dst)
                    entries[rel] = {"sha256": sha256_file(dst), "size": dst.stat().st_size}  # hash the COPY: it is what a rollback restores
                    copied += 1
                except OSError:
                    continue  # vanished while we were copying
        write_json(vd / "code.json", {"count": len(entries), "bytes": sum(x["size"] for x in entries.values()),
                                      "files": entries})
        write_lock(vd / "deps.lock", lock or {})
        man = src_doc.get("manifest") or {}
        cfg = {"datasets": datasets, "accelerator": man.get("accelerator"), "accelerator_id": man.get("accelerator_id"),
               "kernel": man.get("kernel"), "python": (info or {}).get("python") or man.get("python"),
               "docker_image": (info or {}).get("image") or man.get("docker_image"),
               "secrets": (src_doc.get("secrets") or {}).get("names", []), "lock_known": lock is not None}
        write_json(vd / "config.json", {k: x for k, x in cfg.items() if x is not None})
        meta = proj.meta(v)
        meta.update(message=message, parent=head, branch=proj.current_branch(), committed=iso(), source=w,
                    files=len(entries))
        if auto:
            meta["auto"] = True
        write_json(vd / "meta.json", {k: x for k, x in meta.items() if x is not None})

        def fn(doc):
            m = dict(man)
            m.update(version=v, id=meta["id"], created=iso(), updated=iso())
            m.pop("name", None)
            doc["manifest"] = m
            doc["dependencies"] = {"detected": dict((src_doc.get("dependencies") or {}).get("detected") or {}),
                                   "lock": dict(lock or {})}
            dec = dict((src_doc.get("dependencies") or {}).get("declared") or {})
            if dec:
                doc["dependencies"]["declared"] = dec  # packages the user declared by hand travel with every commit
            doc["io"] = {"inputs": {"datasets": datasets,
                                    "mounted": sorted((info or {}).get("mounted") or
                                                      ((src_doc.get("io") or {}).get("inputs") or {}).get("mounted", []))},
                         "outputs": {}}
            doc["secrets"] = {"names": list((src_doc.get("secrets") or {}).get("names", []))}
            doc["sessions"] = {"file": "sessions.json", "count": 0, "total_s": 0}
            doc["metrics"] = dict(metrics)

        CoreStore(proj, v).update(fn)
        if w and upto:
            proj.set_meta(w, metrics_upto=upto)  # those results now belong to the new version
    except BaseException:
        shutil.rmtree(vd, ignore_errors=True)
        raise
    proj.set_active(v)
    return {"version": v, "head": head, "files": len(entries), "linked": linked, "copied": copied, "skipped": skipped,
            "changed": changed, "lock": len(lock or {}), "lock_known": lock is not None, "metrics": metrics,
            "branch": proj.current_branch(), "session_version": w if live else None}


def working_dirty(proj, head, files, datasets, metrics):
    prev = proj.load_snapshot(head) if head and proj.has_snapshot(head) else None
    if prev is None:
        return bool(files) or bool(metrics)
    cfg = read_json(proj.kdir / head / "config.json", {})
    return any(diff_local(prev, files)) or bool(metrics) or datasets != cfg.get("datasets", datasets)


def kernel_remove(ep, rels):
    ks = load_ksync(ep.name)
    for i in range(0, len(rels), 100):
        part = rels[i:i + 100]
        with ep.call("/exec", payload={"cmd": "rm -f -- " + " ".join(shlex.quote(r) for r in part)}, timeout=120) as r:
            r.read()
    for r in rels:
        ks["pushed"].pop(r, None)
        ks["kbase"].pop(r, None)
    save_ksync(ep.name, ks)


def rebuild_env(ep, proj, v, full=False, dry_run=False, warn_env=True):
    """Best effort: make the kernel's packages match the version's deps.lock plus its declared packages (only what
    the project needs and what is missing; the rest of Kaggle's image is left alone), then run `pip check`.
    Warns when Kaggle's Python or Docker image differ from what the version recorded. -> number of packages."""
    want = read_lock(proj.kdir / v / "deps.lock")
    doc = CoreStore(proj, v).load()
    declared = dict((doc.get("dependencies") or {}).get("declared") or {})
    info = ep.json_call("/coreinfo", payload={"modules": []}, timeout=60, tries=2)
    if warn_env:
        env_mismatch(recorded_env(proj, v), info)
    if not want and not declared:
        say(c("[kenv] this version has no dependency lock or declared packages, so there is nothing to rebuild", "33"))
        return 0
    have = {norm_pkg(k): x for k, x in info["lock"].items()}
    direct = set()
    for spec in ((doc.get("dependencies") or {}).get("detected") or {}).values():
        m = re.match(r"^(.+?)==", str(spec))
        if m:
            direct.add(norm_pkg(m.group(1)))
    todo = [f"{k}=={x}" for k, x in sorted(want.items()) if norm_pkg(k) not in have or
            (have[norm_pkg(k)] != x and (full or norm_pkg(k) in direct))]
    names = {norm_pkg(re.split(r"[=<>!~]", t, 1)[0]) for t in todo}
    for k, sp in sorted(declared.items()):
        sp = str(sp)
        if norm_pkg(k) in names:
            continue
        if norm_pkg(k) not in have:
            todo.append(k if sp in ("", "*") else k + sp)
        elif sp.startswith("==") and have[norm_pkg(k)] != sp[2:]:
            todo.append(k + sp)
    if dry_run:
        say(f"[kenv] dry run: {len(todo)} package(s) would be installed" + (": " + ", ".join(todo[:20]) + (" ..." if len(todo) > 20 else "") if todo else ""))
        return len(todo)
    if not todo:
        say("[kenv] the kernel already has the packages this version needs")
        return 0
    if len(todo) > 40 and not full:
        say(c(f"[kenv] {len(todo)} packages differ from the lock - too many to install one by one. The kernel's image "
              f"differs from the one this version was locked on; install what you need with `kenv exec pip install ...`, "
              f"or force everything with `kenv rebuild --all`.", "33"))
        return 0
    say(f"[kenv] installing {len(todo)} package(s) from the version's lock and declared packages ...")
    rc = stream_exec(ep, "python -m pip install -q --no-deps " + " ".join(shlex.quote(t) for t in todo))
    if rc != 0:
        say(c("[kenv] pip reported a problem; the session keeps running", "33"))
    say("[kenv] checking for dependency conflicts (pip check) ...")
    if stream_exec(ep, "python -m pip check") == 0:
        say(c("[kenv] pip check: no broken requirements", "32"))
    else:
        say(c("[kenv] pip check reported conflicts (above). Conflicts inside Kaggle's own image are not caused by your code.", "33"))
    return len(todo)


# ---- diff

def side_view(proj, v):
    has = proj.has_snapshot(v)
    cfg = read_json(proj.kdir / v / "config.json", {})
    return {"label": proj.label(v), "v": v, "snap": has,
            "code": {rel: e.get("sha256") for rel, e in proj.load_snapshot(v).items()},
            "libs": read_lock(proj.kdir / v / "deps.lock"), "inputs": sorted(cfg.get("datasets") or []),
            "metrics": metrics_of(proj, v)}


def working_view(proj):
    w, live = working_version(proj)
    lock, _ = working_lock(proj, w, live)
    return {"label": "working files", "v": None, "snap": True,
            "code": {r: e["sha256"] for r, e in tracked_files(proj).items()},
            "libs": lock or {}, "inputs": project_datasets(proj), "metrics": pending_metrics(proj, w)[0]}


def side_text(proj, side, rel):
    p = (proj.kdir / side["v"] / "code" / rel) if side["v"] else (proj.root / rel)
    try:
        if p.stat().st_size > 1024 * 1024:
            return None
        raw = p.read_bytes()
    except OSError:
        return None
    return None if b"\0" in raw else raw.decode("utf-8", "replace").splitlines()


def fmt_metric(x):
    return "(none)" if x is None else (f"{x:.6g}" if isinstance(x, float) else str(x))


def cmd_diff(a):
    proj = current_project()
    refs = a.args
    if len(refs) > 2:
        raise KenvError("Usage: kenv diff [<from>] [<to>]     (versions: v2, a name, a tag, kv:...; one argument compares with "
                        "your working files, none compares the active version with them)")
    if len(refs) == 2:
        A, B = side_view(proj, proj.find_version(refs[0])), side_view(proj, proj.find_version(refs[1]))
    else:
        head = proj.find_version(refs[0]) if refs else proj.last_active()
        if not head:
            raise KenvError("No versions yet - `kenv init` creates v1.")
        A, B = side_view(proj, head), working_view(proj)
    say(f"Diff  {A['label']}  ->  {B['label']}")
    for s in (A, B):
        if not s["snap"]:
            say(c(f"  note: {s['label']} has no snapshot (never committed) - shown as an empty project; its metrics are the ones "
                  f"recorded in it", "33"))
    ac, bc = A["code"], B["code"]
    add = sorted(r for r in bc if r not in ac)
    rem = sorted(r for r in ac if r not in bc)
    mod = sorted(r for r in bc if r in ac and ac[r] != bc[r])
    say(f"\nCode: {len(add)} added, {len(mod)} changed, {len(rem)} removed" + ("" if add or mod or rem else "  (identical)"))
    patches = []
    for tag_, names in (("A", add), ("M", mod), ("D", rem)):
        for rel in names[:60]:
            extra = ""
            ta, tb = side_text(proj, A, rel) if tag_ != "A" else [], side_text(proj, B, rel) if tag_ != "D" else []
            if ta is not None and tb is not None:
                d = list(difflib.unified_diff(ta, tb, f"{A['label']}/{rel}", f"{B['label']}/{rel}", lineterm="", n=2))
                p_, m_ = sum(1 for x in d if x.startswith("+") and not x.startswith("+++")), \
                    sum(1 for x in d if x.startswith("-") and not x.startswith("---"))
                extra = f"  (+{p_} -{m_})"
                patches.append(d)
            say(f"  {tag_} {rel}{extra}")
        if len(names) > 60:
            say(f"  ... and {len(names) - 60} more {tag_}")
    if a.patch:
        n = 0
        for d in patches:
            for ln in d:
                if n >= 500:
                    say("... (patch output cut at 500 lines)")
                    break
                say(ln)
                n += 1
    la, lb = {norm_pkg(k): (k, x) for k, x in A["libs"].items()}, {norm_pkg(k): (k, x) for k, x in B["libs"].items()}
    ladd = sorted(k for k in lb if k not in la)
    lrem = sorted(k for k in la if k not in lb)
    lchg = sorted(k for k in lb if k in la and la[k][1] != lb[k][1])
    say(f"\nLibraries: {len(ladd)} added, {len(lchg)} upgraded/downgraded, {len(lrem)} removed"
        + ("" if ladd or lchg or lrem else "  (identical)"))
    if not A["libs"] or not B["libs"]:
        say(c("  note: a dependency lock is empty on one side (no session has recorded packages for it yet)", "33"))
    for k in ladd[:40]:
        say(f"  + {lb[k][0]}=={lb[k][1]}")
    for k in lrem[:40]:
        say(f"  - {la[k][0]}=={la[k][1]}")
    for k in lchg[:40]:
        say(f"  ~ {lb[k][0]} {la[k][1]} -> {lb[k][1]}")
    if len(ladd) > 40 or len(lrem) > 40 or len(lchg) > 40:
        say("  ... (first 40 of each shown)")
    ia, ib = set(A["inputs"]), set(B["inputs"])
    say("\nInputs (datasets): " + ("identical" if ia == ib else f"{len(ib - ia)} added, {len(ia - ib)} removed"))
    for r in sorted(ib - ia):
        say(f"  + {r}")
    for r in sorted(ia - ib):
        say(f"  - {r}")
    ma, mb = A["metrics"], B["metrics"]
    names = sorted(set(ma) | set(mb))
    diffs = [n for n in names if ma.get(n) != mb.get(n)]
    say("\nMetrics: " + (f"{len(diffs)} differ" if diffs else f"no change ({len(names)} compared)" if names else "none recorded"))
    for n in diffs:
        x, y = ma.get(n), mb.get(n)
        delta = ""
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            delta = f"   ({y - x:+.6g})"
        say(f"  {n}: {fmt_metric(x)} -> {fmt_metric(y)}{delta}")
    return 0


# ---- commands

def cmd_commit(a):
    """kenv commit -m "message" """
    proj = current_project()
    msg = a.message or (" ".join(a.args) if a.args else "") or f"snapshot {time.strftime('%Y-%m-%d %H:%M')}"
    r = do_commit(proj, msg)
    if r is None:
        say(f"[kenv] nothing to commit: your working files, packages, datasets and metrics match {proj.label(proj.last_active())}")
        return 0
    v = r["version"]
    say(f"[kenv] Committed {proj.label(v)}  \"{msg}\"   (branch {r['branch']}, parent {r['head'] or '-'}, id {proj.meta(v)['id']})")
    ch = r["changed"]
    delta = f"; {len(ch[0])} new, {len(ch[1])} changed, {len(ch[2])} removed since {r['head']}" if ch else ""
    say(f"  code    : {r['files']} file(s){delta} - {r['linked']} shared with the previous snapshot, {r['copied']} copied")
    say(f"  packages: {r['lock']} in deps.lock" if r["lock_known"] else
        c("  packages: unknown yet (no session has recorded them) - deps.lock is empty for this version", "33"))
    if r["metrics"]:
        say("  metrics : " + ", ".join(f"{k}={fmt_metric(x)}" for k, x in sorted(r["metrics"].items())))
    for rel in r["skipped"][:5]:
        say(c(f"  not copied (over {human(SNAP_MAX_FILE)}): {rel} - only its hash is recorded", "33"))
    if r["session_version"]:
        say(f"[kenv] The running session keeps working in {r['session_version']}; `kenv init` will resume {v}.")
    try:
        miss, known = missing_imports(proj, v)
        if known and miss:
            say(c(f"  imports : {len(miss)} not in the dependency lock ({', '.join(m for m, _ in miss[:6])}) - see `kenv deps fix`", "33"))
    except Exception:
        pass
    return 0


def cmd_rollback(a):
    """kenv rollback <version|name|tag>: code + dependencies + config come back; kernel state cannot."""
    proj = current_project()
    if not a.args:
        raise KenvError("Usage: kenv rollback <v2 | name | tag>")
    target = proj.find_version(a.args[0])
    if not proj.has_snapshot(target):
        raise KenvError(f"{proj.label(target)} has no snapshot (it was never committed), so there is nothing to restore.")
    head = proj.last_active()
    w, live = working_version(proj)
    ep = None
    if live:
        ep = Endpoint(live["name"], live["secret"])
        if kernel_alive(ep):
            say("[kenv] Pulling the kernel's changes into the project first ...")
            try:
                pull_back(ep, proj, quiet=True)
            except (KenvError, OSError) as e:
                say(c(f"[kenv] Could not pull from the kernel: {e}", "33"))
        else:
            ep = None
    files = tracked_files(proj)
    datasets = project_datasets(proj)
    metrics, _ = pending_metrics(proj, w)
    if target == head and not working_dirty(proj, head, files, datasets, metrics):
        say(f"[kenv] already at {proj.label(target)} and nothing has changed since")
        return 0
    if working_dirty(proj, head, files, datasets, metrics):
        r = do_commit(proj, f"auto-save before rollback to {target}", auto=True)
        say(f"[kenv] Your uncommitted changes are saved as {proj.label(r['version'])} - `kenv rollback {r['version']}` brings them back.")
        proj.set_active(head)  # the safety copy must not become what `activate` points at if the restore fails
        files = tracked_files(proj)
    snap = proj.load_snapshot(target)
    code = proj.kdir / target / "code"
    wrote, removed, missing = [], [], []
    for rel, e in sorted(snap.items()):
        if e.get("skipped"):
            if not (proj.root / rel).is_file():
                missing.append(rel)
            continue
        cur = files.get(rel)
        if cur and cur["sha256"] == e.get("sha256"):
            continue
        dst = proj.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(code / rel, dst)
        wrote.append(rel)
    for rel in sorted(set(files) - set(snap)):
        try:
            (proj.root / rel).unlink()
            removed.append(rel)
            d = (proj.root / rel).parent
            while d != proj.root:
                try:
                    d.rmdir()  # only empty folders go
                except OSError:
                    break
                d = d.parent
        except OSError:
            pass
    cfg = read_json(proj.kdir / target / "config.json", {})
    if isinstance(cfg.get("datasets"), list):
        old = {d["ref"]: d for d in proj.datasets() if d.get("ref")}
        write_json(proj.kdir / "datasets.json",
                   {"datasets": [old.get(r) or {"ref": r, "folder": "", "updated": iso()} for r in cfg["datasets"]]})
    proj.save_sync(tracked_files(proj))
    proj.set_active(target)
    tb = proj.meta(target).get("branch") or "main"
    if tb != proj.current_branch() and (tb == "main" or tb in proj.branches()):
        proj.set_branch(tb)
    say(f"[kenv] Rolled back to {proj.label(target)}: {len(wrote)} file(s) restored, {len(removed)} removed"
        + (f", branch {tb}" if tb != "main" else ""))
    for rel in missing:
        say(c(f"[kenv] {rel} was too big to keep in the snapshot and is not in the folder - restore it yourself", "33"))
    if ep is not None:
        try:
            kernel_remove(ep, removed)
            push_sync(ep, proj)
            rebuild_env(ep, proj, target)
        except (KenvError, OSError) as e:
            say(c(f"[kenv] The project is restored, but updating the kernel failed: {e}", "33"))
        say(f"[kenv] The session keeps working in {w} (a session keeps its version); `kenv init` will resume {target}.")
        if cfg.get("datasets"):
            say("[kenv] Dataset changes are attached at the next session start.")
    else:
        proj.set_meta(target, env_pending=True)
        say(f"[kenv] Packages are rebuilt when a session starts in {target} (`kenv init`).")
    say("[kenv] Only code, dependencies and config come back - a kernel is disposable, so its memory and files cannot be restored.")
    return 0


def cmd_tag(a):
    """kenv tag | kenv tag <version> <tag> | kenv tag <tag> | kenv tag --delete <tag>"""
    proj = current_project()
    tags, args = proj.tags(), list(a.args)
    if a.delete or (args and args[0] in ("rm", "delete")):
        names = args[1:] if args and args[0] in ("rm", "delete") else args
        if not names:
            raise KenvError("Usage: kenv tag --delete <tag>")
        for n in names:
            key = next((k for k in tags if k.lower() == n.lower()), None)
            if key is None:
                raise KenvError(f"There is no tag '{n}'. List them: kenv tag")
            del tags[key]
            say(f"[kenv] tag '{key}' removed")
        proj.save_tags(tags)
        return 0
    if not args:
        if not tags:
            say("No tags yet. Create one:  kenv tag v2 stable")
            return 0
        for k, v in sorted(tags.items()):
            if (proj.kdir / v).is_dir():
                m = proj.meta(v).get("message")
                say(f"  {k:<18} -> {proj.label(v):<24}" + (f"  \"{m}\"" if m else ""))
        return 0
    if len(args) == 1:
        t = args[0]
        if VERSION_RE.match(t) or t.lower().startswith("kv:"):
            v = proj.find_version(t)
            say(f"  {proj.label(v)}: " + (", ".join(proj.tags_of(v)) or "no tags"))
            return 0
        v, name = proj.last_active(), t
        if not v:
            raise KenvError("No versions yet - `kenv init` creates v1.")
    else:
        v, name = proj.find_version(args[0]), args[1]
    if not NAME_RE.match(name) or VERSION_RE.match(name):
        raise KenvError("Tag names use letters, digits, '.', '_' or '-' (max 40 characters) and cannot look like v1, v2 ...")
    if any((proj.meta(o).get("name") or "").lower() == name.lower() for o in proj.version_names()):
        raise KenvError(f"'{name}' is already the name of a version; tags and version names share one namespace.")
    old = next((k for k in tags if k.lower() == name.lower()), None)
    moved = tags.pop(old) if old else None
    tags[old or name] = v
    proj.save_tags(tags)
    say(f"[kenv] tag '{old or name}' -> {proj.label(v)}" + (f"   (moved from {moved})" if moved and moved != v else ""))
    return 0


def cmd_branch(a):
    """kenv branch | kenv branch <name> | kenv branch switch <name> | kenv branch rm <name>"""
    proj = current_project()
    br, cur, args = proj.branches(), proj.current_branch(), list(a.args)
    if not args:
        say("Branches (* = current). Code is not touched when you switch: use `kenv rollback <tip>` to restore a branch's files.")
        rows = [("main", None)] + sorted(br.items())
        for name, parent in rows:
            vs = proj.branch_versions(name)
            tip = f"tip {vs[-1]}" if vs else "no commits yet"
            frm = f"  from {parent}" if parent else ""
            say(f"  {'*' if name == cur else ' '} {name:<18}{tip:<16}{len(vs)} commit(s){frm}")
        return 0
    if args[0] in ("rm", "delete"):
        if len(args) < 2 or args[1] not in br:
            raise KenvError("Usage: kenv branch rm <name>     (an existing branch; 'main' cannot be removed)")
        if args[1] == cur:
            raise KenvError("That is the current branch; switch away first: kenv branch switch main")
        del br[args[1]]
        proj.save_branches(br)
        say(f"[kenv] branch '{args[1]}' removed (its versions stay)")
        return 0
    if args[0] in ("switch", "use", "checkout"):
        if len(args) < 2 or (args[1] != "main" and args[1] not in br):
            raise KenvError("Usage: kenv branch switch <name>     (list them: kenv branch)")
        proj.set_branch(args[1])
        vs = proj.branch_versions(args[1])
        if vs and not proj.live_session():
            proj.set_active(vs[-1])
        say(f"[kenv] on branch '{args[1]}'" + (f"; active version {proj.label(vs[-1])}" if vs and not proj.live_session() else ""))
        if vs:
            say(f"[kenv] Your files are unchanged. To restore the branch's code:  kenv rollback {vs[-1]}")
        return 0
    if len(args) != 1:
        raise KenvError("Usage: kenv branch <name>     (one name, no spaces)")
    name = args[0]
    if name in ("main", "switch", "use", "checkout", "rm", "delete") or not NAME_RE.match(name) or VERSION_RE.match(name):
        raise KenvError("Branch names use letters, digits, '.', '_' or '-' (max 40 characters), cannot look like v1, "
                        "and 'main', 'switch', 'rm' are reserved.")
    if name in br:
        raise KenvError(f"The branch '{name}' exists. Switch to it: kenv branch switch {name}")
    parent = proj.last_active()
    if not parent:
        raise KenvError("No versions yet - `kenv init` creates v1.")
    br[name] = parent
    proj.save_branches(br)
    proj.set_branch(name)
    say(f"[kenv] branch '{name}' created from {proj.label(parent)} and switched to. Commits made now belong to it.")
    return 0


def cmd_metric(a):
    """kenv metric <name> <value>  |  kenv metric   (list the results waiting for the next commit)"""
    proj = current_project()
    w, live = working_version(proj)
    if not w:
        raise KenvError("No versions yet - `kenv init` creates v1.")
    if not a.args:
        pend, _ = pending_metrics(proj, w)
        if not pend:
            say(f"No metrics recorded in {proj.label(w)} since its last commit.  Record one: kenv metric auc 0.93")
        for k, x in sorted(pend.items()):
            say(f"  {k} = {fmt_metric(x)}")
        return 0
    if len(a.args) != 2:
        raise KenvError("Usage: kenv metric <name> <value>     (quote a name that has spaces)")
    val = parse_number(a.args[1])
    record_metric(proj, w, a.args[0], val, live.get("sid") if live else None)
    say(f"[kenv] metric {a.args[0]} = {val} recorded in {proj.label(w)}; the next `kenv commit` stores it with that version")
    return 0


# ---- logs

LOG_TS = 24  # "2026-09-30T12:00:00.123Z"
ERR_RE = re.compile(r"(?i)\b(traceback|error|exception|fatal|critical|segmentation fault|out of memory|oom-?kill\w*|killed)\b|^\[E ")


def iso_ms(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + ".%03dZ" % int((t % 1) * 1000)


def log_is_error(src, text, st):
    if src == "error":
        return True
    if text.startswith("Traceback (most recent call last)"):
        st["tb"] = True
        return True
    if st.get("tb"):
        if text[:1] in (" ", "\t"):
            return True
        st["tb"] = False
        return True  # the closing "ValueError: ..." line of the traceback
    return src != "metric" and bool(ERR_RE.search(text))


_ERR_STATE = {}


def pull_logs_once(ep, proj, ver, wait=0):
    """Fetch new kernel log lines and append them to .kenv/<ver>/logs/run.log (+ errors.log). A cursor
    (kernel epoch + line number) in logs/.cursor, checked under a lock, keeps reconnects and a second kenv
    process from ever writing a line twice. -> number of lines written."""
    ldir = proj.kdir / ver / LOG_DIR
    ldir.mkdir(parents=True, exist_ok=True)
    cur = read_json(ldir / ".cursor", {})
    r = ep.json_call("/logs", payload={"since": int(cur.get("seq", 0)), "epoch": cur.get("epoch"), "wait": wait,
                                       "limit": 3000}, timeout=wait + 40, tries=1)
    rows = r.get("lines") or []
    if not rows and cur.get("epoch") == r.get("epoch"):
        return 0
    with file_lock(ldir / "run.log"):
        cur = read_json(ldir / ".cursor", {})  # another process may have moved it while we were waiting
        out, err, st = [], [], _ERR_STATE.setdefault(str(ldir), {})
        now = iso_ms(time.time())
        if cur.get("epoch") not in (None, r["epoch"]):
            out.append(f"{now} [kenv] --- the kernel was replaced; its log numbering starts over ---\n")
            cur = {}
        seq0 = int(cur.get("seq", 0)) if cur.get("epoch") == r["epoch"] else 0
        rows = [x for x in rows if x[0] > seq0]
        if rows and seq0 and rows[0][0] > seq0 + 1:
            out.append(f"{now} [kenv] --- {rows[0][0] - seq0 - 1} log line(s) were dropped (the kernel's buffer overflowed) ---\n")
        for seq, ts, src, text in rows:
            text = redact_secrets(scrub(text))  # a log lands in a folder people commit: no tokens
            line = f"{iso_ms(ts)} [{src}] {text}\n"
            out.append(line)
            if log_is_error(src, text, st):
                err.append(line)
        if out:
            with open(ldir / "run.log", "a", encoding="utf-8", newline="\n") as f:
                f.writelines(out)
        if err:
            with open(ldir / "errors.log", "a", encoding="utf-8", newline="\n") as f:
                f.writelines(err)
        write_json(ldir / ".cursor", {"epoch": r["epoch"], "seq": max([seq0] + [x[0] for x in rows])}, indent=None)
    return len(rows)


class LogPuller:
    """Follows the kernel's log for the whole session (a thread of the process that owns it)."""

    def __init__(self, ep, proj, ver, sid):
        self.ep, self.proj, self.ver, self.sid = ep, proj, ver, sid
        self.stop_ev = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        ldir = self.proj.kdir / self.ver / LOG_DIR
        ldir.mkdir(parents=True, exist_ok=True)
        try:
            (ldir / ".cursor").unlink()
        except OSError:
            pass
        for name in ("run.log", "errors.log"):
            if name == "run.log" or (ldir / name).exists():
                with open(ldir / name, "a", encoding="utf-8", newline="\n") as f:
                    f.write(f"{iso_ms(time.time())} [kenv] --- session {self.sid} online ---\n")
        self.thread.start()

    def _loop(self):
        fails = 0
        while not self.stop_ev.is_set():
            try:
                pull_logs_once(self.ep, self.proj, self.ver, wait=8)
                fails = 0
                self.stop_ev.wait(0.3)
            except Exception:
                fails += 1
                if fails % 3 == 0:  # the tunnel address changes when the accelerator is switched
                    try:
                        self.ep.resolve(force=True)
                    except Exception:
                        pass
                self.stop_ev.wait(min(20, 2 * fails))

    def stop(self):
        self.stop_ev.set()
        try:
            self.thread.join(timeout=1.5)
            for _ in range(5):  # the last lines, while the kernel is still there
                if not pull_logs_once(self.ep, self.proj, self.ver, wait=0):
                    break
        except Exception:
            pass


def parse_since(s):
    """'10m' | '2h' | '90s' | '1d' | '2026-09-30' | '2026-09-30 12:00' | '2026-09-30T12:00:00Z' | '12:30' -> epoch seconds.
    Absolute times without a zone are your local time; the log itself is in UTC (the trailing Z)."""
    t = s.strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(s\w*|m\w*|h\w*|d\w*)", t, re.I)
    if m:
        unit = m.group(2)[0].lower()
        return time.time() - float(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", t)
    if m:
        d = datetime.datetime.now().replace(hour=int(m.group(1)), minute=int(m.group(2)), second=int(m.group(3) or 0), microsecond=0)
        return d.timestamp()
    try:
        d = datetime.datetime.fromisoformat(t[:-1] + "+00:00" if t[-1:] in "Zz" else t)
    except ValueError:
        raise KenvError(f"Cannot read the time '{s}'. Use 10m, 2h, 1d, 12:30, 2026-09-30 or 2026-09-30T12:00:00Z.")
    return d.timestamp()  # naive datetimes are taken as local time


def cmd_logs(a):
    """kenv logs [version] [--tail] [--grep PATTERN] [--since TIME] [--errors]"""
    proj = current_project()
    live_now = proj.live_session()
    if a.args:
        v = proj.find_version(a.args[0])
    else:
        v = (live_now or {}).get("version") or proj.last_active()
    if not v:
        raise KenvError("No versions yet - `kenv init` creates v1.")
    path = proj.kdir / v / LOG_DIR / ("errors.log" if a.errors else "run.log")
    since = iso_ms(parse_since(a.since)) if a.since else None
    pat = None
    if a.grep:
        try:
            pat = re.compile(a.grep, re.I)
        except re.error:
            pat = re.compile(re.escape(a.grep), re.I)

    def keep(line):
        if since and line[:LOG_TS] < since:
            return False
        return not pat or bool(pat.search(line))

    if not path.is_file():
        say(f"No {path.name} for {proj.label(v)} yet: logs are recorded while a session for it is online (`kenv init`).")
        return 0
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    hit = [ln for ln in lines if keep(ln)]
    limit = 20 if a.tail else (None if (since or pat) else 500)
    if limit and len(hit) > limit:
        say(c(f"[kenv] ({len(hit) - limit} earlier line(s) not shown - use --since / --grep, or read {path})", "2"))
        hit = hit[-limit:]
    for ln in hit:
        say(ln)
    if not a.tail:
        return 0
    if not (live_now and live_now.get("version") == v):
        say(c(f"[kenv] no session is running in {proj.label(v)}; showing what was recorded", "33"))
        return 0
    say(c(f"[kenv] following {path.name} - Ctrl-C to stop", "2"))
    pos = path.stat().st_size
    buf = b""
    try:
        idle = 0
        while True:
            time.sleep(0.5)
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size < pos:
                pos, buf = 0, b""
            if size > pos:
                with open(path, "rb") as f:
                    f.seek(pos)
                    buf += f.read()
                    pos = f.tell()
                *done, buf = buf.split(b"\n")
                for raw in done:
                    ln = raw.decode("utf-8", "replace").rstrip("\r")
                    if keep(ln):
                        say(ln)
                idle = 0
            else:
                idle += 1
                if idle % 8 == 0:  # every ~4 s: is the session still there?
                    s2 = proj.live_session()
                    if not (s2 and s2.get("version") == v):
                        say(c("[kenv] the session ended", "2"))
                        return 0
    except KeyboardInterrupt:
        say()
        return 0


# ----------------------------------------------------------------------------- commands

def cmd_time(a, op):
    """kenv time_start | time_end | time_output [label]   (the same measurements as kenv.time_* in code)"""
    ep = need_ep()
    r = ep.json_call("/track", payload={"op": op, "label": a.args[0] if a.args else "default"}, timeout=60)
    if r.get("error"):
        raise KenvError(r["error"])
    say(r["text"])


def cmd_core(a):
    """kenv core [version]: refresh (when that version's session is live) and summarise .kenv/vN/core.toml"""
    proj = current_project()
    v, live = (proj.find_version(a.args[0]) if a.args else None), None
    try:
        t = find_target()
    except KenvError:
        t = None
    if t:
        ep = Endpoint(*t)
        ctx = project_ctx(ep)
        if ctx and ctx[0].root == proj.root and ctx[1].get("version"):
            live = (ep, ctx[1])
    v = v or (live[1]["version"] if live else proj.last_active())
    if not v:
        raise KenvError("No versions yet - `kenv init` creates v1.")
    store = CoreStore(proj, v)
    if live and live[1]["version"] == v and kernel_alive(live[0]):
        SessionCore(live[0], proj, v, live[1].get("sid")).refresh(quiet=False)
    doc = store.load()
    if not doc:
        say(f"No core file for {proj.label(v)} yet: it is written once a session for this version is online.")
        return 0
    m, dep, io = doc.get("manifest", {}), doc.get("dependencies", {}), doc.get("io", {})
    res, runs = doc.get("resources", {}), doc.get("runs", {})
    say(f"  Core file : {store.path}")
    say(f"  Version   : {proj.label(v)}   ({m.get('id', '?')})")
    say(f"  Kernel    : {m.get('kernel', '?')}   {m.get('accelerator', '?')}   python {m.get('python', '?')}")
    say(f"  Packages  : {len(dep.get('lock', {}))} locked, {len(dep.get('detected', {}))} imported by your code")
    say(f"  Inputs    : {len((io.get('inputs') or {}).get('datasets', []))} attached dataset(s), "
        f"{len((io.get('inputs') or {}).get('mounted', []))} mounted")
    say(f"  Outputs   : {len(io.get('outputs') or {})} file(s) with sizes and SHA256")
    if res:
        say(f"  Resources : RAM peak {res.get('ram_peak_gb')} GB, VRAM peak {res.get('vram_peak_gb')} GB, "
            f"CPU peak {res.get('cpu_peak_pct')}%, disk {res.get('disk_peak_gb')} GB, {res.get('runtime_min')} min")
    say(f"  Secrets   : {', '.join((doc.get('secrets') or {}).get('names', [])) or 'none referenced'}   (names only)")
    say(f"  Runs      : {len(runs)}")
    for k in list(runs)[-3:]:
        r = runs[k]
        if r.get("kind") == "metric":
            say(f"    {k}  metric {r.get('name')} = {r.get('value')}")
            continue
        say(f"    {k}  {r.get('kind', '?'):<6} {str(r.get('label', '')):<18} {r.get('status', '?'):<7} "
            f"{dur(r.get('duration_s'))}  exit {r.get('exit_code')}" + (f"  {r['error']}" if r.get("error") else ""))
    return 0


def cmd_shim(a):
    proj = current_project()
    if write_shim(proj.root, force=False):
        say(f"[kenv] wrote {proj.root / SHIM_FILE}")
    else:
        say(f"[kenv] {SHIM_FILE} is already in the project ({proj.root})")
    say("Use it in code:   from kenv_shim import kenv")
    say("Outside a kenv session every kenv.* call does nothing (and returns None), so CI and GitHub runs never break.")


def cmd_init(a):
    if os.environ.get("KENV_SESSION"):
        raise KenvError("You are already inside a kenv session. Type `exit` first.")
    proj = Project(Path.cwd())
    proj.guard()
    live = proj.live_session()
    if live:
        raise KenvError(f"A session for this project is already running here ('{live['name']}'). "
                        f"Attach with `kenv -id {live['name']}` or stop it first.")
    name = sanitize(a.name) if a.name else random_name()
    gpu_id = pick_gpu() if a.gpu else None
    banner()
    watch = QuotaWatch()
    quota_register(proj)
    if gpu_id:
        quota_start_check(watch)

    # --- the .kenv folder and the version this session belongs to
    lazy_tunnel = lazy_preflight(proj, a) if a.lazy_local else None   # before any Kaggle quota is spent
    created = proj.ensure()
    clip_session_start(proj)   # Phase 6: kenv lines that `kenv unclip` commented out come back before the files are scanned
    for v, e in proj.repair_crashed():
        say(c(f"[kenv] The last session in {proj.label(v)} ('{e.get('session')}') ended unexpectedly "
              f"(end estimated: {dur(e['duration_s'])} of use). If its kernel is still on Kaggle, `kenv --sweep` removes it.", "33"))
    if not proj.version_names():
        ver = proj.new_version()
        say(f"[kenv] New project folder: created .kenv/ and {ver}. Edit .kenvignore to keep big data/model folders out of the sync.")
    else:
        ver = proj.last_active()
        used = proj.last_ts(ver)
        when = f"last used {ago(used)}" if used else "not used yet"
        say(f"[kenv] Resuming {proj.label(ver)}, {when} - use 'kenv activate' to switch")
        if created:
            say("[kenv] (.kenv/ was created just now)")

    # --- look at the local files BEFORE spending Kaggle quota: the local files are always the source of truth
    old = proj.load_sync()["files"]
    say("[kenv] Scanning project files ...")
    pre = scan_local(proj.root, proj.rules(), old)
    total = sum(v["size"] for r, v in pre.items() if not lazy_tunnel or lazy_eager(r, v["size"]))
    if old:
        add, mod, rem = diff_local(old, pre)
        if add or mod or rem:
            say(f"[kenv] Changed since the last session: {len(add)} new, {len(mod)} modified, {len(rem)} removed - syncing your current local files")
    if total > SYNC_WARN and sys.stdin.isatty():
        say(c(f"[kenv] {len(pre)} files, {human(total)} would be uploaded to the kernel.", "33"))
        say(c("Put big data/model folders in .kenvignore and use `kenv data push <folder>` for inputs.", "33"))
        if not confirm("Upload it all anyway?"):
            raise KenvError("Cancelled before starting a kernel. Edit .kenvignore and run `kenv init` again.")

    sid = secrets.token_hex(4)
    idle = a.idle or 20
    watch.active = lambda: (proj.session_entry(ver, sid) or {}).get("gpu") not in (None, "", "none")
    extra = {"project": str(proj.root), "version": ver, "sid": sid, "kproj": proj.kname}
    if lazy_tunnel:
        extra["lazy"] = True
    datasets = [d["ref"] for d in proj.datasets() if d.get("ref")]
    lazy_box = []   # the LazyAccess of this session (empty when --lazy-local is off or failed)
    with Owner(name, gpu_id, idle, a.startup, extra=extra, kproj=proj.kname, datasets=datasets, lazy=bool(lazy_tunnel)) as o:
        say(f"[kenv] Session name: {c(name, '1')}")
        if datasets:
            say(f"[kenv] Attaching {len(datasets)} project dataset(s): " + ", ".join(datasets))
        o.start()
        proj.open_session(ver, sid, name, idle, gpu_id or "none")

        def on_end(exc, reason, end_ts):
            e = proj.session_entry(ver, sid)
            if not e or e.get("end"):
                return  # `kenv stop` already closed it
            seen = from_iso(e.get("last_seen"))
            proj.close_session(ver, sid, classify_end(exc, reason, o.ep, seen, idle, end_ts), end_ts)

        def on_end_all(exc, reason, end_ts):
            for la in lazy_box:   # the file server and its tunnel never outlive the session
                la.stop()
            on_end(exc, reason, end_ts)

        o.on_end = on_end_all
        say(c("[kenv] Session online", "32;1"))
        say()
        print_info(o.ep)
        say()
        if lazy_tunnel:
            try:
                la = LazyAccess(proj, o.ep, lazy_tunnel)
                lazy_box.append(la)
                la.start()
            except (KenvError, OSError) as e:
                for la in lazy_box:
                    la.stop()
                lazy_box.clear()
                st0 = load_state(o.ep.name)
                if st0:
                    st0["lazy"] = False
                    save_state(st0)
                say(c(f"[kenv] Lazy local access could not start: {e}\n[kenv] Falling back to the normal sync of every non-ignored file.", "33"))
        try:
            push_sync(o.ep, proj, cache=pre)
        except KenvError as e:
            say(c(f"[kenv] The first sync failed: {e}\n[kenv] The session keeps running - retry with `kenv sync`.", "33"))
        try:  # Kaggle controls the Python version and image: compare with what this version recorded BEFORE the core file is refreshed
            env_mismatch(recorded_env(proj, ver), o.ep.json_call("/coreinfo", payload={"modules": []}, timeout=60, tries=2))
        except Exception:
            pass
        core = SessionCore(o.ep, proj, ver, sid)
        core.refresh()  # manifest, dependency lock, I/O map -> .kenv/<version>/core.toml
        if proj.meta(ver).get("env_pending"):  # a rollback restored this version's packages: install them on this fresh kernel
            try:
                rebuild_env(o.ep, proj, ver, warn_env=False)  # the mismatch was already reported above
            except (KenvError, OSError) as e:
                say(c(f"[kenv] Could not rebuild the environment: {e}", "33"))
            proj.set_meta(ver, env_pending=None)
        worker = QueueWorker(o.ep, proj, core)  # answers kenv.cli() calls from code on the kernel
        worker.start()
        puller = LogPuller(o.ep, proj, ver, sid)  # kernel logs -> .kenv/<version>/logs/ while the session lives
        puller.start()
        say(f"Kernel working folder: /kaggle/working/{proj.kname}  (your project; relative paths in notebooks and scripts land here).")
        say("Use an absolute /kaggle/working/... path to write to the Kaggle root.")
        say("You are now inside the session. Try: kenv status | kenv gpu | kenv run file.py | kenv save | kenv --help")
        say("In notebooks and scripts on the kernel:  import kenv   (kenv.cli, kenv.time_start / time_end / time_output)")
        if lazy_box:
            say("Lazy local access: open('data/x.csv') etc. read your local files on demand (kenv.prefetch('data/') warms the cache; kenv.lazy_path(...) for C-level readers).")
        say("Keep this window open: it runs the kenv commands your code sends.")
        say(f"Kernel logs stream to .kenv/{ver}/logs/   (kenv logs --tail | --grep <text> | --since 10m)")
        say("Type `exit` (or close this terminal) to pull your changes back and delete the kernel.")
        say()
        try:
            open_shell(name, o.secret, owner=True, ep=o.ep, on_beat=lambda: (proj.touch_session(ver, sid), core.tick(), watch.check()))
        finally:
            worker.stop()
            puller.stop()
            for la in lazy_box:
                la.stop()
        # normal exit: bring the kernel's new/changed files home before the kernel is deleted
        if kernel_alive(o.ep):
            say("[kenv] Pulling changed files back into the project ...")
            try:
                pull_back(o.ep, proj)
            except (KenvError, OSError) as e:
                say(c(f"[kenv] Could not pull files back: {e}", "33"))
            core.refresh()  # final peaks, run history and dependency lock while the kernel is still there
        else:
            say("[kenv] The kernel is already gone - nothing to pull back.")
    say("[kenv] Session closed. Kaggle kernel deleted.")


def cmd_url(a=None):
    t = find_target()
    if not t:
        live = [st for st in list_states() if pid_alive(st.get("pid"))]
        say("No active kenv session in this terminal.")
        if live:
            say("Sessions running on this machine:")
            for st in live:
                say(f"  {st['name']}   ->  kenv -id {connect_str(st['name'], st['secret'])}")
        else:
            say("Start one with `kenv init`, or attach with `kenv -id <id>`.")
        return 1
    ep = Endpoint(*t)
    ep.resolve(force=True)
    print_info(ep)


def cmd_attach(a):
    if os.environ.get("KENV_SESSION"):
        raise KenvError("You are already inside a kenv session. Type `exit` first.")
    name, secret = parse_target(a.attach)
    ep = Endpoint(name, secret)
    ep.resolve(force=True)
    ep.json_call("/ping", timeout=25)
    banner()
    say(c(f"[kenv] Attached to {name}", "32;1"))
    say()
    print_info(ep)
    say()
    say("Type kenv commands here. `exit` leaves the session running - the window that started it owns the kernel.")
    say()
    open_shell(name, secret, owner=False)


def cmd_status(a):
    ep = need_ep()
    s = ep.json_call("/stats")
    m = ep.info
    cpu, mem = s["cpu"], s["mem"]
    mp = 100.0 * mem["used"] / max(mem["total"], 1)
    say(f"  Session : {c(ep.name, '1')}   up {s['uptime'] // 60} min   [{accel_label(m.get('gpu', 'none'))}]")
    say(f"  Kernel  : {m['ref']}")
    load = " ".join("%.2f" % x for x in cpu["load"])
    say(f"  CPU     : {bar(cpu['pct'])} {cpu['pct']:.0f}%   {cpu['cores']} cores   load {load}")
    say(f"  RAM     : {bar(mp)} {mp:.0f}%   {gb(mem['used'])} / {gb(mem['total'])}")
    for d in s["disk"]:
        dp = 100.0 * d["used"] / max(d["total"], 1)
        say(f"  Disk    : {bar(dp)} {dp:.0f}%   {gb(d['used'])} / {gb(d['total'])}   {d['path']}")
    if s["gpus"]:
        for i, g in enumerate(s["gpus"]):
            gp = 100.0 * g["used"] / max(g["total"], 1)
            say(f"  GPU {i}   : {bar(gp)} {gp:.0f}%   {g['name']}   {g['used']:.0f}/{g['total']:.0f} MiB   "
                f"util {g['util']:.0f}%   {g['temp']:.0f} C")
    else:
        say("  GPU     : none  (run `kenv gpu` to pick one)")


def cmd_gpu(a):
    ep = need_ep()
    info = ep.resolve(force=True)
    gpu_id = pick_gpu(a.args[0] if a.args else None)
    if (gpu_id or "none") == info.get("gpu", "none"):
        say("[kenv] Already on that accelerator.")
        return 0
    ctx = project_ctx(ep)
    say(c("Switching accelerator starts a NEW kernel: everything stored on the current one is lost.", "33"))
    if ctx:
        say(c("Your project files are pulled back first and re-synced to the new kernel; anything in ignored folders is not.", "33"))
    else:
        say(c("Pull what you need first with `kenv save <path>`.", "33"))
    say(c("Notebooks connected to the old Jupyter URL must reconnect.", "33"))
    if not confirm("Continue?"):
        say("Cancelled.")
        return 1
    if ctx:
        try:
            pull_back(ep, ctx[0])
        except (KenvError, OSError) as e:
            say(c(f"[kenv] Could not pull your files back: {e}", "33"))
            if not confirm("Switch anyway?"):
                say("Cancelled.")
                return 1
    require_cli()
    require_relay()
    user, gen = get_username(), info["gen"] + 1
    new_ref = ref_for(user, ep.name, gen)
    idle = a.idle or info.get("idle_min", 20)
    try:
        launch(user, ep.name, ep.secret, gen, gpu_id, idle, ep, a.startup,
               ctx[1].get("kproj") if ctx else None,
               [d["ref"] for d in ctx[0].datasets() if d.get("ref")] if ctx else None)
    except BaseException:
        say("[kenv] Switch failed - removing the new kernel; your current one keeps running.")
        delete_refs([new_ref])
        st = load_state(ep.name)
        if st and new_ref in st["refs"]:
            st["refs"].remove(new_ref)
            save_state(st)
        raise
    ep.info = ep.latest()
    delete_refs([info["ref"]], None)
    st = load_state(ep.name)
    if st:
        st["refs"] = [new_ref]
        save_state(st)
    say(c("[kenv] Accelerator switched", "32;1"))
    if ctx and ctx[1].get("version") and ctx[1].get("sid"):
        record_gpu_switch(ctx[0], ctx[1]["version"], ctx[1]["sid"], gpu_id)  # the GPU clock follows the switch
    print_info(ep)
    if ctx:  # the new kernel is empty: it needs the project again
        save_ksync(ep.name, {"pushed": {}, "kbase": {}})
        try:
            push_sync(ep, ctx[0])
        except KenvError as e:
            say(c(f"[kenv] Re-sync failed: {e} - retry with `kenv sync`.", "33"))


def cmd_exec(a, tail):
    if not tail:
        raise KenvError('Usage: kenv exec <command>   e.g.  kenv exec nvidia-smi   or   kenv exec "pip install torch"')
    cmd = tail[0] if len(tail) == 1 else " ".join(shlex.quote(x) for x in tail)
    return stream_exec(need_ep(), cmd)


def cmd_ls(a):
    p = shlex.quote(a.args[0]) if a.args else "."
    return stream_exec(need_ep(), f"ls -lah {p}")


def cmd_put(a):
    if not a.args:
        raise KenvError("Usage: kenv put <file-or-folder> [more ...] [--dest remote/dir]")
    ep = need_ep()
    data, n = zip_local(a.args)
    if len(data) > MAX_UPLOAD:
        raise KenvError("That upload is over ~95 MB, the limit of the tunnel. Use a Kaggle dataset, "
                        "or fetch it on the kernel:  kenv exec \"wget <url>\"")
    r = ep.json_call("/put", raw=data, query="?dest=" + urllib.parse.quote(a.dest or ""), timeout=600)
    say(f"[kenv] uploaded {n} file(s) -> {r['dest']}")


def cmd_save(a):
    ep = need_ep()
    ctx = project_ctx(ep)
    if not a.args:
        if not ctx:
            raise KenvError("Usage: kenv save <remote path> [more ...] [--to local/folder]\n"
                            "Paths are relative to the kernel's project folder. `kenv save .` grabs everything.")
        pull_back(ep, ctx[0], dest=a.out)  # new/changed files on the kernel -> the project
        return 0
    out = a.out or (str(ctx[0].root) if ctx else ".")  # relative paths mirror the project layout
    names = pull(ep, a.args, out)
    for nme in names[:30]:
        say(f"[kenv] saved {Path(out) / nme}")
    if len(names) > 30:
        say(f"[kenv] ... and {len(names) - 30} more")
    if not names:
        say("[kenv] nothing was saved")
        return 1
    if ctx and not a.out:
        mark_synced(ep, ctx[0], names)


def run_on(ep, script, script_args, out):
    p = Path(script).expanduser()
    if not p.is_file():
        raise KenvError(f"Script not found: {p}")
    if p.suffix not in (".py", ".sh"):
        raise KenvError("kenv run supports .py and .sh files. For notebooks, use the Jupyter URL (kenv --url).")
    ctx = project_ctx(ep)
    if ctx:  # project session: notebook outputs home first, then your latest local edits go up
        proj = ctx[0]
        pull_back(ep, proj, quiet=True)
        push_sync(ep, proj, quiet=True)
        try:
            name = p.resolve().relative_to(proj.root).as_posix()
            if proj.rules().path_ignored(name):
                raise ValueError
        except ValueError:  # outside the project (or ignored): send just this file to the project folder
            data, _ = zip_local([p])
            ep.json_call("/put", raw=data, query="?dest=", timeout=120)
            name = p.name
            snap = ep.json_call("/snapshot", payload={}, timeout=120)
            ks = load_ksync(ep.name)
            if name in snap:
                ks["kbase"][name] = snap[name]
                save_ksync(ep.name, ks)
        before = None
    else:
        name = p.name
        data, _ = zip_local([p])
        ep.json_call("/put", raw=data, query="?dest=", timeout=120)
        before = ep.json_call("/snapshot", payload={})
    runner = "python -u" if p.suffix == ".py" else "bash"
    cmd = f"{runner} {shlex.quote(name)} " + " ".join(shlex.quote(x) for x in script_args)
    say(f"[kenv] running on the kernel: {cmd.strip()}")
    say(c("-" * 60, "2"))
    t0 = time.time()
    rc = stream_exec(ep, cmd)
    say(c("-" * 60, "2"))
    say(f"[kenv] exit code {rc}")
    if ctx:
        record_script_run(ctx[0], ctx[1], name, t0, rc, stream_exec.tail)  # run history in core.toml
    if ctx:
        pull_back(ep, ctx[0], dest=out)
        return rc
    out = out or "kenv_output"
    after = ep.json_call("/snapshot", payload={})
    changed = [k for k, v in after.items() if before.get(k) != v]
    if changed:
        names = pull(ep, changed, out)
        say(f"[kenv] {len(names)} new/changed file(s) saved to {Path(out).resolve()}")
    else:
        say("[kenv] the script produced no new files")
    return rc


def cmd_run(a, script_args):
    if not a.args:
        raise KenvError("Usage: kenv run <script.py> [script args]     (kenv options go before the script)")
    script, out = a.args[0], a.out
    if find_target():
        return run_on(need_ep(), script, script_args, out)
    # No active session: one-shot mode - start a kernel, run, download, delete.
    if not Path(script).expanduser().is_file():
        raise KenvError(f"Script not found: {script}")
    gpu_id = pick_gpu() if a.gpu else None
    banner()
    with Owner(sanitize(a.name) if a.name else random_name(), gpu_id, a.idle or 20, a.startup) as o:
        o.start()
        return run_on(o.ep, script, script_args, out)


def cmd_stop(a):
    ep = need_ep()
    st = load_state(ep.name)
    ctx = project_ctx(ep)
    if ctx and kernel_alive(ep):
        say("[kenv] Pulling changed files back into the project ...")
        try:
            pull_back(ep, ctx[0])
        except (KenvError, OSError) as e:
            say(c(f"[kenv] Could not pull files back: {e}", "33"))
        if st and st.get("version"):
            SessionCore(ep, ctx[0], st["version"], st.get("sid")).refresh()
            try:
                pull_logs_once(ep, ctx[0], st["version"], wait=0)
            except Exception:
                pass
    kill_lazy(st)   # the tunnel to your files goes with the kernel
    refs = st.get("refs", []) if st else [ep.resolve(force=True)["ref"]]
    say("[kenv] Deleting the kernel ...")
    delete_refs(refs, ep)
    remove_state(ep.name)
    if ctx:
        ctx[0].close_session(st.get("version"), st.get("sid"), "clean exit", time.time())
    say("[kenv] Kernel deleted. Type `exit` to leave this shell.")


def sweep():
    require_cli()
    user = get_username()
    p = cli("kernels", "list", "--mine", "-s", "kenv-", "--csv")
    refs = [ln.split(",")[0] for ln in p.stdout.splitlines()[1:] if ln.startswith(f"{user}/kenv-")]
    if not refs:
        say("[kenv] No leftover kenv-* kernels.")
    for ref in refs:
        say(f"[kenv] {'deleted' if cli_delete(ref) else 'FAILED to delete'} {ref}")
    for st in list_states():  # forget sessions whose terminal is gone
        if not pid_alive(st.get("pid")):
            kill_lazy(st)   # a tunnel left behind by a killed kenv
            remove_state(st["name"])


# ----------------------------------------------------------------------------- versions, sync, datasets

def print_version_line(proj, v, active):
    m = proj.meta(v)
    used = proj.last_ts(v)
    tg = proj.tags_of(v)
    msg = m.get("message")
    say(f"  {'*' if active else ' '} {v:<4} {(m.get('name') or '-'):<16} {m['id']:<36} "
        f"{len(proj.sessions(v))} session(s)  {dur(proj.total_time(v)):>8} used  "
        f"{('last ' + ago(used)) if used else 'never used'}"
        + (f"  [{', '.join(tg)}]" if tg else "") + (f"  \"{msg[:40]}\"" if msg else ""))


def cmd_versions(a):
    proj = current_project()
    vs, act = proj.version_names(), proj.last_active()
    if not vs:
        say("No versions yet - `kenv init` creates v1.")
        return 0
    say(f"Project: {proj.root}      (* = active)")
    for v in vs:
        print_version_line(proj, v, v == act)


def cmd_new(a):
    proj = current_project()
    v = proj.new_version(a.args[0] if a.args else None)
    if proj.live_session():
        say(f"[kenv] Created {proj.label(v)}. A session is running, so the active version stays {proj.label(proj.last_active())}.")
    else:
        proj.set_active(v)
        say(f"[kenv] Created {proj.label(v)} and made it the active version.")


def cmd_activate(a):
    """kenv activate v2 | <name> | -id kv:... | <path to a .kenv folder> [version]"""
    if a.attach and not a.attach.lower().startswith("kv:"):
        raise KenvError("`kenv activate -id` takes a VERSION id (kv:...). To attach a terminal to a running "
                        "session use: kenv -id <attach-id>")
    target = a.attach or (a.args[0] if a.args else None)
    if not target:
        raise KenvError("Usage: kenv activate <v2 | version-name | -id kv:... | path/to/.kenv> [version]")
    rest = a.args if a.attach else a.args[1:]
    try:
        proj = current_project()
    except KenvError:
        proj = None
    err = None
    if proj is not None:
        try:
            v = proj.find_version(target)
            if proj.live_session():
                raise KenvError("A session is running for this project - a session keeps its version. "
                                "Exit it (or `kenv stop`), then run `kenv activate` and `kenv init`.")
            proj.set_active(v)
            say(f"[kenv] Active version: {proj.label(v)}   ({proj.meta(v)['id']})")
            return 0
        except KenvError as e:
            if "session is running" in str(e):
                raise
            err = e
    # not a version of this project: maybe a path to a .kenv folder (or to a project that has one)
    path = Path(target).expanduser()
    if path.name != PROJECT_DIR and (path / PROJECT_DIR).is_dir():
        path = path / PROJECT_DIR
    if path.is_dir() and path.name == PROJECT_DIR:
        other = Project(path.resolve().parent)
        v = other.find_version(rest[0]) if rest else other.last_active()
        if v is None:
            raise KenvError(f"{path} has no versions yet.")
        if other.live_session():
            raise KenvError("A session is running in that project; exit it before switching versions.")
        other.set_active(v)
        say(f"[kenv] {other.root}: active version is now {other.label(v)}")
        say(f"[kenv] Start working there with: cd \"{other.root}\" && kenv init")
        return 0
    if err is not None:
        raise err
    raise KenvError(f"'{target}' is not a .kenv folder, and this folder is not a kenv project. "
                    "Run `kenv init` in your project folder.")


def cmd_version_ref(a):
    """`kenv v2 -r name`, `kenv <name> -r new`, `kenv kv:... -r name`; without -r it shows the version."""
    proj = current_project()
    v = proj.find_version(a.command)
    if a.rename:
        old = proj.label(v)
        proj.rename(v, a.rename)
        say(f"[kenv] {old} is now {proj.label(v)}")
        return 0
    m = proj.meta(v)
    say(f"  Version : {proj.label(v)}{'   (active)' if v == proj.last_active() else ''}")
    say(f"  ID      : {m['id']}")
    say(f"  Created : {m.get('created', '?')}")
    if m.get("committed"):
        say(f"  Commit  : \"{m.get('message', '')}\"   branch {m.get('branch') or 'main'}, parent {m.get('parent') or '-'}, "
            f"{m.get('files', '?')} file(s){', auto-saved' if m.get('auto') else ''}")
        mt = metrics_of(proj, v)
        if mt:
            say("  Metrics : " + ", ".join(f"{k}={fmt_metric(x)}" for k, x in sorted(mt.items())))
    elif not proj.has_snapshot(v):
        say("  Commit  : none (never committed; `kenv commit` snapshots your files into the next version)")
    if proj.tags_of(v):
        say(f"  Tags    : {', '.join(proj.tags_of(v))}")
    sess = proj.sessions(v)
    say(f"  Sessions: {len(sess)}   total {dur(proj.total_time(v))}")
    for e in sess[-5:]:
        est = " (end estimated)" if e.get("estimated") else ""
        say(f"    {e.get('start', '?')}  {dur(e.get('duration_s')) if e.get('end') else 'running':>8}  "
            f"{e.get('ended_by') or '-'}{est}   [{e.get('session')}]")


def cmd_sync(a):
    ep = need_ep()
    ctx = project_ctx(ep)
    if not ctx:
        raise KenvError("This session was not started from a local project on this machine, so there is nothing to sync.")
    pull_back(ep, ctx[0])
    push_sync(ep, ctx[0])


def dataset_state(ref):
    p = cli("datasets", "status", ref)
    text = (p.stdout + p.stderr).strip().lower()
    if p.returncode != 0 or not text or any(w in text for w in ("404", "not found", "forbidden", "403", "could not find")):
        return None
    return text


def cmd_data(a):
    sub = a.args[0] if a.args else None
    if sub == "list":
        proj = current_project()
        ds = proj.datasets()
        if not ds:
            say("No datasets attached. Upload one with: kenv data push <folder>")
        for d in ds:
            say(f"  {d['ref']}   (from {d.get('folder', '?')})   -> /kaggle/input/{d['ref'].split('/')[1]}")
        return 0
    if sub != "push" or len(a.args) < 2:
        raise KenvError("Usage: kenv data push <folder>     upload a folder once as a private Kaggle dataset\n"
                        "       kenv data list               datasets attached to this project")
    folder = Path(a.args[1]).expanduser().resolve()
    if not folder.is_dir():
        raise KenvError(f"Not a folder: {folder}")
    require_cli()
    user = get_username()
    try:
        proj = current_project()
    except KenvError:
        proj = Project(Path.cwd().resolve())
    proj.ensure()
    slug = re.sub(r"[^a-z0-9]+", "-", f"kenv-{proj.kname}-{folder.name}".lower()).strip("-")[:50].strip("-")
    ref = f"{user}/{slug}"
    meta_path = folder / "dataset-metadata.json"
    if meta_path.exists():
        raise KenvError(f"{folder} already has a dataset-metadata.json. Move it away (kenv writes its own, "
                        "temporarily) or upload that dataset with the kaggle CLI yourself.")
    exists = dataset_state(ref) is not None
    say(f"[kenv] {'New version of' if exists else 'Creating private dataset'} {ref} from {folder} ...")
    say(c("[kenv] Subfolders are included. The first upload of a big folder can take a while.", "2"))
    write_json(meta_path, {"title": slug, "id": ref, "licenses": [{"name": "CC0-1.0"}]})
    try:
        cmd = ["kaggle", "datasets", "version", "-p", str(folder), "-m", "kenv data push", "-r", "zip"] if exists \
            else ["kaggle", "datasets", "create", "-p", str(folder), "-r", "zip"]
        rc = subprocess.run(cmd).returncode  # inherit the terminal so Kaggle's upload progress is visible
    finally:
        try:
            meta_path.unlink()
        except OSError:
            pass
    if rc != 0:
        raise KenvError("The kaggle CLI could not upload the dataset (see its message above).")
    t0 = time.time()
    while True:
        st = dataset_state(ref) or ""
        if "ready" in st:
            break
        if "error" in st or time.time() - t0 > 900:
            raise KenvError(f"Kaggle did not finish processing {ref} (status: {st or 'unknown'}). "
                            f"Check https://www.kaggle.com/datasets/{ref}")
        say(f"[kenv] Kaggle is processing the dataset ({st or 'waiting'}) ...")
        time.sleep(15)
    try:
        rel = folder.relative_to(proj.root).as_posix()
    except ValueError:
        rel = folder.name
    proj.add_dataset(ref, rel)
    say(c(f"[kenv] Dataset ready: {ref}", "32;1"))
    say(f"[kenv] It appears under /kaggle/input/{slug} in every new session (listed in .kenv/datasets.json).")
    say("[kenv] Add the folder to .kenvignore so it is not also synced through the tunnel. A running kernel "
        "does not get it - start the next session to attach it.")


def cmd_cred(a=None):
    banner()
    sysname = {"Darwin": "macOS"}.get(platform.system(), platform.system())
    say(f"Detected: {c(sysname, '1')} {platform.release()} ({platform.machine()}), Python {platform.python_version()}")
    say()
    ok, bad = c("[ok]", "32;1"), c("[missing]", "31;1")
    missing = []

    py_ok = sys.version_info >= (3, 8)
    say(f"  {ok if py_ok else bad}  Python 3.8+")
    if not py_ok:
        missing.append("python")

    kpath = shutil.which("kaggle")
    kver = ""
    if kpath:
        try:
            kver = (cli("--version", timeout=20).stdout or "").strip()
        except Exception:
            pass
    say(f"  {ok if kpath else bad}  kaggle CLI (pip package `kaggle`)  {kver}")
    if not kpath:
        missing.append("kaggle")
    elif cli("kernels", "delete", "--help").returncode != 0:
        say(f"  {bad}  this kaggle CLI is too old (no `kernels delete`) - upgrade it")
        missing.append("kaggle")

    if os.name != "nt":
        has_bash = bool(shutil.which("bash"))
        say(f"  {ok if has_bash else bad}  bash (used for the kenv shell)")
        if not has_bash:
            missing.append("bash")
    try:
        with http(f"{RELAY}/v1/health", timeout=8) as r:
            r.read()
        say(f"  {ok}  relay reachable ({RELAY})")
    except Exception:
        say(f"  {bad}  relay NOT reachable ({RELAY}) - needed to find your kernel's URLs")

    cr = find_creds()
    say(f"  {ok if cr else bad}  Kaggle credentials" + (f"  ({cr[0]}, from {cr[1]})" if cr else ""))
    say(c("  optional: VS Code / Cursor with the Jupyter extension, to use a session as a notebook kernel", "2"))
    say()

    if missing:
        hints = {
            "Linux": ["sudo apt install python3 python3-pip", "pip install -U kaggle"],
            "macOS": ["brew install python", "pip3 install -U kaggle"],
            "Windows": ["winget install Python.Python.3.12", "py -m pip install -U kaggle"],
        }.get(sysname, ["install Python 3.8+", "pip install -U kaggle"])
        box(f"Install what is missing ({sysname})", hints)
        say()

    if not cr:
        here = lambda k: c("   <- your system", "36") if k == sysname else ""
        box("Add your Kaggle credentials", [
            "Either place your kaggle.json (Kaggle -> Settings -> Create New Token) at:",
            "",
            "  Linux/macOS: ~/.kaggle/kaggle.json" + here("Linux") + here("macOS"),
            r"  Windows: C:\Users\<you>\.kaggle\kaggle.json" + here("Windows"),
            "",
            "or set environment variables:",
            "",
            "  export KAGGLE_USERNAME=your_username",
            "  export KAGGLE_KEY=your_api_key",
            "",
            "  (Windows cmd: setx KAGGLE_USERNAME your_username   /   setx KAGGLE_KEY your_api_key)",
            "  (Linux/macOS: also run  chmod 600 ~/.kaggle/kaggle.json)",
        ])
        return 1
    if kpath:
        try:
            p = cli("kernels", "list", "--mine", timeout=45)
            if p.returncode == 0:
                say(f"  {ok}  Kaggle accepted your credentials. You are ready: run `kenv init`")
            else:
                say(f"  {bad}  Kaggle rejected them: {(p.stderr or p.stdout).strip()[:200]}")
                return 1
        except Exception as e:
            say(f"  (could not verify with Kaggle right now: {e})")
    return 1 if missing else 0



# ----------------------------------------------------------------------------- Phase 4: portability and safety
#
# export / import / rebuild / convert, the approximate Kaggle quota, secret scanning, missing imports,
# dependency conflicts and `kenv doctor`. The secret scan is a SAFETY NET, not a guarantee.

QUOTA_FILE = Path.home() / ".kenv" / "quota.json"

# ---- secret scanning (patterns + entropy)

# High-confidence patterns first; a hit here is very likely a real credential.
SECRET_PATTERNS = [
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("AWS secret key", re.compile(r"(?i)\baws[_-]?secret[_-]?(?:access[_-]?)?key\b\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b")),
    ("GitHub fine-grained", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{82}\b")),
    ("OpenAI key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("Anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    ("Stripe key", re.compile(r"\bsk_live_[A-Za-z0-9]{24,}\b")),
    ("Kaggle key", re.compile(r"(?i)\bkaggle[_-]?(?:key|api[_-]?key)\b\s*[:=]\s*['\"]?([A-Za-z0-9]{32,})")),
    ("Private key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("Bearer token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]{30,}")),
    ("Generic assignment",
     re.compile(r"(?i)\b(?:api[_-]?key|apikey|secret|password|passwd|token)\b\s*[:=]\s*['\"]([A-Za-z0-9\-._~+/=]{24,})['\"]")),
]
ENTROPY_MIN = 4.5           # bits/char; a random 32-byte base64 string is ~5.2, an English word is ~3
ENTROPY_MIN_LEN = 24
ENTROPY_TOKEN = re.compile(r"(?<![A-Za-z0-9+/=_\-])[A-Za-z0-9+/=_\-]{%d,128}(?![A-Za-z0-9+/=_\-])" % ENTROPY_MIN_LEN)
SECRETS_ALLOWLIST = "secrets-allowlist"


def _entropy(s):
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = float(len(s))
    return -sum((v / n) * math.log2(v / n) for v in counts.values())


def _looks_like_word(s):
    return bool(re.fullmatch(r"[A-Za-z]+", s)) or bool(re.fullmatch(r"[a-z0-9]+_[a-z0-9_]+", s))


def secret_fp(text):
    """12-hex fingerprint of a flagged string: what the allowlist stores instead of the secret itself."""
    return hashlib.sha256(str(text).encode("utf-8", "replace")).hexdigest()[:12]


def _allowed(text, allow):
    return secret_fp(text) in allow


def scan_secret_text(text, allow=()):
    """-> [(pattern_name, matched_text, line_no), ...]. `allow` is a list of literal substrings to skip."""
    if not text or len(text) > 2 * 1024 * 1024:
        return []
    lines = text.splitlines()
    findings, seen = [], []

    def line_of(pos):
        return text.count("\n", 0, pos) + 1

    for name, rx in SECRET_PATTERNS:
        for m in rx.finditer(text):
            hit = m.group(0)
            if _allowed(hit, allow):
                continue
            ln = line_of(m.start())
            findings.append((name, hit, ln))
            seen.append((m.start(), m.end()))
    for m in ENTROPY_TOKEN.finditer(text):
        tok = m.group(0)
        if _allowed(tok, allow) or _looks_like_word(tok) or text[max(0, m.start() - 3):m.start()] == "kv:":
            continue  # kv:<id> is kenv's own version id: random-looking, but not a secret
        if any(s <= m.start() < e for s, e in seen):
            continue
        if _entropy(tok) < ENTROPY_MIN:
            continue
        findings.append(("high-entropy string", tok, line_of(m.start())))
    # dedupe by (line, first 20 chars)
    out, keys = [], set()
    for name, hit, ln in findings:
        if 0 < ln <= len(lines) and "kenv:allow" in lines[ln - 1]:
            continue  # the line carries a `# kenv:allow` marker
        k = (ln, hit[:20])
        if k in keys:
            continue
        keys.add(k)
        out.append((name, hit, ln))
    return out


def scan_secret_file(path, allow=()):
    try:
        if Path(path).stat().st_size > 4 * 1024 * 1024:
            return []
        raw = Path(path).read_bytes()
    except OSError:
        return []
    if b"\0" in raw[:8192]:
        return []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", "replace")
    return scan_secret_text(text, allow)


def secret_allowlist(proj):
    """Set of allowed fingerprints (12 hex chars per line; `#` comments). Anything else in the file is ignored."""
    try:
        lines = (proj.kdir / SECRETS_ALLOWLIST).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return set()
    out = set()
    for ln in lines:
        t = ln.split("#", 1)[0].strip().lower()
        if re.fullmatch(r"[0-9a-f]{12}", t):
            out.add(t)
    return out


def _scan_files(proj, rels):
    allow = secret_allowlist(proj)
    out = []
    for rel in rels:
        hits = scan_secret_file(proj.root / rel, allow)
        if hits:
            out.append((rel, hits))
    return out

def _mask(s):
    s = str(s)
    return (s[:4] + "..." + s[-2:]) if len(s) > 12 else "..."


def redact_secrets(text):
    """Pattern hits (no entropy guessing) -> <redacted>. Used on text that is about to be written into .kenv."""
    for _name, rx in SECRET_PATTERNS:
        text = rx.sub("<redacted>", text)
    return text


def report_secret_findings(findings, header="[kenv] Possible secrets found:"):
    if not findings:
        return
    say(c(header, "33;1"))
    for rel, hits in findings:
        say(f"  {c(rel, '1')}")
        for name, hit, ln in hits[:5]:
            say(f"    line {ln}: {name}  ({_mask(hit)})  fingerprint {secret_fp(hit)}")
        if len(hits) > 5:
            say(f"    ... and {len(hits) - 5} more")
    say("  Move the value out of the file (Kaggle Secrets + kaggle_secrets, or an environment variable). A false alarm: add\n"
        "  `# kenv:allow` to that line, or run `kenv scan allow` (it stores the 12-character fingerprint, never the text).")


def secret_gate(proj, rels, action):
    """Raise KenvError when any of the project files `rels` look like they hold a secret."""
    hits = _scan_files(proj, rels)
    if hits:
        report_secret_findings(hits)
        raise KenvError(f"{action} stopped: {sum(len(h) for _, h in hits)} possible secret(s) in {len(hits)} file(s). "
                        "Nothing was written. Kenv's scan is a safety net, not a guarantee - always review what you share.")


def kenv_meta_files(proj, limit=400):
    """Files under .kenv that get committed or shared (core files, configs, locks, logs)."""
    out = []
    for v in proj.version_names():
        for rel in (CORE_FILE, "config.json", "meta.json", "deps.lock", f"{LOG_DIR}/run.log", f"{LOG_DIR}/errors.log"):
            if (proj.kdir / v / rel).is_file():
                out.append(f"{PROJECT_DIR}/{v}/{rel}")
    return out[:limit]


def cmd_secret_scan(a):
    """kenv secret-scan [paths...] [--staged]"""
    proj = current_project()
    if a.staged:
        p = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM", "--relative"],
                           cwd=str(proj.root), capture_output=True, text=True)
        if p.returncode != 0:
            raise KenvError("`git diff --cached` failed (is this folder inside a git repository?)")
        rels = [x.strip() for x in p.stdout.splitlines() if x.strip()]
    elif a.args:
        rels = [str(Path(x)).replace("\\", "/") for x in a.args]
    else:
        rels = sorted(tracked_files(proj)) + kenv_meta_files(proj)
    rels = [r for r in rels if (proj.root / r).is_file() and not r.startswith(f"{PROJECT_DIR}/{SECRETS_ALLOWLIST}")]
    hits = _scan_files(proj, rels)
    if not hits:
        say(f"[kenv] no secret-looking strings in {len(rels)} file(s). (Pattern + entropy scan: a safety net, not a guarantee.)")
        return 0
    report_secret_findings(hits)
    return 1


def cmd_scan_allow(a):
    """kenv scan allow [paths | <fingerprint> ...]: allow current findings; only 12-hex fingerprints are stored"""
    proj = current_project()
    args = a.args[1:]
    known = secret_allowlist(proj)
    if args and all(re.fullmatch(r"[0-9a-fA-F]{12}", x) for x in args):
        fps = [x.lower() for x in args]
    else:
        rels = [str(Path(x)).replace("\\", "/") for x in args] or sorted(tracked_files(proj)) + kenv_meta_files(proj)
        hits = _scan_files(proj, [r for r in rels if (proj.root / r).is_file()])
        fps = sorted({secret_fp(h) for _, hs in hits for _n, h, _l in hs})
        if not fps:
            say("[kenv] no findings, nothing to allow")
            return 0
        report_secret_findings(hits, "[kenv] These findings would be allowed:")
        if sys.stdin.isatty() and not confirm("Allow all of them? (only their fingerprints are stored)"):
            return 1
    new = [f for f in fps if f not in known]
    p = proj.kdir / SECRETS_ALLOWLIST
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        if not p.stat().st_size:
            f.write("# kenv secrets allowlist: 12-character fingerprints only (a hash of the flagged text), never the text itself\n")
        f.writelines(x + "\n" for x in new)
    say(f"[kenv] {len(new)} fingerprint(s) added to .kenv/{SECRETS_ALLOWLIST}" + (f" ({len(fps) - len(new)} were already there)" if len(fps) != len(new) else ""))
    return 0


def cmd_scan(a):
    """kenv scan [paths] [--staged] | scan allow [...] | scan hook | scan unhook"""
    sub = a.args[0].lower() if a.args else ""
    if sub == "allow":
        return cmd_scan_allow(a)
    if sub in ("hook", "unhook"):
        a.args = ["install" if sub == "hook" else "remove"]
        return cmd_secret_hook(a)
    return cmd_secret_scan(a)


HOOK_MARK = "# kenv-secret-hook"


def cmd_secret_hook(a):
    """kenv secret-hook install | remove | status"""
    proj = current_project()
    act = (a.args[0] if a.args else "status").lower()
    gitdir = proj.root / ".git"
    if not gitdir.is_dir():
        raise KenvError("No .git folder in the project root, so there is no git repository to hook.")
    hook = gitdir / "hooks" / "pre-commit"
    mine = hook.is_file() and HOOK_MARK in hook.read_text(encoding="utf-8", errors="replace")
    if act == "status":
        say("[kenv] the pre-commit secret hook is " + ("installed" if mine else "not installed"))
        return 0
    if act == "install":
        if hook.exists() and not mine and not a.force:
            raise KenvError("A different pre-commit hook already exists. Keep it, or replace it with `kenv secret-hook install --force`.")
        hook.parent.mkdir(parents=True, exist_ok=True)
        script = (f"#!/bin/sh\n{HOOK_MARK}\ncd \"{proj.root.as_posix()}\" || exit 1\n"
                  f"exec \"{Path(sys.executable).as_posix()}\" \"{Path(__file__).resolve().as_posix()}\" scan --staged\n")
        hook.write_bytes(script.encode("utf-8"))
        try:
            hook.chmod(0o755)
        except OSError:
            pass
        say("[kenv] installed .git/hooks/pre-commit: commits with secret-looking strings are blocked (bypass once: git commit --no-verify)")
        return 0
    if act in ("remove", "uninstall"):
        if mine:
            hook.unlink()
            say("[kenv] removed the pre-commit secret hook")
        else:
            say("[kenv] no kenv hook to remove")
        return 0
    raise KenvError("Usage: kenv scan hook | unhook   (or: kenv secret-hook install | remove | status)")


# ---- Kaggle quota (an ESTIMATE from kenv's own session logs; Kaggle has no official API for it)

QUOTA_DEFAULT = {"weekly_gpu_hours": 30.0, "warn": [80, 95], "projects": []}


def quota_cfg():
    d = read_json(QUOTA_FILE, {})
    d = d if isinstance(d, dict) else {}
    cfg = dict(QUOTA_DEFAULT)
    try:
        cfg["weekly_gpu_hours"] = max(0.1, float(d.get("weekly_gpu_hours", cfg["weekly_gpu_hours"])))
    except (TypeError, ValueError):
        pass
    w = d.get("warn")
    if isinstance(w, list) and w and all(isinstance(x, (int, float)) and 0 < x <= 100 for x in w):
        cfg["warn"] = sorted(int(x) for x in w)
    cfg["projects"] = [p for p in d.get("projects", []) if isinstance(p, str)]
    return cfg


def quota_save(cfg):
    QUOTA_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_json(QUOTA_FILE, cfg)


def quota_register(proj):
    """Remember this project so the estimate also counts its GPU sessions when you run `kenv quota` elsewhere."""
    try:
        cfg = quota_cfg()
        if str(proj.root) not in cfg["projects"]:
            cfg["projects"].append(str(proj.root))
            quota_save(cfg)
    except OSError:
        pass


def _gpu_segments(e, now):
    """[(start, end)] spans of one session that ran on a GPU. A mid-session `kenv gpu` switch closes a span
    (gpu_log) and opens a new one (gpu, gpu_since), so the GPU clock starts and stops at the right moments."""
    out = []
    for sg in e.get("gpu_log") or []:
        a_, b_ = from_iso(sg.get("from")), from_iso(sg.get("to"))
        if a_ and b_ and sg.get("gpu") not in (None, "", "none"):
            out.append((a_, b_))
    if e.get("gpu") not in (None, "", "none"):
        start = from_iso(e.get("gpu_since")) or from_iso(e.get("start"))
        if start:
            end = from_iso(e.get("end")) or (now if pid_alive(e.get("pid")) else (from_iso(e.get("last_seen")) or start))
            out.append((start, end))
    return out


def record_gpu_switch(proj, v, sid, new_gpu):
    sess = proj.sessions(v)
    for e in sess:
        if e.get("sid") == sid and not e.get("end"):
            e.setdefault("gpu_log", []).append({"gpu": e.get("gpu") or "none", "from": e.get("gpu_since") or e.get("start"), "to": iso()})
            e.update(gpu=new_gpu or "none", gpu_since=iso())
            proj._save_sessions(v, sess)
            return


def gpu_seconds(cfg, now=None, window=7 * 86400):
    now = now or time.time()
    lo, total = now - window, 0.0
    for root in cfg["projects"]:
        p = Project(root)
        if not p.kdir.is_dir():
            continue
        for v in p.version_names():  # imported sessions live in sessions-archived.json and never count here
            for e in p.sessions(v):
                for a_, b_ in _gpu_segments(e, now):
                    total += max(0.0, min(b_, now) - max(a_, lo))
    return total


def quota_status(now=None):
    cfg = quota_cfg()
    used = gpu_seconds(cfg, now) / 3600.0
    lim = cfg["weekly_gpu_hours"]
    return {"used_h": used, "limit_h": lim, "left_h": max(0.0, lim - used), "pct": used / lim * 100.0, "warn": cfg["warn"]}


def quota_line(q):
    return (f"GPU this week (approximate, rolling 7 days, from kenv's own logs): {q['used_h']:.1f} h of {q['limit_h']:g} h "
            f"used, about {q['left_h']:.1f} h left ({q['pct']:.0f}%)")


class QuotaWatch:
    """Prints each warning threshold once per kenv process: at session start and while a session runs."""

    def __init__(self):
        self.said, self.t, self.active = set(), 0.0, None

    def check(self, force=False):
        now = time.time()
        if not force and now - self.t < 300:
            return
        self.t = now
        if self.active and not force and not self.active():
            return  # the session is not on a GPU right now
        try:
            q = quota_status()
        except Exception:
            return
        for th in q["warn"]:
            if q["pct"] >= th and th not in self.said:
                self.said.add(th)
                say(c(f"\n[kenv] {quota_line(q)} - past your {th}% warning. Kaggle's real limit may differ; check kaggle.com/settings.", "33;1"))


def cmd_quota(a):
    """kenv quota | quota set limit <h> | quota set warn <a,b> | quota reset   (also: quota set --limit <h> --warn a,b)"""
    cfg = quota_cfg()
    args = [x.lower() if i == 0 else x for i, x in enumerate(a.args)]
    if args[:1] == ["reset"]:
        quota_save({**QUOTA_DEFAULT, "projects": cfg["projects"]})
        say("[kenv] quota settings reset: 30 h per week, warnings at 80% and 95%")
        return 0
    if args[:1] == ["set"]:
        limit, warn, rest = a.limit, a.warn, args[1:]
        while rest:
            key = rest[0].lower()
            if key in ("limit", "warn") and len(rest) > 1:
                if key == "limit":
                    try:
                        limit = float(rest[1])
                    except ValueError:
                        raise KenvError("limit takes a number of hours, e.g. kenv quota set limit 30")
                else:
                    warn = rest[1]
                rest = rest[2:]
            else:
                raise KenvError("Usage: kenv quota set limit <hours> | kenv quota set warn 80,95")
        if limit is None and not warn:
            raise KenvError("Usage: kenv quota set limit <hours> | kenv quota set warn 80,95 | kenv quota reset")
        if limit is not None:
            if limit <= 0:
                raise KenvError("the limit must be a positive number of hours")
            cfg["weekly_gpu_hours"] = float(limit)
        if warn:
            try:
                w = sorted({int(x) for x in warn.split(",") if x.strip()})
            except ValueError:
                raise KenvError("warn takes percentages like 80,95")
            if not w or any(not 0 < x <= 100 for x in w):
                raise KenvError("warn percentages must be between 1 and 100")
            cfg["warn"] = w
        quota_save(cfg)
        say(f"[kenv] weekly GPU limit {cfg['weekly_gpu_hours']:g} h, warnings at {', '.join(str(x) + '%' for x in cfg['warn'])}")
        return 0
    try:
        quota_register(current_project())
    except KenvError:
        pass
    q = quota_status()
    say(quota_line(q))
    say(f"  {bar(min(100, q['pct']))}   warnings at {', '.join(str(x) + '%' for x in q['warn'])}")
    say("  This is an ESTIMATE: kenv counts the GPU sessions it logged in the projects it knows about. Kaggle has no\n"
        "  official API for your quota, other GPU use (notebooks in the browser) is not included, and the week may\n"
        "  reset on a different day. Set your own limit: kenv quota set limit 30")
    return 1 if q["pct"] >= 100 else 0


def quota_start_check(watch):
    q = quota_status()
    say(c(f"[kenv] {quota_line(q)}", "33" if q["pct"] >= min(q["warn"]) else "0"))
    watch.check(force=True)
    if q["pct"] >= 100 and sys.stdin.isatty() and not confirm("You are at or past your GPU limit. Start a GPU session anyway?"):
        raise KenvError("Cancelled before starting a GPU kernel.")


# ---- missing imports and dependency conflicts

IMPORT_ALIASES = {"cv2": "opencv-python", "PIL": "pillow", "sklearn": "scikit-learn", "yaml": "pyyaml", "bs4": "beautifulsoup4",
                  "skimage": "scikit-image", "dateutil": "python-dateutil", "dotenv": "python-dotenv", "attr": "attrs",
                  "jwt": "pyjwt", "serial": "pyserial", "OpenSSL": "pyopenssl", "Crypto": "pycryptodome", "git": "gitpython",
                  "umap": "umap-learn", "google": "google-api-python-client", "MySQLdb": "mysqlclient", "fitz": "pymupdf"}


def _stdlib_names():
    names = set(getattr(sys, "stdlib_module_names", ())) | set(sys.builtin_module_names)
    if not getattr(sys, "stdlib_module_names", None):  # Python < 3.10
        import sysconfig
        d = sysconfig.get_paths().get("stdlib") or ""
        for base in (d, os.path.join(d, "lib-dynload")):
            try:
                names |= {x.split(".")[0] for x in os.listdir(base) if not x.startswith("_") or x.endswith(".py")}
            except OSError:
                pass
    return names


def missing_imports(proj, v):
    """-> (list of (import name, likely distribution), lock_known). Imports the code uses that the version's lock lacks."""
    mods, _ = scan_code(proj)
    lock = read_lock(proj.kdir / v / "deps.lock") if v else {}
    if not lock:
        return [], False
    have = {norm_pkg(k) for k in lock}
    std = _stdlib_names()
    out = []
    for m in mods:
        if m in std or norm_pkg(m) in have or norm_pkg(IMPORT_ALIASES.get(m, m)) in have or norm_pkg(m.replace("_", "-")) in have:
            continue
        out.append((m, IMPORT_ALIASES.get(m, m)))
    return out, True


def cmd_deps(a):
    """kenv deps check | fix | conflicts"""
    proj = current_project()
    sub = (a.args[0] if a.args else "check").lower()
    w, live = working_version(proj)
    if not w:
        raise KenvError("This project has no version yet. Run `kenv init` first.")
    if sub == "add":
        req = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*((?:(?:==|>=|<=|~=|!=|>|<)\s*[^\s;,]+\s*,?\s*)+)?$")
        add = {}
        for t in a.args[1:]:
            m = req.match(t.strip())
            if not m:
                raise KenvError(f"'{t}' is not a package requirement (examples: rich, rich==13.7.1, rich>=13)")
            add[m.group(1)] = re.sub(r"\s+", "", m.group(2) or "*").rstrip(",")
        if not add:
            raise KenvError("Usage: kenv deps add <package[==version]> [more ...]")

        def fn(doc):
            d = doc.setdefault("dependencies", {})
            d["declared"] = dict(sorted({**(d.get("declared") or {}), **add}.items()))
        CoreStore(proj, w).update(fn)
        say(f"[kenv] declared in {proj.label(w)}: " + ", ".join(k + ("" if x == "*" else x) for k, x in sorted(add.items()))
            + "\n  Commits carry them forward and `kenv rebuild` installs them on the kernel.")
        return 0
    if sub in ("check", "fix"):
        miss, known = missing_imports(proj, w)
        if not known:
            say(c(f"[kenv] {proj.label(w)} has no dependency lock yet (it is written during a session or a commit), so imports cannot be compared.", "33"))
            return 0
        if not miss:
            say(f"[kenv] every import in your code is covered by {proj.label(w)}'s dependency lock")
            return 0
        say(c(f"[kenv] {len(miss)} import(s) are not in {proj.label(w)}'s dependency lock:", "33"))
        for m, dist in miss:
            say(f"  {m}" + (f"   (package: {dist})" if dist != m else ""))
        if sub == "check":
            say("Add them with `kenv deps fix`.")
            return 1
        if not live:
            say("Start a session (`kenv init`) and run `kenv deps fix` inside it: kenv looks the installed versions up on the kernel.\n"
                "Or install by hand: pip install " + " ".join(d for _, d in miss))
            return 1
        if sys.stdin.isatty() and not confirm("Add them to the dependency lock?"):
            return 1
        info = Endpoint(live["name"], live["secret"]).json_call("/coreinfo", payload={"modules": [m for m, _ in miss]}, timeout=60, tries=2)
        add, gone = {}, []
        for m, dist in miss:
            spec = (info.get("detected") or {}).get(m, "")
            mm = re.match(r"^(.+?)==(\S+)$", spec)
            if mm and mm.group(2) != "?":
                add[mm.group(1)] = mm.group(2)
            else:
                gone.append((m, dist))
        if add:
            lock = read_lock(proj.kdir / w / "deps.lock")
            lock.update(add)
            write_lock(proj.kdir / w / "deps.lock", lock)

            def fn(doc):
                d = doc.setdefault("dependencies", {})
                d["lock"] = dict(sorted({**(d.get("lock") or {}), **add}.items()))
                det = dict(d.get("detected") or {})
                det.update({m: f"{k}=={x}" for m, dist in miss for k, x in add.items() if norm_pkg(k) == norm_pkg(dist) or k == m})
                d["detected"] = det
            CoreStore(proj, w).update(fn)
            say(f"[kenv] added {len(add)} package(s) to the lock: " + ", ".join(f"{k}=={x}" for k, x in sorted(add.items())))
        for m, dist in gone:
            say(c(f"[kenv] '{m}' is not installed on the kernel - install it first: kenv exec pip install {dist}", "33"))
        return 1 if gone else 0
    if sub == "conflicts":
        ep = need_ep()
        say("[kenv] running `pip check` on the kernel ...")
        rc = stream_exec(ep, "python -m pip check")
        tail = getattr(stream_exec, "tail", "")
        n = len([ln for ln in tail.splitlines() if " requires " in ln or "has requirement" in ln])
        if rc == 0:
            say(c("[kenv] no broken requirements on the kernel", "32"))
            return 0
        say(c(f"[kenv] pip check reported conflicts (about {n} line(s) above). Conflicts inside Kaggle's own image are not caused by "
              "your code; the ones that matter are for packages you installed or locked yourself.", "33"))
        return 1
    raise KenvError("Usage: kenv deps check | fix | add <package> | conflicts")


# ---- environment: compare with what Kaggle provides now, rebuild from the lock

def env_mismatch(cfg, info):
    """Warn when the kernel's Python / Docker image differ from what the version recorded. Kaggle controls both."""
    warns = []
    old_py, new_py = str(cfg.get("python") or ""), str((info or {}).get("python") or "")
    if old_py and new_py and old_py not in ("unknown", "") and old_py.split(".")[:2] != new_py.split(".")[:2]:
        warns.append(f"Python differs: recorded {old_py}, Kaggle provides {new_py} now")
    old_im, new_im = str(cfg.get("docker_image") or ""), str((info or {}).get("image") or "")
    if old_im and new_im and "unknown" not in (old_im, new_im) and old_im != new_im:
        warns.append(f"Docker image differs: recorded {old_im}, Kaggle provides {new_im} now")
    for w in warns:
        say(c(f"[kenv] WARNING: {w}. Kaggle controls both, so locked packages may not install or behave the same.", "33;1"))
    return warns


def cmd_rebuild(a):
    """kenv rebuild [version] [--dry-run] [--all|--full]: install the version's locked and declared packages on the kernel"""
    proj = current_project()
    v = proj.find_version(a.args[0]) if a.args else working_version(proj)[0]
    if not v:
        raise KenvError("This project has no version yet.")
    ep = need_ep()
    n = rebuild_env(ep, proj, v, full=a.all, dry_run=a.dry_run)
    if not a.dry_run:
        say(f"[kenv] rebuild finished for {proj.label(v)}" + (f": {n} package(s) installed" if n else ""))
    return 0


def recorded_env(proj, v):
    cfg = dict(CoreStore(proj, v).load().get("manifest") or {})
    cfg.update({k: x for k, x in read_json(proj.kdir / v / "config.json", {}).items() if k in ("python", "docker_image") and x})
    return cfg


# ---- convert: requirements.txt / environment.yml / kernel-metadata.json

def _direct_specs(proj, v, everything=False):
    doc = CoreStore(proj, v).load()
    lock = read_lock(proj.kdir / v / "deps.lock") or dict((doc.get("dependencies") or {}).get("lock") or {})
    declared = dict((doc.get("dependencies") or {}).get("declared") or {})
    dec = {k: (str(x)[2:] if str(x).startswith("==") else "*") for k, x in declared.items()}
    if everything:
        return {**dec, **lock}
    out = dict(dec)
    for spec in ((doc.get("dependencies") or {}).get("detected") or {}).values():
        m = re.match(r"^(.+?)==(\S+)$", str(spec))
        if m and m.group(2) != "?":
            out[m.group(1)] = m.group(2)
    return out or dict(lock)


def to_requirements(specs):
    return "".join((f"{k}=={x}\n" if x != "*" else f"{k}\n") for k, x in sorted(specs.items(), key=lambda kv: kv[0].lower()))


def to_environment_yml(specs, name, python):
    py = ".".join(str(python).split(".")[:2]) if python and python != "unknown" else "3.10"
    body = f"name: {re.sub(r'[^A-Za-z0-9_.-]+', '-', name) or 'kenv-project'}\nchannels:\n  - conda-forge\ndependencies:\n  - python={py}\n  - pip\n"
    if specs:
        body += "  - pip:\n" + "".join((f"      - {k}=={x}\n" if x != "*" else f"      - {k}\n") for k, x in sorted(specs.items(), key=lambda kv: kv[0].lower()))
    return body


def parse_requirements(text):
    """-> (pinned {name: version}, unpinned [name])"""
    pinned, loose = {}, []
    for ln in text.splitlines():
        ln = ln.split("#", 1)[0].strip()
        if not ln or ln.startswith(("-", "git+", "http")):
            continue
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*==\s*([^\s;,]+)", ln)
        if m:
            pinned[m.group(1)] = m.group(2)
            continue
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", ln)
        if m:
            loose.append(m.group(1))
    return pinned, loose


def parse_environment_yml(text):
    """Small, dependency-free reader for the usual environment.yml shape (conda list + a pip: sub-list)."""
    pinned, loose, py, in_pip = {}, [], None, False
    for raw in text.splitlines():
        ln = raw.rstrip()
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if re.match(r"^\S", ln):  # a top-level key
            in_pip = False
            continue
        if not s.startswith("-"):
            continue
        item = s[1:].strip()
        if item.startswith("pip:"):
            in_pip = True
            continue
        if in_pip or raw.startswith("      "):
            p, lo = parse_requirements(item)
            pinned.update(p)
            loose += lo
            continue
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*=\s*=?\s*([0-9][^\s=]*)", item)
        if m and m.group(1) == "python":
            py = m.group(2)
        elif m:
            pinned[m.group(1)] = m.group(2).rstrip("=")  # conda pins: name=1.2.3 (build strings are dropped)
        else:
            m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", item)
            if m and m.group(1) not in ("pip", "python"):
                loose.append(m.group(1))
    return pinned, loose, py


def kernel_metadata_for(proj, v, user):
    cfg = read_json(proj.kdir / v / "config.json", {})
    slug = re.sub(r"[^a-z0-9-]+", "-", proj.kname.lower()).strip("-") or "kenv-project"
    gpu = cfg.get("accelerator_id") not in (None, "", "none")
    return {"id": f"{user}/{slug}", "title": slug, "code_file": "main.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": bool(gpu), "enable_tpu": False, "enable_internet": True,
            "dataset_sources": list(cfg.get("datasets") or []), "competition_sources": [], "kernel_sources": [],
            "model_sources": []}


def cmd_convert(a):
    """kenv convert --to requirements.txt|environment.yml|kernel-metadata.json [version] [--out file] [--all]
       kenv convert --from <requirements.txt|environment.yml|kernel-metadata.json>"""
    proj = current_project()
    if bool(a.convert_from) == bool(a.to_fmt):
        raise KenvError("Usage: kenv convert --to requirements.txt|environment.yml|kernel-metadata.json   or   kenv convert --from <file>")
    if a.to_fmt:
        fmt = Path(a.to_fmt).name.lower()
        v = proj.find_version(a.args[0]) if a.args else working_version(proj)[0]
        if not v:
            raise KenvError("This project has no version yet.")
        cfg = read_json(proj.kdir / v / "config.json", {})
        man = CoreStore(proj, v).load().get("manifest") or {}
        if fmt.startswith("requirements"):
            text, dest = to_requirements(_direct_specs(proj, v, a.all)), "requirements.txt"
        elif fmt.startswith("environment"):
            text = to_environment_yml(_direct_specs(proj, v, a.all), proj.kname, cfg.get("python") or man.get("python"))
            dest = "environment.yml"
        elif fmt.startswith("kernel-metadata"):
            try:
                user = get_username()
            except KenvError:
                user = "your-kaggle-username"
            text, dest = json.dumps(kernel_metadata_for(proj, v, user), indent=2) + "\n", "kernel-metadata.json"
        else:
            raise KenvError("--to takes requirements.txt, environment.yml or kernel-metadata.json")
        if not text.strip():
            raise KenvError(f"{proj.label(v)} has no dependency data yet, so there is nothing to convert.")
        if not text.startswith("{") and secret_gate_text(text):
            raise KenvError("The output looks like it contains a secret; nothing was written.")
        target = Path(a.out) if a.out else proj.root / dest
        if target.is_dir():
            target = target / dest
        if target.exists() and not a.force:
            raise KenvError(f"{target} already exists. Use --out <other file> or --force to overwrite it.")
        target.write_text(text, encoding="utf-8", newline="\n")
        say(f"[kenv] wrote {target}  (from {proj.label(v)})")
        return 0
    src = Path(a.convert_from)
    if not src.is_file():
        raise KenvError(f"Cannot find {src}")
    text = src.read_text(encoding="utf-8", errors="replace")
    w = working_version(proj)[0]
    if not w:
        raise KenvError("This project has no version yet. Run `kenv init` first.")
    name = src.name.lower()
    if name.endswith(".json"):
        try:
            md = json.loads(text)
        except ValueError as e:
            raise KenvError(f"{src} is not valid JSON ({e})")
        if not isinstance(md, dict):
            raise KenvError(f"{src} is not a kernel-metadata.json object")
        cfg = read_json(proj.kdir / w / "config.json", {})
        ds = [d for d in md.get("dataset_sources", []) if isinstance(d, str)]
        cfg["datasets"] = sorted(set(cfg.get("datasets") or []) | set(ds))
        cfg["accelerator"] = "GPU" if md.get("enable_gpu") else "CPU"
        write_json(proj.kdir / w / "config.json", cfg)
        say(f"[kenv] {proj.label(w)}: {len(ds)} dataset source(s) and accelerator '{cfg['accelerator']}' read from {src.name}. "
            "Datasets must be readable by the account you run kenv with.")
        return 0
    if name.endswith((".yml", ".yaml")):
        pinned, loose, py = parse_environment_yml(text)
    else:
        (pinned, loose), py = parse_requirements(text), None
    lock = read_lock(proj.kdir / w / "deps.lock")
    lock.update(pinned)
    write_lock(proj.kdir / w / "deps.lock", lock)

    def fn(doc):
        d = doc.setdefault("dependencies", {})
        d["lock"] = dict(sorted({**(d.get("lock") or {}), **pinned}.items()))
        dec = dict(d.get("declared") or {})
        dec.update({k: "==" + x for k, x in pinned.items()})
        dec.update({k: "*" for k in loose if k not in dec})
        d["declared"] = dict(sorted(dec.items()))
        if py:
            doc.setdefault("manifest", {})["python"] = py
    CoreStore(proj, w).update(fn)
    say(f"[kenv] {proj.label(w)}: {len(pinned)} pinned package(s) merged into the dependency lock" + (f", Python {py} recorded" if py else ""))
    if loose:
        say(f"  {len(loose)} package(s) without a pinned version were recorded as declared packages: {', '.join(loose[:8])}"
            + (" ..." if len(loose) > 8 else "") + "\n  `kenv rebuild` installs them on the kernel; pin them (name==version) for exact reproducibility.")
    return 0


def secret_gate_text(text):
    return bool([1 for _n, rx in SECRET_PATTERNS if rx.search(text)])


# ---- export / import

EXPORT_FORMAT = 1
SUMS_NAME = "kenv-sha256.json"
EXPORT_MAX = 400 * 1024 * 1024


def cmd_export(a):
    """kenv export [version] [--out file.zip]"""
    proj = current_project()
    v = proj.find_version(a.args[0]) if a.args else proj.last_active()
    if not v:
        raise KenvError("This project has no version to export yet.")
    vd = proj.kdir / v
    snap = proj.load_snapshot(v)
    doc = CoreStore(proj, v).load()
    files = [(rel, vd / "code" / rel) for rel, e in sorted(snap.items()) if not e.get("skipped") and (vd / "code" / rel).is_file()]
    skipped = [rel for rel, e in snap.items() if e.get("skipped")]
    if not files and not proj.has_snapshot(v):
        say(c(f"[kenv] {proj.label(v)} has no code snapshot (`kenv commit` makes one). Exporting the core file, lock and logs only.", "33"))
    # secrets: scan everything that goes into the zip BEFORE it is written
    hits = []
    allow = secret_allowlist(proj)
    for rel, fp in files:
        h = scan_secret_file(fp, allow)
        if h:
            hits.append((f"code/{rel}", h))
    for rel in (CORE_FILE, "config.json", "deps.lock"):  # logs are redacted instead of refused
        if (vd / rel).is_file():
            h = scan_secret_file(vd / rel, allow)
            if h:
                hits.append((rel, h))
    if hits:
        report_secret_findings(hits)
        raise KenvError("Export stopped: possible secrets found. No zip was written. Remove them, or allowlist false alarms. "
                        "Kenv's scan is a safety net, not a guarantee.")
    meta = proj.meta(v)
    out = Path(a.out) if a.out else proj.root / f"{proj.kname}-{v}.kenv.zip"
    if out.is_dir():
        out = out / f"{proj.kname}-{v}.kenv.zip"
    if out.exists() and not a.force:
        raise KenvError(f"{out} already exists. Use --out <other file> or --force.")
    sessions = [{k: e.get(k) for k in ("sid", "start", "end", "duration_s", "ended_by", "gpu")} for e in proj.sessions(v)]
    info = {"format": EXPORT_FORMAT, "kenv": VERSION, "exported": iso(), "project": proj.kname, "version": v,
            "name": meta.get("name"), "message": meta.get("message"), "committed": meta.get("committed"),
            "python": (doc.get("manifest") or {}).get("python"), "docker_image": (doc.get("manifest") or {}).get("docker_image"),
            "accelerator_id": (doc.get("manifest") or {}).get("accelerator_id"),
            "datasets": (read_json(vd / "config.json", {}).get("datasets") or []),
            "secret_names": (doc.get("secrets") or {}).get("names", []), "sessions": sessions,
            "skipped_big_files": skipped, "note": "Secret values are never exported: only their names."}
    tmp = out.with_name(out.name + ".part")
    total = 0
    parts = {"kenv-export.json": json.dumps(info, indent=2).encode("utf-8")}
    for rel in ("deps.lock", "code.json", "meta.json"):
        if (vd / rel).is_file():
            parts[rel] = (vd / rel).read_bytes()
    if (vd / "config.json").is_file():
        cj = read_json(vd / "config.json", {})
        cj.pop("kernel", None)  # the exporter's kernel reference contains their username
        parts["config.json"] = json.dumps(cj, indent=2).encode("utf-8")
    if doc:
        d2 = json.loads(json.dumps(doc))
        (d2.get("manifest") or {}).pop("kernel", None)
        parts[CORE_FILE] = redact_secrets(toml_dumps(canon(d2), CORE_HEADER)).encode("utf-8")
    for rel in ("run.log", "errors.log"):
        fp = vd / LOG_DIR / rel
        if fp.is_file():
            parts[f"{LOG_DIR}/{rel}"] = redact_secrets(fp.read_text(encoding="utf-8", errors="replace")).encode("utf-8")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            sums = {}

            def add(name, data):
                z.writestr(name, data)
                sums[name] = hashlib.sha256(data).hexdigest()
            for name, data in parts.items():
                add(name, data)
            for rel, fp in files:
                data = fp.read_bytes()
                total += len(data)
                if total > EXPORT_MAX:
                    raise KenvError(f"The code snapshot is bigger than {human(EXPORT_MAX)}; keep big data out with .kenvignore and `kenv data push`.")
                add(f"code/{rel}", data)
            z.writestr(SUMS_NAME, json.dumps(sums, indent=2))  # SHA256 of every other entry: import verifies them all
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    ign = proj.root / IGNORE_FILE
    try:
        if ign.is_file() and "*.kenv.zip" not in ign.read_text(encoding="utf-8", errors="replace"):
            with open(ign, "a", encoding="utf-8", newline="\n") as f:
                f.write("\n# kenv exports\n*.kenv.zip\n")
    except OSError:
        pass
    say(f"[kenv] exported {proj.label(v)}: {len(files)} file(s), {human(out.stat().st_size)} -> {out}")
    if skipped:
        say(c(f"  {len(skipped)} file(s) above the snapshot size limit are listed but not included: {', '.join(skipped[:5])}", "33"))
    if info["secret_names"]:
        say("  Secret NAMES the code uses (add them in Kaggle > Add-ons > Secrets on the new account): " + ", ".join(info["secret_names"]))
    return 0


def _safe_rel(name):
    p = posixpath.normpath(name)
    return not ("\\" in name or p.startswith(("/", "..")) or "/../" in "/" + p or ":" in p.split("/")[0] or p == ".")


def cmd_import(a):
    """kenv import <file.zip> [-n name]: recreate the version in THIS folder; the next session builds a kernel under your account"""
    if not a.args:
        raise KenvError("Usage: kenv import <file.zip> [--name <version-name>]")
    zp = Path(a.args[0])
    if not zp.is_file():
        raise KenvError(f"Cannot find {zp}")
    if a.out:  # --to <dir>: import into another folder
        target = Path(a.out).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        proj = Project(target)
    else:
        proj = current_project(create=True)
    proj.guard()
    cr = find_creds()
    say(f"[kenv] importing into this project; the kernel will be created under "
        + (f"your Kaggle account '{cr[0]}'" if cr else "your Kaggle account (no credentials found yet: run `kenv --cred`)"))
    try:
        z = zipfile.ZipFile(zp)
    except zipfile.BadZipFile:
        raise KenvError(f"{zp} is not a zip file")
    with z:
        try:
            info = json.loads(z.read("kenv-export.json"))
        except (KeyError, ValueError):
            raise KenvError(f"{zp.name} is not a kenv export (kenv-export.json is missing or damaged)")
        if info.get("format") != EXPORT_FORMAT:
            raise KenvError(f"Unsupported export format {info.get('format')}; this kenv reads format {EXPORT_FORMAT}.")
        names = [n for n in z.namelist() if not n.endswith("/")]
        bad = [n for n in names if not _safe_rel(n)]
        if bad:
            raise KenvError(f"The zip contains unsafe paths (e.g. {bad[0]}); import refused.")
        if sum(i.file_size for i in z.infolist()) > 2 * EXPORT_MAX:
            raise KenvError("The zip expands to more than the size kenv accepts.")
        try:
            sums = json.loads(z.read(SUMS_NAME))
        except (KeyError, ValueError):
            raise KenvError(f"{zp.name} has no checksum manifest ({SUMS_NAME}); it was not made by kenv export, or it was edited. Import refused.")
        for n in names:
            if n != SUMS_NAME and sums.get(n) != _zip_sha(z, n):
                raise KenvError(f"Import refused: {n} does not match its checksum (the zip is damaged or was edited).")
        gone_entries = [n for n in sums if n not in names]
        if gone_entries:
            raise KenvError(f"Import refused: {gone_entries[0]} is listed in the checksums but missing from the zip.")
        # secret gate on the incoming files (a zip from someone else can hold anything)
        hits = []
        for n in names:
            if n.startswith("code/") or n in (CORE_FILE, "config.json", "deps.lock"):
                h = scan_secret_text(z.read(n).decode("utf-8", "replace"), secret_allowlist(proj)) if z.getinfo(n).file_size < 4 * 1024 * 1024 else []
                if h:
                    hits.append((n, h))
        if hits:
            report_secret_findings(hits, "[kenv] The export contains possible secrets:")
            raise KenvError("Import refused: remove the secrets from the export (or allowlist false alarms), then import again.")
        created = proj.ensure()
        if a.name:
            v = proj.new_version(a.name)
        else:
            try:
                v = proj.new_version(info.get("name") or None)
            except KenvError:  # that name is taken here (or not a valid name): plain numbered version
                v = proj.new_version()
        vd = proj.kdir / v
        snap = read_json_bytes(z, "code.json")
        entries, wrote, kept = {}, 0, []
        for n in names:
            if not n.startswith("code/"):
                continue
            rel = n[5:]
            data = z.read(n)
            dst = vd / "code" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            h = hashlib.sha256(data).hexdigest()
            entries[rel] = {"sha256": h, "size": len(data)}
            live_p = proj.root / rel
            if live_p.exists():
                if sha256_file(live_p) != h:
                    alt = proj.root / alt_name(rel).replace(".kenv-remote", ".kenv-import")
                    alt.parent.mkdir(parents=True, exist_ok=True)
                    alt.write_bytes(data)
                    kept.append(rel)
            else:
                live_p.parent.mkdir(parents=True, exist_ok=True)
                live_p.write_bytes(data)
                wrote += 1
        for rel, e in ((snap.get("files") or {}) if isinstance(snap, dict) else {}).items():
            if isinstance(e, dict) and e.get("skipped"):
                entries.setdefault(rel, e)  # too big to export: only its hash is kept, like in the source project
        write_json(vd / "code.json", {"count": len(entries), "bytes": sum(x.get("size", 0) for x in entries.values()), "files": entries})
        lock_text = z.read("deps.lock").decode("utf-8", "replace") if "deps.lock" in names else ""
        (vd / "deps.lock").write_bytes(lock_text.encode("utf-8"))
        cfg = read_json_bytes(z, "config.json") or {}
        cfg.pop("kernel", None)  # the old account's kernel: a new one is created in your account
        keep_ds, gone_ds = check_datasets(list(cfg.get("datasets") or []))
        if gone_ds:
            cfg["datasets"] = keep_ds
        cfg["lock_known"] = bool(read_lock(vd / "deps.lock"))
        write_json(vd / "config.json", cfg)
        old_doc = {}
        if CORE_FILE in names:
            try:
                old_doc = toml_loads(z.read(CORE_FILE).decode("utf-8"))
            except Exception:
                old_doc = {}
        meta = proj.meta(v)
        meta.update(message=f"imported from {zp.name} ({info.get('project')} {info.get('version')})",
                    committed=iso(), branch=proj.current_branch(), env_pending=True,
                    imported={"from": zp.name, "exported": info.get("exported"), "project": info.get("project"),
                              "version": info.get("version"), "python": info.get("python"), "docker_image": info.get("docker_image")})
        write_json(vd / "meta.json", {k: x for k, x in meta.items() if x is not None})
        write_json(vd / "sessions.json", {"sessions": []})
        write_json(vd / "sessions-archived.json", {"note": "Sessions of the exporting account. They never count toward your quota.",
                                                   "sessions": info.get("sessions") or []})

        def fn(doc):
            m = dict(old_doc.get("manifest") or {})
            m.pop("kernel", None)
            m.pop("name", None)
            m.update(version=v, id=meta["id"], created=iso(), updated=iso())
            doc["manifest"] = m
            doc["dependencies"] = old_doc.get("dependencies") or {"detected": {}, "lock": read_lock(vd / "deps.lock")}
            doc["io"] = {"inputs": (old_doc.get("io") or {}).get("inputs", {}), "outputs": {}}
            doc["secrets"] = {"names": list((old_doc.get("secrets") or {}).get("names", info.get("secret_names") or []))}
            doc["sessions"] = {"file": "sessions.json", "count": 0, "total_s": 0}
            doc["metrics"] = {}
        CoreStore(proj, v).update(fn)
        # logs: a new kernel cannot inherit old run times, so they are archived, original timestamps untouched
        arch = vd / LOG_DIR / "archive"
        for rel in ("run.log", "errors.log"):
            if f"{LOG_DIR}/{rel}" in names:
                arch.mkdir(parents=True, exist_ok=True)
                (arch / rel).write_bytes(redact_secrets(z.read(f"{LOG_DIR}/{rel}").decode("utf-8", "replace")).encode("utf-8"))
        if arch.is_dir():
            stamps = []
            for rel in ("run.log", "errors.log"):
                try:
                    ls = [ln for ln in (arch / rel).read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
                except OSError:
                    continue
                if ls:
                    stamps.append({"file": rel, "lines": len(ls), "first": ls[0][:LOG_TS], "last": ls[-1][:LOG_TS]})
            write_json(arch / "archive.json", {"note": "Imported log; the lines keep the timestamps they had on the original kernel.",
                                               "original_project": info.get("project"), "original_version": info.get("version"),
                                               "exported": info.get("exported"), "files": stamps,
                                               "original_sessions": info.get("sessions") or []})
    proj.set_active(v)
    quota_register(proj)
    say(f"[kenv] imported as {proj.label(v)}: {wrote} file(s) written into the project, {len(entries)} in the snapshot")
    if kept:
        say(c(f"  {len(kept)} file(s) already existed and differ; yours were kept and the imported copies saved as *.kenv-import.* ({', '.join(kept[:4])})", "33"))
    if gone_ds:
        say(c("  Dropped (your account cannot read them): " + ", ".join(gone_ds) +
              "\n  Attach your own copies with `kenv data push <folder>`.", "33"))
    if keep_ds:
        say("  Kaggle datasets kept: " + ", ".join(keep_ds))
    if info.get("secret_names"):
        say("  Add these secrets in your Kaggle account (names only were exported): " + ", ".join(info["secret_names"]))
    if info.get("python"):
        say(f"  Recorded on the original kernel: Python {info['python']}, image {info.get('docker_image') or 'unknown'}. "
            "The first session compares them with what Kaggle provides now and warns on a mismatch.")
    if a.init:
        say("[kenv] starting a session ...")
        os.chdir(proj.root)
        a.name = None  # --name named the version above, not the session
        return cmd_init(a)
    say("  Next: `kenv init` starts a kernel under YOUR credentials and installs the locked packages"
        + (f" (run it inside {proj.root})" if a.out else "") + ".")
    return 0


def _zip_sha(z, name):
    h = hashlib.sha256()
    with z.open(name) as f:
        for blk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def check_datasets(refs):
    """-> (kept, dropped). A dataset the importing account cannot read is dropped; when the kaggle CLI is missing or
    the check is inconclusive (network), the ref is kept."""
    if not refs:
        return [], []
    if not shutil.which("kaggle"):
        say(c("[kenv] the kaggle CLI is not installed, so the dataset references cannot be checked; all are kept", "33"))
        return list(refs), []
    keep, gone = [], []
    for r in refs:
        try:
            p = cli("datasets", "files", r, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            keep.append(r)
            continue
        txt = (p.stdout + p.stderr).lower()
        if p.returncode != 0 and any(x in txt for x in ("403", "404", "401", "not found", "forbidden", "no such", "could not find")):
            gone.append(r)
        else:
            keep.append(r)
    return keep, gone


def read_json_bytes(z, name):
    try:
        return json.loads(z.read(name).decode("utf-8"))
    except (KeyError, ValueError):
        return {}


# ---- doctor

def cmd_doctor(a):
    """kenv doctor: check credentials, the Kaggle CLI and the integrity of .kenv. Exits 1 on problems."""
    problems, warns, fixes = [], [], []
    scope = (a.args[0].lower() if a.args else "all")
    if scope not in ("all", "env", "files"):
        raise KenvError("Usage: kenv doctor [env | files] [--fix]")

    def fixed(msg):
        fixes.append(msg)
        say(f"  {c('[fixed]', '36;1')}  {msg}")

    def ok(msg):
        say(f"  {c('[ok]', '32;1')}  {msg}")

    def bad(msg, fix):
        problems.append(msg)
        say(f"  {c('[problem]', '31;1')}  {msg}\n              fix: {fix}")

    def warn(msg, fix=""):
        warns.append(msg)
        say(f"  {c('[warn]', '33;1')}  {msg}" + (f"\n              fix: {fix}" if fix else ""))

    def env_checks():
        say(c("Environment", "1"))
        if sys.version_info >= (3, 8):
            ok(f"Python {platform.python_version()}")
        else:
            bad("Python is older than 3.8", "install Python 3.8 or newer")
        if not shutil.which("kaggle"):
            bad("Kaggle CLI not found", "pip install -U kaggle")
        else:
            try:
                p = cli("--version", timeout=30)
                txt = (p.stdout + p.stderr).strip()
                m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", txt)
                ver = tuple(int(x or 0) for x in m.groups()) if m else None
                if p.returncode != 0 or not ver:
                    warn(f"Could not read the Kaggle CLI version ({txt[:80] or 'no output'})", "pip install -U kaggle")
                elif ver < (1, 6, 0):
                    warn(f"Kaggle CLI {'.'.join(map(str, ver))} is old (GPU selection may not work)", "pip install -U kaggle")
                else:
                    ok(f"Kaggle CLI {'.'.join(map(str, ver))}")
                if cli("kernels", "delete", "--help", timeout=30).returncode != 0:
                    bad("This Kaggle CLI has no `kernels delete` (kenv cleanup needs it)", "pip install -U kaggle")
            except (OSError, subprocess.TimeoutExpired) as e:
                warn(f"Kaggle CLI did not answer ({e})", "reinstall it: pip install -U kaggle")
        cr = find_creds()
        if cr:
            ok(f"Kaggle credentials for '{cr[0]}' ({cr[1]})")
            f = cred_file()
            if f.exists() and os.name != "nt":
                try:
                    if f.stat().st_mode & 0o077:
                        warn(f"{f} is readable by other users", f"chmod 600 {f}")
                except OSError:
                    pass
        else:
            bad("No Kaggle credentials found", "run `kenv --cred` (or set KAGGLE_USERNAME and KAGGLE_KEY)")
        try:
            with http(f"{RELAY}/v1/health", timeout=6) as r:
                r.read()
            ok(f"relay {RELAY} reachable")
        except Exception as e:
            warn(f"relay {RELAY} not reachable ({str(e)[:60]})", "check your connection, or set KENV_RELAY")


    def file_checks():
        say(c("Project", "1"))
        try:
            proj = current_project()
        except KenvError:
            say("  (not inside a kenv project - project checks skipped)")
            proj = None
        if proj:
            for fn_ in ("sync.json", "datasets.json", TAGS_FILE, BRANCHES_FILE):
                p = proj.kdir / fn_
                if p.is_file():
                    try:
                        json.loads(p.read_text(encoding="utf-8"))
                    except (ValueError, OSError) as e:
                        bad(f".kenv/{fn_} is not valid JSON ({str(e)[:50]})", f"restore it from git, or delete it (kenv recreates an empty one)")
            versions = proj.version_names()
            ok(f"{len(versions)} version(s): {', '.join(versions) or 'none yet'}")
            for v in versions:
                vd = proj.kdir / v
                try:
                    mj = json.loads((vd / "meta.json").read_text(encoding="utf-8"))
                    if not isinstance(mj, dict) or not mj.get("id"):
                        raise ValueError("no id")
                except (OSError, ValueError):
                    warn(f"{v}: meta.json is missing or damaged", f"kenv repairs it the next time it opens {v}; or restore it from git")
                    mj = {}
                try:
                    json.loads((vd / "sessions.json").read_text(encoding="utf-8"))
                except FileNotFoundError:
                    pass
                except (OSError, ValueError):
                    bad(f"{v}: sessions.json is damaged", f"restore it from git, or replace it with {{\"sessions\": []}}")
                doc = None
                if (vd / CORE_FILE).is_file():
                    try:
                        doc = CoreStore(proj, v).load()
                        mid = (doc.get("manifest") or {}).get("id")
                        if mid and mj.get("id") and mid != mj["id"]:
                            warn(f"{v}: core file id does not match meta.json", f"open {CORE_FILE} and check it belongs to {v}")
                    except KenvError as e:
                        bad(f"{v}: core file does not parse - {str(e)[:110]}", f"restore .kenv/{v}/{CORE_FILE} from git, or delete it (kenv writes a new one in the next session)")
                if proj.has_snapshot(v):
                    snap, badh, miss = proj.load_snapshot(v), [], []
                    for rel, e in snap.items():
                        if e.get("skipped"):
                            continue
                        fp = vd / "code" / rel
                        if not fp.is_file():
                            miss.append(rel)
                        elif e.get("sha256") and sha256_file(fp) != e["sha256"]:
                            badh.append(rel)
                    if miss or badh:
                        bad(f"{v}: snapshot is damaged ({len(miss)} missing, {len(badh)} changed: {', '.join((miss + badh)[:3])})",
                            "restore .kenv from git or a backup; a rollback to this version would restore wrong files")
                if doc:
                    outs = ((doc.get("io") or {}).get("outputs") or {})
                    stale = [k for k, e in outs.items() if isinstance(e, dict) and (proj.root / k).is_file() and e.get("sha256")
                             and sha256_file(proj.root / k) != e["sha256"]]
                    if stale:
                        warn(f"{v}: {len(stale)} output file(s) differ from the hashes in the I/O map (edited since, or corrupted): {', '.join(stale[:3])}",
                             "expected if you edited them; otherwise re-run to regenerate them")
            tags = read_json(proj.kdir / TAGS_FILE, {})
            for t, tv in list(tags.items() if isinstance(tags, dict) else []):
                if not isinstance(tv, str) or tv not in versions:
                    if a.fix:
                        tags.pop(t, None)
                        proj.save_tags({k: x for k, x in tags.items() if isinstance(x, str)})
                        fixed(f"removed tag '{t}' (it pointed to '{tv}', which does not exist)")
                    else:
                        bad(f"tag '{t}' points to '{tv}', which does not exist", f"kenv doctor --fix   (or: kenv tag -d {t})")
            br = read_json(proj.kdir / BRANCHES_FILE, {})
            for b, parent in list(br.items() if isinstance(br, dict) else []):
                if isinstance(parent, str) and VERSION_RE.match(parent) and parent not in versions:
                    if a.fix:
                        br.pop(b, None)
                        proj.save_branches(br)
                        fixed(f"removed branch '{b}' (it started from {parent}, which does not exist)")
                    else:
                        bad(f"branch '{b}' starts from {parent}, which does not exist", f"kenv doctor --fix   (or: kenv branch rm {b})")
            try:
                la = (proj.kdir / "last_active").read_text().strip()
                if la and la not in versions and la not in tags and not any((proj.meta(x).get("name") or "") == la for x in versions):
                    if a.fix and versions:
                        newest = max(versions, key=lambda x: int(x[1:]))
                        proj.set_active(newest)
                        fixed(f"last_active pointed to '{la}', which does not exist; it now points to {newest}")
                    else:
                        bad(f"last_active points to '{la}', which does not exist", "kenv doctor --fix   (or: kenv activate <version>)")
            except OSError:
                pass
            try:
                cb = (proj.kdir / "branch").read_text().strip()
                if cb and cb != "main" and cb not in (br if isinstance(br, dict) else {}):
                    if a.fix:
                        (proj.kdir / "branch").write_text("main\n")
                        fixed(f"the current branch '{cb}' did not exist; switched to main")
                    else:
                        warn(f"the current branch '{cb}' does not exist (kenv falls back to main)", "kenv doctor --fix")
            except OSError:
                pass
            n_dead = sum(1 for v in versions for e in proj.sessions(v) if not e.get("end") and not pid_alive(e.get("pid")))
            if n_dead:
                warn(f"{n_dead} session(s) never recorded an end (crash or kill)", "the next `kenv init` repairs them; `kenv --sweep` removes leftover kernels")
            leftovers = [p for p in proj.kdir.rglob("*") if p.is_file() and "code" not in p.relative_to(proj.kdir).parts
                         and re.search(r"\.\d+\.tmp$|\.part$", p.name)]
            if leftovers:
                if a.fix:
                    for p in leftovers:
                        try:
                            p.unlink()
                        except OSError:
                            pass
                    fixed(f"removed {len(leftovers)} leftover temp file(s) from interrupted writes")
                else:
                    warn(f"{len(leftovers)} leftover temp file(s) in .kenv (an interrupted write)", "kenv doctor --fix")
            try:
                junk = [x for x in (proj.kdir / SECRETS_ALLOWLIST).read_text(encoding="utf-8", errors="replace").splitlines()
                        if x.split("#", 1)[0].strip() and not re.fullmatch(r"[0-9a-fA-F]{12}", x.split("#", 1)[0].strip())]
                if junk:
                    warn(f"{SECRETS_ALLOWLIST} has {len(junk)} entr(ies) that are not fingerprints (ignored; they may contain a secret)",
                         "delete those lines and run `kenv scan allow` to add fingerprints")
            except OSError:
                pass
            sh = _scan_files(proj, kenv_meta_files(proj))
            if sh:
                report_secret_findings(sh, "[kenv] Possible secrets inside .kenv:")
                bad(f"{sum(len(h) for _, h in sh)} secret-looking string(s) inside .kenv files", "remove them from those files and rotate the key")
            else:
                ok("no secret-looking strings in .kenv files")

    if scope in ("all", "env"):
        env_checks()
    if scope in ("all", "files"):
        file_checks()
    say()
    if problems:
        say(c(f"{len(problems)} problem(s), {len(warns)} warning(s)" + (f", {len(fixes)} fixed" if fixes else "") + ".", "31;1"))
        return 1
    say(c(f"All good ({len(warns)} warning(s)" + (f", {len(fixes)} fixed" if fixes else "") + ").", "32;1"))
    return 0


# ----------------------------------------------------------------------------- Phase 6: kenv.clip() / kenv unclip
#
#   kenv.clip()                 at the top of a file: "the kenv lines in this file may be commented out"
#   kenv unclip                 comment out every kenv line of those files (ast finds the statements), auto-saved
#   session start               the commented lines come back (kenv finds them by their marker)
#   .kenv/clip                  JSON: which files and which line numbers - refreshed whenever kenv looks at a file
#
# A clipped line is   <indent>#kenv:clip# <the original line>     (the marker sits after the indentation, so the
# reverse is exact). When commenting leaves a block empty, kenv adds   pass  #kenv:clip-pass   and removes it again.
# Notebooks are JSON: every code cell is handled on its own and the file keeps its formatting.

CLIP_FILE = "clip"
CLIP_MARK = "#kenv:clip# "
CLIP_MARK_RE = re.compile(r"^([ \t\f]*)#kenv:clip# ?")
CLIP_PASS_LINE = "pass  #kenv:clip-pass"
CLIP_PASS_RE = re.compile(r"^[ \t\f]*pass  #kenv:clip-pass[ \t]*(?:\r\n|\r|\n)?$")
CLIP_BACKUP = "clip-backup"
CLIP_MAX_FILE = 4 * 1024 * 1024
CLIP_SKIP_DIRS = set(DEFAULT_IGNORES) | {"site-packages", ".tox", ".mypy_cache", ".pytest_cache", "__pypackages__", ".eggs"}
CLIP_HOOK_BEGIN, CLIP_HOOK_END = "# kenv-clip-hook begin", "# kenv-clip-hook end"
_LINE_RE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+")
_EOL_RE = re.compile(r"(\r\n|\r|\n)$")


def split_lines(text):
    """Lines with their line endings, split the way Python's tokenizer does (not like str.splitlines)."""
    return _LINE_RE.findall(text)


def _eol(line):
    m = _EOL_RE.search(line)
    return m.group(1) if m else ""


def _ws(line):
    return re.match(r"[ \t\f]*", line).group(0)


# ---- finding the kenv statements

def _clip_names(tree):
    """Names bound to kenv in this code: ({module alias}, {function alias: original name})."""
    mods, funcs = set(), {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for al in n.names:
                if al.name == "kenv":
                    mods.add(al.asname or "kenv")
        elif isinstance(n, ast.ImportFrom) and n.level == 0:
            if n.module == "kenv_shim":
                for al in n.names:
                    if al.name == "kenv":
                        mods.add(al.asname or "kenv")
            elif n.module == "kenv":
                for al in n.names:
                    if al.name != "*":
                        funcs[al.asname or al.name] = al.name
    return mods, funcs


def _kenv_call(node, mods, funcs):
    """The kenv function a call node calls ('cli', 'clip', ...) or None."""
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in mods:
        return f.attr
    if isinstance(f, ast.Name) and f.id in funcs:
        return funcs[f.id]
    return None


def _kenv_import(st):
    """True when this import statement binds nothing but kenv (so commenting it out is safe)."""
    if isinstance(st, ast.Import):
        return len(st.names) == 1 and st.names[0].name == "kenv"
    if isinstance(st, ast.ImportFrom) and st.level == 0:
        if st.module == "kenv_shim":
            return len(st.names) == 1 and st.names[0].name == "kenv"
        if st.module == "kenv":
            return all(al.name != "*" for al in st.names)
    return False


def _touches_kenv_import(st):
    if isinstance(st, ast.Import):
        return any(al.name == "kenv" for al in st.names)
    if isinstance(st, ast.ImportFrom) and st.level == 0:
        return st.module in ("kenv", "kenv_shim")
    return False


def _alone_on_lines(lines, st):
    """The statement is the only code on its physical lines (nothing before it, at most a comment after it)."""
    try:
        first, last = lines[st.lineno - 1], lines[st.end_lineno - 1]
        before = first.encode("utf-8", "surrogatepass")[:st.col_offset]
        after = last.encode("utf-8", "surrogatepass")[st.end_col_offset:].decode("utf-8", "replace")
    except (IndexError, AttributeError):
        return False
    return before.strip(b" \t\f") == b"" and re.match(r"^[ \t\f]*(#.*)?(\r\n|\r|\n)?$", after) is not None


def clip_plan(text, extra=None):
    """Find the statements `kenv unclip` would comment out in Python source.
    -> {optin, targets: [(first_line, last_line)], passes: {line: indent}, warnings: [(line, msg)], blocked, error, names}"""
    plan = {"optin": False, "targets": [], "passes": {}, "warnings": [], "blocked": False, "error": None,
            "names": (set(), {})}
    lines = split_lines(text)
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
        plan["error"] = "cannot parse: %s" % (str(e).splitlines() or [type(e).__name__])[0]
        return plan
    mods, funcs = _clip_names(tree)
    if extra:
        mods, funcs = mods | set(extra[0]), {**funcs, **extra[1]}
    plan["names"] = (mods, funcs)
    if not mods and not funcs:
        return plan
    stmt_lists = []
    for node in ast.walk(tree):
        for fld in ("body", "orelse", "finalbody"):
            v = getattr(node, fld, None)
            if isinstance(v, list) and v and isinstance(v[0], ast.stmt):
                stmt_lists.append((node, v))
    targets, all_stmts = {}, []
    for _, lst in stmt_lists:
        all_stmts += lst
    for st in all_stmts:
        what = None
        if isinstance(st, ast.Expr):
            fn = _kenv_call(st.value, mods, funcs)
            if fn:
                what = fn
                if fn == "clip":
                    plan["optin"] = True
        elif _kenv_import(st):
            what = "import"
        if what is None:
            if _touches_kenv_import(st):
                plan["warnings"].append((st.lineno, "an import that also imports something else cannot be commented out"))
            continue
        if not _alone_on_lines(lines, st):
            plan["warnings"].append((st.lineno, "shares its line with other code (put the kenv statement on its own line)"))
            continue
        targets[id(st)] = st
    ranges = sorted((t.lineno, t.end_lineno) for t in targets.values())
    plan["targets"] = ranges

    def inside(n):
        return any(a <= n <= b for a, b in ranges)

    seen = {ln for ln, _ in plan["warnings"]}
    for node in ast.walk(tree):  # kenv used in a way that cannot be commented out (x = kenv.cli(...), with kenv.timed(): ...)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and (node.id in mods or node.id in funcs):
            if not inside(node.lineno) and node.lineno not in seen:
                seen.add(node.lineno)
                plan["warnings"].append((node.lineno, "kenv is used in an expression (%s) that cannot be commented out"
                                         % lines[node.lineno - 1].strip()[:50]))
    for parent, lst in stmt_lists:
        if isinstance(parent, ast.Module):
            continue
        if all(id(s) in targets for s in lst):  # the block would be empty: keep it valid with a pass
            plan["passes"][lst[-1].end_lineno] = _ws(lines[lst[0].lineno - 1])
    left = [w for w in plan["warnings"]]
    plan["blocked"] = bool(left)
    return plan


def clip_apply_lines(lines, plan):
    clipped = set()
    for a, b in plan["targets"]:
        clipped.update(range(a, b + 1))
    out = []
    for i, line in enumerate(lines, 1):
        if i in clipped:
            ws = _ws(line)
            out.append(ws + CLIP_MARK + line[len(ws):])
        else:
            out.append(line)
        if i in plan["passes"]:
            eol = _eol(line)
            if not eol:  # last line of the file has no line ending: keep it that way for the pass line
                out[-1] += "\n"
                out.append(plan["passes"][i] + CLIP_PASS_LINE)
            else:
                out.append(plan["passes"][i] + CLIP_PASS_LINE + eol)
    return out


def clip_restore_lines(lines):
    out, n = [], 0
    for line in lines:
        if CLIP_PASS_RE.match(line):
            if not _eol(line) and out and _eol(out[-1]):
                out[-1] = _EOL_RE.sub("", out[-1])
            continue
        m = CLIP_MARK_RE.match(line)
        if m:
            out.append(m.group(1) + line[m.end():])
            n += 1
        else:
            out.append(line)
    return out, n


def has_clip_marks(text):
    return "#kenv:clip#" in text or "#kenv:clip-pass" in text


# ---- whole files: .py text and .ipynb JSON

def _py_decode(data):
    try:
        enc, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        return data.decode(enc), enc
    except (SyntaxError, LookupError, UnicodeDecodeError) as e:
        raise ValueError("cannot decode: %s" % e)


def _nb_sanitize(text):
    """IPython magics are not Python: swap them for `pass` (same line, same indentation) just for parsing."""
    out = []
    for line in split_lines(text):
        if line.lstrip().startswith(("%", "!")):
            out.append(_ws(line) + "pass" + _eol(line))
        else:
            out.append(line)
    return "".join(out)


def _nb_cell_text(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def _nb_set_cell(cell, text):
    if isinstance(cell.get("source"), list):
        cell["source"] = re.findall(r"[^\n]*\n|[^\n]+", text)
    else:
        cell["source"] = text


def _nb_format(raw_text):
    """(indent, sort_keys, ensure_ascii) that reproduce this notebook's text exactly, or None."""
    m = re.match(r'\{\n([ \t]+)"', raw_text)
    indent = (m.group(1) if m.group(1).startswith("\t") else len(m.group(1))) if m else 1
    tail = "\n" if raw_text.endswith("\n") else ""
    obj = json.loads(raw_text)
    for sk in (True, False):
        for ea in (False, True):
            if json.dumps(obj, indent=indent, sort_keys=sk, ensure_ascii=ea) + tail == raw_text:
                return indent, sk, ea, tail
    return indent, False, False, tail


def clip_file_bytes(data, rel, mode, force=False):
    """mode 'clip' | 'restore' | 'scan'.  -> (new_bytes or None when nothing changes, info).
    info: optin, statements, lines (for .py) / cells (for .ipynb), warnings, blocked, error, restored."""
    info = {"kind": "ipynb" if rel.endswith(".ipynb") else "py", "optin": False, "statements": 0, "lines": [],
            "cells": {}, "warnings": [], "blocked": False, "error": None, "restored": 0}
    try:
        if info["kind"] == "py":
            return _clip_py(data, mode, force, info)
        return _clip_nb(data, mode, force, info)
    except ValueError as e:
        info["error"] = str(e)
        return None, info


def _clip_py(data, mode, force, info):
    text, enc = _py_decode(data)
    if mode == "restore":
        new, n = clip_restore_lines(split_lines(text))
        info["restored"] = n
        return ("".join(new).encode(enc) if n or len(new) != len(split_lines(text)) else None), info
    plan = clip_plan(text)
    info.update(optin=plan["optin"], warnings=plan["warnings"], blocked=plan["blocked"], error=plan["error"],
                statements=len(plan["targets"]))
    info["lines"] = [n for a, b in plan["targets"] for n in range(a, b + 1)]
    if plan["error"] or mode == "scan" or not plan["targets"]:
        return None, info
    if not plan["optin"]:
        return None, info
    if plan["blocked"] and not force:
        return None, info
    lines = split_lines(text)
    new_lines = clip_apply_lines(lines, plan)
    new_text = "".join(new_lines)
    try:  # the result must still be valid Python, and restoring must give back exactly what we started with
        ast.parse(new_text)
    except (SyntaxError, ValueError):
        raise ValueError("internal check failed: the commented file would not parse; nothing was changed")
    back, _ = clip_restore_lines(split_lines(new_text))
    if "".join(back) != text:
        raise ValueError("internal check failed: the change would not restore exactly; nothing was changed")
    return new_text.encode(enc), info


def _clip_nb(data, mode, force, info):
    try:
        raw = data.decode("utf-8")
        nb = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as e:
        raise ValueError("cannot read notebook: %s" % e)
    cells = nb.get("cells") if isinstance(nb, dict) else None
    if not isinstance(cells, list):
        raise ValueError("unsupported notebook format (only nbformat 4)")
    fmt = _nb_format(raw)
    changed = False
    code = [(i, cl) for i, cl in enumerate(cells) if isinstance(cl, dict) and cl.get("cell_type") == "code"]
    if mode == "restore":
        for i, cl in code:
            text = _nb_cell_text(cl)
            if has_clip_marks(text):
                new, n = clip_restore_lines(split_lines(text))
                if n or len(new) != len(split_lines(text)):
                    _nb_set_cell(cl, "".join(new))
                    info["restored"] += n
                    changed = True
        return (_nb_dump(nb, fmt) if changed else None), info
    mods, funcs, plans = set(), {}, {}
    for i, cl in code:  # kenv may be imported in one cell and used in another
        text = _nb_cell_text(cl)
        if text.lstrip().startswith("%%") or "kenv" not in text:
            continue
        p = clip_plan(_nb_sanitize(text))
        if p["error"]:
            info["warnings"].append((i, "cell %d: %s" % (i, p["error"])))
            continue
        mods |= p["names"][0]
        funcs.update(p["names"][1])
    for i, cl in code:
        text = _nb_cell_text(cl)
        if text.lstrip().startswith("%%") or "kenv" not in text:
            continue
        p = clip_plan(_nb_sanitize(text), extra=(mods, funcs))
        if p["error"]:
            continue
        plans[i] = (text, p)
        info["optin"] = info["optin"] or p["optin"]
        info["statements"] += len(p["targets"])
        if p["targets"]:
            info["cells"][str(i)] = [n for a, b in p["targets"] for n in range(a, b + 1)]
        for ln, msg in p["warnings"]:
            info["warnings"].append((ln, "cell %d, line %d: %s" % (i, ln, msg)))
            info["blocked"] = True
    if mode == "scan" or not info["optin"] or not info["statements"] or (info["blocked"] and not force):
        return None, info
    for i, (text, p) in plans.items():
        if not p["targets"]:
            continue
        new_text = "".join(clip_apply_lines(split_lines(text), p))
        try:
            ast.parse(_nb_sanitize(new_text))
        except (SyntaxError, ValueError):
            raise ValueError("internal check failed: cell %d would not parse; nothing was changed" % i)
        back, _ = clip_restore_lines(split_lines(new_text))
        if "".join(back) != text:
            raise ValueError("internal check failed: cell %d would not restore exactly; nothing was changed" % i)
        _nb_set_cell(cells[i], new_text)
        changed = True
    return (_nb_dump(nb, fmt) if changed else None), info


def _nb_dump(nb, fmt):
    indent, sk, ea, tail = fmt
    return (json.dumps(nb, indent=indent, sort_keys=sk, ensure_ascii=ea) + tail).encode("utf-8")


# ---- the .kenv/clip file

class ClipStore:
    def __init__(self, proj):
        self.path = proj.kdir / CLIP_FILE

    def load(self):
        d = read_json(self.path, {})
        if not isinstance(d, dict):
            d = {}
        d.setdefault("version", 1)
        d.setdefault("state", "active")   # active = the kenv lines are in the files; clipped = `kenv unclip` commented them out
        if not isinstance(d.get("files"), dict):
            d["files"] = {}
        return d

    def save(self, d):
        d["updated"] = iso()
        write_json(self.path, d)

    def ensure(self):
        if not self.path.exists():
            self.save(self.load())

    def record(self, d, rel, info, clipped):
        ent = {"kind": info["kind"], "statements": info["statements"], "seen": iso(), "clipped": clipped}
        if info["kind"] == "ipynb":
            ent["cells"] = info["cells"]
        else:
            ent["lines"] = info["lines"]
        d["files"][rel] = ent


def clip_walk(proj):
    """Every .py / .ipynb file of the project that could hold kenv lines."""
    out = []
    for cur, dirs, files in os.walk(str(proj.root)):
        dirs[:] = [d for d in dirs if d not in CLIP_SKIP_DIRS and not os.path.islink(os.path.join(cur, d))]
        for f in files:
            if not f.endswith((".py", ".ipynb")):
                continue
            fp = os.path.join(cur, f)
            try:
                if os.path.islink(fp) or os.path.getsize(fp) > CLIP_MAX_FILE:
                    continue
            except OSError:
                continue
            rel = os.path.relpath(fp, str(proj.root)).replace(os.sep, "/")
            if rel == SHIM_FILE:
                continue
            out.append(rel)
    return sorted(out)


def clip_candidates(proj, store, explicit=None):
    """Files `kenv unclip` / the session start look at: the ones given, else the recorded ones plus any file whose
    text mentions kenv (the final decision is made by parsing it)."""
    if explicit:
        rels = []
        for x in explicit:
            p = Path(x).expanduser()
            p = p if p.is_absolute() else Path.cwd() / p
            try:
                rel = p.resolve().relative_to(proj.root).as_posix()
            except (ValueError, OSError):
                raise KenvError(f"{x} is not inside the project folder.")
            if not (proj.root / rel).is_file():
                raise KenvError(f"{x}: no such file.")
            rels.append(rel)
        return rels
    found = set(store.load()["files"])
    for rel in clip_walk(proj):
        try:
            data = (proj.root / rel).read_bytes()
        except OSError:
            continue
        if b"kenv" in data:
            found.add(rel)
    return sorted(r for r in found if (proj.root / r).is_file())


# ---- writing many files without ever losing one

def clip_stale_backups(proj, include_done=False):
    """Backups of operations that did not finish. Ones that were rolled back cleanly are not a problem."""
    d = proj.kdir / CLIP_BACKUP
    if not d.is_dir():
        return []
    out = []
    for p in sorted(x for x in d.iterdir() if x.is_dir()):
        if include_done or read_json(p / "manifest.json", {}).get("status") != "rolled-back":
            out.append(p)
    return out


def _atomic_write(fp, data, like=None):
    tmp = fp.with_name(fp.name + f".kenv-tmp{os.getpid()}")
    try:
        tmp.write_bytes(data)
        if like is not None:
            try:
                shutil.copymode(str(like), str(tmp))
            except OSError:
                pass
        return tmp
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def clip_write_all(proj, changes, label):
    """changes: {rel: new_bytes}. Backup first, then temp files, then replace; if anything fails every file is put back.
    The backup in .kenv/clip-backup is deleted only after everything succeeded."""
    if not changes:
        return
    originals = {rel: (proj.root / rel).read_bytes() for rel in changes}
    bdir = proj.kdir / CLIP_BACKUP / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    bdir.mkdir(parents=True, exist_ok=True)
    for rel, data in originals.items():
        (bdir / "files" / rel).parent.mkdir(parents=True, exist_ok=True)
        (bdir / "files" / rel).write_bytes(data)
    write_json(bdir / "manifest.json", {"op": label, "created": iso(), "files": sorted(changes), "status": "in-progress"})
    temps, done = {}, []
    try:
        for rel, data in changes.items():
            temps[rel] = _atomic_write(proj.root / rel, data, like=proj.root / rel)
        fail_at = os.environ.get("KENV_CLIP_TEST_FAIL")   # test hook: simulate a write error at the n-th file
        for n, rel in enumerate(changes, 1):
            if fail_at and fail_at.isdigit() and int(fail_at) == n:
                raise OSError("simulated write error (KENV_CLIP_TEST_FAIL)")
            os.replace(temps[rel], proj.root / rel)
            done.append(rel)
    except BaseException as e:
        problems = []
        for rel in done:  # put back what was already replaced
            try:
                t = _atomic_write(proj.root / rel, originals[rel], like=proj.root / rel)
                os.replace(t, proj.root / rel)
            except OSError as e2:
                problems.append(f"{rel}: {e2}")
        for t in temps.values():
            try:
                if t.exists():
                    t.unlink()
            except OSError:
                pass
        if not problems:
            try:
                write_json(bdir / "manifest.json", {"op": label, "created": iso(), "files": sorted(changes),
                                                    "status": "rolled-back"})
            except OSError:
                pass
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        msg = f"{label} failed ({e}); every file was left as it was."
        if problems:
            msg = f"{label} failed ({e}) and these files could not be put back: " + "; ".join(problems)
        raise KenvError(msg + f"\nYour originals are kept in {bdir}")
    for old in clip_stale_backups(proj, include_done=True):  # this run and earlier cleanly rolled-back runs are done
        if old == bdir or read_json(old / "manifest.json", {}).get("status") == "rolled-back":
            shutil.rmtree(old, ignore_errors=True)
    try:
        (proj.kdir / CLIP_BACKUP).rmdir()   # only succeeds when empty
    except OSError:
        pass


def clip_recover(proj):
    """Copy the files of interrupted backups back over the project."""
    dirs = clip_stale_backups(proj)
    if not dirs:
        say("[kenv] no interrupted unclip to recover from")
        return 0
    n = 0
    for d in dirs:
        man = read_json(d / "manifest.json", {})
        for rel in man.get("files", []):
            src = d / "files" / rel
            if src.is_file():
                fp = proj.root / rel
                fp.parent.mkdir(parents=True, exist_ok=True)
                os.replace(_atomic_write(fp, src.read_bytes(), like=fp if fp.exists() else None), fp)
                n += 1
        shutil.rmtree(d, ignore_errors=True)
    try:
        (proj.kdir / CLIP_BACKUP).rmdir()
    except OSError:
        pass
    say(f"[kenv] put back {n} file(s) from the backup")
    return 0


# ---- operations

def clip_scan_project(proj, store, rels, quiet=False):
    """Read-only: record which kenv lines each file holds. -> {rel: info}"""
    d = store.load()
    res = {}
    for rel in rels:
        try:
            data = (proj.root / rel).read_bytes()
        except OSError:
            continue
        _, info = clip_file_bytes(data, rel, "scan")
        was_clipped = has_clip_marks(data.decode("utf-8", "replace"))
        if info["error"]:
            res[rel] = info
            continue
        if info["optin"] or rel in d["files"] or was_clipped:
            res[rel] = info
            if info["optin"] or rel in d["files"]:
                if not was_clipped:  # do not overwrite the recorded line numbers of a clipped file
                    store.record(d, rel, info, False)
    store.save(d)
    return res


def clip_do_unclip(proj, rels, dry=False, force=False, label="unclip"):
    """Comment out the kenv lines of the opted-in files -> (changed files, skipped [(rel, why)], infos)."""
    store = ClipStore(proj)
    d = store.load()
    changes, skipped, infos = {}, [], {}
    for rel in rels:
        try:
            data = (proj.root / rel).read_bytes()
        except OSError as e:
            skipped.append((rel, str(e)))
            continue
        new, info = clip_file_bytes(data, rel, "clip", force=force)
        infos[rel] = info
        if info["error"]:
            skipped.append((rel, info["error"]))
        elif not info["optin"]:
            if has_clip_marks(data.decode("utf-8", "replace")):
                continue  # already clipped
            if rel in d["files"]:
                skipped.append((rel, "no kenv.clip() in this file any more"))
        elif info["blocked"] and not force:
            why = "; ".join(dict.fromkeys(msg for _, msg in info["warnings"][:3]))
            skipped.append((rel, f"not changed - {why}  (use -f to comment out what can be, or `from kenv_shim import kenv`)"))
        elif new is not None:
            changes[rel] = new
    if dry or not changes:
        return changes, skipped, infos, d
    clip_write_all(proj, changes, label)
    for rel in changes:
        store.record(d, rel, infos[rel], True)
    d["state"] = "clipped"
    store.save(d)
    return changes, skipped, infos, d


def clip_restore_project(proj, rels=None, quiet=False):
    """Bring the commented kenv lines back. -> (files restored, lines restored)"""
    store = ClipStore(proj)
    d = store.load()
    cand = rels if rels is not None else clip_candidates(proj, store)
    changes, counts = {}, {}
    for rel in cand:
        try:
            data = (proj.root / rel).read_bytes()
        except OSError:
            continue
        if b"#kenv:clip" not in data:
            continue
        new, info = clip_file_bytes(data, rel, "restore")
        if info["error"]:
            if not quiet:
                say(c(f"[kenv] could not restore kenv lines in {rel}: {info['error']}", "33"))
            continue
        if new is not None:
            changes[rel] = new
            counts[rel] = info["restored"]
    if changes:
        clip_write_all(proj, changes, "restoring kenv lines")
        for rel, data in changes.items():
            _, info = clip_file_bytes(data, rel, "scan")
            store.record(d, rel, info, False)
    for rel in [r for r in d["files"] if not (proj.root / r).is_file()]:
        del d["files"][rel]
    if d.get("state") != "active" or changes:
        d["state"] = "active"
        for ent in d["files"].values():
            ent["clipped"] = False
        store.save(d)
    return len(changes), sum(counts.values())


def clip_session_start(proj):
    """`kenv init`: create .kenv/clip and put back the kenv lines that `kenv unclip` commented out. Never blocks a session."""
    try:
        ClipStore(proj).ensure()
        stale = clip_stale_backups(proj)
        if stale:
            say(c(f"[kenv] An unclip/restore was interrupted earlier: your original files are in {stale[-1]}.\n"
                  "       `kenv unclip --recover` puts them back.", "33"))
        nf, nl = clip_restore_project(proj)
        if nf:
            say(f"[kenv] Restored {nl} commented-out kenv line(s) in {nf} file(s) (clipped by `kenv unclip`)")
    except (KenvError, OSError) as e:
        say(c(f"[kenv] Could not restore the kenv lines: {e}", "33"))


# ---- commands

def _print_warnings(rel, info):
    for ln, msg in info["warnings"]:
        if isinstance(ln, int) and not msg.startswith("cell"):
            say(c(f"  {rel}:{ln}: {msg}", "33"))
        else:
            say(c(f"  {rel}: {msg}", "33"))


def cmd_clip(a):
    """kenv clip [files]   - record which files opted in with kenv.clip() and which lines are kenv commands (read-only)"""
    proj = current_project()
    store = ClipStore(proj)
    store.ensure()
    rels = clip_candidates(proj, store, a.args or None)
    res = clip_scan_project(proj, store, rels)
    d = store.load()
    shown = 0
    for rel, info in sorted(res.items()):
        if info["error"]:
            say(c(f"  {rel}: {info['error']}", "33"))
            continue
        if not (info["optin"] or rel in d["files"]):
            continue
        shown += 1
        where = (f"cells {', '.join(sorted(info['cells'], key=int))}" if info["kind"] == "ipynb"
                 else (f"lines {_fmt_lines(info['lines'])}" if info["lines"] else "no kenv lines"))
        mark = "opted in" if info["optin"] else "clipped"
        say(f"  {rel}: {info['statements']} kenv statement(s), {where}   [{mark}]")
        _print_warnings(rel, info)
    if not shown:
        say("[kenv] no file has kenv.clip() yet. Put `import kenv` and `kenv.clip()` at the top of a file that uses kenv commands.")
    else:
        say(f"[kenv] state: {d['state']}   (`kenv unclip` comments these lines out; a session start brings them back)")
    return 0


def _fmt_lines(nums):
    nums = sorted(set(nums))
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(parts)


def cmd_unclip(a):
    """kenv unclip [files] [--dry-run] [-f] | --undo | --staged | --recover | hook | unhook"""
    sub = a.args[0].lower() if a.args else ""
    if sub in ("hook", "unhook"):
        return clip_hook(a, sub == "hook")
    proj = current_project()
    if a.recover:
        return clip_recover(proj)
    if a.staged:
        return clip_staged(proj, a)
    store = ClipStore(proj)
    store.ensure()
    stale = clip_stale_backups(proj)
    if stale:
        say(c(f"[kenv] An earlier unclip/restore was interrupted; your originals are in {stale[-1]}\n"
              "       Put them back first:  kenv unclip --recover", "33"))
        return 1
    if a.undo:
        nf, nl = clip_restore_project(proj, clip_candidates(proj, store, a.args or None) if a.args else None)
        say(f"[kenv] restored {nl} kenv line(s) in {nf} file(s)" if nf else "[kenv] nothing to restore")
        return 0
    live = proj.live_session()
    rels = clip_candidates(proj, store, a.args or None)
    changes, skipped, infos, d = clip_do_unclip(proj, rels, dry=a.dry_run, force=a.force)
    for rel in sorted(changes):
        inf = infos[rel]
        where = (f"{inf['statements']} statement(s) in {len(inf['cells'])} cell(s)" if inf["kind"] == "ipynb"
                 else f"{inf['statements']} statement(s), lines {_fmt_lines(inf['lines'])}")
        say(f"  {'would comment out' if a.dry_run else 'commented out'}: {rel}  ({where})")
        _print_warnings(rel, inf)
    for rel, why in skipped:
        say(c(f"  skipped {rel}: {why}", "33"))
    if not changes:
        if not skipped:
            say("[kenv] nothing to comment out (no file has kenv.clip() with kenv lines; see `kenv clip`)")
        return 1 if skipped else 0
    if a.dry_run:
        say(f"[kenv] dry run: {len(changes)} file(s) would change")
        return 0
    say(f"[kenv] {len(changes)} file(s) saved. A new kenv session (or `kenv unclip --undo`) brings the lines back.")
    if live:
        say(c(f"[kenv] A session is running ('{live['name']}'): the next sync would send the commented files to the kernel. "
              "Run `kenv unclip --undo` before you carry on working.", "33"))
    return 0


def _git(proj, *args, input=None):
    try:
        return subprocess.run(["git", *args], cwd=str(proj.root), input=input, capture_output=True)
    except OSError as e:
        raise KenvError(f"git is not available: {e}")


def clip_staged(proj, a):
    """The commit gets the commented version; your working files stay as they are (only the git index changes)."""
    p = _git(proj, "diff", "--cached", "--name-only", "--diff-filter=ACM", "--relative", "-z")
    if p.returncode != 0:
        raise KenvError("`git diff --cached` failed (is this folder inside a git repository?)")
    prefix = _git(proj, "rev-parse", "--show-prefix").stdout.decode("utf-8", "replace").strip()
    rels = [x.decode("utf-8", "replace") for x in p.stdout.split(b"\0") if x]
    n = 0
    for rel in rels:
        if not rel.endswith((".py", ".ipynb")):
            continue
        blob = _git(proj, "show", f":./{rel}")
        if blob.returncode != 0:
            continue
        new, info = clip_file_bytes(blob.stdout, rel, "clip", force=a.force)
        if info["error"]:
            say(c(f"[kenv] {rel}: {info['error']}", "33"))
            continue
        if info["optin"] and info["blocked"] and not a.force:
            say(c(f"[kenv] {rel}: has kenv uses that cannot be commented out ({info['warnings'][0][1]}); committed as it is", "33"))
            continue
        if new is None:
            continue
        sha = _git(proj, "hash-object", "-w", "--stdin", input=new)
        ls = _git(proj, "ls-files", "-s", "--", rel)
        if sha.returncode != 0 or ls.returncode != 0 or not ls.stdout.strip():
            raise KenvError(f"could not update the staged copy of {rel}")
        mode = ls.stdout.split()[0].decode()
        up = _git(proj, "update-index", "--cacheinfo", f"{mode},{sha.stdout.decode().strip()},{prefix}{rel}")
        if up.returncode != 0:
            raise KenvError(f"git update-index failed for {rel}: {up.stderr.decode('utf-8', 'replace').strip()}")
        n += 1
        say(f"[kenv] commit gets {rel} with its kenv lines commented out (your working file is unchanged)")
    if not n:
        say("[kenv] no staged file needed unclipping")
    return 0


def _hook_block(proj):
    return (f"{CLIP_HOOK_BEGIN}\n(cd \"{proj.root.as_posix()}\" && \"{Path(sys.executable).as_posix()}\" "
            f"\"{Path(__file__).resolve().as_posix()}\" unclip --staged) || exit 1\n{CLIP_HOOK_END}\n")


def split_clip_block(text):
    """-> (text without the kenv clip block, the block or '')"""
    a, b = text.find(CLIP_HOOK_BEGIN), text.find(CLIP_HOOK_END)
    if a < 0 or b < a:
        return text, ""
    end = b + len(CLIP_HOOK_END)
    if text[end:end + 1] == "\n":
        end += 1
    return text[:a] + text[end:], text[a:end]


def _insert_block(rest, block):
    if not rest.strip():
        return "#!/bin/sh\n" + block
    first, _, tail = rest.partition("\n")
    return first + "\n" + block + tail


def clip_hook(a, install):
    """kenv unclip hook | unhook: a pre-commit hook that comments out kenv lines in what is being committed."""
    proj = current_project()
    gitdir = proj.root / ".git"
    if not gitdir.is_dir():
        raise KenvError("No .git folder in the project root, so there is no git repository to hook.")
    hook = gitdir / "hooks" / "pre-commit"
    text = hook.read_text(encoding="utf-8", errors="replace") if hook.is_file() else ""
    rest, block = split_clip_block(text)
    if not install:
        if not block:
            say("[kenv] no kenv unclip hook to remove")
            return 0
        if rest.strip() in ("", "#!/bin/sh"):
            hook.unlink()
        else:
            hook.write_text(rest, encoding="utf-8")
        say("[kenv] removed the pre-commit unclip hook")
        return 0
    if block:
        hook.write_text(_insert_block(rest, _hook_block(proj)), encoding="utf-8")
        say("[kenv] the pre-commit unclip hook is already installed (refreshed)")
        return 0
    hook.parent.mkdir(parents=True, exist_ok=True)
    if not text:
        hook.write_text("#!/bin/sh\n" + _hook_block(proj), encoding="utf-8")
    else:
        first = text.split("\n", 1)[0]
        if not (first.startswith("#!") and "sh" in first) or not (HOOK_MARK in text or a.force):
            raise KenvError("A different pre-commit hook already exists. Keep it, or add the unclip step to it with "
                            "`kenv unclip hook --force` (it is inserted after the first line of a shell hook).")
        hook.write_text(_insert_block(text, _hook_block(proj)), encoding="utf-8")
    try:
        hook.chmod(0o755)
    except OSError:
        pass
    say("[kenv] installed the pre-commit hook: commits get a copy of your files with the kenv lines commented out;\n"
        "       your working files are not touched (bypass once: git commit --no-verify)")
    return 0


# ----------------------------------------------------------------------------- Phase 6: lazy local access (experimental)
#
#   kenv init --lazy-local [--tunnel cloudflared|ngrok]
#
# Your machine runs a READ-ONLY file server limited to the project folder, protected by a random token and reachable
# through a tunnel (cloudflared by default). The kernel fetches files on demand (kenv_lazy.py, patched open/os.*),
# caches them, and only small code/config files are synced the normal way. The server, the tunnel and the token live
# exactly as long as the session. Nothing outside the project folder is ever served.

LAZY_EAGER_EXT = {".py", ".ipynb", ".sh", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".txt", ".md", ".json", ".rst"}
LAZY_EAGER_MAX = 2 * 1024 * 1024   # bigger files (and all other types) are fetched on demand
LAZY_DENY_DIRS = {".kenv", ".git", ".ssh", ".aws", ".gnupg"}
LAZY_DENY_NAMES = ("*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "id_ed25519*", "id_ecdsa*", "id_dsa*", ".env", ".env.*",
                   "*.env", ".netrc", ".pypirc", "kaggle.json", ".git-credentials", "credentials", "credentials.json",
                   "secrets.json", "secrets.yaml", "secrets.yml", "secrets.toml")
LAZY_LIST_MAX = 200000


def lazy_eager(rel, size):
    """Files that are uploaded the normal way in lazy mode: small code and config files."""
    return size <= LAZY_EAGER_MAX and posixpath.splitext(rel)[1].lower() in LAZY_EAGER_EXT


class LazyDenied(Exception):
    def __init__(self, code, msg):
        Exception.__init__(self, msg)
        self.code, self.msg = code, msg


def _lazy_name_denied(name):
    n = name.lower()
    return any(fnmatch.fnmatch(n, pat) for pat in LAZY_DENY_NAMES)


def lazy_resolve(root_real, rel):
    """A project-relative path as sent by the kernel -> an absolute path INSIDE the project, or LazyDenied.
    400 = malformed or traversal, 403 = a place that is never served (secrets, .git, .kenv, symlinks leaving the project)."""
    if not isinstance(rel, str) or "\x00" in rel or len(rel) > 4096:
        raise LazyDenied(400, "bad path")
    if rel.endswith("/") and len(rel) > 1:
        rel = rel[:-1]
    if rel in ("", "."):
        parts = []
    else:
        if rel.startswith("/") or "\\" in rel or re.match(r"^[A-Za-z]:", rel):
            raise LazyDenied(400, "paths must be relative to the project")
        parts = rel.split("/")
        if any(p in ("", ".", "..") for p in parts):
            raise LazyDenied(400, "path traversal is not allowed")
    if any(p.lower() in LAZY_DENY_DIRS for p in parts) or (parts and _lazy_name_denied(parts[-1])):
        raise LazyDenied(403, "not served")
    target = os.path.realpath(os.path.join(root_real, *parts))
    try:
        inside = os.path.commonpath([os.path.normcase(root_real), os.path.normcase(target)]) == os.path.normcase(root_real)
    except ValueError:
        inside = False
    if not inside:
        raise LazyDenied(403, "outside the project")
    if target != root_real:
        rp = os.path.relpath(target, root_real).replace(os.sep, "/").split("/")
        if any(p.lower() in LAZY_DENY_DIRS for p in rp) or _lazy_name_denied(rp[-1]):
            raise LazyDenied(403, "not served")
    return target


class LazyServer:
    """Read-only HTTP file server for one folder. Every request needs `Authorization: Bearer <token>`."""

    def __init__(self, root):
        self.root = os.path.realpath(str(root))
        self.token = secrets.token_hex(24)
        self.stats = {"requests": 0, "files": 0, "bytes": 0, "denied": 0}
        self.httpd = None
        self.port = None

    def start(self):
        srv = self
        root = self.root

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(self, code, obj=None, extra=None, body=None):
                data = body if body is not None else json.dumps(obj if obj is not None else {}).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(data)

            def _auth(self):
                got = self.headers.get("Authorization", "")
                got = got[7:] if got.startswith("Bearer ") else ""
                if not hmac.compare_digest(got.encode("utf-8", "replace"), srv.token.encode()):
                    srv.stats["denied"] += 1
                    time.sleep(0.25)   # slows down guessing
                    self._send(401, {"error": "unauthorized"})
                    return False
                return True

            def _refuse(self):
                self._send(405, {"error": "this server is read-only"}, {"Allow": "GET, HEAD"})

            do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _refuse

            def do_HEAD(self):
                self.do_GET()

            def do_GET(self):
                if not self._auth():
                    return
                srv.stats["requests"] += 1
                u = urllib.parse.urlparse(self.path)
                q = urllib.parse.parse_qs(u.query, keep_blank_values=True)
                rel = (q.get("p") or [""])[0]
                try:
                    if u.path == "/ping":
                        return self._send(200, {"ok": True})
                    if u.path not in ("/stat", "/list", "/file"):
                        return self._send(404, {"error": "unknown endpoint"})
                    try:
                        target = lazy_resolve(root, rel)
                    except LazyDenied as e:
                        srv.stats["denied"] += 1
                        return self._send(e.code, {"error": e.msg})
                    if u.path == "/stat":
                        return self._stat(target)
                    if u.path == "/list":
                        return self._list(target)
                    return self._file(target)
                except (BrokenPipeError, ConnectionError):
                    return
                except OSError:
                    try:
                        self._send(404, {"error": "not found"})
                    except OSError:
                        pass

            def _stat(self, target):
                try:
                    st = os.stat(target)
                except OSError:
                    return self._send(200, {"type": "none"})
                if stat.S_ISDIR(st.st_mode):
                    return self._send(200, {"type": "dir", "mtime_ns": st.st_mtime_ns})
                if stat.S_ISREG(st.st_mode):
                    return self._send(200, {"type": "file", "size": st.st_size, "mtime_ns": st.st_mtime_ns})
                return self._send(200, {"type": "none"})

            def _list(self, target):
                if not os.path.isdir(target):
                    return self._send(404, {"error": "not a directory"})
                ents = []
                with os.scandir(target) as it:
                    for e in it:
                        if len(ents) >= LAZY_LIST_MAX:
                            break
                        if e.name.lower() in LAZY_DENY_DIRS or _lazy_name_denied(e.name):
                            continue
                        try:
                            real = os.path.realpath(e.path)
                            if os.path.commonpath([os.path.normcase(root), os.path.normcase(real)]) != os.path.normcase(root):
                                continue   # a symlink that leaves the project
                            st = os.stat(real)
                        except (OSError, ValueError):
                            continue
                        if stat.S_ISDIR(st.st_mode):
                            ents.append([e.name, "dir", 0, st.st_mtime_ns])
                        elif stat.S_ISREG(st.st_mode):
                            ents.append([e.name, "file", st.st_size, st.st_mtime_ns])
                self._send(200, {"entries": ents})

            def _file(self, target):
                flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
                try:
                    fd = os.open(target, flags)
                except OSError:
                    return self._send(404, {"error": "not found"})
                with os.fdopen(fd, "rb") as f:
                    st = os.fstat(f.fileno())
                    if not stat.S_ISREG(st.st_mode):
                        return self._send(404, {"error": "not a file"})
                    size, start, end, code = st.st_size, 0, st.st_size - 1, 200
                    rng = self.headers.get("Range", "")
                    m = re.match(r"^bytes=(\d*)-(\d*)$", rng.strip())
                    if m and (m.group(1) or m.group(2)):
                        if m.group(1):
                            start = int(m.group(1))
                            end = int(m.group(2)) if m.group(2) else size - 1
                        else:   # the last N bytes
                            start, end = max(0, size - int(m.group(2))), size - 1
                        end = min(end, size - 1)
                        if start >= size or start > end:
                            return self._send(416, {"error": "range not satisfiable"}, {"Content-Range": "bytes */%d" % size})
                        code = 206
                    n = end - start + 1 if size else 0
                    self.send_response(code)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(n))
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("X-Kenv-Size", str(size))
                    self.send_header("X-Kenv-Mtime-Ns", str(st.st_mtime_ns))
                    if code == 206:
                        self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
                    self.end_headers()
                    if self.command == "HEAD":
                        return
                    f.seek(start)
                    left = n
                    while left > 0:
                        blk = f.read(min(1024 * 1024, left))
                        if not blk:
                            break
                        self.wfile.write(blk)
                        left -= len(blk)
                        srv.stats["bytes"] += len(blk)
                    srv.stats["files"] += 1

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def stop(self):
        if self.httpd is not None:
            h, self.httpd = self.httpd, None
            try:
                h.shutdown()
                h.server_close()
            except Exception:
                pass


# ---- tunnels from the local machine

def find_cloudflared(allow_download=True):
    p = shutil.which("cloudflared")
    if p:
        return p
    binname = "cloudflared.exe" if os.name == "nt" else "cloudflared"
    dest = Path.home() / ".kenv" / "bin" / binname
    if dest.is_file():
        return str(dest)
    if not allow_download:
        return None
    sysn, mach = platform.system().lower(), platform.machine().lower()
    arch = "arm64" if mach in ("arm64", "aarch64") else ("386" if mach in ("i386", "i686", "x86") else "amd64")
    asset = {"windows": f"cloudflared-windows-{arch}.exe", "darwin": f"cloudflared-darwin-{arch}.tgz",
             "linux": f"cloudflared-linux-{arch}"}.get(sysn)
    if not asset:
        raise KenvError(f"No cloudflared download for {platform.system()}. Install cloudflared yourself, or use --tunnel ngrok.")
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/" + asset
    say(f"[kenv] downloading cloudflared ({asset}) from github.com/cloudflare/cloudflared ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".download")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=180) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        if asset.endswith(".tgz"):
            import tarfile
            with tarfile.open(tmp) as t:
                member = next((m for m in t.getmembers() if m.isfile() and os.path.basename(m.name) == "cloudflared"), None)
                if member is None:
                    raise KenvError("the cloudflared archive did not contain the program")
                with t.extractfile(member) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
        else:
            os.replace(tmp, dest)
        dest.chmod(0o755)
    except (OSError, urllib.error.URLError) as e:
        raise KenvError(f"Could not download cloudflared: {e}\nInstall it from https://developers.cloudflare.com/cloudflare-one/"
                        "connections/connect-networks/downloads/ or use --tunnel ngrok.")
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    return str(dest)


def _pdeathsig():
    """Linux: the tunnel dies with kenv even if kenv is killed."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
    except Exception:
        pass


def _popen_tunnel(cmd, env=None):
    kw = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True, "env": env}
    if sys.platform.startswith("linux"):
        kw["preexec_fn"] = _pdeathsig
    return subprocess.Popen(cmd, **kw)


def start_cloudflared_tunnel(port):
    cf = find_cloudflared()
    last = ""
    for attempt in range(3):
        p = _popen_tunnel([cf, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate", "--protocol", "http2"])
        found, lines = [], []

        def pump(p=p, found=found, lines=lines):
            for ln in p.stdout:   # keep draining so the pipe never fills
                lines.append(ln)
                del lines[:-20]
                m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", ln)
                if m and not found:
                    found.append(m.group(0))

        threading.Thread(target=pump, daemon=True).start()
        t0 = time.time()
        while time.time() - t0 < 60 and not found and p.poll() is None:
            time.sleep(0.4)
        if found:
            return found[0], p
        last = "".join(lines)[-300:]
        stop_proc(p)
    raise KenvError("cloudflared could not open a tunnel" + (f":\n{last.strip()}" if last.strip() else "."))


def start_ngrok_tunnel(port):
    ng = shutil.which("ngrok")
    if not ng:
        raise KenvError("ngrok was not found on your PATH. Install it from https://ngrok.com/download, sign up (free) and run\n"
                        "  ngrok config add-authtoken <your token>\n"
                        "The free plan has bandwidth and connection limits; cloudflared (the default) needs no account.")
    p = _popen_tunnel([ng, "http", f"127.0.0.1:{port}", "--log=stdout", "--log-format=json"])
    found, lines = [], []

    def pump():
        for ln in p.stdout:
            lines.append(ln)
            del lines[:-20]
            try:
                ev = json.loads(ln)
            except ValueError:
                continue
            u = ev.get("url") or ""
            if u.startswith("https://") and not found:
                found.append(u)

    threading.Thread(target=pump, daemon=True).start()
    t0 = time.time()
    while time.time() - t0 < 45 and not found and p.poll() is None:
        if time.time() - t0 > 4:   # the local API of ngrok is the other place the address shows up
            try:
                with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=3) as r:
                    for t in json.loads(r.read()).get("tunnels", []):
                        if str(t.get("public_url", "")).startswith("https://") and f":{port}" in str(t.get("config", {}).get("addr", "")):
                            found.append(t["public_url"])
                            break
            except Exception:
                pass
        time.sleep(0.5)
    if found:
        return found[0], p
    out = "".join(lines)[-400:]
    stop_proc(p)
    hint = "\nAdd your auth token: ngrok config add-authtoken <token>" if re.search(r"authtoken|ERR_NGROK_4018", out, re.I) else ""
    raise KenvError("ngrok could not open a tunnel." + hint + (f"\n{out.strip()}" if out.strip() else ""))


def stop_proc(p):
    if p is None:
        return
    try:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=4)
            except subprocess.TimeoutExpired:
                p.kill()
    except OSError:
        pass


def _proc_looks_like_tunnel(pid):
    try:
        name = Path(f"/proc/{int(pid)}/comm").read_text().strip().lower()
        return name.startswith(("cloudflared", "ngrok"))
    except OSError:
        return True   # cannot tell (Windows, macOS): trust the recorded pid


def kill_lazy(st):
    """Stop the tunnel recorded in a session's state (kenv stop, --sweep)."""
    pid = (st or {}).get("lazy_pid")
    if pid and pid_alive(pid) and _proc_looks_like_tunnel(pid):
        try:
            os.kill(int(pid), signal.SIGTERM)
        except OSError:
            pass


def lazy_on(ep):
    return bool((load_state(ep.name) or {}).get("lazy"))


def lazy_preflight(proj, a):
    """Before any Kaggle quota is spent: say what lazy mode exposes, ask, make sure the tunnel program exists."""
    tunnel = (a.tunnel or "cloudflared").lower()
    if tunnel not in ("cloudflared", "ngrok"):
        raise KenvError("--tunnel must be cloudflared or ngrok")
    say(c("[kenv] --lazy-local is EXPERIMENTAL.", "33;1"))
    say(f"  * For this session the kernel can READ the files of {proj.root} through a tunnel ({tunnel}).")
    say("  * The address is random and every request needs a random token that only the kernel gets; the server is read-only,")
    say("    serves nothing outside the folder (no .kenv, .git, key files, .env), and stops with the session.")
    say("  * Only small code/config files are uploaded the normal way; everything else is fetched when your code reads it.")
    say("  * C-level file readers bypass the patch (use kenv.lazy_path), and cache misses cross your home internet connection.")
    if sys.stdin.isatty() and not a.force:
        if not confirm("Let the kernel read this folder for the session?"):
            raise KenvError("Cancelled before starting a kernel.")
    if tunnel == "cloudflared" and find_cloudflared(allow_download=False) is None:
        if sys.stdin.isatty() and not a.force and not confirm("cloudflared (Cloudflare's tunnel program) is not installed. "
                                                              "Download it to ~/.kenv/bin?"):
            raise KenvError("Install cloudflared yourself, or use --tunnel ngrok.")
        find_cloudflared()
    elif tunnel == "ngrok" and not shutil.which("ngrok"):
        start_ngrok_tunnel(0)   # raises with the install instructions
    return tunnel


class LazyAccess:
    """The server + tunnel of one session; posts the address to the kernel and watches the tunnel."""

    def __init__(self, proj, ep, tunnel):
        self.proj, self.ep, self.kind = proj, ep, tunnel
        self.server = LazyServer(proj.root)
        self.proc, self.url = None, None
        self._stop, self._lock = threading.Event(), threading.Lock()
        self.stopped = False

    def _open_tunnel(self):
        self.url, self.proc = (start_ngrok_tunnel if self.kind == "ngrok" else start_cloudflared_tunnel)(self.server.port)
        st = load_state(self.ep.name)
        if st is not None:
            st["lazy_pid"] = self.proc.pid
            save_state(st)

    def _post(self):
        r = self.ep.json_call("/lazy/config", payload={"url": self.url, "token": self.server.token, "root": str(self.proj.root),
                                                       "style": "nt" if os.name == "nt" else "posix"}, timeout=180, tries=2)
        if not r.get("ok"):
            raise KenvError(f"the kernel did not accept lazy access: {r.get('error')}")
        if not r.get("reachable"):
            raise KenvError("the kernel cannot reach the tunnel to your machine "
                            f"({r.get('error', 'no answer')}). Try again, or use --tunnel ngrok / a normal sync.")

    def start(self):
        say(f"[kenv] Lazy local access: starting a read-only file server for {self.proj.root} ...")
        self.server.start()
        try:
            self._open_tunnel()
            self._post()
        except BaseException:
            self.stop()
            raise
        threading.Thread(target=self._watch, daemon=True).start()
        say(c("[kenv] Lazy local access is on", "32;1") + "  (the kernel reads files of this folder on demand; nothing is served outside it)")

    def _watch(self):
        bad = 0
        while not self._stop.wait(45):
            if self.proc is not None and self.proc.poll() is not None:   # the tunnel died: open a new one
                try:
                    with self._lock:
                        self._open_tunnel()
                        self._post()
                    say(c("[kenv] Lazy local access: the tunnel was restarted", "33"))
                except (KenvError, OSError):
                    pass
            if kernel_alive(self.ep, timeout=15):
                bad = 0
            else:
                bad += 1
                if bad >= 4:   # the kernel is gone: nothing needs your files any more
                    self.stop()
                    return

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        self._stop.set()
        stop_proc(self.proc)
        self.server.stop()
        s = self.server.stats
        if s["requests"]:
            say(f"[kenv] Lazy local access closed: {s['files']} file(s), {human(s['bytes'])} served to the kernel")
        st = load_state(self.ep.name)
        if st and st.get("lazy_pid"):
            st.pop("lazy_pid", None)
            save_state(st)


# ----------------------------------------------------------------------------- Phase 5: dashboard (kenv ui)
#
# A local, READ-ONLY web dashboard over .kenv. This file only holds the stdlib server and the JSON API; the
# React + Ant Design bundle is a separate folder (kenv_ui/, built from kenv_ui_src/) so kenv.py stays stdlib-only.
#   * binds to 127.0.0.1 only, random port, random access token (URL, cookie or X-Kenv-Token header)
#   * GET only; serves files from the asset folder and answers /api/* from .kenv/ (never from the project's own files)
#   * every string that leaves the server passes through a credential filter; secrets appear by NAME only

UI_MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
           ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon", ".woff2": "font/woff2",
           ".json": "application/json", ".txt": "text/plain; charset=utf-8"}
UI_LOG_WINDOW = 16 * 1024 * 1024    # the newest 16 MB of a log are searchable; older text is only on disk
UI_LOG_MAX = 2000
UI_PATCH_FILES = 40
UI_PATCH_LINES = 400
UI_MAX_RUNS = 600
UI_BG_IDLE_MIN = 60                 # a dashboard started from code stops itself after this many idle minutes
UI_ANNOUNCE = "KENV-UI-URL "
UI_LOG_LINE = re.compile(r"^(\d{4}-\d\d-\d\dT[\d:.]+Z) \[([^\]]*)\] ?(.*)$")
UI_EXTRA_REDACT = [re.compile(r"kenv://\S+"), re.compile(r"\b[a-z0-9][a-z0-9-]*#[0-9a-f]{32}\b")]


class UiNotFound(Exception):
    pass


def ui_asset_dirs():
    out = []
    if os.environ.get("KENV_UI_DIR"):
        out.append(Path(os.environ["KENV_UI_DIR"]).expanduser())
    out.append(Path(__file__).resolve().parent / "kenv_ui")
    out.append(Path.home() / ".kenv" / "ui")
    return out


def find_ui_assets():
    for d in ui_asset_dirs():
        if (d / "index.html").is_file() and (d / "app.js").is_file():
            return d.resolve()
    return None


def ui_missing_message():
    where = "\n".join(f"    {d}" for d in ui_asset_dirs())
    return ("The dashboard files (the prebuilt React + Ant Design bundle) were not found. kenv.py itself stays stdlib-only, so they\n"
            "ship as a separate folder, kenv_ui/, with index.html and app.js. Install them one of these ways:\n"
            "  1. Copy the kenv_ui folder that came with kenv next to kenv.py.\n"
            "  2. Put it in ~/.kenv/ui, or point KENV_UI_DIR at it.\n"
            "  3. Build it yourself (needs Node 18+):  cd kenv_ui_src && npm install && npm run build\n"
            "kenv looked in:\n" + where)


def ui_redact(s):
    s = redact_secrets(s)
    for rx in UI_EXTRA_REDACT:
        s = rx.sub("<redacted>", s)
    return s


def ui_clean(o):
    if isinstance(o, str):
        return ui_redact(o)
    if isinstance(o, dict):
        return {str(k): ui_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [ui_clean(v) for v in o]
    return o


def ui_core(proj, v):
    try:
        return CoreStore(proj, v).load(), None
    except KenvError as e:
        return {}, str(e)


def ui_resolve(proj, ref, only):
    if not ref:
        raise UiNotFound("missing version")
    try:
        v = proj.find_version(ref)
    except KenvError as e:
        raise UiNotFound(str(e))
    if only and v != only:
        raise UiNotFound(f"this dashboard shows only {only}")
    return v


def ui_sessions(proj, v):
    out, now = [], time.time()
    for e in proj.sessions(v):
        start, end = from_iso(e.get("start")), from_iso(e.get("end"))
        running = not e.get("end") and pid_alive(e.get("pid"))
        secs = e.get("duration_s")
        if not e.get("end"):
            end = now if running else (from_iso(e.get("last_seen")) or start)
            secs = int(max(0, (end or 0) - (start or 0)))
        out.append({"sid": e.get("sid"), "start": e.get("start"), "end": iso(end) if end else None, "duration_s": secs,
                    "state": "running" if running else ("ended" if e.get("end") else "lost"),
                    "ended_by": e.get("ended_by"), "estimated": bool(e.get("estimated")) or (not e.get("end") and not running),
                    "gpu": e.get("gpu"), "gpu_log": e.get("gpu_log") or [], "idle_min": e.get("idle_min")})
    return out


def ui_version_row(proj, v):
    m = proj.meta(v)
    doc, err = ui_core(proj, v)
    last = proj.last_ts(v)
    return {"version": v, "id": m["id"], "name": m.get("name"), "label": proj.label(v), "tags": proj.tags_of(v),
            "branch": m.get("branch") or "main", "parent": m.get("parent"), "message": m.get("message"),
            "committed": m.get("committed"), "auto": bool(m.get("auto")), "created": m.get("created"),
            "files": m.get("files"), "sessions": len(proj.sessions(v)), "total_s": proj.total_time(v),
            "last_used": iso(last) if last else None, "active": v == proj.last_active(),
            "metrics": metrics_of(proj, v) if not err else {}, "resources": doc.get("resources") or {},
            "accelerator": (doc.get("manifest") or {}).get("accelerator"), "core_error": err}


def ui_overview(proj, only):
    vs = [only] if only else proj.version_names()
    rows = [ui_version_row(proj, v) for v in vs]
    keep = set(vs)
    br = proj.branches()
    branches = [{"name": "main", "parent": None, "versions": [v for v in proj.branch_versions("main") if v in keep]}]
    branches += [{"name": n, "parent": p, "versions": [v for v in proj.branch_versions(n) if v in keep]} for n, p in sorted(br.items())]
    if only:
        branches = [b for b in branches if b["versions"] or b["name"] == (proj.meta(only).get("branch") or "main")]
    return {"project": proj.root.name, "kenv": VERSION, "only": only, "active": proj.last_active(),
            "branch": proj.current_branch(), "live": bool(proj.live_session()), "versions": rows, "branches": branches,
            "tags": [{"name": k, "version": x} for k, x in sorted(proj.tags().items()) if x in keep],
            "quota": not only, "generated": iso()}


def ui_version(proj, v):
    m = proj.meta(v)
    doc, err = ui_core(proj, v)
    deps, io = doc.get("dependencies") or {}, doc.get("io") or {}
    ins, outs, runs = io.get("inputs") or {}, io.get("outputs") or {}, doc.get("runs") or {}
    lock = read_lock(proj.kdir / v / "deps.lock") or dict(deps.get("lock") or {})
    run_list = []
    for k in sorted((k for k in runs if re.fullmatch(r"r\d+", k)), key=lambda k: int(k[1:]), reverse=True)[:UI_MAX_RUNS]:
        if isinstance(runs[k], dict):
            run_list.append({"id": k, **{x: y for x, y in runs[k].items() if x != "evid"}})
    code = proj.load_snapshot(v)
    logdir = proj.kdir / v / LOG_DIR
    return {"version": v, "id": m["id"], "name": m.get("name"), "label": proj.label(v), "active": v == proj.last_active(),
            "tags": proj.tags_of(v), "core_error": err,
            "meta": {k: m.get(k) for k in ("created", "committed", "message", "parent", "branch", "auto", "files", "source")},
            "manifest": doc.get("manifest") or {}, "config": read_json(proj.kdir / v / "config.json", {}),
            "dependencies": {"lock": lock, "detected": deps.get("detected") or {}, "declared": deps.get("declared") or {}},
            "io": {"datasets": list(ins.get("datasets") or []), "mounted": list(ins.get("mounted") or []),
                   "outputs": [{"path": p, "size": e.get("size"), "sha256": e.get("sha256")}
                               for p, e in sorted(outs.items()) if isinstance(e, dict)]},
            "resources": doc.get("resources") or {}, "secrets": list((doc.get("secrets") or {}).get("names") or []),
            "sessions": ui_sessions(proj, v), "metrics": metrics_of(proj, v) if not err else {}, "runs": run_list,
            "code": [{"path": p, "size": e.get("size"), "sha256": e.get("sha256"), "skipped": bool(e.get("skipped"))}
                     for p, e in sorted(code.items()) if isinstance(e, dict)],
            "logs": {"run": (logdir / "run.log").is_file(), "errors": (logdir / "errors.log").is_file()}}


def ui_logs(proj, v, q="", errors=False, limit=500, skip=0):
    path = proj.kdir / v / LOG_DIR / ("errors.log" if errors else "run.log")
    if not path.is_file():
        return {"exists": False, "lines": [], "total": 0, "truncated": False}
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size > UI_LOG_WINDOW:
            f.seek(size - UI_LOG_WINDOW)
            f.readline()
        raw = f.read()
    needle = q.lower()
    hit = [ln for ln in raw.decode("utf-8", "replace").splitlines() if not needle or needle in ln.lower()]
    limit, skip = max(1, min(UI_LOG_MAX, limit)), max(0, skip)
    end = len(hit) - skip
    start = max(0, end - limit)
    rows = []
    for ln in (hit[start:end] if end > 0 else []):
        m = UI_LOG_LINE.match(ln)
        rows.append({"ts": m.group(1), "src": m.group(2), "text": m.group(3)} if m else {"ts": None, "src": "", "text": ln})
    return {"exists": True, "lines": rows, "total": len(hit), "older": start, "truncated": size > UI_LOG_WINDOW, "size": size}


def _ui_text(proj, v, rel):
    root = proj.kdir / v / "code"
    if not _inside(root, rel):
        return None
    return side_text(proj, {"v": v}, rel)


def ui_diff(proj, va, vb):
    A, B = side_view(proj, va), side_view(proj, vb)
    ac, bc = A["code"], B["code"]
    files = [{"path": r, "status": "added"} for r in sorted(r for r in bc if r not in ac)]
    files += [{"path": r, "status": "modified"} for r in sorted(r for r in bc if r in ac and ac[r] != bc[r])]
    files += [{"path": r, "status": "removed"} for r in sorted(r for r in ac if r not in bc)]
    for i, f in enumerate(files):
        if i >= UI_PATCH_FILES:
            f["patch"] = None
            continue
        ta = [] if f["status"] == "added" else _ui_text(proj, va, f["path"])
        tb = [] if f["status"] == "removed" else _ui_text(proj, vb, f["path"])
        if ta is None or tb is None:
            f["binary"] = True
            continue
        d = list(difflib.unified_diff(ta, tb, f"{va}/{f['path']}", f"{vb}/{f['path']}", lineterm="", n=2))
        f["plus"] = sum(1 for x in d if x.startswith("+") and not x.startswith("+++"))
        f["minus"] = sum(1 for x in d if x.startswith("-") and not x.startswith("---"))
        f["patch"], f["cut"] = d[:UI_PATCH_LINES], len(d) > UI_PATCH_LINES
    la, lb = {norm_pkg(k): (k, x) for k, x in A["libs"].items()}, {norm_pkg(k): (k, x) for k, x in B["libs"].items()}
    libs = [{"name": lb[k][0], "from": None, "to": lb[k][1]} for k in sorted(lb) if k not in la]
    libs += [{"name": la[k][0], "from": la[k][1], "to": None} for k in sorted(la) if k not in lb]
    libs += [{"name": lb[k][0], "from": la[k][1], "to": lb[k][1]} for k in sorted(lb) if k in la and la[k][1] != lb[k][1]]
    ma, mb = A["metrics"], B["metrics"]
    metrics = []
    for n in sorted(set(ma) | set(mb)):
        x, y = ma.get(n), mb.get(n)
        metrics.append({"name": n, "from": x, "to": y, "delta": (y - x) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None})
    ia, ib = set(A["inputs"]), set(B["inputs"])
    return {"from": va, "to": vb, "from_label": A["label"], "to_label": B["label"], "snapshots": [A["snap"], B["snap"]],
            "files": files, "libs": libs, "inputs": {"added": sorted(ib - ia), "removed": sorted(ia - ib)}, "metrics": metrics}


def ui_gpu_days(cfg, now):
    days, base = [0.0] * 7, int(now // 86400) * 86400
    for root in cfg["projects"]:
        p = Project(root)
        if not p.kdir.is_dir():
            continue
        for v in p.version_names():
            for e in p.sessions(v):
                for a_, b_ in _gpu_segments(e, now):
                    for i in range(7):
                        lo = base - (6 - i) * 86400
                        days[i] += max(0.0, min(b_, now, lo + 86400) - max(a_, lo))
    return [{"day": iso(base - (6 - i) * 86400)[:10], "hours": round(days[i] / 3600, 2)} for i in range(7)]


def ui_quota():
    cfg, now = quota_cfg(), time.time()
    q = quota_status(now)
    return {**q, "approximate": True, "days": ui_gpu_days(cfg, now), "projects_counted": len(cfg["projects"]),
            "note": ("An ESTIMATE from kenv's own session logs in the projects it knows about. Kaggle has no official API for "
                     "your quota, GPU use in the browser is not included, and the week may reset on another day.")}


def ui_api(proj, only, name, q):
    def one(k, d=""):
        return (q.get(k) or [d])[0]
    if name == "overview":
        return ui_overview(proj, only)
    if name == "version":
        return ui_version(proj, ui_resolve(proj, one("v"), only))
    if name == "logs":
        try:
            lim, skip = int(one("limit", "500")), int(one("skip", "0"))
        except ValueError:
            raise UiNotFound("limit and skip are numbers")
        return ui_logs(proj, ui_resolve(proj, one("v"), only), one("q")[:200], one("errors") == "1", lim, skip)
    if name == "diff":
        if only:
            raise UiNotFound("diffs compare two versions; this dashboard shows only " + only)
        return ui_diff(proj, ui_resolve(proj, one("from"), only), ui_resolve(proj, one("to"), only))
    if name == "quota":
        if only:
            raise UiNotFound("the quota estimate is not part of a single-version dashboard")
        return ui_quota()
    raise UiNotFound("unknown endpoint")


class UiServer:
    def __init__(self, proj, assets, only=None, port=0, idle_min=0):
        self.proj, self.assets, self.only, self.idle_min = proj, Path(assets), only, idle_min
        self.token = secrets.token_urlsafe(24)
        self.last = time.time()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), self._handler())   # loopback only, never 0.0.0.0
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.cookie = f"kenv_ui_{self.port}"

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/?t={self.token}"

    def _handler(self):
        srv = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "kenv-ui"

            def log_message(self, *a):
                pass

            def _send(self, code, body=b"", ctype="text/plain; charset=utf-8", extra=()):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                                 "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
                for k, v in extra:
                    self.send_header(k, v)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def _json(self, code, obj):
                self._send(code, json.dumps(ui_clean(obj), default=str).encode("utf-8"), "application/json; charset=utf-8")

            def _authorized(self, params):
                cands = [params.get("t", [""])[0], self.headers.get("X-Kenv-Token", "")]
                auth = self.headers.get("Authorization", "")
                if auth.lower().startswith("bearer "):
                    cands.append(auth[7:].strip())
                for part in self.headers.get("Cookie", "").split(";"):
                    k, _, val = part.strip().partition("=")
                    if k == srv.cookie:
                        cands.append(val)
                return any(hmac.compare_digest(c.encode("utf-8", "replace"), srv.token.encode()) for c in cands if c)

            def do_GET(self):
                srv.last = time.time()
                if self.headers.get("Host", "") not in (f"127.0.0.1:{srv.port}", f"localhost:{srv.port}"):
                    return self._send(403, b"kenv ui: wrong Host header")   # DNS-rebinding guard
                u = urllib.parse.urlsplit(self.path)
                params = urllib.parse.parse_qs(u.query)
                if not self._authorized(params):
                    return self._send(401, b"kenv ui: missing or wrong access token. Open the link that `kenv ui` printed.")
                path = urllib.parse.unquote(u.path)
                if "t" in params and not path.startswith("/api/"):   # first visit: trade the URL token for a cookie, hide it
                    rest = urllib.parse.urlencode([(k, x) for k, vs in params.items() if k != "t" for x in vs])
                    return self._send(302, b"", extra=[("Location", path + ("?" + rest if rest else "")),
                                      ("Set-Cookie", f"{srv.cookie}={srv.token}; Path=/; HttpOnly; SameSite=Strict")])
                if path.startswith("/api/"):
                    try:
                        return self._json(200, ui_api(srv.proj, srv.only, path[5:].strip("/"), params))
                    except UiNotFound as e:
                        return self._json(404, {"error": str(e)})
                    except KenvError as e:
                        return self._json(422, {"error": str(e)})
                    except Exception as e:
                        return self._json(500, {"error": f"{type(e).__name__}: {e}"})
                return self._static(path)

            do_HEAD = do_GET

            def _static(self, path):
                rel = "index.html" if path in ("", "/") else path.lstrip("/")
                fp = (srv.assets / rel)
                try:
                    fp = fp.resolve()
                    fp.relative_to(srv.assets)
                except (ValueError, OSError):
                    return self._send(404, b"not found")
                mime = UI_MIME.get(fp.suffix.lower())
                if not mime or not fp.is_file():
                    return self._send(404, b"not found")
                self._send(200, fp.read_bytes(), mime)

            def _no(self):
                self._send(405, b"kenv ui is read-only", extra=[("Allow", "GET, HEAD")])

            do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _no

        return H

    def serve(self):
        if self.idle_min:
            def watch():
                while time.time() - self.last < self.idle_min * 60:
                    time.sleep(5)
                self.httpd.shutdown()
            threading.Thread(target=watch, daemon=True).start()
        try:
            self.httpd.serve_forever(poll_interval=0.5)
        finally:
            self.httpd.server_close()


def ui_open_browser(url):
    try:
        import webbrowser
        return bool(webbrowser.open(url))
    except Exception:
        return False


def ui_spawn_background(a, proj):
    """`kenv.cli("kenv ui")` runs inside a command that must finish: start the server as a detached process, wait for its
    URL, open the browser, and keep the token out of the output (the output goes back to the kernel, i.e. to Kaggle)."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "ui", "--no-open", "--idle", str(a.idle or UI_BG_IDLE_MIN)]
    if a.uri or a.args:
        cmd += ["-uri", a.uri or a.args[0]]
    if a.port:
        cmd += ["--port", str(a.port)]
    env = {**os.environ, "KENV_UI_CHILD": "1", "NO_COLOR": "1", "PYTHONIOENCODING": "utf-8"}
    env.pop("KENV_FROM_CODE", None)
    kw = {"creationflags": 0x00000008 | 0x00000200} if os.name == "nt" else {"start_new_session": True}
    p = subprocess.Popen(cmd, cwd=str(proj.root), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, **kw)
    box = []
    t = threading.Thread(target=lambda: box.append(p.stdout.readline().decode("utf-8", "replace").strip()), daemon=True)
    t.start()
    t.join(20)
    line = box[0] if box else ""
    if not line.startswith(UI_ANNOUNCE):
        with contextlib.suppress(Exception):
            p.kill()
        raise KenvError("The dashboard did not start. Run `kenv ui` in your kenv terminal to see why.")
    ok = ui_open_browser(line[len(UI_ANNOUNCE):])
    say("[kenv] dashboard " + ("opened in the browser on your machine" if ok else "is running on your machine, but no browser could be opened; "
        "run `kenv ui` in your terminal to get the link") + f" (it stops after {a.idle or UI_BG_IDLE_MIN} idle minutes)")
    return 0


def cmd_ui(a):
    """kenv ui [-uri kv:<id>] [--port N] [--no-open] [--idle MIN]"""
    proj = current_project()
    assets = find_ui_assets()
    if not assets:
        raise KenvError(ui_missing_message())
    uri = a.uri or (a.args[0] if a.args else None)
    only = proj.find_version(uri) if uri else None      # kv:<id>, vN, a name or a tag: one version only
    if os.environ.get("KENV_FROM_CODE") and not os.environ.get("KENV_UI_CHILD"):
        return ui_spawn_background(a, proj)
    child = bool(os.environ.get("KENV_UI_CHILD"))
    if only and not child:
        if proj.live_session():
            say(c(f"[kenv] A session is running, so the active version stays {proj.label(proj.last_active())}; showing {proj.label(only)} only.", "33"))
        else:
            proj.set_active(only)
            say(f"[kenv] Active version: {proj.label(only)}   ({proj.meta(only)['id']})")
    try:
        srv = UiServer(proj, assets, only=only, port=int(a.port or 0), idle_min=int(a.idle or 0))
    except OSError as e:
        raise KenvError(f"Cannot start the dashboard server on 127.0.0.1:{a.port or 'any'} ({e}). Try another --port.")
    if child:
        say(UI_ANNOUNCE + srv.url)
        sys.stdout = sys.stderr = open(os.devnull, "w")   # the parent is gone: nothing may block on a closed pipe
    else:
        say(f"[kenv] Dashboard for {proj.root.name}" + (f" - only {proj.label(only)}" if only else "") + "  (read-only, this machine only)")
        say(f"  {c(srv.url, '36;1')}")
        say(c("  The link carries a private access token; anyone without it is rejected. Ctrl-C stops the server.", "2"))
        if not a.no_open and not ui_open_browser(srv.url):
            say("  (no browser could be opened - open the link yourself)")
    try:
        srv.serve()
    except KeyboardInterrupt:
        pass
    if not child:
        say("\n[kenv] dashboard stopped")
    return 0


HELP = """\
{banner}
USAGE
  kenv <command> [options]

START / JOIN A SESSION   (a SESSION = one disposable running kernel)
  kenv init                    start a session in THIS folder (the project) and open a kenv shell.
                               Creates .kenv/ the first time; later it resumes the last-used version.
  kenv -n "<name>"             start a session called <name>   (also: kenv init -n "<name>")
  kenv init --gpu              same, but pick a GPU at startup
  kenv --url                   show the Kaggle link, the Jupyter URL and the attach ID
  kenv -id <attach-id>         attach THIS terminal to a running session (from another window)
  kenv stop                    pull your changes back, then delete the session's kernel now

PROJECT VERSIONS   (a VERSION = persistent, lives in .kenv/vN and survives every session)
  kenv versions                list versions: number, name, id (kv:...), sessions, time used
  kenv new [name]              create the next version and make it active
  kenv activate v2             switch the active version - by number ...
  kenv activate model-champ    ... by name ...
  kenv activate -id kv:<id>    ... by version id ...
  kenv activate <path/.kenv>   ... or point at another project's .kenv folder
  kenv v2 -r model-champ       rename a version (also: kenv <name> -r <new>, kenv kv:<id> -r <name>)
  kenv v2                      show a version: id, sessions, how each ended
  (Two different ids: `kenv -id <attach-id>` = attach to a running session, name#secret.
   `kenv activate -id kv:...` = a project version. Never mixed up: version ids start with kv:.)

PROJECT SYNC
  The project is mirrored to /kaggle/working/<project-folder-name> on the kernel and notebooks/scripts
  start there, so relative paths behave like your local folder. Write to /kaggle/working/... to use
  Kaggle's root. Changed/new kernel files come back on `kenv run`, `kenv save`, `kenv stop` and exit.
  kenv sync                    pull the kernel's new/changed files, then push your local edits
  kenv save                    pull new/changed files (see INSIDE A SESSION for explicit paths)
  .kenvignore                  what is NOT synced, .gitignore syntax (big data and model folders)
  kenv data push <folder>      upload a folder ONCE as a private Kaggle dataset; it is attached to every
                               later session at /kaggle/input/<name>       (kenv data list)

INSIDE A SESSION  (type these in the kenv shell; the session's Jupyter URL works in VS Code)
  kenv status                  CPU / RAM / disk / GPU usage of the kernel
  kenv gpu [t4|l4|none]        switch accelerator (prompts if you leave it out)
  kenv exec <command>          run a shell command on the kernel     e.g. kenv exec nvidia-smi
  kenv ls [path]               list files on the kernel (in the project folder)
  kenv put <files/folders>     upload to the kernel                  [--dest remote/dir]
  kenv save <paths>            download these paths from the kernel to your project   [--to local/dir]
  kenv run <script.py> [args]  sync, run a script on the kernel, stream the output, and pull every
                               new/changed file back into the project   [--out dir]
                               (with no active session it starts a temporary one, saves to
                               ./kenv_output and deletes the kernel)

RUN KENV FROM CODE   (on the kernel: notebook cells, scripts, `kenv exec`)   -   see also CLIP / UNCLIP below
  import kenv                  installed by every session - no pip needed
  kenv.cli("kenv v2 -r champ") any command from the terminal; the ones that need your files run on YOUR machine,
                               through a queue answered by the window that started the session (keep it open).
                               Also: kenv.cli("kenv.time_start"), kenv.cli(["kenv", "sync"]), timeout=..., capture=True
  kenv.time_start()            record time, RAM, VRAM, disk, CPU     (label: kenv.time_start("fit"))
  kenv.time_end()              record the second snapshot
  kenv.time_output()           print the difference; every measurement is stored in the version's run history
  with kenv.timed("fit"): ...  the three above in one block
  kenv.status()                same as `kenv status`, from code
  kenv.metric("auc", 0.93)     record a result for `kenv diff` (also kenv.cli("kenv metric auc 0.93"))
  kenv.log("epoch 3 done")     add a line to the session log (kenv.log("oops", "error") also lands in errors.log)
  kenv.cli('kenv commit -m "msg"')   commit / diff / rollback / branch / tag / logs work from code too
  kenv.kaggle_root()           "/kaggle/working"; kenv.kaggle_root(chdir=True) switches there
  kenv time_start|time_end|time_output [label]     the same measurements from the terminal
  from kenv_shim import kenv   the one-line import that keeps your code working where kenv does not exist
                               (CI, GitHub): every call is a no-op returning None there   (kenv shim writes the file)

VERSIONING   (a commit snapshots code + packages + config into the NEXT numbered version)
  kenv commit -m "msg"         snapshot your tracked files (.kenvignore applies), the package lock (deps.lock), the
                               attached datasets and the metrics recorded since the last commit -> .kenv/vN/
                               `kenv init` resumes the newest commit; syncing always starts from your current files
  kenv diff v2 v3              what changed: code (files added/removed/changed), libraries (added/removed/upgraded),
                               input datasets and metric values.   kenv diff v2 = v2 vs your working files,
                               kenv diff = active version vs working files.   -p / --patch adds the text diff
  kenv rollback v2             restore that version's code, datasets and packages (your uncommitted changes are saved
                               to a safety version first). A kernel is disposable: its memory/files are NOT restored.
                               Packages are rebuilt on the running kernel, or at the next `kenv init`
  kenv branch                  list branches;  kenv branch <name> creates one (records its parent) and switches to it
  kenv branch switch <name>    change branch (files are not touched - use `kenv rollback <tip>`)   kenv branch rm <name>
  kenv tag                     list tags;  kenv tag v2 stable  attaches one;  kenv tag <tag>  tags the active version;
                               kenv tag --delete <tag>.   A tag works anywhere a version does:
                               kenv diff stable v3   kenv activate best-auc   kenv rollback stable
  kenv metric auc 0.93         record a result (kenv metric = list the waiting ones); from code: kenv.metric("auc", 0.93)

LOGS   (streamed live from the kernel while the session runs; plain text, UTC timestamps on every line)
  .kenv/vN/logs/run.log        agent + Jupyter messages, the output of `kenv run` / `kenv exec`, metrics, kenv.log(...) lines
  .kenv/vN/logs/errors.log     the error lines and tracebacks of the same stream
  kenv logs [version]          print the log         --errors  read errors.log instead
  kenv logs --tail             follow it live (Ctrl-C to stop)
  kenv logs --grep <pattern>   only matching lines (regex, case-insensitive)
  kenv logs --since 10m        relative (30s, 10m, 2h, 1d) or absolute (2026-09-30, "2026-09-30 12:00", 12:30 today)
  (Text printed by a notebook cell stays in the notebook; use kenv.log("...") or a script run with `kenv run`.)

CORE FILE   (.kenv/vN/core.toml - readable, diffable, one per version)
  kenv core [version]          summary: manifest, packages, inputs/outputs, peak resources, run history
                               Holds kernel, accelerator, python, exact package versions + imports found in your code,
                               datasets, output files (size + SHA256), peak RAM/VRAM/CPU/disk, every run, and the
                               NAMES of secrets your code reads - never their values.

PORTABILITY   (Phase 4)
  kenv export [version]        zip a version: core file, code snapshot, deps.lock, config, logs, metadata and a SHA256
                               manifest. No secret values (scanned first; logs are redacted) and no kernel reference of
                               yours. Default: the active version.   --out <file.zip>  --force
  kenv import <zip> [--to <dir>] [--init]    verify every hash, refuse unsafe paths, recreate the version in this folder
                               (or <dir>). The first session builds the kernel under YOUR account and installs the
                               locked packages. Old logs are archived with their original timestamps; datasets your
                               account cannot read are dropped.   --name <version-name>   --init starts a session now
  kenv rebuild [version]       install the locked AND declared packages on the running kernel, run `pip check`, and warn
                               when Kaggle's Python or Docker image differ from what was recorded.
                               --dry-run  only show what would be installed      --all | --full   every differing package
  kenv convert --to requirements.txt | environment.yml | kernel-metadata.json   [version] [--file <out>] [--all] [--force]
  kenv convert --from <requirements.txt | environment.yml | kernel-metadata.json>   read it into the active version
                               (pinned packages go into the lock, all of them into the declared packages)

SAFETY & QUALITY   (Phase 4)
  kenv quota                   weekly GPU hours used / left - an APPROXIMATE estimate from kenv's own session logs
  kenv quota set limit <hours> | quota set warn 80,95 | quota reset        (also: quota set --limit H --warn A,B)
  kenv deps                    imports in your code that are missing from the lock        kenv deps fix   add them
  kenv deps add <pkg[==ver]>   declare a package by hand (kept in [dependencies.declared], carried by every commit)
  kenv deps conflicts          run `pip check` on the kernel and report conflicts
  kenv scan [paths] [--staged]    look for keys and tokens: known key formats + high-entropy strings
  kenv scan allow [paths | <fingerprint>]   allow findings (only a 12-character fingerprint is stored)
  kenv scan hook | unhook      install / remove a git pre-commit hook that runs `scan --staged`
  kenv doctor [env | files] [--fix]   credentials, Kaggle CLI, relay; core files, snapshot and I/O hashes, tags, branches,
                               last_active. Prints the fix for each problem and exits 1. --fix repairs dangling tags,
                               last_active, dangling branches and leftover temp files.
  From code (kenv.cli): doctor, export, convert, quota, deps, scan, rebuild (also --dry-run). import, quota set/reset,
  scan allow/hook and paths outside the project are blocked there.

CLIP / UNCLIP   (Phase 6, optional - the no-op `from kenv_shim import kenv` stub is the simpler way to keep kenv lines in code)
  kenv.clip()                  put it at the top of a file that has kenv lines (after `import kenv`). It opts the file in.
  kenv clip [files]            list the opted-in files and the line numbers of their kenv statements -> .kenv/clip (read-only)
  kenv unclip [files]          comment out every kenv statement (import kenv, kenv.cli(...), kenv.time_start() ...) of the
                               opted-in files, multi-line calls and notebook cells included, and save the files.
                               Each line becomes  #kenv:clip# <the original line>.   --dry-run  show it only
  kenv unclip --undo           bring the lines back by hand (every `kenv init` also does it at session start)
  kenv unclip --staged         comment out the kenv lines in what is STAGED for a git commit; your working files are untouched
  kenv unclip hook | unhook    install / remove a git pre-commit hook that runs `unclip --staged`
  kenv unclip --recover        put back the originals of an unclip that was interrupted (kept in .kenv/clip-backup until it succeeds)
  A statement kenv cannot comment out safely (x = kenv.cli(...), `with kenv.timed():`, two statements on one line) stops the
  whole file and is reported; fix the line, or use -f to comment out the rest.

DASHBOARD   (Phase 5, read-only, this machine only)
  kenv ui                      open a local dashboard in your browser: versions, branches and tags; the session timeline; run history
                               and metrics; RAM / VRAM / CPU / disk / runtime charts; the I/O map (sizes, hashes); the dependency lock;
                               logs with search; diffs between versions; the GPU quota estimate (approximate)
  kenv ui -uri kv:<id>         the same, for ONE version only (also a vN, name or tag). Makes it the active version when no session runs.
  kenv ui --port N | --no-open | --idle <min>     fixed port, print the link only, stop after N idle minutes
  * Needs the separate kenv_ui/ folder (prebuilt React + Ant Design) next to kenv.py, in ~/.kenv/ui, or at $KENV_UI_DIR; without it kenv
    says what to install. Build it with: cd kenv_ui_src && npm install && npm run build   (kenv.py itself stays stdlib-only)
  * Binds to 127.0.0.1 only, GET only, behind a random token in the link. It answers from .kenv/ alone and shows secret NAMES, never values.
  * From code, kenv.cli("kenv ui") starts it in the background (stops after 60 idle minutes) and keeps the link out of the output.

LAZY LOCAL ACCESS   (Phase 6, EXPERIMENTAL, off by default)
  kenv init --lazy-local [--tunnel cloudflared|ngrok]   do not upload the heavy files: the kernel reads them from YOUR
                               machine on demand, caches them, and sees `os.getcwd()` as your local project path.
                               Only small code/config files are synced the normal way.
  On the kernel: kenv.prefetch("data/") warms the cache   kenv.lazy_path("img/a.png") real path for C-level loaders
                               kenv.lazy_status() hits / misses / bytes
  * A read-only file server on your machine, limited to the project folder, behind a random token and a tunnel (cloudflared by
    default; ngrok needs an account and auth token and its free plan has bandwidth caps). It rejects ../ paths, symlinks that
    leave the folder, .kenv, .git and key/.env files, and it stops with the session.
  * C-level file access (some image/audio/video loaders, memory maps) bypasses the patch. Cache misses cross your home
    internet connection: loops that re-read big files every epoch are slow until the cache is warm.
  * Writes never go to your machine; they land in the kernel's project folder and come back with the normal sync.
  * A real mount (sshfs/FUSE) is not used: Kaggle kernels are assumed not to allow FUSE.

SETUP & CLEANUP
  kenv --cred                  check OS, packages and Kaggle credentials (shows how to fix them)
  kenv --sweep                 delete leftover kenv-* kernels (after kill -9 / power loss)
  kenv --help | -h             this screen          kenv --version

OPTIONS
  -n, --name <name>   session name          --gpu              choose a GPU when starting
  -r, --rename <name> new version name (with a version target)
  --to/--from <fmt/file>  (convert)   --limit <h> --warn <pct,pct>  (quota)   --staged  (scan)   -f, --force
  --all/--full  --dry-run  (rebuild)   --init  (import)   --fix  (doctor)
  -m, --message <msg> commit message                 -p, --patch      show text diffs (kenv diff)
  --tail  --grep <re>  --since <time>  --errors      log options (kenv logs)         -d, --delete   remove a tag
  --lazy-local  --tunnel <cloudflared|ngrok>  (init)    --undo  --recover  --staged  (unclip)
  -uri <version>  --port <n>  --no-open  (ui)
  --idle <min>        stop the kernel after this many minutes with no activity (default 20)
  --startup <sec>     how long to wait for the kernel to come online (default 900)

NOTES
  * SECRET SCANNING IS A SAFETY NET, NOT A GUARANTEE. It matches common key formats and high-entropy strings, so it can miss
    a secret and flag a harmless string (`# kenv:allow` on the line, or `kenv scan allow`, which stores a fingerprint only). Commits and exports
    stop when it finds something; only secret NAMES are ever stored in .kenv. Always review what you share.
  * The GPU quota is an ESTIMATE from kenv's own logs; Kaggle has no official API for it.
  * Kernels are private and always deleted when you `exit`, Ctrl-C or close the terminal.
    The kernel also stops itself after --idle minutes if your machine vanishes.
  * kenv options go BEFORE the script in `kenv run`; everything after the script is passed to it.
  * Files on the kernel disappear with it. Project files sync back automatically at the end of a run,
    on `kenv save`, `kenv stop` and a normal exit; a kernel that dies on its own cannot be pulled from.
  * Every session uploads the non-ignored project files to a fresh kernel (kernels cannot be restored);
    only files that changed since are re-sent within a session. Keep big folders in .kenvignore.
  * kenv.cli() from code refuses anything that starts, stops or attaches a session (init, stop, gpu, exec, run,
    -id ...), `logs --tail`, and any path outside the project folder. With no listening kenv window it raises an error at once.
  * If you edit a file locally while the kernel changed it too, your file wins and the kernel's copy is
    saved as <name>.kenv-remote<ext>.
"""


def build_parser():
    ap = argparse.ArgumentParser(prog="kenv", add_help=False)
    ap.add_argument("command", nargs="?")
    ap.add_argument("args", nargs="*")
    ap.add_argument("-n", "--name")
    ap.add_argument("-id", "--id", dest="attach")
    ap.add_argument("-r", "--rename", dest="rename")
    ap.add_argument("-m", "--message", dest="message")  # for `kenv commit -m "..."` (Phase 3)
    ap.add_argument("--since")               # kenv logs --since 10m
    ap.add_argument("--grep")                # kenv logs --grep <pattern>
    ap.add_argument("--tail", action="store_true")
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("-d", "--delete", action="store_true")   # kenv tag --delete <tag>
    ap.add_argument("-p", "--patch", action="store_true")    # kenv diff --patch
    ap.add_argument("--to-fmt", dest="to_fmt")              # kenv convert --to <format>  (main() renames --to for convert)
    ap.add_argument("--from", dest="convert_from")          # kenv convert --from <file>
    ap.add_argument("--limit", type=float)                  # kenv quota set --limit <hours>
    ap.add_argument("--warn")                               # kenv quota set --warn 80,95
    ap.add_argument("--staged", action="store_true")        # kenv secret-scan --staged
    ap.add_argument("-f", "--force", action="store_true")
    ap.add_argument("--all", "--full", action="store_true", dest="all")   # kenv rebuild --all|--full / kenv convert --all
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")     # kenv rebuild --dry-run
    ap.add_argument("--init", action="store_true")                        # kenv import <zip> --init
    ap.add_argument("--fix", action="store_true")                         # kenv doctor --fix
    ap.add_argument("--lazy-local", action="store_true", dest="lazy_local")   # kenv init --lazy-local (experimental)
    ap.add_argument("--tunnel")                                           # kenv init --lazy-local --tunnel cloudflared|ngrok
    ap.add_argument("--undo", action="store_true")                        # kenv unclip --undo
    ap.add_argument("--recover", action="store_true")                     # kenv unclip --recover
    ap.add_argument("-uri", "--uri", dest="uri")                          # kenv ui -uri kv:<id>  (one version only)
    ap.add_argument("--port", type=int)                                   # kenv ui --port N
    ap.add_argument("--no-open", action="store_true", dest="no_open")     # kenv ui --no-open
    ap.add_argument("--url", action="store_true")
    ap.add_argument("--cred", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", "--to", "--file", dest="out")
    ap.add_argument("--dest", default="")
    ap.add_argument("--idle", type=int)
    ap.add_argument("--startup", type=int, default=900)
    ap.add_argument("-h", "--help", action="store_true")
    ap.add_argument("-v", "--version", action="store_true")
    return ap


def split_passthrough(argv):
    """`kenv run x.py --epochs 3` and `kenv exec ls -la`: everything after the script/command is not ours."""
    i = 0
    while i < len(argv):
        t = argv[i]
        if t in VALUE_OPTS:
            i += 2
        elif t.startswith("-"):
            i += 1
        else:
            if t == "run":
                return argv[:i + 2], argv[i + 2:]
            if t == "exec":
                return argv[:i + 1], argv[i + 1:]
            break
    return argv, []


def main():
    if os.name == "nt":
        os.system("")  # enables ANSI colours in the Windows console
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    head, tail = split_passthrough(sys.argv[1:])
    if "convert" in head:  # `--to` means "output folder" elsewhere; for convert it names the target format
        head = ["--to-fmt" if t == "--to" else ("--to-fmt=" + t[5:] if t.startswith("--to=") else t) for t in head]
    ap = build_parser()
    try:
        a = ap.parse_intermixed_args(head)  # `kenv tag --delete x`, `kenv logs v2 --tail`: flags may sit anywhere
    except TypeError:
        a = ap.parse_args(head)
    cmd = a.command
    if cmd and cmd.lower().startswith("kenv."):  # kenv.time_start typed as in code
        cmd = cmd[5:]
    try:
        if a.help or cmd == "help":
            say(HELP.format(banner="\n".join(_glyphs()) + "\n  By EpicRaven\n"))
            return 0
        if a.version:
            say(f"kenv {VERSION}")
            return 0
        if a.cred or cmd == "cred":
            return cmd_cred(a)
        if a.sweep or cmd == "sweep":
            return sweep()
        if cmd == "activate":
            return cmd_activate(a)
        if cmd and (VERSION_RE.match(cmd) or cmd.lower().startswith("kv:") or a.rename):
            if a.rename and not cmd:
                raise KenvError("Usage: kenv <v2 | version-name | kv:id> -r <new-name>")
            return cmd_version_ref(a)
        if a.attach:
            return cmd_attach(a)
        if a.url or cmd == "url":
            return cmd_url(a)
        if cmd == "init" or (a.name and not cmd):
            return cmd_init(a)
        if cmd is None:
            banner()
            say("Start a session with `kenv init`, or see everything with `kenv --help`.")
            return 0
        table = {"status": cmd_status, "gpu": cmd_gpu, "ls": cmd_ls, "put": cmd_put,
                 "save": cmd_save, "stop": cmd_stop, "versions": cmd_versions, "new": cmd_new,
                 "sync": cmd_sync, "data": cmd_data, "core": cmd_core, "shim": cmd_shim,
                 "commit": cmd_commit, "diff": cmd_diff, "rollback": cmd_rollback, "branch": cmd_branch,
                 "tag": cmd_tag, "metric": cmd_metric, "logs": cmd_logs, "log": cmd_logs,
                 "doctor": cmd_doctor, "export": cmd_export, "import": cmd_import, "convert": cmd_convert,
                 "quota": cmd_quota, "deps": cmd_deps, "rebuild": cmd_rebuild,
                 "secret-scan": cmd_secret_scan, "secret-hook": cmd_secret_hook, "scan": cmd_scan,
                 "clip": cmd_clip, "unclip": cmd_unclip, "ui": cmd_ui}
        if cmd and cmd.lower().replace("-", "_") in ("time_start", "time_end", "time_output"):
            return cmd_time(a, cmd.lower().replace("-", "_")[5:])
        if cmd == "run":
            return cmd_run(a, tail)
        if cmd == "exec":
            return cmd_exec(a, tail)
        if cmd in table:
            return table[cmd](a)
        die(f"Unknown command '{cmd}'. See: kenv --help")
    except KenvError as e:
        die(f"[kenv] {e}")
    except KeyboardInterrupt:
        die("\n[kenv] Cancelled. Kaggle kernel cleaned up.", 130)


if __name__ == "__main__":
    sys.exit(main() or 0)
