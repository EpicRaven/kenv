#!/usr/bin/env python3
"""
kenv - disposable Kaggle kernels for your terminal.                     By EpicRaven

Start a session, get a Jupyter URL for your local notebook, run scripts on Kaggle's
CPU/GPU, pull files back to your own folder. The kernel is ALWAYS deleted afterwards
(exit, Ctrl-C, closing the terminal, kill). Run `kenv --help` for every command.

Needs only: Python 3.8+, `pip install -U kaggle`, and Kaggle credentials (`kenv --cred`).
"""
import argparse
import base64
import hashlib
import io
import json
import os
import platform
import random
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

VERSION = "1.0.0"
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
VALUE_OPTS = {"-n", "--name", "-id", "--id", "--out", "--to", "--dest", "--idle", "--startup"}


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

    def json_call(self, path, payload=None, raw=None, query="", timeout=60):
        with self.call(path, payload=payload, raw=raw, query=query, timeout=timeout) as r:
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
    try:
        state_path(name).unlink()
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


# ----------------------------------------------------------------------------- the agent (runs ON the kernel)

AGENT_SRC = r'''
import base64, hmac, io, json, os, re, shutil, subprocess, sys, threading, time, urllib.request, zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

CFG = json.loads(base64.b64decode("__CFG__"))
SECRET, TOPIC, RELAY = CFG["secret"], CFG["topic"], CFG["relay"]
WORK = CFG.get("work", "/kaggle/working")
AGENT_PORT, JUP_PORT = CFG.get("agent_port", 8899), CFG.get("jup_port", 8898)
SKIP = {"__pycache__", ".ipynb_checkpoints", ".git"}
START = time.time()
STATE = {"last": time.time(), "busy": 0, "stop": False}
LOCK = threading.Lock()
PROCS = []


def log(*a):
    print("[kenv-agent]", *a, flush=True)


def touch():
    STATE["last"] = time.time()


def resolve(p):
    p = p if os.path.isabs(p) else os.path.join(WORK, p)
    return os.path.normpath(p)


def arcname(ap):
    try:
        if os.path.commonpath([ap, WORK]) == WORK:
            return os.path.relpath(ap, WORK)
    except ValueError:
        pass
    return ap.lstrip("/")


def snapshot():
    out = {}
    for root, dirs, files in os.walk(WORK):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for f in files:
            p = os.path.join(root, f)
            try:
                s = os.stat(p)
                out[os.path.relpath(p, WORK)] = [s.st_size, int(s.st_mtime)]
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
            elif path == "/snapshot":
                self._send(200, json.dumps(snapshot()).encode())
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

    def do_exec(self):
        req = json.loads(self._body() or b"{}")
        cwd = resolve(req["cwd"]) if req.get("cwd") else WORK
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with LOCK:
            STATE["busy"] += 1
        rc, p = -1, None
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
                self.wfile.write(chunk)
                self.wfile.flush()
            rc = p.wait()
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
            "--ServerApp.trust_xheaders=True", "--ServerApp.root_dir=" + WORK]
    code = "from jupyter_server.serverapp import main; main()"
    PROCS.append(subprocess.Popen([sys.executable, "-c", code] + args,
                                  stdout=open("/tmp/jupyter.log", "w"), stderr=subprocess.STDOUT))


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
    os.makedirs(WORK, exist_ok=True)
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


# ----------------------------------------------------------------------------- launching kernels

def ref_for(user, name, gen):
    return f"{user}/kenv-{name}" + (f"-g{gen}" if gen > 1 else "")


def push_kernel(ref, cfg, gpu_id):
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
        (d / "kernel-metadata.json").write_text(json.dumps(meta))
        p = cli(*args)
        out = p.stdout + p.stderr
        if p.returncode != 0 and gpu_id and "--accelerator" in out:  # older CLI: metadata alone is enough
            p = cli("kernels", "push", "-p", str(d))
            out = p.stdout + p.stderr
        if p.returncode != 0 or "error" in out.lower():
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


def launch(user, name, secret, gen, gpu_id, idle_min, ep, startup):
    ref = ref_for(user, name, gen)
    cfg = {"secret": secret, "topic": ep.topic, "relay": RELAY, "gen": gen, "ref": ref,
           "gpu": gpu_id or "none", "idle_min": idle_min, "max_hours": 8, "work": "/kaggle/working"}
    st = load_state(name)
    if st and ref not in st["refs"]:  # recorded BEFORE pushing so an interrupt mid-push still gets cleaned up
        st["refs"].append(ref)
        save_state(st)
    say(f"[kenv] Pushing {ref} ...")
    push_kernel(ref, cfg, gpu_id)
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

    def __init__(self, name, gpu_id=None, idle_min=20, startup=900):
        self.user = get_username()
        self.name, self.secret = name, secrets.token_hex(16)
        self.gpu_id, self.idle_min, self.startup = gpu_id, idle_min, startup
        self.ep = Endpoint(name, self.secret)
        self._done = False

    def __enter__(self):
        require_cli()
        require_relay()
        old = load_state(self.name)
        if old and pid_alive(old.get("pid")):
            raise KenvError(f"A session called '{self.name}' is already running on this machine.")
        save_state({"name": self.name, "secret": self.secret, "user": self.user, "refs": [],
                    "pid": os.getpid(), "created": int(time.time())})
        install_handlers(self.cleanup)
        return self

    def start(self):
        launch(self.user, self.name, self.secret, 1, self.gpu_id, self.idle_min, self.ep, self.startup)

    def cleanup(self):
        if self._done:
            return
        self._done = True
        for n in ("SIGINT", "SIGHUP", "SIGTERM", "SIGBREAK"):  # nothing may abort cleanup
            if hasattr(signal, n):
                try:
                    signal.signal(getattr(signal, n), signal.SIG_IGN)
                except (ValueError, OSError):
                    pass
        st = load_state(self.name)
        if st:  # no state file means `kenv stop` already deleted everything
            delete_refs(st.get("refs", []), self.ep if st.get("refs") else None)
            remove_state(self.name)

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            say("\n[kenv] Stopping. Cleaning up Kaggle resources (please wait) ...")
        self.cleanup()
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


def open_shell(name, secret, owner, ep=None):
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


def extract_zip(data, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root, names = dest.resolve(), []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for m in z.infolist():
            if m.is_dir():
                continue
            t = (root / m.filename).resolve()
            if t != root and root not in t.parents:  # zip-slip guard
                continue
            t.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as s, open(t, "wb") as d:
                shutil.copyfileobj(s, d)
            names.append(m.filename)
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
    payload = {"cmd": cmd}
    if cwd:
        payload["cwd"] = cwd
    r = ep.call("/exec", payload=payload, timeout=1800)
    out, buf = sys.stdout.buffer, b""
    try:
        while True:
            chunk = r.read1(4096)
            if not chunk:
                break
            buf += chunk
            if len(buf) > 64:  # hold back the tail: it carries the exit marker
                out.write(buf[:-64])
                out.flush()
                buf = buf[-64:]
    finally:
        r.close()
    m = re.search(rb"\n@@KENV_EXIT:(-?\d+)\s*$", buf)
    out.write(buf[:m.start()] if m else buf)
    out.flush()
    if not m:
        say("\n[kenv] connection to the kernel was lost before the command finished")
        return 1
    return int(m.group(1))


# ----------------------------------------------------------------------------- commands

def cmd_init(a):
    if os.environ.get("KENV_SESSION"):
        raise KenvError("You are already inside a kenv session. Type `exit` first.")
    name = sanitize(a.name) if a.name else random_name()
    gpu_id = pick_gpu() if a.gpu else None
    banner()
    with Owner(name, gpu_id, a.idle or 20, a.startup) as o:
        say(f"[kenv] Session name: {c(name, '1')}")
        o.start()
        say(c("[kenv] Session online", "32;1"))
        say()
        print_info(o.ep)
        say()
        say("You are now inside the session. Try: kenv status | kenv gpu | kenv run file.py | kenv save <file> | kenv --help")
        say("Type `exit` (or close this terminal) to delete the kernel.")
        say()
        open_shell(name, o.secret, owner=True, ep=o.ep)
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
    say(c("Switching accelerator starts a NEW kernel: everything stored on the current one is lost.", "33"))
    say(c("Pull what you need first with `kenv save <path>`. Notebooks connected to the old Jupyter URL must reconnect.", "33"))
    if not confirm("Continue?"):
        say("Cancelled.")
        return 1
    require_cli()
    require_relay()
    user, gen = get_username(), info["gen"] + 1
    new_ref = ref_for(user, ep.name, gen)
    idle = a.idle or info.get("idle_min", 20)
    try:
        launch(user, ep.name, ep.secret, gen, gpu_id, idle, ep, a.startup)
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
    print_info(ep)


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
    if not a.args:
        raise KenvError("Usage: kenv save <remote path> [more ...] [--to local/folder]\n"
                        "Paths are relative to /kaggle/working. `kenv save .` grabs everything.")
    ep = need_ep()
    out = a.out or "."
    names = pull(ep, a.args, out)
    for nme in names[:30]:
        say(f"[kenv] saved {Path(out) / nme}")
    if len(names) > 30:
        say(f"[kenv] ... and {len(names) - 30} more")
    if not names:
        say("[kenv] nothing was saved")
        return 1


def run_on(ep, script, script_args, out):
    p = Path(script).expanduser()
    if not p.is_file():
        raise KenvError(f"Script not found: {p}")
    if p.suffix not in (".py", ".sh"):
        raise KenvError("kenv run supports .py and .sh files. For notebooks, use the Jupyter URL (kenv --url).")
    data, _ = zip_local([p])
    ep.json_call("/put", raw=data, query="?dest=", timeout=120)
    before = ep.json_call("/snapshot")
    runner = "python -u" if p.suffix == ".py" else "bash"
    cmd = f"{runner} {shlex.quote(p.name)} " + " ".join(shlex.quote(x) for x in script_args)
    say(f"[kenv] running on the kernel: {cmd.strip()}")
    say(c("-" * 60, "2"))
    rc = stream_exec(ep, cmd)
    say(c("-" * 60, "2"))
    after = ep.json_call("/snapshot")
    changed = [k for k, v in after.items() if before.get(k) != v]
    say(f"[kenv] exit code {rc}")
    if changed:
        names = pull(ep, changed, out)
        say(f"[kenv] {len(names)} new/changed file(s) saved to {Path(out).resolve()}")
    else:
        say("[kenv] the script produced no new files")
    return rc


def cmd_run(a, script_args):
    if not a.args:
        raise KenvError("Usage: kenv run <script.py> [script args]     (kenv options go before the script)")
    script, out = a.args[0], a.out or "kenv_output"
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
    refs = st.get("refs", []) if st else [ep.resolve(force=True)["ref"]]
    say("[kenv] Deleting the kernel ...")
    delete_refs(refs, ep)
    remove_state(ep.name)
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
            remove_state(st["name"])


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


HELP = """\
{banner}
USAGE
  kenv <command> [options]

START / JOIN A SESSION
  kenv init                    start a session with a random name, then open a kenv shell
  kenv -n "<name>"             start a session called <name>   (also: kenv init -n "<name>")
  kenv init --gpu              same, but pick a GPU at startup
  kenv --url                   show the Kaggle link, the Jupyter URL and the attach ID
  kenv -id <attach-id>         attach THIS terminal to a running session (from another window)
  kenv stop                    delete the session's kernel now

INSIDE A SESSION  (type these in the kenv shell; the session's Jupyter URL works in VS Code)
  kenv status                  CPU / RAM / disk / GPU usage of the kernel
  kenv gpu [t4|l4|none]        switch accelerator (prompts if you leave it out)
  kenv exec <command>          run a shell command on the kernel     e.g. kenv exec nvidia-smi
  kenv ls [path]               list files on the kernel (/kaggle/working)
  kenv put <files/folders>     upload to the kernel                  [--dest remote/dir]
  kenv save <paths>            download from the kernel to YOUR folder   [--to local/dir]
  kenv run <script.py> [args]  run a script on the kernel, stream the output, and save every
                               new/changed file to ./kenv_output   [--out dir]
                               (with no active session it starts a temporary one and deletes it)

SETUP & CLEANUP
  kenv --cred                  check OS, packages and Kaggle credentials (shows how to fix them)
  kenv --sweep                 delete leftover kenv-* kernels (after kill -9 / power loss)
  kenv --help | -h             this screen          kenv --version

OPTIONS
  -n, --name <name>   session name          --gpu              choose a GPU when starting
  --idle <min>        stop the kernel after this many minutes with no activity (default 20)
  --startup <sec>     how long to wait for the kernel to come online (default 900)

NOTES
  * Kernels are private and always deleted when you `exit`, Ctrl-C or close the terminal.
    The kernel also stops itself after --idle minutes if your machine vanishes.
  * kenv options go BEFORE the script in `kenv run`; everything after the script is passed to it.
  * Files on the kernel disappear with it - `kenv save` or `kenv run` what you want to keep.
"""


def build_parser():
    ap = argparse.ArgumentParser(prog="kenv", add_help=False)
    ap.add_argument("command", nargs="?")
    ap.add_argument("args", nargs="*")
    ap.add_argument("-n", "--name")
    ap.add_argument("-id", "--id", dest="attach")
    ap.add_argument("--url", action="store_true")
    ap.add_argument("--cred", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", "--to", dest="out")
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
    a = build_parser().parse_args(head)
    cmd = a.command
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
                 "save": cmd_save, "stop": cmd_stop}
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
