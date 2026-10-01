#!/usr/bin/env python3
"""Seed a throw-away kenv project (no Kaggle needed) so `kenv ui` has real .kenv data to show.
    HOME=/tmp/kenv-home python3 kenv_ui_src/seed_demo.py /tmp/kenv-demo/lagos-price-model
then:  cd /tmp/kenv-demo/lagos-price-model && HOME=/tmp/kenv-home python3 /path/to/kenv.py ui
(Setting HOME keeps the demo out of your real ~/.kenv.)"""
import importlib.util, json, os, shutil, sys, time
from pathlib import Path

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kenv", here.parent / "kenv.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/kenv-demo/lagos-price-model").resolve()
shutil.rmtree(root, ignore_errors=True); root.mkdir(parents=True)
proj = k.Project(root); proj.ensure()
now = time.time(); H = 3600

CODE = {
 "v1": {"train.py": "import pandas as pd\nLR = 0.1\nEPOCHS = 5\nprint('training on naija-houses')\n", "utils.py": "def clean(df):\n    return df.dropna()\n"},
 "v2": {"train.py": "import pandas as pd\nLR = 0.05\nEPOCHS = 12\nprint('training on naija-houses')\n", "utils.py": "def clean(df):\n    return df.dropna()\n"},
 "v3": {"train.py": "import pandas as pd\nLR = 0.03\nEPOCHS = 20\nprint('training on naija-houses')\n", "utils.py": "def clean(df):\n    return df.dropna().reset_index(drop=True)\n", "eval.py": "print('rmse')\n"},
}
LOCK = {"v1": {"pandas": "2.1.4", "numpy": "1.26.4", "scikit-learn": "1.3.2"},
        "v2": {"pandas": "2.2.3", "numpy": "1.26.4", "scikit-learn": "1.5.2", "lightgbm": "4.3.0"},
        "v3": {"pandas": "2.2.3", "numpy": "2.0.2", "scikit-learn": "1.5.2", "lightgbm": "4.5.0", "xgboost": "2.1.1"}}
METRICS = {"v1": {"rmse": 41.2, "r2": 0.71}, "v2": {"rmse": 33.8, "r2": 0.80}, "v3": {"rmse": 31.4, "r2": 0.83}}
PEAKS = {"v1": (4.1, 0.0, 88, 3.2, 14), "v2": (7.8, 5.2, 96, 5.1, 31), "v3": (9.6, 8.4, 99, 6.0, 52)}

def mk(v, name=None, branch="main", parent=None, msg=None, age_days=6):
    ver = proj.new_version(name)
    assert ver == v, (ver, v)
    vd = proj.kdir / v
    entries = {}
    for rel, text in CODE[v].items():
        (vd / "code").mkdir(exist_ok=True); (vd / "code" / rel).write_text(text)
        entries[rel] = {"sha256": k.sha256_file(vd / "code" / rel), "size": len(text)}
    k.write_json(vd / "code.json", {"count": len(entries), "bytes": sum(e["size"] for e in entries.values()), "files": entries})
    k.write_lock(vd / "deps.lock", LOCK[v])
    k.write_json(vd / "config.json", {"datasets": ["adaeze/naija-houses", "chinedu/lagos-rents"], "accelerator": "GPU", "accelerator_id": "t4",
                 "python": "3.11.13", "secrets": ["HF_TOKEN", "WANDB_API_KEY"], "lock_known": True})
    proj.set_meta(v, message=msg, parent=parent, branch=branch, committed=k.iso(now - age_days * 86400), files=len(entries))
    ram, vram, cpu, disk, rt = PEAKS[v]
    runs = {"manifest": None}
    def fn(doc):
        doc["manifest"] = {"version": v, "id": proj.meta(v)["id"], "kernel": "adaeze/kenv-demo", "accelerator": "GPU", "accelerator_id": "t4",
                           "gpu_model": "Tesla T4", "python": "3.11.13", "docker_image": "kaggle-gpu-2025-09", "created": k.iso(now - age_days * 86400)}
        doc["dependencies"] = {"detected": {"pandas": "2.2.3"}, "lock": LOCK[v]}
        doc["io"] = {"inputs": {"datasets": ["adaeze/naija-houses", "chinedu/lagos-rents"], "mounted": ["naija-houses", "lagos-rents"]},
                     "outputs": {f"outputs/model_{v}.pkl": {"size": 4_200_000 * int(v[1:]), "sha256": k.hashlib.sha256(v.encode()).hexdigest()},
                                 "outputs/preds.csv": {"size": 880_000, "sha256": k.hashlib.sha256(b"p" + v.encode()).hexdigest()}}}
        doc["resources"] = {"session": "s1", "runtime_min": rt, "ram_peak_gb": ram, "vram_peak_gb": vram, "cpu_peak_pct": cpu, "disk_peak_gb": disk}
        doc["secrets"] = {"names": ["HF_TOKEN", "WANDB_API_KEY"]}
        doc["metrics"] = METRICS[v]
        rr = {}
        for i in range(1, 6):
            rr[f"r{i:04d}"] = {"kind": "metric", "label": "rmse", "name": "rmse", "value": round(METRICS[v]["rmse"] + (6 - i) * 1.3, 2),
                               "session": "s1", "started": k.iso(now - age_days * 86400 + i * 600), "status": "ok"}
        rr["r0006"] = {"kind": "timed", "label": "fit", "session": "s1", "started": k.iso(now - age_days * 86400 + 4000), "duration_s": 60.0 * rt / 5,
                       "status": "ok", "exit_code": 0, "ram_delta_mb": 900.5, "ram_peak_gb": ram, "vram_delta_mb": 700.0, "vram_peak_gb": vram, "disk_delta_mb": 120.0, "cpu_avg_pct": int(cpu * .6)}
        rr["r0007"] = {"kind": "script", "label": "train.py", "session": "s1", "started": k.iso(now - age_days * 86400 + 9000), "duration_s": 61.5,
                       "status": "failed" if v == "v1" else "ok", "exit_code": 1 if v == "v1" else 0, **({"error": "ValueError: bad shape"} if v == "v1" else {})}
        doc["runs"] = rr
    k.CoreStore(proj, v).update(fn)
    sess = [{"sid": f"{v}a", "session": "swift-raven", "start": k.iso(now - age_days * 86400), "end": k.iso(now - age_days * 86400 + rt * 60),
             "duration_s": rt * 60, "ended_by": "user", "last_seen": k.iso(now - age_days * 86400 + rt * 60), "idle_min": 20, "gpu": "t4", "pid": 1},
            {"sid": f"{v}b", "session": "calm-otter", "start": k.iso(now - age_days * 86400 + 8 * H), "end": k.iso(now - age_days * 86400 + 8 * H + 1500),
             "duration_s": 1500, "ended_by": "idle", "last_seen": k.iso(now - age_days * 86400 + 8 * H + 1500), "idle_min": 20, "gpu": "none", "pid": 1}]
    proj._save_sessions(v, sess)
    ld = vd / "logs"; ld.mkdir()
    lines = []
    for i in range(1, 60):
        t = now - age_days * 86400 + i * 30
        lines.append(f"{k.iso_ms(t)} [stdout] epoch {i % 12 + 1}/12 loss={1 / (i + 1):.4f}\n")
    lines.insert(30, f"{k.iso_ms(now - age_days * 86400 + 900)} [stderr] Traceback (most recent call last):\n")
    lines.insert(31, f"{k.iso_ms(now - age_days * 86400 + 901)} [stderr] ValueError: bad shape token=ghp_{'A' * 36}\n")
    lines.insert(32, f"{k.iso_ms(now - age_days * 86400 + 902)} [kenv] attach kenv://swift-raven#{'ab12' * 8}\n")
    (ld / "run.log").write_text("".join(lines)); (ld / "errors.log").write_text("".join(lines[30:32]))
    return v

mk("v1", msg="baseline", age_days=6)
mk("v2", msg="lower learning rate", parent="v1", age_days=3)
proj.save_tags({"champion": "v2"}); proj.save_branches({"tuning": "v2"})
mk("v3", branch="tuning", msg="xgboost experiment", parent="v2", age_days=1)
v4 = proj.new_version("scratch"); proj.set_active("v3")
k.QUOTA_FILE.parent.mkdir(parents=True, exist_ok=True)
k.write_json(k.QUOTA_FILE, {"weekly_gpu_hours": 30.0, "warn": [80, 95], "projects": [str(root)]})
print("seeded", root, "versions:", proj.version_names())
