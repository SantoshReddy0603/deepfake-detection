"""Run train_v2.py jobs on shared lab GPU machines without touching anyone else's files.

Every remote action happens inside one private directory, ~/dfd_work_<tag>, that this tool
creates and marks with an owner file. Nothing outside it is read, written, listed or killed:
caches (pip, torch hub) are redirected into it, processes are tracked by our own pid files,
and `cleanup` stops only those pids and deletes only that directory.

    python gpu/cluster.py probe   --access "GPU ACCESS.xlsx"
    python gpu/cluster.py deploy  --access ... --data_root DATA --split fold0.csv --gpus 6
    python gpu/cluster.py launch  --access ... --jobs jobs.txt
    python gpu/cluster.py status  --access ...
    python gpu/cluster.py fetch   --access ... --dest runs/ [--members]
    python gpu/cluster.py cleanup --access ...

Credentials are read from the access sheet at run time and never printed or stored.
"""
import argparse
import io
import json
import shlex
import tarfile
import time
from pathlib import Path

import openpyxl
import paramiko

CODE = Path(__file__).resolve().parents[1]
STATE = CODE.parent / ".build" / "gpu_state.json"
TAG = "dfd_work_v2"
REMOTE = f"~/{TAG}"
OWNER = "OWNER_dfd_v2"
ENV = (f"export TORCH_HOME=$HOME/{TAG}/cache/torch PIP_CACHE_DIR=$HOME/{TAG}/cache/pip "
       f"XDG_CACHE_HOME=$HOME/{TAG}/cache; cd $HOME/{TAG}")


def machines(access):
    sheet = openpyxl.load_workbook(access, read_only=True).active
    rows = list(sheet.iter_rows(values_only=True))
    found = {}
    for group, ip, user, password in (r[:4] for r in rows[1:]):
        if ip and user and password:
            found[f"{ip}|{user}"] = {"group": int(group), "ip": str(ip).strip(), "user": str(user).strip(),
                                     "password": str(password)}
    return found


def connect(machine, timeout=8):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(machine["ip"], username=machine["user"], password=machine["password"], timeout=timeout,
                   allow_agent=False, look_for_keys=False)
    client.get_transport().set_keepalive(30)
    return client


def run(client, command, timeout=600):
    _, out, err = client.exec_command(command, timeout=timeout)
    code = out.channel.recv_exit_status()
    return code, out.read().decode(errors="replace"), err.read().decode(errors="replace")


def remote_home(client):
    """Absolute $HOME; the SFTP start directory can differ from it, so never rely on that."""
    home = run(client, "echo $HOME")[1].strip()
    if not home.startswith("/home/") and not home.startswith("/root"):
        raise RuntimeError("unexpected home directory")
    return home


def guard(client):
    code, _, _ = run(client, f"test -f {REMOTE}/{OWNER}")
    if code:
        raise RuntimeError("private work directory missing or not ours; refusing to act")


def probe(args):
    slots, seen = [], set()
    for key, m in machines(args.access).items():
        if m["ip"] in seen or (args.ips and m["ip"] not in args.ips):
            continue
        try:
            with connect(m) as client:
                code, out, _ = run(client, "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,"
                                           "memory.total --format=csv,noheader,nounits")
                _, disk, _ = run(client, "df -Pk ~ | tail -1")
        except Exception as exc:
            print(f"group {m['group']:>2} {m['ip']}: unreachable ({type(exc).__name__})")
            continue
        if code:
            print(f"group {m['group']:>2} {m['ip']}: no nvidia-smi")
            continue
        seen.add(m["ip"])
        free_gb = int(disk.split()[3]) / 1e6 if disk.split() else 0
        for line in out.strip().splitlines():
            index, name, util, used, total = [v.strip() for v in line.split(",")]
            if not (util.isdigit() and used.isdigit() and total.isdigit()):
                print(f"group {m['group']:>2} {m['ip']} gpu{index} {name}: stats unavailable, skipped")
                continue
            slots.append({"key": key, "group": m["group"], "ip": m["ip"], "gpu": int(index), "name": name,
                          "util": int(util), "used_mb": int(used), "total_mb": int(total), "disk_free_gb": free_gb})
            print(f"group {m['group']:>2} {m['ip']} gpu{index} {name}: util {util}% mem {used}/{total} MB, "
                  f"home free {free_gb:.0f} GB")
    slots.sort(key=lambda s: (s["util"], s["used_mb"], -s["total_mb"]))
    return slots


def deploy(args):
    slots = [s for s in probe(args) if s["util"] <= args.max_util and s["used_mb"] <= args.max_used_mb
             and s["total_mb"] >= 10000 and s["disk_free_gb"] >= 5]
    chosen, per_machine = [], {}
    for slot in slots:
        if len(chosen) == args.gpus:
            break
        if per_machine.get(slot["key"], 0) < args.per_machine:
            per_machine[slot["key"]] = per_machine.get(slot["key"], 0) + 1
            chosen.append(slot)
    if not chosen:
        raise SystemExit("no idle GPU found")
    bundle = io.BytesIO()
    with tarfile.open(fileobj=bundle, mode="w:gz") as tar:
        for path in CODE.glob("*.py"):
            tar.add(path, arcname=f"code/{path.name}")
        data = Path(args.data_root)
        tar.add(data / "train_metadata.csv", arcname="data/train_metadata.csv")
        tar.add(data / "train", arcname="data/train")
        tar.add(args.split, arcname="data/split.csv")
        for extra in args.extra or []:
            tar.add(extra, arcname=f"data/{Path(extra).name}")
    payload = bundle.getvalue()
    machines_all = machines(args.access)
    check = ("cd $HOME/" + TAG + " && venv/bin/python -c 'import torch,torchvision,pandas,timm,huggingface_hub;torch.zeros(1).cuda();"
             "print(\"torch\",torch.__version__,torch.cuda.get_device_name(0))'")

    def install(key):
        m = machines_all[key]
        with connect(m) as client:
            code, _, _ = run(client, f"test -e {REMOTE} && ! test -f {REMOTE}/{OWNER}")
            if code == 0:
                raise RuntimeError(f"{REMOTE} exists on {m['ip']} but is not ours")
            run(client, f"mkdir -p {REMOTE}/runs {REMOTE}/logs {REMOTE}/cache && touch {REMOTE}/{OWNER}")
            home = remote_home(client)
            with client.open_sftp() as sftp:
                sftp.putfo(io.BytesIO(payload), f"{home}/{TAG}/bundle.tgz")
            run(client, f"{ENV} && rm -rf code data && tar xzf bundle.tgz && rm bundle.tgz")
            code, out, _ = run(client, check)
            if code == 0:
                return out.strip()
        setup = (f"{ENV} && rm -rf venv && "
                 "export UV_CACHE_DIR=$PWD/cache/uv UV_PYTHON_INSTALL_DIR=$PWD/python UV_INSTALL_DIR=$PWD/bin "
                 "UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 UV_HTTP_TIMEOUT=900 UV_CONCURRENT_DOWNLOADS=4 && "
                 "(test -x bin/uv || curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet) && "
                 "bin/uv venv -q --python 3.12 venv && "
                 "for attempt in 1 2 3 4; do bin/uv pip install -q --python venv/bin/python torch==2.9.1 "
                 "torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128 && break; done && "
                 "bin/uv pip install -q --python venv/bin/python pandas pillow numpy timm==1.0.31 huggingface_hub safetensors > logs/install.log 2>&1")
        with connect(m) as client:
            # detached so a dropped SSH session cannot interrupt the install
            run(client, f"cd {REMOTE} && nohup setsid bash -c {shlex.quote(setup)} > logs/install.log 2>&1 &")
        for _ in range(120):
            time.sleep(30)
            try:
                with connect(m) as client:
                    code, out, _ = run(client, check)
                    if code == 0:
                        return out.strip()
                    if run(client, "pgrep -f '[u]v pip install|[u]v venv|[i]nstall.sh' -u $(id -u) >/dev/null")[0]:
                        tail = run(client, f"tail -c 300 {REMOTE}/logs/install.log")[1]
                        raise RuntimeError("install stopped: " + tail)
            except (paramiko.SSHException, EOFError, OSError):
                continue
        raise RuntimeError("install timed out")

    from concurrent.futures import ThreadPoolExecutor
    keys = sorted({s["key"] for s in chosen})
    ready = set()
    with ThreadPoolExecutor(args.parallel) as pool:
        for key, future in zip(keys, [pool.submit(install, k) for k in keys]):
            try:
                print(f"{machines_all[key]['ip']}: ready ({future.result()})")
                ready.add(key)
            except Exception as exc:
                print(f"{machines_all[key]['ip']}: FAILED {type(exc).__name__}: {str(exc)[-300:]}")
    chosen = [s for s in chosen if s["key"] in ready]
    if STATE.exists():
        earlier = [s for s in json.loads(STATE.read_text(encoding="utf-8")) if s["key"] not in ready]
        chosen = earlier + chosen
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps([{k: s[k] for k in ("key", "group", "ip", "gpu", "name")} for s in chosen],
                                indent=1), encoding="utf-8")
    print(f"{len(chosen)} GPU slots recorded in {STATE}")


MINUTES = {"convnext_small": 25, "swin_t": 17, "convnext_tiny": 13, "resnet50": 11, "efficientnet_b0": 8,
           "efficientnet_v2_s": 16, "maxvit_t": 24, "vit_base_in21k": 18, "deit3_base_in22k": 18,
           "convnext_small_in22k": 25, "convnextv2_tiny_in22k": 16, "swin_small_in22k": 28}


def estimate(job):
    arch = job.split("--arch")[1].split()[0]
    return MINUTES.get(arch, 15) * (1.25 if "--fold all" in job else 1.0)


def launch(args):
    slots = [s for s in json.loads(STATE.read_text(encoding="utf-8")) if not args.ips or s["ip"] in args.ips]
    jobs = [line.strip() for line in Path(args.jobs).read_text().splitlines() if line.strip()]
    queues, load = {i: [] for i in range(len(slots))}, [0.0] * len(slots)
    for job in sorted(jobs, key=estimate, reverse=True):
        i = load.index(min(load))
        queues[i].append(job)
        load[i] += estimate(job)
    stamp = time.strftime("%H%M%S")
    machines_all = machines(args.access)
    for i, slot in enumerate(slots):
        if not queues[i]:
            continue
        name = f"queue_{stamp}_gpu{slot['gpu']}"
        lines = [f"{ENV}", f"PY=$HOME/{TAG}/venv/bin/python", "cd code"]
        if args.after:
            lines.append("until [ $(grep -l QUEUE_DONE ../logs/queue_*.log | wc -l) -ge "
                         "$(ls ../logs/queue_*.log | wc -l) ]; do sleep 20; done")
        for job in queues[i]:
            lines.append(f"CUDA_VISIBLE_DEVICES={slot['gpu']} $PY train_v2.py --data_root ../data "
                         f"--split ../data/split.csv --out ../runs {job} >> ../logs/{name}.log 2>&1 "
                         f"|| echo JOB_FAILED {shlex.quote(job)} >> ../logs/{name}.log")
        lines.append(f"echo QUEUE_DONE >> ../logs/{name}.log")
        script = "\n".join(lines) + "\n"
        with connect(machines_all[slot["key"]]) as client:
            guard(client)
            home = remote_home(client)
            with client.open_sftp() as sftp:
                sftp.putfo(io.BytesIO(script.encode()), f"{home}/{TAG}/{name}.sh")
            run(client, f"cd {REMOTE} && (setsid nohup bash {name}.sh > /dev/null 2>&1 & echo $! > {name}.pid)")
        print(f"{slot['ip']} gpu{slot['gpu']}: {len(queues[i])} jobs, about {load[i]:.0f} min")


def status(args):
    machines_all = machines(args.access)
    for i, slot in enumerate(json.loads(STATE.read_text(encoding="utf-8"))):
        with connect(machines_all[slot["key"]]) as client:
            guard(client)
            _, out, _ = run(client, f"for f in {REMOTE}/logs/queue_*.log; do grep -v '%' $f | tail -n 2; done; "
                                    f"nvidia-smi -i {slot['gpu']} --query-gpu=utilization.gpu,memory.used "
                                    "--format=csv,noheader")
        print(f"[{slot['ip']} gpu{slot['gpu']}]\n{out.strip()}")


def fetch(args):
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    machines_all = machines(args.access)
    for key in sorted({s["key"] for s in json.loads(STATE.read_text(encoding="utf-8"))}):
        with connect(machines_all[key]) as client:
            guard(client)
            with client.open_sftp() as sftp:
                home = remote_home(client)
                for folder in sorted(n for n in sftp.listdir(f"{home}/{TAG}") if n.startswith("runs")):
                    remote, local = f"{home}/{TAG}/{folder}", dest / folder[len("runs"):].lstrip("_")
                    local.mkdir(parents=True, exist_ok=True)
                    for name in sftp.listdir(remote):
                        wanted = name.endswith((".csv", ".json")) or (args.members and name.startswith("member_"))
                        if wanted:
                            sftp.get(f"{remote}/{name}", str(local / name))
                for name in sftp.listdir(f"{home}/{TAG}/logs"):
                    sftp.get(f"{home}/{TAG}/logs/{name}", str(dest / f"{machines_all[key]['ip']}_{name}"))
    print(f"fetched into {dest} (runs/ -> {dest}, runs_<name>/ -> {dest}/<name>)")


def cleanup(args):
    machines_all = machines(args.access)
    for key in sorted({s["key"] for s in json.loads(STATE.read_text(encoding="utf-8"))}):
        m = machines_all[key]
        with connect(m) as client:
            if run(client, f"test -f {REMOTE}/{OWNER}")[0]:
                print(f"{m['ip']}: nothing of ours left")
                continue
            # stop only the process groups we started (setsid makes each queue its own group)
            run(client, f"for f in {REMOTE}/queue_*.pid; do [ -f \"$f\" ] && kill -- -$(cat \"$f\") 2>/dev/null; done; "
                        "sleep 3; true")
            code, _, err = run(client, f"rm -rf -- {REMOTE} && ! test -e {REMOTE}")
            print(f"{m['ip']}: {'removed' if code == 0 else 'cleanup failed ' + err[-200:]}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["probe", "deploy", "launch", "status", "fetch", "cleanup"])
    p.add_argument("--access", required=True, help="xlsx with Group, IP Address, Username, Password")
    p.add_argument("--data_root")
    p.add_argument("--split")
    p.add_argument("--extra", nargs="*", help="extra files copied into the remote data folder")
    p.add_argument("--gpus", type=int, default=6)
    p.add_argument("--per_machine", type=int, default=2)
    p.add_argument("--max_util", type=int, default=10, help="only GPUs at or below this utilisation")
    p.add_argument("--max_used_mb", type=int, default=2000)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--ips", nargs="*", help="only consider these machines")
    p.add_argument("--jobs")
    p.add_argument("--after", action="store_true", help="start once the queues already there have finished")
    p.add_argument("--dest", default="runs")
    p.add_argument("--members", action="store_true", help="fetch refit member weights too")
    a = p.parse_args(argv)
    {"probe": probe, "deploy": deploy, "launch": launch, "status": status, "fetch": fetch, "cleanup": cleanup}[a.command](a)


if __name__ == "__main__":
    main()
