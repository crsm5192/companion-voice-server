"""TTS 常驻服务看门狗

作用：
  1) 源码变动自动重启 —— 监视 tts_server.py 的 mtime，一改就重启服务（免手动）
  2) 掉线自愈 —— 8001 没有 LISTENING 就立刻拉起
  3) 等待 /health 通过，记录到 watch_tts_server.log

⚠ 无窗口纪律（用户反馈「桌面会闪 cmd 窗口」后修正）：
  **本脚本一切探测/杀进程动作都走 psutil，绝不调用 netstat.exe / tasklist.exe / taskkill.exe。**
  原因：本进程由 pythonw.exe（无控制台）启动，一旦 subprocess 拉起控制台程序（netstat 等），
  Windows 会为它分配一个新控制台 → 桌面闪出转瞬即逝的黑框。每 8 秒闪一次 = 无法工作。
  唯一允许的 subprocess 是「拉起服务进程」，且必须带 CREATE_NO_WINDOW | DETACHED_PROCESS。

用法（后台无窗口，推荐）：
    <venv>/Scripts/pythonw.exe watch_tts_server.py

前台调试 / 只看一次：
    python watch_tts_server.py --once

停止：删除或建立 STOP_WATCHDOG 文件；或结束该 pythonw 进程
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

try:
    import psutil
except ImportError:  # 理论上不会发生（venv 已装 psutil 7.x）
    psutil = None

WORKDIR = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_server.py")
PYW = os.path.join(WORKDIR, ".venv", "Scripts", "pythonw.exe")
LOG = os.path.join(WORKDIR, "watch_tts_server.log")
STOP_FLAG = os.path.join(WORKDIR, "STOP_WATCHDOG")
LOCK_FILE = os.path.join(WORKDIR, "watch_tts_server.lock")
BIND_HOST = "0.0.0.0"      # 服务监听地址
PROBE_HOST = "127.0.0.1"   # 健康探测地址（连 0.0.0.0 会失败，必须分开）
PORT = 8001
POLL_SECONDS = 8
HEALTH_WAIT_SECONDS = 120

CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _listen_pids(port: int = PORT) -> set[int]:
    """用 psutil 查监听端口的 PID —— 不调用 netstat，不会闪窗。"""
    pids: set[int] = set()
    if psutil is None:
        return pids
    try:
        for c in psutil.net_connections(kind="tcp"):
            if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port and c.pid:
                pids.add(c.pid)
    except (psutil.Error, OSError):
        pass
    return pids


def listening_pid(port: int = PORT) -> int | None:
    pids = _listen_pids(port)
    return sorted(pids)[0] if pids else None


def health_ok(timeout: float = 4.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://{PROBE_HOST}:{PORT}/health", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace")).get("ok") is True
    except Exception:
        return False


def kill_pid(pid: int) -> None:
    """用 psutil 结束进程 —— 不调用 taskkill，不会闪窗。"""
    if psutil is None:
        return
    try:
        p = psutil.Process(pid)
        p.terminate()
        try:
            p.wait(timeout=8)
        except psutil.TimeoutExpired:
            p.kill()
            p.wait(timeout=8)
    except psutil.NoSuchProcess:
        pass
    except psutil.Error as exc:
        log(f"kill {pid} 失败：{exc}")


def launch() -> None:
    if not os.path.exists(PYW):
        log(f"缺失 pythonw.exe：{PYW}")
        return
    args = [PYW, SERVER, "--host", BIND_HOST, "--port", str(PORT), "--device", "auto"]
    # 唯一的 subprocess：必须无窗口 + 脱离本进程，否则会闪 cmd 框
    subprocess.Popen(
        args, cwd=WORKDIR,
        creationflags=CREATE_NO_WINDOW | DETACHED_PROCESS,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
        close_fds=True,
    )
    log(f"已拉起服务：{SERVER} --host {BIND_HOST} --port {PORT} --device auto")


def restart(reason: str) -> None:
    log(f"重启（原因：{reason}）")
    pid = listening_pid()
    if pid:
        kill_pid(pid)
        time.sleep(3)
    launch()
    deadline = time.time() + HEALTH_WAIT_SECONDS
    while time.time() < deadline:
        if health_ok():
            log("重启完成，/health OK")
            return
        time.sleep(3)
    log(f"警告：{HEALTH_WAIT_SECONDS}s 内 /health 未通过，请查 tts_server.log")


def pid_alive(pid: int) -> bool:
    if psutil is None:
        return False
    try:
        return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
    except psutil.Error:
        return False


def acquire_singleton() -> bool:
    """独占锁：同一时间只允许一个看门狗（避免重复实例互相抢着重启）。"""
    if os.path.exists(LOCK_FILE):
        try:
            old_pid = int((open(LOCK_FILE, encoding="utf-8").read().strip() or "0"))
        except (OSError, ValueError):
            old_pid = 0
        if old_pid and old_pid != os.getpid() and pid_alive(old_pid):
            log(f"已有看门狗在跑（PID {old_pid}），本次退出")
            return False
    try:
        with open(LOCK_FILE, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except OSError:
        pass
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="只做一轮检查后退出（调试用）")
    args = ap.parse_args()

    if psutil is None:
        log("缺少 psutil，无法做到无窗口探测，退出")
        return 1
    if not os.path.exists(SERVER):
        log(f"找不到服务脚本：{SERVER}")
        return 1

    if not args.once and not acquire_singleton():
        return 0

    last_mtime = os.path.getmtime(SERVER)
    log(f"看门狗启动（监视 {SERVER}，轮询 {POLL_SECONDS}s，无窗口模式）")

    if not health_ok():
        restart("启动时服务未就绪")

    if args.once:
        return 0

    while True:
        if os.path.exists(STOP_FLAG):
            log("检测到 STOP_WATCHDOG，看门狗退出")
            return 0
        time.sleep(POLL_SECONDS)
        try:
            cur_mtime = os.path.getmtime(SERVER)
        except OSError:
            continue
        if cur_mtime != last_mtime:
            last_mtime = cur_mtime
            restart(f"源码变动 {datetime.fromtimestamp(cur_mtime):%H:%M:%S}")
            continue
        if listening_pid() is None or not health_ok():
            restart("服务掉线")


if __name__ == "__main__":
    sys.exit(main())
