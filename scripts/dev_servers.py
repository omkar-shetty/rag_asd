"""Start the local dev stack: DynamoDB Local, the read server and the actions server.

Run from the repo root:  python scripts/dev.py
Ctrl+C stops both Python servers. DynamoDB Local keeps running; stop it with
`docker stop dynamodb-local`.
"""

import os
import subprocess
import sys
import socket
import time

ENV = {
    **os.environ,
    "MCP_TRANSPORT": "streamable-http",
    "DYNAMODB_ENDPOINT": os.getenv("DYNAMODB_ENDPOINT", "http://127.0.0.1:8001"),
}

MODULES = ["src.mcp_server.server", "src.actions_server.server"]
SERVERS = [
    ("read", "src.mcp_server.server", 8000),
    ("actions", "src.actions_server.server", 8002),
]


def wait_for_port(port, proc, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:        # the server already exited
            return False
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return True
        except OSError:
            time.sleep(1)
    return False


def main():
    subprocess.run(["docker", "start", "dynamo-local"], check=True)
    procs = [subprocess.Popen([sys.executable, "-m", module], env=ENV)
             for _, module, _ in SERVERS]
    try:
        for (name, _, port), proc in zip(SERVERS, procs):
            ok = wait_for_port(port, proc)
            print(f"{name} server ready" if ok else f"{name} server FAILED to start")
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait()


if __name__ == "__main__":
    main()