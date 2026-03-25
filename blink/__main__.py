"""
blink.__main__
==============
CLI entry point for the blink-gpu package.

Usage
-----
    # Quick prediction from the command line
    blink predict resnet50 --batch-size 32

    # Start the Streamlit dashboard
    blink dashboard

    # Start the FastAPI server
    blink-server
    blink-server --host 0.0.0.0 --port 8000 --workers 4

    # Check installed version
    blink --version
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Entry point for the ``blink`` CLI command."""
    parser = argparse.ArgumentParser(
        prog="blink",
        description="Blink — GPU performance predictor for PyTorch models",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    sub = parser.add_subparsers(dest="command")

    # ── predict ───────────────────────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="Predict GPU usage for a named model")
    p_pred.add_argument("model", help='Model name, e.g. "resnet50"')
    p_pred.add_argument("--batch-size", "-b", type=int, default=32)
    p_pred.add_argument("--all-batches", action="store_true",
                        help="Also show batch sizes 1,2,4,8,16,64")

    # ── dashboard ─────────────────────────────────────────────────────────────
    p_dash = sub.add_parser("dashboard", help="Launch the Streamlit dashboard")
    p_dash.add_argument("--port", type=int, default=8501)

    # ── server ────────────────────────────────────────────────────────────────
    p_srv = sub.add_parser("server", help="Launch the FastAPI REST server")
    p_srv.add_argument("--host", default="0.0.0.0")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--workers", "-w", type=int, default=1)

    args = parser.parse_args()

    # ── --version ──────────────────────────────────────────────────────────────
    if args.version or args.command is None:
        from blink._version import __version__
        print(f"blink-gpu {__version__}")
        if args.command is None and not args.version:
            parser.print_help()
        return

    # ── predict ───────────────────────────────────────────────────────────────
    if args.command == "predict":
        from blink._predictor import BlinkPredictor
        p = BlinkPredictor()
        batch_sizes = ([1, 2, 4, 8, 16, args.batch_size, 64]
                       if args.all_batches else [args.batch_size])
        batch_sizes = sorted(set(batch_sizes))

        print(f"\n🔮 Blink prediction for '{args.model}'\n")
        print(f"{'Batch':>6}  {'Exec (ms)':>10}  {'Memory (MB)':>12}  CI-Exec (80%)")
        print("-" * 60)
        for bs in batch_sizes:
            try:
                r = p.predict(args.model, batch_size=bs)
                ci = f"[{r['exec_time_lower']:.1f} – {r['exec_time_upper']:.1f}]"
                print(f"{bs:>6}  {r['exec_time_ms']:>10.2f}  {r['memory_mb']:>12.1f}  {ci}")
            except Exception as e:
                print(f"{bs:>6}  ERROR: {e}")
        print()
        return

    # ── dashboard ─────────────────────────────────────────────────────────────
    if args.command == "dashboard":
        import subprocess
        from pathlib import Path
        dash = Path(__file__).parent.parent / "dashboard.py"
        print(f"Launching Blink dashboard on http://localhost:{args.port} ...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dash),
            f"--server.port={args.port}",
        ], check=True)
        return

    # ── server ────────────────────────────────────────────────────────────────
    if args.command == "server":
        serve(host=args.host, port=args.port, workers=args.workers)
        return


def serve(host: str = "0.0.0.0", port: int = 8000, workers: int = 1) -> None:
    """Entry point for the ``blink-server`` CLI command."""
    import uvicorn
    print(f"🚀 Starting Blink REST API on http://{host}:{port} | docs: /docs")
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
