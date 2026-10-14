"""Command line entry points: fetch the data, run the study, launch the app."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="shotsense")
    sub = parser.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="rebuild data/shots.csv.gz from the public event data")
    f.add_argument("--clone", default=".cache/open-data")
    f.add_argument("--out", default="data/shots.csv.gz")

    r = sub.add_parser("run", help="train, evaluate and write outputs/ and assets/")
    r.add_argument("--data", default="data/shots.csv.gz")
    r.add_argument("--out", default="outputs")
    r.add_argument("--assets", default="assets")
    r.add_argument("--quick", action="store_true", help="fewer bootstrap draws")

    a = sub.add_parser("app", help="launch the Gradio scorer")
    a.add_argument("--model", default="outputs/model.joblib")

    args = parser.parse_args(argv)
    if args.cmd == "fetch":
        from .extract import build
        build(Path(args.clone), Path(args.out))
    elif args.cmd == "run":
        from .pipeline import run
        run(Path(args.data), Path(args.out), Path(args.assets), quick=args.quick)
    else:
        from .app import build_app
        build_app(Path(args.model)).launch()


if __name__ == "__main__":
    main()
