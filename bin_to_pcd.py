#!/usr/bin/env python3
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm


def load_kitti_bin(bin_path: Path) -> np.ndarray:
    arr = np.fromfile(str(bin_path), dtype=np.float32)

    if arr.size % 4 != 0:
        raise ValueError(
            f"{bin_path} size is not divisible by 4. "
            f"Expected KITTI-style float32 Nx4 bin, got {arr.size} floats."
        )

    points = arr.reshape(-1, 4)

    # remove NaN / inf
    mask = np.isfinite(points).all(axis=1)
    return points[mask].astype(np.float32, copy=False)


def save_pcd_binary(points: np.ndarray, pcd_path: Path) -> None:
    """
    Save Nx4 [x, y, z, intensity] as binary PCD.
    """
    pcd_path.parent.mkdir(parents=True, exist_ok=True)
    n = points.shape[0]

    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {n}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {n}
DATA binary
"""

    with open(pcd_path, "wb") as f:
        f.write(header.encode("ascii"))
        points.tofile(f)


def save_pcd_ascii(points: np.ndarray, pcd_path: Path) -> None:
    """
    Save Nx4 [x, y, z, intensity] as ASCII PCD.
    Slower and larger than binary.
    """
    pcd_path.parent.mkdir(parents=True, exist_ok=True)
    n = points.shape[0]

    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {n}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {n}
DATA ascii
"""

    with open(pcd_path, "w") as f:
        f.write(header)
        np.savetxt(f, points, fmt="%.6f %.6f %.6f %.6f")


def convert_one(args_tuple):
    bin_path, out_path, overwrite, pcd_format = args_tuple

    bin_path = Path(bin_path)
    out_path = Path(out_path)

    if out_path.exists() and not overwrite:
        return str(bin_path), str(out_path), "skip", 0

    points = load_kitti_bin(bin_path)

    if pcd_format == "binary":
        save_pcd_binary(points, out_path)
    elif pcd_format == "ascii":
        save_pcd_ascii(points, out_path)
    else:
        raise ValueError(f"Unsupported format: {pcd_format}")

    return str(bin_path), str(out_path), "ok", points.shape[0]


def build_tasks(input_path: Path, output_path: Path, overwrite: bool, pcd_format: str):
    if input_path.is_file():
        if output_path.suffix.lower() == ".pcd":
            out_file = output_path
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            out_file = output_path / f"{input_path.stem}.pcd"

        return [(input_path, out_file, overwrite, pcd_format)]

    if input_path.is_dir():
        bin_files = sorted(input_path.glob("*.bin"))
        if not bin_files:
            raise FileNotFoundError(f"No .bin files found in {input_path}")

        output_path.mkdir(parents=True, exist_ok=True)

        return [
            (bin_file, output_path / f"{bin_file.stem}.pcd", overwrite, pcd_format)
            for bin_file in bin_files
        ]

    raise FileNotFoundError(input_path)


def main():
    parser = argparse.ArgumentParser(
        description="Parallel KITTI-style .bin to .pcd converter"
    )
    parser.add_argument("--input", required=True, help="Input .bin file or directory")
    parser.add_argument("--output", required=True, help="Output .pcd file or directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument(
        "--backend",
        choices=["process", "thread"],
        default="process",
        help="Parallel backend. process is usually faster for ASCII; thread is safer for I/O."
    )
    parser.add_argument(
        "--format",
        choices=["binary", "ascii"],
        default="binary",
        help="PCD save format"
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .pcd files")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    tasks = build_tasks(input_path, output_path, args.overwrite, args.format)

    Executor = ProcessPoolExecutor if args.backend == "process" else ThreadPoolExecutor

    ok = 0
    skipped = 0
    failed = 0

    with Executor(max_workers=args.workers) as ex:
        futures = [ex.submit(convert_one, task) for task in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
            try:
                bin_file, out_file, status, n_points = fut.result()

                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skipped += 1

            except Exception as e:
                failed += 1
                print(f"[ERROR] {e}")

    print("Done")
    print(f"ok      : {ok}")
    print(f"skipped : {skipped}")
    print(f"failed  : {failed}")


if __name__ == "__main__":
    main()
