"""Benchmark helpers for TauFactor convergence studies."""

from __future__ import annotations

import gc
import inspect
import io
import itertools
import os
import time
from contextlib import redirect_stdout

import torch

import taufactor as tau
from taufactor.utils import create_fcc_cube, create_2d_diagonals, create_3d_diagonals, create_stacked_blocks, create_2d_zigzag

DEFAULT_OUTFILE = "taufactor_benchmark_results.txt"

# Benchmarkable solvers exposed by the top-level package.
SOLVER_REGISTRY = {
    name: cls
    for name, cls in vars(tau).items()
    if name.endswith("Solver") and isinstance(cls, type) and not inspect.isabstract(cls)
}


def resolve_solver(solver: str | type | None) -> type:
    """Resolve a solver provided either as class object or string name."""
    if solver is None:
        return tau.PeriodicSolver
    if isinstance(solver, str):
        if solver not in SOLVER_REGISTRY:
            available = ", ".join(sorted(SOLVER_REGISTRY))
            raise ValueError(
                f"Unknown solver '{solver}'. Available solvers: {available}"
            )
        return SOLVER_REGISTRY[solver]
    if isinstance(solver, type):
        return solver
    raise TypeError("solver must be None, a solver class, or a solver name string")


def write_header_if_missing(outfile: str = DEFAULT_OUTFILE) -> None:
    """Create output file and write header if it does not exist."""
    if not os.path.exists(outfile):
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(
                f"{'N':>4} {'struct':>10} {'solver':>16} {'dev':>4} {'conv':>6} "
                f"{'Ttime(s)':>9} {'Wtime(s)':>9} {'iters':>6} {'tau':>8} "
                f"{'VRAM(cur)':>10} {'VRAM(max)':>10} {'VRAM(res)':>10}\n"
            )
            f.write("=" * 120 + "\n")


def append_row_to_file(row: dict, outfile: str = DEFAULT_OUTFILE) -> None:
    """Format a benchmark result row and append it to the output file."""
    line = (
        f"{row['N']:4d} {row['structure'][:10]:>10} {row['solver'][:16]:>16} {row['device'][:4]:>4} {row['conv_crit']:.4f} "
        f"{row['total_time']:9.3f} {row['solve_time']:9.3f} {row['iterations']:6d} {row['taufactor']:8.3f} "
        f"{row['torch_cur']:10.2f} {row['torch_max']:10.2f} {row['torch_res']:10.2f}\n"
    )
    with open(outfile, "a", encoding="utf-8") as f:
        f.write(line)


def run_benchmark_case(
    N: int,
    device: str,
    conv_crit: float,
    structure: str = "fcc",
    features: int | None = None,
    iter_limit: int = 10000,
    solver: str | type | None = None,
    solver_kwargs: dict | None = None,
    solve_kwargs: dict | None = None,
) -> dict:
    """Run a single structure benchmark case."""
    if structure == "fcc":
        cube = create_fcc_cube(N, overlap=0.05)
        cube = (cube==0).astype(int)
    elif structure == "blocks":
        cube = create_stacked_blocks(N, features=features)
    elif structure == "diagonal2d":
        cube = create_2d_diagonals(N, features=features)
    elif structure == "zigzag":
        cube = create_2d_zigzag(N, features=features)
    elif structure == "diagonal3d":
        cube = create_3d_diagonals(N, features=features)
    else:
        raise ValueError(
            f"Unknown structure '{structure}'. Supported: fcc, blocks, diagonal2d, zigzag, diagonal3d"
        )

    solver_cls = resolve_solver(solver)
    solver_kwargs = dict(solver_kwargs or {})
    solve_kwargs = dict(solve_kwargs or {})

    if device == "cuda":
        torch.cuda.empty_cache()
    torch._dynamo.reset()
    gc.collect()

    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    buf = io.StringIO()
    with redirect_stdout(buf):
        solver = solver_cls(cube, device=device, **solver_kwargs)
        solver.solve(iter_limit=iter_limit, conv_crit=conv_crit, **solve_kwargs)

        if device == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()

    out = buf.getvalue().splitlines()

    conv_line = next(line for line in out if line.startswith("converged to"))
    torch_line = next((line for line in out if "GPU-RAM" in line), "")

    iterations = int(conv_line.split()[4])
    wall_time = float(conv_line.split()[7])
    taufactor = float(solver.tau[0])

    if torch_line:
        parts = torch_line.replace("(", "").replace(")", "").replace(",", "").split()
        torch_cur = float(parts[2])
        torch_max = float(parts[6])
        torch_res = float(parts[8])
    else:
        torch_cur = 0.0
        torch_max = 0.0
        torch_res = 0.0

    return {
        "N": N,
        "structure": structure,
        "solver": solver_cls.__name__,
        "device": device,
        "conv_crit": conv_crit,
        "total_time": end_time - start_time,
        "solve_time": wall_time,
        "iterations": iterations,
        "taufactor": taufactor,
        "torch_cur": torch_cur,
        "torch_max": torch_max,
        "torch_res": torch_res,
    }


def run_benchmark_study(
    Ns: list[int] | tuple[int, ...] = (100, 128, 200, 256, 300, 384, 400),
    devices: list[str] | tuple[str, ...] = ("cuda",),
    conv_crit_values: list[float] | tuple[float, ...] = (1e-3,),
    structure: str = "fcc",
    features: int | None = None,
    outfile: str = DEFAULT_OUTFILE,
    write_file: bool = True,
    iter_limit: int = 10000,
    solver: str | type | None = None,
    solver_kwargs: dict | None = None,
    solve_kwargs: dict | None = None,
) -> list[dict]:
    """Run a convergence benchmark on synthetic structures."""
    rows: list[dict] = []

    if write_file:
        write_header_if_missing(outfile=outfile)

    for N, device, conv_crit in itertools.product(Ns, devices, conv_crit_values):
        if device == "cuda" and not torch.cuda.is_available():
            print(f"Skipping N={N} on CUDA (not available)")
            continue

        row = run_benchmark_case(
            N=N,
            device=device,
            conv_crit=conv_crit,
            structure=structure,
            features=features,
            iter_limit=iter_limit,
            solver=solver,
            solver_kwargs=solver_kwargs,
            solve_kwargs=solve_kwargs,
        )
        rows.append(row)

        if write_file:
            append_row_to_file(row, outfile=outfile)

    return rows
