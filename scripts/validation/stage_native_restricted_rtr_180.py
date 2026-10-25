"""Stage one pinned #180 trust-region experiment on a separate Slurm node.

The checked-in scientific package in the archived source and the original
producer stay untouched. Only the new diagnostic driver and its three pinned
support modules are copied into the job-local checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_plan(config: dict) -> dict:
    replay = config["schema"] == "native180-restricted-rtr-replay-v1"
    assert config["schema"] in (
        "native180-restricted-rtr-v1",
        "native180-restricted-rtr-replay-v1",
    )
    assert config["limits"] == {
        "cpus": 8,
        "memory_GiB": 64,
        "wall_seconds": 2700 if replay else 5400,
    }
    if replay:
        assert config["experiment"] == {"mode": "checkpoint_evaluation_no_solve"}
        for name in ("report", "checkpoint", "arrays"):
            assert (
                digest(config["prior"][name]["path"]) == config["prior"][name]["sha256"]
            ), name
    else:
        assert config["experiment"] == {"max_steps": 200, "max_tcg_steps": 10}
    assert digest(config["source_archive"]) == config["source_archive_sha256"]
    for name, path in config["input_files"].items():
        assert digest(path) == config["input_sha256"][name], name
    for name, tool in config["tools"].items():
        assert digest(tool["path"]) == tool["sha256"], name
    producer_dir = Path(config["producer"]["producer_dir"])
    assert (
        digest(producer_dir / "manifest.json")
        == config["producer"]["producer_manifest_sha256"]
    )
    producer = json.loads((producer_dir / "manifest.json").read_text())
    assert (
        producer["source_environment"]["source_files_sha256"]
        == config["source_identity"]["source_files_sha256"]
    )
    assert (
        producer["source_environment"]["installed_distributions"]
        == config["source_identity"]["installed_distributions"]
    )
    assert producer["inputs"]["sha256"] == config["input_sha256"]
    assert producer["per_point"][0]["grid_value"] == 0.050088084721517094
    assert (
        producer["per_point"][0]["parameter_checkpoint_sha256"]
        == config["producer"]["checkpoint_sha256"]
    )
    for name, sha in producer["outputs_sha256"].items():
        assert Path(name).name == name and digest(producer_dir / name) == sha, name
    return producer


def main() -> None:
    if sys.flags.optimize:
        raise RuntimeError("provenance assertions must not run under python -O")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    producer = verify_plan(config)
    if args.check_only:
        print(json.dumps({"ready": True, "config_sha256": digest(args.config)}))
        return

    cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    allowed = sorted(os.sched_getaffinity(0))
    assert cpus == 8 and len(allowed) >= cpus and sys.version_info[:3] == (3, 11, 6)
    os.sched_setaffinity(0, allowed[:cpus])
    job_id = os.environ["SLURM_JOB_ID"]
    stage = Path(f"/local/job{job_id}/native180-restricted-rtr")
    output = args.config.parent / "results" / f"job-{job_id}"
    stage.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "complete": False,
        "job_id": job_id,
        "config_sha256": digest(args.config),
        "affinity": sorted(os.sched_getaffinity(0)),
        "source_environment": config["source_identity"],
        "input_sha256": config["input_sha256"],
        "producer_manifest_sha256": config["producer"]["producer_manifest_sha256"],
    }
    write(output / "index.json", record)
    started = time.perf_counter()
    try:
        for name in ("source", "input", "tmp", "cache", "external-guard"):
            (stage / name).mkdir()
        with tarfile.open(config["source_archive"]) as archive:
            archive.extractall(stage / "source", filter="data")
        source_root = stage / "source/src/emu_gmm"
        source_hashes = {
            str(path.relative_to(source_root)): digest(path)
            for path in sorted(source_root.rglob("*.py"))
        }
        assert source_hashes == config["source_identity"]["source_files_sha256"]
        validation = stage / "source/scripts/validation"
        assert (
            digest(validation / "native_projected_set_180.py")
            == config["source_identity"]["driver_sha256"]
        )
        local_inputs = {}
        for name, source in config["input_files"].items():
            target = stage / "input" / Path(source).name
            shutil.copyfile(source, target)
            assert digest(target) == config["input_sha256"][name]
            local_inputs[name] = target

        env = os.environ.copy()
        for name in (
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "PIP_INDEX_URL",
            "PIP_EXTRA_INDEX_URL",
            "PIP_FIND_LINKS",
            "PIP_REQUIREMENT",
            "PIP_CONSTRAINT",
        ):
            env.pop(name, None)
        env.update(
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
            PIP_CONFIG_FILE="/dev/null",
            PIP_CACHE_DIR=str(stage / "cache"),
            PIP_DISABLE_PIP_VERSION_CHECK="1",
            PIP_NO_INPUT="1",
            TMPDIR=str(stage / "tmp"),
        )
        venv = stage / ".venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv)], env=env, check=True)
        python = venv / "bin/python"
        bundle = Path(config["bundle_dir"])
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--only-binary=:all:",
                "--find-links",
                str(bundle / "wheelhouse"),
                "--requirement",
                str(bundle / "metadata/consumer-index-requirements.txt"),
            ],
            env=env,
            check=True,
            stdout=(output / "install.log").open("wb"),
        )
        installed = {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions(
                path=[str(venv / "lib/python3.11/site-packages")]
            )
        }
        assert installed == config["source_identity"]["installed_distributions"]
        local_tools = {}
        for name, tool in config["tools"].items():
            directory = stage / "external-guard" if name == "guard" else validation
            target = directory / Path(tool["path"]).name
            shutil.copyfile(tool["path"], target)
            assert digest(target) == tool["sha256"]
            local_tools[name] = target
        env.update(
            PYTHONPATH=str(stage / "source/src"),
            SUCODER_LOCAL_ROOT=str(stage),
            JAX_PLATFORMS="cpu",
            JAX_ENABLE_X64="1",
            XLA_FLAGS="--xla_force_host_platform_device_count=1",
            JAX_COMPILATION_CACHE_DIR=str(stage / "cache/jax"),
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
        )
        expected_env = config["source_identity"]["numerical_environment"]
        assert {key: env.get(key) for key in expected_env} == expected_env
        command = [
            str(python),
            "-u",
            str(local_tools["driver"]),
            "--producer-dir",
            config["producer"]["producer_dir"],
            "--manifest-sha256",
            config["producer"]["producer_manifest_sha256"],
            "--guard",
            str(local_tools["guard"]),
            "--guard-sha256",
            config["tools"]["guard"]["sha256"],
            "--data",
            str(local_inputs["data"]),
            "--report",
            str(local_inputs["report"]),
            "--warm",
            str(local_inputs["warm"]),
            "--output-dir",
            str(output / "experiment"),
        ]
        if config["schema"] == "native180-restricted-rtr-replay-v1":
            command += [
                "--replay-report",
                config["prior"]["report"]["path"],
                "--replay-report-sha256",
                config["prior"]["report"]["sha256"],
                "--replay-checkpoint",
                config["prior"]["checkpoint"]["path"],
                "--replay-arrays",
                config["prior"]["arrays"]["path"],
                "--original-driver-sha256",
                config["prior"]["driver_sha256"],
            ]
        else:
            command += [
                "--max-steps",
                str(config["experiment"]["max_steps"]),
                "--max-tcg-steps",
                str(config["experiment"]["max_tcg_steps"]),
            ]
        record["command"] = command
        record["experiment_processes_started"] = 1
        write(output / "index.json", record)
        with (
            (output / "experiment.out").open("wb") as stdout,
            (output / "experiment.err").open("wb") as stderr,
        ):
            run = subprocess.run(
                command, env=env, cwd=stage, stdout=stdout, stderr=stderr
            )
        record["returncode"] = run.returncode
        assert run.returncode == 0, f"single experiment returned {run.returncode}"
        report = json.loads((output / "experiment/rtr-probe.json").read_text())
        assert (
            "execution_failure" not in report
            and report["preflight"]["original_objective_and_k_parity"]
        )
        assert report["driver_sha256"] == config["tools"]["driver"]["sha256"]
        if config["schema"] == "native180-restricted-rtr-replay-v1":
            assert report["new_optimizer_solves"] == 0
            assert report["checkpoint_evaluations"] == 1
            assert report["postfit_parity"] is True
            assert not (output / "experiment/optimizer.jsonl").exists()
        else:
            assert (
                len((output / "experiment/optimizer.jsonl").read_text().splitlines())
                == 1
            )
        assert report["producer_record"] == producer["per_point"][0]
        for name, sha in report["outputs_sha256"].items():
            assert digest(output / "experiment" / name) == sha, name
        record["complete"] = True
        record["report_sha256"] = digest(output / "experiment/rtr-probe.json")
    except BaseException as exc:
        record["execution_failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        write(output / "index.json", record)


if __name__ == "__main__":
    main()
