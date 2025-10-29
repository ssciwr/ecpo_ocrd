import functools
import os
import shutil
import subprocess
import venv


_prefix = None


@functools.lru_cache()
def create_env(env_identifier):
    envdir = _prefix / "envs" / env_identifier
    if envdir.exists():
        return envdir

    builder = venv.EnvBuilder(with_pip=True)
    builder.create(envdir)
    return envdir


@functools.lru_cache()
def install_package(package, envdir):
    python_executable = envdir / "bin" / "python"
    subprocess.run(
        [str(python_executable), "-m", "pip", "install", package],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def install_ocrd_tool(tool, package=None, env_identifier="core", models=[]):
    assert package is not None

    # Ensure existence of the environment
    env = create_env(env_identifier=env_identifier)

    # Ensure installation of the package
    install_package(package, env)

    # Download models (if any)
    for model in models:
        subprocess.run(
            [str(env / "bin" / "ocrd"), "resmgr", "download", tool, model],
            check=True,
        )

    # Ensure existence of bin directory
    (_prefix / "bin").mkdir(parents=True, exist_ok=True)

    # Ensure that a wrapper is written
    tool_path = _prefix / "bin" / tool
    with open(tool_path, "w") as f:
        f.write(
            f"""#!/bin/bash
source "{str(env / 'bin' / 'activate')}"
"{tool}" "$@"
"""
        )
    os.chmod(tool_path, 0o755)

    # Check that the tool is callable
    subprocess.run([str(tool_path), "--help"], check=True, stdout=subprocess.DEVNULL)


def install(prefix):
    # We store the prefix globally
    global _prefix
    _prefix = prefix

    install_ocrd_tool("ocrd-segment-extract-regions", package="ocrd_segment")
    install_ocrd_tool("ocrd-skimage-denoise-raw", package="ocrd_wrap")
    install_ocrd_tool("ocrd-skimage-normalize", package="ocrd_wrap")
    install_ocrd_tool(
        "ocrd-sbb-binarize", package="eynollah", env_identifier="eynollah", models=["*"]
    )
    install_ocrd_tool("ocrd-cis-ocropy-denoise", package="ocrd_cis")
    install_ocrd_tool("ocrd-cis-ocropy-deskew", package="ocrd_cis")


def uninstall(prefix):
    envs_dir = prefix / "envs"
    if envs_dir.exists():
        shutil.rmtree(envs_dir)

    bin_dir = prefix / "bin"
    if bin_dir.exists():
        shutil.rmtree(bin_dir)
