"""Build the original HW3 C++ backend using the active Python interpreter."""
import importlib.util
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sysconfig


ROOT = Path(__file__).resolve().parents[1]


def main():
    spec = importlib.util.find_spec("pybind11")
    if spec is not None:
        import pybind11
        include = Path(pybind11.get_include())
    else:
        # Reuse installed headers when available; torch is not a runtime dependency.
        spec = importlib.util.find_spec("torch")
        include = (Path(next(iter(spec.submodule_search_locations))) / "include") if spec else None
        if include is None or not (include / "pybind11/pybind11.h").is_file():
            raise SystemExit("Install the build dependency first: python -m pip install pybind11")
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise SystemExit("A C++ compiler is required.")
    output = ROOT / "python/needle/backend_ndarray" / ("ndarray_backend_cpu" + sysconfig.get_config_var("EXT_SUFFIX"))
    command = [compiler, "-std=c++17", "-O2", "-shared", "-fPIC",
               "-I" + str(include), "-I" + sysconfig.get_path("include"),
               str(ROOT / "src/ndarray_backend_cpu.cc"), "-o", str(output)]
    if platform.system() == "Darwin":
        sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
        command.extend(["-isysroot", sdk])
        cpp_headers = Path(sdk) / "usr/include/c++/v1"
        if cpp_headers.is_dir():
            command.extend(["-isystem", str(cpp_headers)])
        command.extend(["-undefined", "dynamic_lookup"])
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)
    print("Built:", output.relative_to(ROOT))


if __name__ == "__main__":
    main()
