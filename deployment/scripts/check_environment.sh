#!/usr/bin/env bash
# Read-only environment report for Jetson/TensorRT deployment readiness.

set -u

run_if_available() {
  local command_name="$1"
  shift
  if command -v "$command_name" >/dev/null 2>&1; then
    "$command_name" "$@" 2>&1 || true
  else
    echo "not found: $command_name"
  fi
}

echo "== System =="
run_if_available uname -a
if [ -r /etc/os-release ]; then
  sed -n '1,12p' /etc/os-release
fi

echo
echo "== Jetson / L4T =="
if [ -r /etc/nv_tegra_release ]; then
  cat /etc/nv_tegra_release
else
  echo "not found: /etc/nv_tegra_release (this may not be a Jetson)"
fi
run_if_available nvpmodel -q

echo
echo "== CUDA / GPU =="
run_if_available nvcc --version
run_if_available nvidia-smi
run_if_available tegrastats --interval 1000 --count 1

echo
echo "== TensorRT =="
run_if_available trtexec --version
if command -v dpkg-query >/dev/null 2>&1; then
  dpkg-query -W 'libnvinfer*' 'tensorrt*' 2>/dev/null || true
fi

echo
echo "== Python packages =="
python3 --version 2>&1 || true
python3 - <<'PY' 2>/dev/null || true
for package in ("ultralytics", "onnx", "onnxruntime", "tensorrt"):
    try:
        module = __import__(package)
        print(f"{package}: {getattr(module, '__version__', 'installed')}")
    except Exception as exc:
        print(f"{package}: unavailable ({exc.__class__.__name__})")
PY

echo
echo "== ROS2 =="
echo "ROS_DISTRO=${ROS_DISTRO:-not set}"
run_if_available ros2 --help

echo
echo "== Camera devices =="
if compgen -G '/dev/video*' >/dev/null; then
  ls -l /dev/video*
else
  echo "no /dev/video* devices found"
fi

echo
echo "== Storage and memory =="
run_if_available df -h .
run_if_available free -h
