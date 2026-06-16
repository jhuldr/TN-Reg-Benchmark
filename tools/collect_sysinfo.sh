#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/TN_Reg}"
OUTDIR="$ROOT/sysinfo"
mkdir -p "$OUTDIR"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="$OUTDIR/system_report_${TS}.txt"

default_if() { ip route | awk '/default/ {print $5; exit}'; }

{
  echo "===== SYSTEM REPORT @ $(date) ====="
  echo "Root: $ROOT"
  echo

  echo "## Host & OS"
  hostnamectl 2>/dev/null || true
  echo

  echo "## Kernel"
  uname -a
  echo

  echo "## CPU"
  lscpu
  echo

  echo "## Memory"
  free -h
  echo

  echo "## Disks / Filesystems"
  df -hT
  echo
  lsblk -o NAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINT
  echo
  nvme list 2>/dev/null || true
  echo

  echo "## GPU (NVIDIA)"
  nvidia-smi
  echo
  nvidia-smi topo -m 2>/dev/null || true
  echo

  echo "## CUDA / Driver"
  nvcc --version 2>/dev/null || echo "nvcc not found"
  cat /usr/local/cuda/version.txt 2>/dev/null || true
  echo

  echo "## Network"
  ip -br a
  echo
  ip route
  echo
  IFACE="$(default_if || true)"
  if [[ -n "${IFACE:-}" ]]; then
    echo "### Default interface: $IFACE"
    ethtool -i "$IFACE" 2>/dev/null || true
    ethtool "$IFACE" 2>/dev/null | head -n 40 || true
  fi
  echo

  echo "## PCIe summary"
  lspci | egrep -i "nvidia|ethernet|infiniband|nvme|raid|sata" || true
  echo

  echo "## Limits"
  ulimit -a
  echo

  echo "## Python / Conda / Libraries"
  which python || true
  python --version || true
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu_count', torch.cuda.device_count())" 2>/dev/null || true
  python -c "import numpy; print('numpy', numpy.__version__)" 2>/dev/null || true
  python -c "import nibabel, SimpleITK; print('nibabel', nibabel.__version__, 'SimpleITK', SimpleITK.Version_VersionString())" 2>/dev/null || true
  echo

  echo "## ANTs"
  which antsRegistration 2>/dev/null || true
  antsRegistration --version 2>/dev/null || true
  echo

  echo "===== END ====="
} | tee "$OUT"

echo
echo "Saved to: $OUT"
