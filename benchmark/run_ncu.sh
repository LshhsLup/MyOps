#!/bin/bash
# =============================================================================
# Nsight Compute Profiling Script for reduce_sum kernel
#
# This script profiles the reduce_sum kernel using ncu
# (NVIDIA Nsight Compute Command Line Profiler).
#
# Usage:
#   cd benchmark/
#   chmod +x run_ncu.sh
#   ./run_ncu.sh [metric_set]
#
# Examples:
#   ./run_ncu.sh basic      # Profile with basic metrics (throughput + memory)
#   ./run_ncu.sh full       # Profile with full section-based analysis (recommended)
#   ./run_ncu.sh warp       # Profile with warp stall analysis only
#   ./run_ncu.sh roofline   # Profile with roofline metrics
#   ./run_ncu.sh save       # Save full report to .ncu-rep file for GUI analysis
#
# NOTE: If you see "n/a" for smsp__* metrics, it's a permission issue.
#       Fix: sudo ./run_ncu.sh full
#       Or use "warp" mode which uses --section (works without sudo).
#
# You can also run from project root:
#   ./benchmark/run_ncu.sh basic
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_SCRIPT="${SCRIPT_DIR}/profile_reduce.py"

METRIC_SET=${1:-basic}

echo "=========================================="
echo "Profiling reduce_sum"
echo "Metric set: ${METRIC_SET}"
echo "=========================================="

# =============================================================================
# NCU (Nsight Compute) 参数详解
# =============================================================================
#
# --target-processes all
#     分析所有进程（包括子进程）。PyTorch 启动 CUDA 时会有子进程，
#     这个参数确保能抓到 kernel。
#
# --kernel-name "regex:reduceSumKernel"
#     只分析名称匹配正则表达式的 kernel。
#     必须加 "regex:" 前缀才能使用正则匹配。
#     例如 "regex:reduceSumKernel" 匹配 reduceSumKernel_V0/V1/V2。
#
# --kernel-id <id>
#     只分析指定 ID 的 kernel（ID 从 Available Kernels 列表中获取）。
#     注意：这个参数是按 kernel 名称在程序中的出现顺序编号，不是 launch 顺序。
#     通常不如 --launch-skip + --launch-count 灵活。
#
# --set full
#     使用预定义的完整指标集，包含所有性能指标。
#     其他可选值：basic, detailed, roofline
#
# --metrics <metric_list>
#     手动指定要收集的指标（逗号分隔）。
#     常用指标见下方说明。
#
# -o <output_file>
#     将 profiling 结果保存到 .ncu-rep 文件，可用 Nsight Compute GUI 打开。
#
# --launch-count 1
#     只分析前 1 次 kernel launch。
#
# --launch-skip 1
#     跳过前 1 次 kernel launch（用于跳过 warmup）。
#
# =============================================================================

# =============================================================================
# 常用性能指标说明
# =============================================================================
#
# 吞吐量指标 (Throughput):
#   sm__throughput.avg.pct_of_peak_sustained_elapsed
#       SM (Streaming Multiprocessor) 吞吐量，占峰值的百分比。
#       反映计算单元的利用率。
#
#   dram__throughput.avg.pct_of_peak_sustained_elapsed
#       DRAM (显存) 吞吐量，占峰值的百分比。
#       Reduce 是 memory-bound，这个指标很重要。
#
#   l1tex__throughput.avg.pct_of_peak_sustained_elapsed
#       L1 Cache 吞吐量。
#
#   lts__throughput.avg.pct_of_peak_sustained_elapsed
#       L2 Cache 吞吐量。
#
# 内存指标 (Memory):
#   dram__bytes.sum
#       总 DRAM 访问字节数（读 + 写）。
#
#   dram__bytes_read.sum
#       DRAM 读取字节数。
#
#   dram__bytes_write.sum
#       DRAM 写入字节数。
#
# Warp 调度状态 (Warp Scheduler Stalls):
#   这些指标告诉你为什么 warp 没有在执行，帮助定位瓶颈。
#
#   smsp__warps_issue_stalled_long_scoreboard_per_warp_active.pct
#       等待全局内存 (DRAM) 的百分比。Reduce kernel 通常这个很高。
#
#   smsp__warps_issue_stalled_short_scoreboard_per_warp_active.pct
#       等待共享内存 (Shared Memory) 的百分比。
#
#   smsp__warps_issue_stalled_membar_per_warp_active.pct
#       等待内存屏障 (__syncthreads) 的百分比。
#
#   smsp__warps_issue_stalled_barrier_per_warp_active.pct
#       等待同步屏障的百分比。
#
#   smsp__warps_issue_stalled_wait_per_warp_active.pct
#       等待固定延迟指令（如常量内存）的百分比。
#
#   smsp__warps_issue_stalled_mio_throttle_per_warp_active.pct
#       等待 MIO (Memory Input/Output) 的百分比。
#
#   smsp__warps_issue_stalled_no_instruction_per_warp_active.pct
#       没有可用指令的百分比（通常是 warp 调度问题）。
#
#   smsp__warps_issue_stalled_not_selected_per_warp_active.pct
#       没有被调度器选中的百分比。
#
#   smsp__warps_issue_stalled_math_pipe_throttle_per_warp_active.pct
#       等待数学流水线的百分比。
#
#   smsp__warps_issue_stalled_drain_per_warp_active.pct
#       等待指令完成的百分比。
#
#   smsp__warps_issue_stalled_sleeping_per_warp_active.pct
#       warp 处于睡眠状态的百分比。
#
# 占用率指标 (Occupancy):
#   sm__warps_active.avg.pct_of_peak_sustained_active
#       活跃 warp 占峰值的百分比。越高越好。
#
#   launch__occupancy.avg.pct_of_peak_sustained
#       理论占用率（基于 register/shared memory 使用量）。
#
# =============================================================================

# =============================================================================
# 为什么某些指标显示 n/a？
# =============================================================================
#
# smsp__* 系列指标（warp stall 原因）需要访问 SM 调度器的 PC sampling 硬件计数器。
# 在以下环境中可能无法采集：
#   - 容器中运行（缺少 /dev/nvidia* 权限）
#   - 非 root 用户
#   - 安全策略限制了 perfmon 访问
#
# 解决方案：
#   1. sudo ./run_ncu.sh 1 full     # 用 root 权限采集
#   2. ./run_ncu.sh 1 warp          # 用 --section 方式，自动跳过不可用指标
#   3. 在宿主机上运行（非容器）
#
# --section 方式更推荐：ncu 会自动选择当前环境可用的指标，并给出优化建议。
# =============================================================================

case ${METRIC_SET} in
  basic)
    # 基础指标：吞吐量 + 内存带宽
    echo "Collecting basic metrics: throughput + memory..."
    ncu \
      --target-processes all \
      --kernel-name "regex:reduceSumKernel" \
      --launch-skip 1 \
      --launch-count 1 \
      --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum,\
launch__occupancy.avg.pct_of_peak_sustained \
      python ${PROFILE_SCRIPT}
    ;;

  full)
    # 完整分析：使用 --set full 让 ncu 自动选择可用指标
    # 注意：smsp__* 指标可能需要 sudo 权限，否则会显示 n/a
    echo "Collecting full metrics (may need sudo for smsp__* counters)..."
    ncu \
      --target-processes all \
      --kernel-name "regex:reduceSumKernel" \
      --launch-skip 1 \
      --launch-count 1 \
      --set full \
      python ${PROFILE_SCRIPT}
    ;;

  warp)
    # Warp 状态分析：使用 --section 方式，不需要特殊权限
    # 会显示 warp stall 原因和优化建议
    echo "Collecting warp state statistics (no sudo needed)..."
    ncu \
      --target-processes all \
      --kernel-name "regex:reduceSumKernel" \
      --launch-skip 1 \
      --launch-count 1 \
      --section WarpStateStats \
      --section MemoryWorkloadAnalysis \
      python ${PROFILE_SCRIPT}
    ;;

  roofline)
    # Roofline 指标：用于绘制 roofline 模型
    echo "Collecting roofline metrics..."
    ncu \
      --target-processes all \
      --kernel-name "regex:reduceSumKernel" \
      --launch-skip 1 \
      --launch-count 1 \
      --set roofline \
      python ${PROFILE_SCRIPT}
    ;;

  save)
    # 保存报告到 .ncu-rep 文件，可用 GUI 打开
    OUTPUT_FILE="reduce.ncu-rep"
    echo "Saving full report to ${OUTPUT_FILE}..."
    ncu \
      --target-processes all \
      --kernel-name "regex:reduceSumKernel" \
      --launch-skip 1 \
      --launch-count 1 \
      --set full \
      -o "reduce" \
      python ${PROFILE_SCRIPT}
    echo "Report saved to ${OUTPUT_FILE}"
    echo "Open it with: ncu-ui ${OUTPUT_FILE}"
    ;;

  *)
    echo "Unknown metric set: ${METRIC_SET}"
    echo "Available: basic, full, warp, roofline, save"
    exit 1
    ;;
esac

echo ""
echo "=========================================="
echo "Profiling complete!"
echo "=========================================="
