# Operational Runbook 05: Compute Worker CPU Saturation & Thermal Throttling Stall
## Metadata
- **Runbook ID:** RB-GRID-005
- **Severity Level:** MEDIUM / HIGH
- **Component Subsystem:** Batch Worker Node / Cluster Compute Slot
- **Target Failure Mode:** Mode B (Temporal Oscillation & CPU Thermal Stall)
- **Primary Telemetry Signature:** Persistent elevated `src_cpu_load_curr > 95.0%`, continuous multi-step load degradation, high temporal sequence reconstruction error in `LSTM Autoencoder`, low queue depth progression.

## Symptom Signature
Compute worker nodes executing heavy simulation or matrix reconstruction routines can suffer from kernel thrashing, memory lock contention, or CPU frequency throttling under thermal stress. The LSTM Autoencoder detects this pattern through persistent deviations from nominal periodic cyclic behavior across consecutive window steps. Unchecked CPU stalls lead to job heart-beat dropouts, slot evacuation, and compute resource waste.

## Diagnostic Verification
1. **Temporal Sequence Inspection:** Examine the LSTM reconstruction error trajectory over the past 6 sliding time steps.
2. **Core Frequency and Temperature Audit:** Check hardware sensor metrics (MSR counters, `turbostat`, and package temperatures).
3. **Run-Queue Length:** Validate if thread run-queue depth exceeds physical hyperthread allocation ($> 2\times$ core count).
4. **I/O Wait Contention:** Ensure the CPU load is not inflated by uninterrupted D-state kernel waits on NFS/Ceph mounts.

## Remediation & Recovery Procedure
Execute process isolation, cgroup capping, and graceful slot de-allocation:

```bash
# 1. Query top CPU-consuming jobs and check cgroup limits
top -b -n 1 -o %CPU | head -n 15

# 2. Check CPU frequency governor and thermal throttling alerts
dmesg -T | grep -i -E "thermal|throttling|mcheck"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq

# 3. Soft-throttle rogue worker process cgroups to prevent node crash
cgset -r cpu.cfs_quota_us=400000 /worker_job_cgroup

# 4. Gracefully drain current compute slots and mark node unschedulable
drain_worker_node --graceful <WORKER_NODE_NAME>

# 5. Reset cooling policy and fan speed profile via IPMI if thermal alarm tripped
ipmitool sdr type "Temperature"
```

## Recovery Criteria
- Source CPU load decreases to $< 80\%$.
- LSTM sequence reconstruction error drops below the 90th percentile baseline.
- CPU operating frequency returns to nominal base clock without thermal throttling flags.

## Prevention & Hardening
- **SLA Resolution Target:** 30 minutes.
- **Escalation Path:** If thermal alarm persists after draining jobs, open a hardware ticket with Data Center Facilities for heatsink/chassis fan inspection.
