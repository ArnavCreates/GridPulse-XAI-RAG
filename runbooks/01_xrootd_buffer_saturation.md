# Operational Runbook 01: XRootD Storage Buffer Saturation & Ingress Overflow
## Metadata
- **Runbook ID:** RB-GRID-001
- **Severity Level:** CRITICAL / HIGH
- **Component Subsystem:** Storage Element (SE) / XRootD Data Gateway
- **Target Failure Mode:** Mode A (Queue Saturation & Buffer Overflow)
- **Primary Telemetry Signature:** `protocol_type == "XROOTD_STREAM"`, elevated `payload_mb`, sudden spike in `dst_queue_depth_curr - dst_queue_depth_prev > 40.0`, high `src_cpu_load_curr > 85.0%`.

## Symptom Signature
During bulk data replication across primary and secondary storage clusters, incoming high-throughput streams can overwhelm the XRootD staging chunk cache. When client socket write buffers fill faster than the destination block storage driver flushes dirty blocks to disk arrays, the pending I/O queue depth surges exponentially. Unmitigated queue exhaustion results in TCP zero-window deadlocks, stream timeouts, and cascading transfer retries that magnify site-wide transfer degradation.

## Diagnostic Verification
1. **Queue Gradient Analysis:** Check if `dst_queue_depth_curr - dst_queue_depth_prev` exceeds threshold within a single telemetry window step.
2. **Transfer Rate Verification:** Confirm incoming payload size (`payload_mb > 150 MB`) is stressing the storage node's NVMe write pool.
3. **Storage Daemon Socket Inspection:** Run socket diagnostic utilities on the storage node to inspect TCP retransmission counters and local buffer ring allocation.
4. **Bandwidth Throttling State:** Assess whether concurrent file copy streams exceed the storage node's physical backplane limit.

## Remediation & Recovery Procedure
Execute the following procedures on the affected storage element node:

```bash
# 1. Inspect current active XRootD transfer streams and queued file descriptors
xrdfs localhost:1094 query stats a | grep -E "oss.space|ofs.tpc"

# 2. Dynamically throttle concurrent connection limit to prevent buffer blow-out
xrdctl localhost:1094 config set pss.concurrency 64

# 3. Flush transient dirty memory buffers to NVMe storage array
sync && echo 3 > /proc/sys/vm/drop_caches

# 4. Temporarily rate-limit ingress streams at the traffic control layer
tc qdisc change dev eth0 root tbf rate 8gbit burst 64mb latency 50ms
```

## Recovery Criteria
- Destination queue depth (`dst_queue_depth_curr`) declines below 15 pending operations.
- Source CPU utilization drops into the nominal operational envelope (30% - 65%).
- Zero TCP socket drop errors reported across subsequent telemetry steps.

## Prevention & Hardening
- **SLA Resolution Target:** 15 minutes.
- **Escalation Path:** If queue depth remains above 80 after throttling, escalate to the On-Call Storage Operations Team and route new stream allocations to backup storage cluster pool.
