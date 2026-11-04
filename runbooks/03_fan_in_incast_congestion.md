# Operational Runbook 03: Fan-In Incast Congestion & Storage Pool Bottleneck
## Metadata
- **Runbook ID:** RB-GRID-003
- **Severity Level:** HIGH
- **Component Subsystem:** Central Storage Aggregator / File Catalog Pool
- **Target Failure Mode:** Mode D (Fan-In Incast Saturation)
- **Primary Telemetry Signature:** High target node in-degree (`target_fan_in_degree > 10`), concentrated traffic towards a single destination node (`SE_POOL_*`), high PageRank centrality, destination queue depth escalation.

## Symptom Signature
When dozens of distributed compute worker nodes finish data reduction phases simultaneously, they initiate concurrent barrier-synchronized transfers to a common destination storage pool or metadata catalog aggregator. This incast scenario saturates the receiver's top-of-rack switch buffer, creating microburst packet drops, TCP throughput collapse (due to synchronization timeout), and elevated end-to-end task tail latency.

## Diagnostic Verification
1. **In-Degree Centrality Evaluation:** Identify destination nodes where `target_fan_in_degree` surges $> 3\times$ above the cluster median.
2. **Switch Queue Telemetry:** Verify Explicit Congestion Notification (ECN) marking rate on the top-of-rack switch uplink.
3. **Client Stream Scheduling:** Check workload orchestrator batch job schedules to detect unjittered batch task completion.
4. **Queue Growth Gradient:** Monitor destination node I/O scheduler queue backpressure metrics.

## Remediation & Recovery Procedure
Execute load rebalancing and randomized backoff at the aggregator and client endpoints:

```bash
# 1. Inspect top incoming connections to the congested storage pool
netstat -ant | grep :1094 | awk '{print $5}' | cut -d: -f1 | sort | uniq -c | sort -nr | head -n 15

# 2. Enable Data Center TCP (DCTCP) and explicit congestion notification (ECN)
sysctl -w net.ipv4.tcp_congestion_control=dctcp
sysctl -w net.ipv4.tcp_ecn=1

# 3. Trigger orchestrator dynamic backoff for concurrent writing jobs
curl -X POST http://orchestrator.grid.internal:9090/v1/jobs/apply-jitter \
  -H "Content-Type: application/json" \
  -d '{"target_pool": "SE_POOL_007", "jitter_window_sec": 45, "max_concurrent_streams": 6}'

# 4. Activate secondary overflow storage volume in storage router
xrootd-admin route-table divert --from "SE_POOL_CONGESTED" --to "SE_POOL_OVERFLOW" --percentage 50
```

## Recovery Criteria
- Destination node in-degree returns below the nominal fan-in threshold ($\le 8$).
- Destination queue depth stabilizes below 20 operations.
- ECN marking percentage drops below 2% of total packets.

## Prevention & Hardening
- **SLA Resolution Target:** 20 minutes.
- **Escalation Path:** Alert Grid Storage Infrastructure Lead if incast buffer drops trigger batch job timeouts across more than 3 compute clusters.
