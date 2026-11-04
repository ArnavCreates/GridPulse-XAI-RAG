# Operational Runbook 04: Fan-Out Replication Flood & Runaway Cache Multicast
## Metadata
- **Runbook ID:** RB-GRID-004
- **Severity Level:** HIGH
- **Component Subsystem:** Cache Replication Service / Data Dispatcher
- **Target Failure Mode:** Mode D (Fan-Out Flood / Runaway Replication)
- **Primary Telemetry Signature:** `protocol_type == "CACHE_REPLICATION"`, high source out-degree (`source_fan_out_degree > 12`), sudden spike in aggregate bandwidth utilization, elevated source node CPU load.

## Symptom Signature
A cache replication coordinator or dataset distributor initiates a runaway fan-out broadcast to a large set of worker cache nodes simultaneously. This can occur when a new dataset partition is marked for aggressive pre-fetching or when a replication failure causes repetitive global re-broadcasts. The source node's NIC is pinned at 100% saturation, dropping outbound packets, starving local control threads, and triggering false node-offline alerts.

## Diagnostic Verification
1. **Out-Degree Anomaly Detection:** Check if `source_fan_out_degree` exceeds the safe fan-out threshold.
2. **Replication Queue Audit:** Inspect the cache manager queue for duplicate replication tasks targeting identical dataset hashes.
3. **Bandwidth Saturation Rate:** Correlate `payload_mb * source_fan_out_degree` against interface physical wire speed.
4. **Cache Policy Evaluation:** Determine if an aggressive 'replicate-to-all' prewarm policy was erroneously deployed.

## Remediation & Recovery Procedure
Execute broadcast throttling and replication ring shedding:

```bash
# 1. Identify active fan-out replication child processes
ps aux | grep -E "grid-cache-replicator|xrdcp" | head -n 20

# 2. Pause active cache replication queue on the flooding node
systemctl stop grid-cache-replicator.service

# 3. Limit replication worker concurrency to a maximum of 4 concurrent targets
sed -i 's/MAX_FANOUT_PEERS=.*/MAX_FANOUT_PEERS=4/' /etc/grid/cache-replicator.conf

# 4. Clear pending uncommitted replication requests from local Redis/etcd queue
redis-cli -h cache-broker.grid.internal LTRIM "replication_queue:broadcast" 0 10

# 5. Restart replicator service with tree-based hierarchical distribution
systemctl start grid-cache-replicator.service
```

## Recovery Criteria
- Source node out-degree drops to $\le 6$ concurrent egress destinations.
- Source CPU load decreases to $< 60\%$.
- Cache replication latency adheres to standard distribution bounds without socket buffer errors.

## Prevention & Hardening
- **SLA Resolution Target:** 15 minutes.
- **Escalation Path:** If network link congestion impacts inter-site wide-area links (inter-cluster WAN / core interconnects), contact Network Operations Center (NOC) immediately.
