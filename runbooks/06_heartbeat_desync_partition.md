# Operational Runbook 06: Control-Plane Heartbeat Desynchronization & Partition
## Metadata
- **Runbook ID:** RB-GRID-006
- **Severity Level:** CRITICAL
- **Component Subsystem:** Grid Information Service / Consensus Coordinator
- **Target Failure Mode:** Mode B / Mode C (Control-Plane Desynchronization)
- **Primary Telemetry Signature:** `protocol_type == "HEARTBEAT_SYNC"`, minimal `payload_mb`, irregular temporal jitter, sporadic queue spikes on target consensus coordinator, elevated SHAP importance for protocol and window step features.

## Symptom Signature
Cluster consensus state engines (e.g. Raft/etcd/ZooKeeper clusters managing resource catalogs) rely on deterministic periodic heartbeat exchanges. When network jitter, clock drift, or garbage collection pauses delay `HEARTBEAT_SYNC` messages, follower nodes trigger leader election campaigns. Successive desynchronization waves cause split-brain risk, lock acquisition failures, and widespread cluster stall.

## Diagnostic Verification
1. **Heartbeat Inter-Arrival Variance:** Check variance in successive `window_step` arrival timestamps for heartbeat protocol events.
2. **NTP/PTP Precision Check:** Verify local chrony or NTP time offsets against atomic stratum-1 reference clocks.
3. **Consensus Peer Latency:** Test round-trip latency to all quorum members to detect asymmetric link drops.
4. **JVM/Runtimes GC Stalls:** Check coordinator process pause times (Stop-The-World GC events).

## Remediation & Recovery Procedure
Execute clock resynchronization and consensus quorum recovery:

```bash
# 1. Inspect local clock offset against reference grid stratum servers
chronyc sources -v
chronyc tracking

# 2. Check consensus leader health and quorum membership status
etcdctl endpoint health --cluster -w table

# 3. Increase heartbeat election timeout dynamically to absorb transient jitter
etcdctl member update <MEMBER_ID> --peer-urls="http://<NODE_IP>:2380"
# (Alternatively update heartbeat interval in coordinator config)

# 4. Flush transient packet drop buffer on management network interface
ethtool -S eth1 | grep -i -E "drop|miss|err"

# 5. Force step-sync NTP if clock offset exceeds 50 milliseconds
systemctl stop chrony && chronyd -q 'server pool.ntp.org iburst' && systemctl start chrony
```

## Recovery Criteria
- Clock offset remains within $\pm 1.0\text{ ms}$.
- Consensus cluster reports quorum healthy with zero uncommitted log entries.
- Heartbeat protocol events exhibit steady periodic intervals with nominal queue depth ($< 5$).

## Prevention & Hardening
- **SLA Resolution Target:** 10 minutes.
- **Escalation Path:** If quorum is permanently lost, invoke disaster recovery protocol for coordinator state store snapshot restoration.
