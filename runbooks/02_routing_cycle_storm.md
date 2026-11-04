# Operational Runbook 02: Directed Packet Routing Cycle & Forwarding Loop Storm
## Metadata
- **Runbook ID:** RB-GRID-002
- **Severity Level:** CRITICAL
- **Component Subsystem:** SDN Virtual Fabric / Overlay Control Plane
- **Target Failure Mode:** Mode C (Directed Routing Cycle Storm)
- **Primary Telemetry Signature:** `protocol_type in {"RPC_CONTROL", "CACHE_REPLICATION"}`, `NetworkX` directed cycle detected (`in_routing_cycle == True`, cycle length 3 or 4), abnormal repetition of source-target interactions.

## Symptom Signature
A routing loop or circular forwarding anomaly in the software-defined overlay network causes inter-node RPC messages or data replication packets to traverse a closed feedback cycle (e.g., $Node_A \to Node_B \to Node_C \to Node_A$). This creates an amplification loop, exhausting virtual routing switch tables, inflating link latency, and preventing valid control-plane directives from reaching worker daemons.

## Diagnostic Verification
1. **Topology Cycle Extraction:** Query the directed graph topology analyzer to inspect identified cycle node sequences (`cycle_path`).
2. **SDN Flow Table Audit:** Validate OpenFlow or BGP EVPN routing tables for conflicting forward-next-hop rules.
3. **Hop-Limit Inspection:** Check packet TTL / hop limits on inter-cluster gateway interfaces to confirm loop persistence.
4. **Overlay Encapsulation Check:** Inspect Geneve/VXLAN encapsulation headers for circular tunnel encapsulation.

## Remediation & Recovery Procedure
Execute targeted topology route severance on the primary ingress node:

```bash
# 1. Trace the circular forwarding loop across the overlay fabric
traceroute -m 10 -p 8443 <TARGET_NODE_IP>

# 2. Invalidate the stale OpenFlow route causing the looping directive
ovs-ofctl del-flows br-int "table=0,ip,nw_dst=<TARGET_SUBNET>"

# 3. Force re-computation of Dijkstra shortest-path spanning tree in SDN controller
curl -X POST http://sdn-controller.grid.internal:8080/v1/topology/recompute \
  -H "Content-Type: application/json" \
  -d '{"invalidate_subgraph": ["NODE_A", "NODE_B", "NODE_C"], "reason": "routing_cycle_storm"}'

# 4. Enforce strict hop limit decrements on virtual switch ports
ovs-vsctl set bridge br-int other-config:max-idle=30
```

## Recovery Criteria
- Graph topology analyzer reports zero simple cycles across the active sliding window.
- RPC control packet latency returns to $< 2.5\text{ ms}$.
- BGP routing reconvergence completed with single unique path per egress prefix.

## Prevention & Hardening
- **SLA Resolution Target:** 10 minutes.
- **Escalation Path:** If automated flow table flush fails to resolve the cycle within 5 minutes, sever the loop manually by administrative down on the non-primary transit link.
