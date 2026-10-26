---
title: "AI-Factory Best Practices: The Order of Operations for NVAIE on Bare-Metal OpenShift"
date: 2026-10-26
tags: AI, NVAIE, OpenShift, GPU, RDMA, NCCL, Rail-Optimized, Multus
excerpt: Standing up a GPU training cluster isn't a checklist you can run in any order. Rail-optimized fabric design, two operators that both touch the same kernel modules, secondary CNIs so pods can see RDMA NICs, and a validation stage that has to prove the fabric before anyone trains on it — sequence is the whole story. Here's the build order, end to end.
image: assets/images/blog/covers/2026-06-21-ai-factory-best-practices-nvaie-openshift.jpg
---

# AI-Factory Best Practices: The Order of Operations for NVAIE on Bare-Metal OpenShift

Most NVIDIA AI Enterprise (NVAIE) deployment guides read like a parts catalog: here's the Network Operator, here's the GPU Operator, here's how to run an NCCL test. What they don't tell you is that **two of those operators write to the same kernel modules on the same nodes**, and if you install them in the wrong order — or without the one label that gates them — you get an intermittent driver-load race that looks like flaky hardware and isn't.

This is the build sequence I use, written up as a public reference: [**maheshpabba/AI-Factory-Best-Practices**](https://github.com/maheshpabba/AI-Factory-Best-Practices). Every section below links to the actual manifests in that repo. The sequence matters more than any individual YAML file, so that's the spine of this post.

---

## Why Sequence Is the Whole Problem

A GPU worker node in this design ends up running, at minimum: the OpenShift SDN, the NMState Operator, a custom `MachineConfigPool` with blacklisted kernel modules, the Node Feature Discovery operator, the NVIDIA Maintenance Operator, the NVIDIA Network Operator (which loads the MOFED/DOCA-OFED driver), and the NVIDIA GPU Operator (which loads the NVIDIA driver). Every one of those has a dependency on the step before it:

1. **Network planes and node identity have to exist before the cluster does.** The Assisted Installer binds static IPs to MAC addresses, not hostnames — get that table wrong and you silently hand the wrong IP to the wrong physical box.
2. **GPU nodes have to be labeled and isolated into their own `MachineConfigPool` before any kernel-level config lands**, or every node in the cluster reboots for a change that only GPU workers need.
3. **Node Feature Discovery has to run before the Network Operator and GPU Operator do anything**, because both of them schedule based on labels NFD writes (`feature.node.kubernetes.io/pci-15b3.present=true` and friends). Skip it and their pods either don't schedule, or schedule onto nodes without the hardware they need.
4. **The Network Operator's NIC driver has to be ready before the GPU Operator's driver loads** — not "roughly around the same time," but gated on an actual Kubernetes label. This is the one most guides skip, and it's the one that actually breaks clusters.
5. **The fabric has to be validated with a real multi-node, multi-GPU workload before anyone trusts it for training** — a cluster that *installs* cleanly and a cluster that's *ready for distributed training* are different claims, and the gap between them is exactly where RDMA, QoS, and NCCL tuning live.

Everything below follows that order.

---

## Step 1: Rail-Optimized Topology — Design the Fabric Before You Rack Anything

Before any of the Kubernetes-level configuration below matters, the physical network has to be designed correctly, and for multi-node GPU training that means a **rail-optimized topology**.

The idea: each GPU in a node has its own dedicated high-speed NIC (a ConnectX adapter, one per GPU, not shared). In a rail-optimized fabric, **GPU *N* on every node connects to the same leaf switch** — GPU 0 across all nodes lands on "rail 0," GPU 1 across all nodes lands on "rail 1," and so on. NCCL's collective algorithms (ring and tree reductions) are rail-aware: when GPU 0 on node A needs to talk to GPU 0 on node B, that traffic stays on a single leaf switch instead of crossing a spine. The result is lower latency and higher effective bandwidth for the exact communication pattern distributed training actually produces, instead of a topology that's merely "fast" for generic traffic.

### Before the Rail: GPUs Already Talk to Each Other Inside the Node

Rail-optimization is about traffic that *leaves* a node. Before NCCL ever touches the RDMA fabric, it has a much faster option available for GPUs sitting in the same server: NVSwitch.

On an 8-GPU platform like NVIDIA's HGX B300, every GPU connects to **two NVSwitch chips** over fifth-generation NVLink — a non-blocking, all-to-all crossbar inside the node, rated at 1,800 GB/s GPU-to-GPU and roughly 14.4 TB/s aggregate across the baseboard. That's an order of magnitude more bandwidth than any inter-node RDMA link, which is exactly why NCCL treats the two domains so differently: collectives are built to do as much of the reduction as possible over NVLink/NVSwitch *inside* a node first, and only hand off across the rail-optimized fabric for the inter-node portion of the same collective. NCCL discovers this topology automatically (via NVML and the Fabric Manager topology file), so a ring or tree algorithm effectively has two tiers baked in — a fast intra-node hop over NVSwitch, then a slower inter-node hop over RDMA — without anything in the workload needing to know the difference.

NVSwitch isn't just a crossbar, either — it can do math. NVLink SHARP (NVLS) lets the switch itself perform the reduction (the sum in an AllReduce, for example) on data as it streams through, instead of every GPU doing the full reduction in its own memory. NCCL exposes this as the NVLS transport, controlled by `NCCL_NVLS_ENABLE` — defaulting to `2` ("auto") in current NCCL versions, meaning it enables itself automatically whenever the hardware and topology support it, with no action needed on your part.

Worth being honest about what that bought in practice on this fabric: across repeated AllReduce runs, bus bandwidth consistently landed around **778 GB/s whether NVLS was enabled or explicitly disabled** (`NCCL_NVLS_ENABLE=0`, visible in the hand-written MPIJob manifests in Step 6). The reduction offload is real, but on this topology and message-size range it didn't show up as a measurable win in aggregate bandwidth — the run was bound by something else in the path, not by GPU-side reduction math. That's not a reason to disable NVLS; it's a reason to actually measure it on your own fabric rather than assume the feature is moving your BusBW number just because it's enabled.

This is also *why* the node-level NIC inventory in the next step matters so much: if you don't know precisely which physical NIC maps to which GPU and which rail, you can't validate that the physical cabling matches the intended topology, and a single miscabled rail shows up later as an unexplained bandwidth cliff on one specific NCCL collective — exactly the kind of problem that's expensive to diagnose after the cluster is in production and cheap to catch with an inventory pass before it is.

The practical first step, before touching OpenShift at all, is inventorying every node's RDMA-capable NICs — PCIe address, interface name, and adapter family (ConnectX-7 vs. ConnectX-8 vs. a BlueField DPU, which is easy to get wrong because a BlueField's adapter string often still contains "ConnectX-7"). I got tired of doing that inventory by hand with `oc debug`/`lspci`/`ethtool` on every node, so I created [`Scripts/cluster-utility.sh`](https://github.com/maheshpabba/AI-Factory-Best-Practices/blob/Dev/Scripts/cluster-utility.sh) (and its PowerShell twin) to automate it — it's turned out to be far more handy than doing this manually: discover nodes by name or role, print a full RDMA device table filtered by adapter family, and generate the NMState manifests from what it actually finds on the hardware rather than from a spreadsheet that can drift out of sync with reality.

---

## Step 2: Three Network Planes, Declared Before the Cluster Exists

Every GPU worker node in this design carries three logically separate network planes:

- **Frontend / primary CNI** — the OpenShift SDN plane: node management, the Kubernetes API, pod-to-pod traffic that doesn't need RDMA.
- **Compute (scale-out) fabric** — the rail-optimized RDMA fabric from Step 1, carrying NCCL collectives (AllReduce, AllGather, AllToAll, ReduceScatter) between nodes.
- **Storage / backend fabric** — kept on its own subnet and its own NICs, isolated from the compute fabric so storage I/O never competes with GPU collective traffic for bandwidth.

Before the discovery ISO ever boots, build a single inventory table — one row per host, with columns for hostname, MAC address, static IP/subnet/gateway for each of the three planes above, plus the cluster-wide DNS servers, NTP servers, and API/Ingress VIPs. That table is what gets converted to NMState-formatted YAML and pasted into the Assisted Installer's static network configuration step, so every host comes up with the right plane wired correctly on first boot — no per-host manual network setup in the UI.

---

## Step 3: Isolate GPU Nodes Before Any Kernel Change Lands

Once the cluster is up, two things happen exactly once, before any operator touches the nodes:

**Label the GPU workers into their own role.** This is what lets every GPU-specific `MachineConfig` target only those nodes:

```bash
oc label node worker-0 node-role.kubernetes.io/gpuworker=
oc label node worker-1 node-role.kubernetes.io/gpuworker=
```

**Create a dedicated `MachineConfigPool`** scoped to that role, then apply a `MachineConfig` that blacklists the inbox `irdma`/`rpcrdma` kernel modules (they conflict with the NVIDIA/Mellanox MOFED stack), disables IOMMU, and configures huge pages:

```yaml
apiVersion: machineconfiguration.openshift.io/v1
kind: MachineConfigPool
metadata:
  name: gpu-worker
  labels:
    worker.type: gpu-worker
spec:
  machineConfigSelector:
    matchExpressions:
      - key: machineconfiguration.openshift.io/role
        operator: In
        values: [worker, gpu-worker]
  nodeSelector:
    matchLabels:
      node-role.kubernetes.io/gpu-worker: ""
```

### Why the Label and the Pool, Not Just a `MachineConfig`

This is worth spelling out, because the payoff isn't just "fewer reboots" — it's specifically about how OpenShift's Machine Config Operator (MCO) rolls out changes.

Every node in a given `MachineConfigPool` shares one rendered configuration, and the MCO updates that pool as its own independent rolling sequence: cordon a node, drain it, apply the new config, reboot, uncordon, move to the next node — one at a time (governed by `maxUnavailable`, which defaults to 1 per pool). Without the `gpu-worker` label and pool, these nodes are just ordinary members of the default `worker` pool, which means two things go wrong at once:

- **Every change to the default `worker` pool reboots GPU nodes too** — a CRI-O config tweak, a kernel argument for an unrelated team's use case, a routine OS update — none of it is GPU-specific, but all of it would still cordon and reboot your GPU capacity along with the rest of the fleet.
- **An OpenShift cluster upgrade itself rolls node-by-node *within* a pool.** If GPU nodes are mixed into the general `worker` pool, they're interleaved into that same rolling sequence with every other compute node, with no way to reason about GPU availability independently of the rest of the cluster.

Splitting GPU nodes into their own pool fixes both: GPU-specific `MachineConfig` objects (the kernel module blacklist above, driver prep, etc.) only ever target the `gpu-worker` pool, so non-GPU workers never reboot for them — and, just as importantly, **the GPU pool rolls through its own nodes one at a time**, independent of the rest of the cluster's upgrade sequence. For a multi-node training job spread across several GPU workers, that's the difference between losing one node's worth of GPU capacity at a time during an upgrade and having the scheduler's GPU nodes reboot in an order you don't control relative to unrelated cluster changes. See [`Misc/custom-role-gpuworker.yaml`](https://github.com/maheshpabba/AI-Factory-Best-Practices/blob/Dev/Misc/custom-role-gpuworker.yaml) for the full pool definition.

---

## Step 4: The NVAIE Operator Stack — Install in This Order, Not Alphabetical Order

Four operators, and the order they're installed in is not interchangeable:

1. **Node Feature Discovery** — inspects every node's PCI devices, CPU features, and kernel/OS version and writes the labels everything downstream depends on. No configuration needed; just verify `feature.node.kubernetes.io/pci-15b3.present=true` shows up on the nodes with Mellanox NICs before moving on.
2. **NVIDIA Maintenance Operator** — provides the cordon/drain coordination the Network Operator needs for firmware updates and driver reloads, so a maintenance action drains a node before disrupting it instead of after.
3. **NVIDIA Network Operator** — provisions the MOFED/DOCA-OFED driver, the RDMA device plugin, and NIC QoS on the ConnectX/BlueField NICs inventoried in Step 1.
4. **NVIDIA GPU Operator** — provisions the GPU driver, container toolkit, device plugin, and DCGM monitoring.

Steps 3 and 4 are where the actual conflict lives, because **both operators load kernel modules on the same GPU nodes**, and without coordination they can do it at the same time.

### The Fix: One Label as a Gate

The Network Operator's `NicNodePolicy` writes a node label once its MOFED/DOCA-OFED driver is ready: `network.nvidia.com/operator.mofed.wait=false`. The GPU Operator's driver is configured to *wait for that label* before it ever loads:

```yaml
# ClusterPolicy: configures every GPU Operator component EXCEPT the driver
apiVersion: nvidia.com/v1
kind: ClusterPolicy
metadata:
  name: gpu-cluster-policy
spec:
  driver:
    enabled: false          # <-- deliberately not managed here
    useNvidiaDriverCRD: true
  toolkit:
    enabled: true
  devicePlugin:
    enabled: true
  dcgm:
    enabled: true
  gfd:
    enabled: true
```

```yaml
# NVIDIADriver: a separate, per-node-group object that supplies the driver —
# gated on the Network Operator's own readiness label
apiVersion: nvidia.com/v1alpha1
kind: NVIDIADriver
metadata:
  name: gpu-driver-workers
spec:
  default: false
  nodeSelector:
    network.nvidia.com/operator.mofed.wait: "false"
```

Read that `nodeSelector` literally: the GPU driver pod is not created at all until the Network Operator has finished its own kernel module work on that node and flipped the gate. Two operators, one shared resource, one explicit ordering condition instead of a hope that timing works out. The full manifests are in [`GPU-Operator/gpu-cluster-policy.yaml`](https://github.com/maheshpabba/AI-Factory-Best-Practices/blob/Dev/GPU-Operator/gpu-cluster-policy.yaml) and [`GPU-Operator/gpu-driver.yaml`](https://github.com/maheshpabba/AI-Factory-Best-Practices/blob/Dev/GPU-Operator/gpu-driver.yaml).

On the Network Operator side, the split that matters is `NicClusterPolicy` (singleton, cluster-wide — DOCA telemetry, the NIC Configuration Operator) versus `NicNodePolicy` (per-node — the DOCA-OFED driver itself, the RDMA shared device plugin). Defining the same per-node component in both is rejected outright by the admission webhook, not silently merged, which is a useful forcing function: get the split wrong and you find out immediately instead of debugging a stale config weeks later. See [`network-Operator/nicclusterpolicy.yaml`](https://github.com/maheshpabba/AI-Factory-Best-Practices/blob/Dev/network-Operator/nicclusterpolicy.yaml) and [`network-Operator/nicNodePolicy.yaml`](https://github.com/maheshpabba/AI-Factory-Best-Practices/blob/Dev/network-Operator/nicNodePolicy.yaml).

One more field on `NicNodePolicy` that's easy to leave out and expensive to miss: `tolerations` for `node.kubernetes.io/not-ready` and `node.kubernetes.io/unreachable`.

```yaml
  tolerations:
    - effect: NoSchedule
      key: node.kubernetes.io/not-ready
      operator: Exists
    - effect: NoSchedule
      key: node.kubernetes.io/unreachable
      operator: Exists
```

Without them, a MOFED driver pod that's still coming up when the node briefly reports `NotReady` (which happens routinely during the driver load itself) gets evicted by the default node-problem taints before it's finished initializing. The Network Operator's controller retries, the pod gets scheduled again, starts initializing again, and — if the node is still settling — gets evicted again. From the outside this looks like the MOFED pod endlessly restarting rather than converging, and it's easy to misdiagnose as a driver or hardware problem instead of what it actually is: the pod never being given long enough to finish starting on a node that OpenShift hasn't yet marked fully `Ready`.

### QoS: Make the RDMA Plane Actually Lossless

RoCEv2 only behaves as a lossless fabric if Priority Flow Control and ECN are actually configured on the NICs that carry it — left at defaults, this is the single biggest cause of inconsistent NCCL performance and packet loss under load. The Network Operator's `NicConfigurationTemplate` applies that tuning per NIC, matching the switch-side QoS classes it needs to align with. One detail worth getting right: the textbook ToS value for DSCP 24 is `96` (`24 x 4`), but that assumes the two ECN-capable bits are already being set elsewhere in the stack. If your network stack doesn't set them by default, the ToS value needs the +8 that accounts for those bits — which is why the example below uses `104`, not `96`.

```yaml
apiVersion: configuration.net.nvidia.com/v1alpha1
kind: NicConfigurationTemplate
metadata:
  name: compute-fabric-qos
spec:
  nicSelector:
    pciAddresses: ["0000:17:00.0", "0000:17:00.1"]   # example addresses from your inventory
  template:
    rawNvConfig:
      - name: CNP_DSCP_P1       # DSCP marking the NIC uses for outgoing RoCEv2 Congestion
        value: "48"             # Notification Packets (DCQCN) — priority 1
      - name: CNP_DSCP_P2
        value: "48"             # same marking, priority 2
    roceOptimized:
      enabled: true
      qos:
        trust: dscp
        pfc: "0,0,0,1,0,0,0,0"   # PFC enabled on the GPU-direct traffic class
        tos: 104                # 96 (DSCP 24 x 4) is the textbook value; add +8 here if your
                                 # network stack doesn't set the ECN-capable bits by default —
                                 # that's the gap between 96 and 104 above.
```

The `rawNvConfig` block above matters for the same congestion-control loop the ToS tuning is part of: RoCEv2's DCQCN scheme relies on Congestion Notification Packets (CNPs) being recognizable as their own traffic class, not lumped in with ordinary RDMA data — `CNP_DSCP_P1`/`P2` is what tells the NIC firmware which DSCP value to mark those CNPs with on priorities 1 and 2, so the switch (and the receiving NIC) can classify and prioritize them correctly instead of treating congestion feedback as just more best-effort traffic.

---

## Step 5: Secondary CNIs — Giving Pods a Direct Line to the RDMA NICs

Here's the piece that's easy to miss: by default, a pod in OpenShift only sees the primary CNI (OVN-Kubernetes) interface. The compute and storage fabrics from Step 2 are real NICs on the host, but a pod has no way to reach them without a **secondary CNI**.

The Network Operator deploys [Multus](https://github.com/k8snetworkplumbingwg/multus-cni) for exactly this — it lets a pod request *additional* network interfaces beyond its primary one, each defined as a `NetworkAttachmentDefinition` (NAD). For RDMA NICs, a `macvlan`-type NAD creates a pod-visible sub-interface tied to a specific backend or storage NIC:

```yaml
apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: compute-fabric-nic1
  namespace: default
spec:
  config: '{
    "cniVersion": "0.3.1",
    "type": "macvlan",
    "master": "ens1f1np1",
    "mode": "bridge",
    "ipam": { "type": "whereabouts", "range": "10.132.0.0/24" }
  }'
```

A pod attaches to one or more of these via an annotation, and gets both its normal OVN-Kubernetes interface *and* a direct macvlan sub-interface onto the physical RDMA NIC:

```yaml
metadata:
  annotations:
    k8s.v1.cni.cncf.io/networks: '[{"name": "compute-fabric-nic1"}]'
```

The actual device access is brokered by the **RDMA shared device plugin** that the Network Operator's `NicNodePolicy` also deploys — it exposes the backend NICs as a schedulable Kubernetes resource (`rdma/rdmashared`), so a pod requests it in `resources.limits` the same way it requests `nvidia.com/gpu`. But that request only answers *which* RDMA devices the pod is authorized to use on that node — it's a permission/quota check, not a network path. The Multus NAD annotation is the other half: it's what actually attaches the secondary NICs to the pod's network namespace, so the job — and NCCL inside it — has a real, addressable interface to reach those authorized devices through. Put together: GPU resource request from the GPU Operator's device plugin, RDMA resource request (`rdma/rdmashared`) from the Network Operator's device plugin for *authorization*, and a macvlan NAD attachment for the actual *network path* — three separate pieces that all have to line up before a pod can run GPUDirect RDMA traffic at all. Skip the NAD and the pod is authorized to use an RDMA device but has no interface to reach it through; skip the device-plugin resource request and the opposite happens — a wired interface with no authorization behind it.

---

## Step 6: Prove the Fabric With a Real Multi-Node, Multi-GPU Job

A cluster that installs cleanly and a cluster that's ready for distributed training are different claims, and the only way to close that gap is running an actual multi-GPU, multi-node collective — not a single-node smoke test. This cluster validates two ways, deliberately kept independent so one can be a sanity check on the other:

**Hand-written Kubeflow `MPIJob` manifests** — full control over NCCL environment variables and pod-level privileges, run across every GPU on every node in the job (for example, 2 nodes × 8 GPUs = 16 GPUs total), exercising AllGather, AllReduce, AllToAll, and ReduceScatter, plus a short multi-node pretraining smoke test on a real model so an end-to-end training job is confirmed working, not just a synthetic collective. The manifests themselves are in [`nccl-training-tests/`](https://github.com/maheshpabba/AI-Factory-Best-Practices/tree/Dev/nccl-training-tests) — one YAML per collective (`nccl-allgather.yaml`, `nccl-allreduce.yaml`, `nccl-alltoall.yaml`, `nccl-reducescatter.yaml`), the `nemotron5-8b.yaml` training smoke test, the continuous-load QoS pods (`rdma_qos_test.yaml`, `rdma-qos2-worker2.yaml`, `rdma-qos2-worker3.yaml`), and a recorded [`tests-report.txt`](https://github.com/maheshpabba/AI-Factory-Best-Practices/blob/Dev/nccl-training-tests/tests-report.txt) from an actual run of all four collectives plus the training job.

**NVIDIA's own `nvcrectl` Cluster Readiness Engine** — `WorkloadRun`/`Certification` custom resources that run the same style of collectives and produce NVIDIA's standard certification report format. Because `nvcrectl`'s test pods need privileged access and the macvlan attachments from Step 5 (which the CRD itself has no field for), the same `cluster-utility.sh` I created handles that gap too — `cluster-utility.sh -T prepare` sets up a small mutating admission webhook that patches those pods automatically, so testing itself never requires hand-editing security contexts or certs. [`nvcre-workloadruns/`](https://github.com/maheshpabba/AI-Factory-Best-Practices/tree/Dev/nvcre-workloadruns) holds the `WorkloadRun` manifests for both single-node (`nccl-all-reduce.yaml`, etc.) and cross-node, full-fabric scale-out runs (`nccl-all-reduce-scaleout.yaml`, etc.), the `nemotron5-8b.yaml`/`nemotron5-56b.yaml` training `WorkloadRun`s, and the `Certification`/`dcgm-*.yaml` manifests used for the DCGM diagnostic levels and the full certification pass.

### Where RDMA Actually Shows Up in These Tests

Every multi-node NCCL collective in these jobs runs over **GPUDirect RDMA**: data moves from GPU memory on one node, across the compute-fabric NICs from Step 2, directly into GPU memory on another node — without a round-trip through host CPU memory. A handful of NCCL/UCX environment variables are what actually steer this path, and getting them right (or wrong) is the difference between a test that uses the RDMA fabric and one that silently falls back to a slower path:

```bash
NCCL_IB_DISABLE=0            # don't disable the RDMA/IB transport
NCCL_NET_GDR_LEVEL=PHB       # allow GPUDirect RDMA at the PCIe-host-bridge level
NCCL_IB_GID_INDEX=5          # select the correct RoCEv2 GID table entry
UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,...   # explicit list of the RDMA devices from Step 1's inventory
```

Each worker pod attaches every one of the compute-fabric macvlan NADs from Step 5 (not just one), so NCCL has all of the node's rail-optimized NICs available and can spread collective traffic across every rail concurrently — which is the entire point of the topology from Step 1. A test that only attaches a single NIC's NAD will "work" but will never show the fabric's real aggregate bandwidth, which is a common false negative when people first wire these tests up.

### Two Report Formats, Same Two Numbers to Look For

The hand-written MPI path and `nvcrectl` produce differently formatted reports, but both come down to the same two numbers: **average bus bandwidth** across the whole run, and **peak AlgBW/BusBW** at the largest message size. Here's an excerpt from the recorded [`tests-report.txt`](https://github.com/maheshpabba/AI-Factory-Best-Practices/blob/Dev/nccl-training-tests/tests-report.txt) for the hand-written AllReduce job:

```text
┌────────────────────────────────────────────────────────────────┐
│  workloadrun/nccl-allreduce-2node                               │
├────────────────────────────────────────────────────────────────┤
│  Status:    Succeeded                                           │
│  Scale:     cross-node (2 nodes x 8 GPUs)                       │
│  Out of bounds values: 0 OK                                     │
│                                                                  │
│  Avg bus bandwidth: 206.994 GB/s                                │
│                                                                  │
│  Peak (16 GB message):                                          │
│    AlgBW (out-of-place):  414.98 GB/s                           │
│    BusBW (out-of-place):  778.08 GB/s                           │
└──────────────────────────────────────────────────────────────┘
```

`nvcrectl`'s own native report format is laid out differently, but it's the exact same two numbers underneath a `Status`/`Scale` header and a `Summary` block for pass/fail across every category tested — illustrative example, not a specific run:

```text
╔════════════════════════════════════════════════════════════════╗
║                       WorkloadRun Report                        ║
╚════════════════════════════════════════════════════════════════╝

  Name:      nccl-all-reduce-scaleout
  Platform:  onprem
  Nodes:     2

┌────────────────────────────────────────────────────────────────┐
│  workloadrun/nccl-all-reduce-scaleout                           │
├────────────────────────────────────────────────────────────────┤
│  Status:    Succeeded                                           │
│  Scale:     full-scale                                          │
│  Transport: IBext_v11                                           │
│  Bandwidth:                                                     │
│    Size       AlgBW        BusBW        Samples                 │
│    16 GB      <value>      <value>      1                       │
└──────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  Summary                                                        │
├────────────────────────────────────────────────────────────────┤
│  Categories:   1/1 passed                                       │
│  Failed Nodes: none                                              │
│  Result:       PASSED                                           │
└──────────────────────────────────────────────────────────────┘
```

Whichever format you're reading, the same checklist applies: `Status: Succeeded` (not just "finished"), `Out of bounds values: 0` or an equivalent zero-correctness-failures line, a `Categories: N/N passed` with no entries under `Failed Nodes`, and a bandwidth number that's internally consistent with the collective being tested (see below).

### What a Pass Actually Means

For either test path:

- The job reports **Succeeded**, not just "finished" — check the launcher/report exit status, not just that pods terminated.
- **Zero correctness failures** from the NCCL test binary itself. A healthy-looking bandwidth number next to a correctness failure is not a pass.
- Bandwidth figures are internally consistent across collectives — AllToAll will always report a lower number than AllReduce/AllGather/ReduceScatter on the same hardware (its many-to-many pattern produces smaller effective per-pair messages), which is expected, not a fabric problem.
- For a training smoke test: loss decreases every logged iteration, the parameter norm keeps changing (proof weights are actually updating, not a stalled run), and there are zero skipped/NaN iterations for the whole run.

### Sustained Load: QoS Doesn't Count Until It's Tested Under Contention

A single clean NCCL run proves the fabric works once. It doesn't prove the PFC/ECN configuration from Step 4 holds up under *sustained, concurrent* load — which is the normal state of a cluster running real training jobs. The QoS test, [`rdma_qos_test.yaml`](https://github.com/maheshpabba/AI-Factory-Best-Practices/blob/Dev/nccl-training-tests/rdma_qos_test.yaml), runs continuous background RDMA traffic (`ib_write_bw --run_infinitely`) as two independent client/server pairs at once, deliberately without any software-side traffic tagging, so the only thing marking the traffic correctly is the NIC's own RoCE QoS policy from the `NicConfigurationTemplate`. What to check: both pairs sustain comparable bandwidth *simultaneously* — if one collapses while the other climbs, that's queue starvation, meaning PFC is present but not fair between flows, which a single-flow test would never reveal.

---

## The Sequence, End to End

1. Design the rail-optimized topology and inventory every NIC before racking anything.
2. Capture per-host static network config for all three planes; convert to NMState YAML for the Assisted Installer.
3. Label GPU workers and isolate them in their own `MachineConfigPool` before any kernel-level `MachineConfig` lands.
4. Install NFD → Maintenance Operator → Network Operator → GPU Operator, in that order, with the GPU driver gated on the Network Operator's readiness label.
5. Tune RoCEv2 QoS on the compute and storage fabrics.
6. Wire up secondary CNIs (Multus + macvlan NADs) so pods can actually reach the RDMA NICs the Network Operator provisioned.
7. Validate with a real multi-node, multi-GPU job — both a hand-written MPI path and `nvcrectl` — checking correctness, not just bandwidth.
8. Confirm QoS holds under sustained, concurrent load, not just a single clean test run.

Skip a step, or do two of them out of order, and the failure you get almost never points at the step you actually got wrong — it shows up two steps later as an "intermittent" driver issue, a stalled pod with no network path, or a bandwidth number nobody can explain.

This is also the part that's easy to lose in the pitch. Standing up an AI Factory is not a regular OpenShift cluster install with a GPU Operator bolted on, and it's definitely not the same category of problem as opening a browser tab and chatting with an LLM. Every layer above — rail topology, NMState, MachineConfigPools, two operators fighting over the same kernel modules, secondary CNIs, RDMA, sustained QoS — has to be gotten right *and* gotten right in this order, by someone who understands how a failure two layers down surfaces as a symptom two layers up. I wrote more about what that actually looks like day to day, across the rest of the stack this cluster sits inside of, in [Building Enterprise AI Infrastructure: What an Architect Actually Goes Through](../posts/2026-05-20-ai-clusters-cisco-ucs.html). This post is the sequence for one part of that picture — the network/operator build-out — not the whole job.

The full manifests, scripts, and test artifacts for every step above are in [AI-Factory-Best-Practices](https://github.com/maheshpabba/AI-Factory-Best-Practices).

---

*Have questions? [Reach out](../contact.html)*
