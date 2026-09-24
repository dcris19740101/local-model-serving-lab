# local-model-serving-lab

Serving LLMs and traditional ML models locally on a single consumer GPU, using KServe, vLLM, and KEDA on minikube.

This repo is the companion code for a two-part write-up:

- **Part 1**: [Building an AI/ML Workstation with an RTX 5080](https://medium.com/@dcris19740101/building-an-ai-ml-workstation-with-an-rtx-5080-ubuntu-and-two-long-nights-of-gpu-driver-debugging-ee250910a1e3) — Ubuntu, GPU driver debugging, dual-boot setup
- **Part 2**: [Serving LLMs with KServe and vLLM on a Consumer GPU](https://medium.com/@dcris19740101/serving-llms-with-kserve-and-vllm-on-a-consumer-gpu-a-local-kubernetes-build-log-a5c8eeed69f3) — this repo

## What's here

A working (in progress) deployment of:

- **minikube** with GPU passthrough via the NVIDIA Container Toolkit and Kubernetes Device Plugin, no VM, no VFIO, container-level GPU access
- **vLLM**, serving an open-weight model from Hugging Face (PagedAttention + FlashAttention under the hood)
- **KServe**, wrapping vLLM in a standard `InferenceService`, in raw deployment mode
- **KEDA**, scaling the vLLM pod to zero when idle, freeing the only GPU on the machine for other work
- **KServe's sklearn runtime**, serving a traditional ML model trained in [ml-fundamentals](https://github.com/dcris19740101/ml-fundamentals) on the same cluster, CPU-only, no GPU contention
- **Observability**: kube-prometheus-stack for cluster/pod metrics, NVIDIA DCGM exporter for GPU hardware metrics, and vLLM's native `/metrics` endpoint for inference-specific metrics (tokens/sec, time-to-first-token, KV cache pressure, queue depth)

## Hardware this was built and tested on

- AMD Ryzen 9 9950X3D
- NVIDIA RTX 5080 (16GB, Blackwell)
- 64GB DDR5
- Ubuntu 24.04 LTS

## Repo structure

```
.
├── manifests/
│   ├── vllm-inferenceservice.yaml       # vLLM InferenceService (GPU path)
│   ├── vllm-http-scaledobject.yaml      # KEDA HTTP Add-on: real scale-to-zero + wake-on-request
│   ├── sklearn-inferenceservice.yaml    # traditional ML InferenceService (CPU path)
│   ├── hf-token-secret.example.yaml     # template, do not commit a real token
│   └── vllm-servicemonitor.yaml         # wires vLLM's /metrics into Prometheus
├── scripts/
│   ├── 00-install-docker.sh
│   ├── 01-install-nvidia-toolkit.sh
│   ├── 02-setup-minikube.sh
│   ├── 03-setup-huggingface.sh
│   ├── 04-install-monitoring.sh         # Prometheus, Grafana, DCGM, vLLM ServiceMonitor
│   ├── 05-install-keda.sh               # KEDA + HTTP Add-on
│   └── test-vllm-standalone.sh
└── docs/
    ├── architecture-stack.png            # full stack diagram
    └── vllm-kserve-flow.png              # request-path diagram
```

## Quickstart

```bash
git clone https://github.com/<your-username>/local-model-serving-lab.git
cd local-model-serving-lab

# Run in order; each script checkpoints with a verification command
bash scripts/00-install-docker.sh
bash scripts/01-install-nvidia-toolkit.sh
bash scripts/02-setup-minikube.sh
bash scripts/03-setup-huggingface.sh
bash scripts/04-install-monitoring.sh
bash scripts/05-install-keda.sh

# Confirm vLLM works standalone before touching KServe
bash scripts/test-vllm-standalone.sh

# Then deploy through KServe
kubectl create secret generic hf-token-secret --from-literal=token=$(cat ~/.cache/huggingface/token)
kubectl apply -f manifests/vllm-inferenceservice.yaml
kubectl apply -f manifests/vllm-http-scaledobject.yaml
kubectl apply -f manifests/sklearn-inferenceservice.yaml
kubectl apply -f manifests/vllm-servicemonitor.yaml
```

## Status

Work in progress. This README and the manifests will be updated with real command output, actual cold-start numbers, and any errors hit along the way, that's the point of the write-up. Check the article for the narrative version of what breaks and why.

### Real issues hit so far

- **NVIDIA device plugin static manifest 404s.** The old `raw.githubusercontent.com/.../main/nvidia-device-plugin.yml` URL is gone, NVIDIA restructured the repo and now recommends Helm. Fixed in `scripts/02-setup-minikube.sh`.
- **`minikube start --gpus=all` doesn't retroactively apply to an already-running cluster.** If you enabled GPU support after minikube was already created, `minikube delete` and recreate it with the flag from the start.
- **Checking for the GPU resource with a filtered `kubectl describe` can be misleading** if the filter cuts off before reaching `nvidia.com/gpu` (allocatable resources are listed alphabetically, and it sorts after `memory`). `kubectl get node minikube -o jsonpath='{.status.allocatable.nvidia\.com/gpu}'` is unambiguous, it just prints `1` or nothing.
- **Full-precision `Qwen2.5-7B-Instruct` does not fit on a 16GB card.** Confirmed with a real OOM: the RTX 5080 reports 15.45 GiB usable VRAM (not the full 16GB nameplate), the model's weights alone took 14.29 GiB on load, and it crashed trying to allocate a 150 MiB sampler tensor before even reaching KV cache allocation. `--gpu-memory-utilization` doesn't fix this, the weights themselves already exceed budget. Fixed by switching the default model to `Qwen/Qwen2.5-7B-Instruct-AWQ` (4-bit, ~5GB), which leaves 10GB+ free for KV cache and activations.
- **`kubectl apply` on KServe's manifest fails with `metadata.annotations: Too long`.** KServe's `InferenceService` CRD exceeds the 256KiB size limit for client-side apply's change-tracking annotation. Fixed by using `kubectl apply --server-side` instead, KServe's own documented fix. Also needs cert-manager installed and running first (a hard prerequisite, not previously in this guide), or the webhook `Certificate`/`Issuer` resources fail and cascade into unrelated-looking `namespaces "kserve" not found` errors.
- **KServe's manifest doesn't create its own `kserve` namespace.** Run `kubectl create namespace kserve` before applying it.
- **If a plain `apply` was tried before switching to `--server-side`, expect ownership conflicts on the webhook configs.** `--force-conflicts` is the documented fix and worth trying first, but confirmed in practice: once a first attempt fails partway through creating dozens of resources, the partial state can be messier than `--force-conflicts` alone resolves. A clean `kubectl delete -f <url>` followed by a fresh `kubectl apply --server-side -f <url>` is the more reliable fix, not just a fallback.
- **Don't confuse `InferenceService` (what this repo uses) with KServe's newer `LLMInferenceService` CRD.** The latter needs Gateway API, the Gateway API Inference Extension, Envoy Gateway, and LeaderWorkerSet, real requirements, but only for its multi-node/gateway-native serving features, which don't apply to a single-GPU setup. This repo only needs cert-manager.
- **An `InferenceService` sits stuck in `READY: Unknown` with a `ServerlessModeRejected` event.** KServe's default deployment mode is `Serverless` (Knative-based) even when Knative isn't installed. Fixed by patching the `inferenceservice-config` ConfigMap to `RawDeployment` and restarting the controller (now baked into `scripts/02-setup-minikube.sh`), then deleting and recreating any InferenceService that got stuck before the patch took effect.
- **`No ServingRuntimes or ClusterServingRuntimes with the name: kserve-vllmserver`, even though that name is correct.** `kserve.yaml` only installs CRDs, the controller, and webhooks, not the actual built-in runtime definitions. Those ship separately in `kserve-cluster-resources.yaml`, now added as its own step in `scripts/02-setup-minikube.sh`. Confirm with `kubectl get clusterservingruntimes`, it should list around a dozen runtimes once this step runs.
- **Three different ClusterServingRuntimes (`kserve-mlserver`, `kserve-predictiveserver`, `kserve-sklearnserver`) all claim the `sklearn` modelFormat.** Leaving `runtime:` unset on the sklearn InferenceService is genuinely ambiguous, not auto-resolved. `manifests/sklearn-inferenceservice.yaml` now sets `runtime: kserve-sklearnserver` explicitly.
- **The vLLM pod crash-loops with `exec: "python": executable file not found in $PATH`, even with `kserve-vllmserver` correctly installed.** The built-in runtime pins `vllm/vllm-openai:v0.20.0` with a hardcoded `command: [python, ...]`, but that image tag only ships `python3`. It also auto-injects a conflicting `--model=/mnt/models` arg, assuming a `storageUri`-based deployment this guide doesn't use. Fixed by rewriting `manifests/vllm-inferenceservice.yaml` to use `predictor.containers` (a custom container spec) instead of the managed `predictor.model` + runtime pattern, bypassing both problems entirely and reusing the exact image/args already validated in the standalone `docker run` test.
- **The vLLM pod then got `OOMKilled`, a host RAM limit, not GPU VRAM.** Even in the custom-containers pattern, KServe's webhook injects a default resource limit for any container named `kserve-container`, confirmed at exactly `cpu: 1` / `memory: 2Gi`, far too small for an LLM engine's host-side overhead (tokenizer, CUDA graph capture, multiprocessing). Fixed by setting explicit `resources.requests`/`limits` (`cpu: 2`/`4`, `memory: 4Gi`/`12Gi`) in the manifest; `8Gi` got it running but logs showed only `1.71 GiB` free at that point, tight under concurrent load, so bumped to `12Gi` for real headroom.
- **No ingress controller is installed in this guide (deliberately, see raw deployment mode).** The `<ingress-or-nodeport>` placeholder in earlier drafts was never resolved. Both InferenceServices are reached via `kubectl port-forward svc/<name>-predictor <local-port>:80` instead, no NodePort or `minikube ip` needed.
- **Applying the KEDA `ScaledObject` fails: `admission webhook "vscaledobject.kb.io" denied the request: ... already managed by the hpa ...`.** KServe creates its own HPA for the predictor Deployment by default in raw/Standard mode, and KEDA correctly refuses to attach a second autoscaler to the same Deployment. Fixed by adding `serving.kserve.io/autoscalerClass: "external"` to the InferenceService's annotations (now in `manifests/vllm-inferenceservice.yaml`), which tells KServe to create neither its own HPA nor a KServe-managed KEDA object, so the standalone `ScaledObject` can manage it instead.
- **The `ScaledObject` shows `KEDAScalerFailed` / `dial tcp: lookup monitoring-kube-prometheus-prometheus.monitoring.svc ... no such host` even after the HPA conflict is fixed.** With the original Prometheus-based `ScaledObject` design, the trigger depended on `kube-prometheus-stack` and vLLM's `ServiceMonitor`, which didn't exist yet at that point in the original step order. Since fixed twice over: the monitoring stack (Prometheus, DCGM, vLLM's `ServiceMonitor`) now installs as its own dedicated step, `scripts/04-install-monitoring.sh`, before KEDA (`scripts/05-install-keda.sh`) rather than split across two later scripts; and KEDA's scale-from-zero trigger no longer depends on Prometheus at all, see the HTTP Add-on entry below.
- **A Grafana dashboard imported from grafana.com (e.g. NVIDIA DCGM Exporter Dashboard, ID `12239`) shows every panel as "No data," with no error.** kube-prometheus-stack's Prometheus, by default, only scrapes `ServiceMonitor` objects carrying a label matching its own Helm release name (`release: monitoring`). A `ServiceMonitor` created by a completely different chart (like DCGM-exporter's) doesn't carry that label and gets silently ignored. Fixed cluster-wide with `helm upgrade monitoring prometheus-community/kube-prometheus-stack --namespace monitoring --reuse-values --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false`, which opens Prometheus to watch all `ServiceMonitor`s regardless of label. (`scripts/04-install-monitoring.sh` now sets this same flag on the initial install itself, so a fresh setup shouldn't hit this at all.)
- **Even after the fix above, `manifests/vllm-servicemonitor.yaml`'s metrics still don't show up in Prometheus (`Result series: 0`).** Three separate, compounding mismatches, confirmed by checking the real Service directly (`kubectl get svc qwen-vllm-predictor -o yaml`): (1) no `namespaceSelector`, so the `ServiceMonitor` (in `monitoring`) never looked in `default`, where the Service actually lives; (2) the label is `app: isvc.qwen-vllm-predictor` (KServe adds an `isvc.` prefix), not the plain predictor name; (3) the Service's port has no name at all, so matching by `port: http` can never work, `targetPort: 8080` (matching by number) is required instead. All three fixed in the manifest.
- **Grafana dashboard for vLLM's own metrics**: use vLLM's official dashboards, confirmed working, at `examples/observability/dashboards/grafana/{performance_statistics,query_statistics}.json` in the [vLLM repo](https://github.com/vllm-project/vllm) (older versions: `examples/online_serving/dashboards/grafana/`). Import both directly (Dashboards → New → Import → Upload JSON). The community dashboard on grafana.com (ID `25043`) is built for vLLM's separate multi-engine "production-stack" project and left several panels empty (`Available vLLM instances`, `Current QPS`) even with metrics flowing correctly everywhere else, not recommended for a single-engine deployment like this one.
- **A `ScaledObject` scales the pod to zero fine, but a `curl`/`port-forward` afterward just hangs, the pod never wakes back up.** Confirmed against KServe's own docs: `"Scale from Zero is currently not supported in Standard mode for HTTP requests."` Plain KEDA (with a Prometheus trigger or otherwise) can take a Deployment to zero, but nothing then watches for an incoming request against a pod that doesn't exist yet to trigger scale-up, that's a Knative "activator" job, and raw/Standard KServe mode has no equivalent. Fixed by installing **KEDA's HTTP Add-on** (`helm install http-add-on kedacore/keda-add-ons-http --namespace keda`), which adds an interceptor that holds requests during cold start. **Testing must go through the interceptor's own Service** (`kubectl get svc -n keda | grep interceptor`), not the predictor's Service directly, with a `Host` header matching the configured host.
- **A hand-written `InterceptorRoute` (`http.keda.sh/v1beta1`, the current, non-deprecated API per KEDA's docs) plus a hand-written `ScaledObject` with an `external-push` trigger never scales up: `unable to get the linked HTTPScaledObject for ScaledObject`, and the request just hangs forever.** A real version-skew, confirmed by testing: the external-scaler binary shipped with this chart only resolves `GetMetricSpec` for a `ScaledObject` it generated itself from an `HTTPScaledObject`, linked via a field in the trigger metadata a hand-written `ScaledObject` doesn't have. Fixed by using `manifests/vllm-http-scaledobject.yaml` (an `HTTPScaledObject`, the older but actually-working API for this chart version) instead, applying only that one resource and letting its own operator create and own the `ScaledObject`.
- **After switching to `HTTPScaledObject`, a request still hangs forever with zero scale-up activity.** A leftover `InterceptorRoute` from an earlier attempt can keep silently absorbing requests even after its own `ScaledObject` was deleted, confirmed via the interceptor's live queue (`kubectl port-forward svc/keda-add-ons-http-interceptor-admin 9090:9090 -n keda`, then `curl http://localhost:9090/queue`), which showed the request held against the dead route (`RequestCount:1`) while the real, working route sat at zero. `kubectl get interceptorroute` and deleting anything unexpected resolves it; confirmed working end to end afterward with a real cold-start of roughly 5-6 minutes, request in, pod scheduled, model loaded, response returned.

## License

MIT, see [LICENSE](LICENSE).
