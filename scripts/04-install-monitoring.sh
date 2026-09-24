#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing kube-prometheus-stack (Prometheus + Grafana)"
echo "    --set serviceMonitorSelectorNilUsesHelmValues=false: without this, Prometheus"
echo "    only scrapes ServiceMonitors labeled release=monitoring by default, silently"
echo "    ignoring any ServiceMonitor from a different chart (DCGM's, or your own),"
echo "    confirmed by testing to be exactly why Grafana panels showed 'No data' with no error."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

echo "==> Installing NVIDIA DCGM exporter (GPU hardware metrics)"
echo "    --set serviceMonitor.additionalLabels.release=monitoring: DCGM's own"
echo "    chart-native way to satisfy the same label, redundant with the selector"
echo "    flag above but officially documented by NVIDIA, defense in depth."
echo "    Verify current chart repo/name before running, packaging has moved before."
helm repo add nvidia https://nvidia.github.io/dcgm-exporter/helm-charts
helm install dcgm-exporter nvidia/dcgm-exporter --namespace monitoring \
  --set serviceMonitor.additionalLabels.release=monitoring

echo "==> Wiring vLLM's own /metrics into Prometheus"
kubectl apply -f manifests/vllm-servicemonitor.yaml
