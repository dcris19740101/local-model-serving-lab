#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing kubectl, minikube, Helm"

curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm -f kubectl

curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
rm -f minikube-linux-amd64

curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

echo "==> Starting minikube with GPU access"
minikube start --driver=docker --container-runtime=docker --gpus=all

echo "==> Installing NVIDIA Kubernetes Device Plugin via Helm (NVIDIA's preferred method,"
echo "    the old raw-file static manifest URL moved and now 404s)"
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update
helm upgrade -i nvdp nvdp/nvidia-device-plugin \
  --namespace nvidia-device-plugin \
  --create-namespace \
  --version 0.17.1
# Check `helm search repo nvdp` for the current version before running this

echo "==> Checking GPU is schedulable (prints 1 if the device plugin registered it)"
GPU_COUNT=$(kubectl get node minikube -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')
if [ -z "$GPU_COUNT" ]; then
  echo "    GPU not found in Allocatable. Debug with:"
  echo "      docker exec -it minikube nvidia-smi   # host-to-node passthrough"
  echo "      kubectl get pods -n kube-system | grep -i nvidia   # device plugin pod status"
else
  echo "    GPU found: nvidia.com/gpu = ${GPU_COUNT}"
fi

echo "==> Installing cert-manager (hard prerequisite for KServe's webhook certs)"
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.21.2/cert-manager.yaml
kubectl wait --for=condition=ready pod -l app=cert-manager -n cert-manager --timeout=120s
kubectl get pods -n cert-manager
# Check cert-manager's current release tag before running this; KServe needs 1.15.0+

echo "==> Creating the kserve namespace (the release manifest assumes it already exists)"
kubectl create namespace kserve

echo "==> Installing KServe (--server-side avoids the 'annotations too long' error"
echo "    on KServe's large InferenceService CRD, a plain 'apply' fails on it)"
KSERVE_URL="https://github.com/kserve/kserve/releases/download/v0.20.0/kserve.yaml"
# Check the current release tag at https://github.com/kserve/kserve/releases before running this
if ! kubectl apply --server-side "$KSERVE_URL"; then
  echo "==> Conflicts from an earlier partial attempt, if any: --force-conflicts first,"
  echo "    then a clean delete+reapply if conflicts persist (confirmed more reliable than"
  echo "    --force-conflicts alone once a first attempt failed partway through)"
  kubectl apply --server-side --force-conflicts -f "$KSERVE_URL" || {
    kubectl delete -f "$KSERVE_URL"
    kubectl apply --server-side -f "$KSERVE_URL"
  }
fi
# If this still errors with "no matches for kind", the CRDs likely haven't finished
# registering yet, just re-run the same apply command once more.

echo "==> Installing KServe's built-in ClusterServingRuntimes (vLLM, sklearn, etc.)"
echo "    kserve.yaml alone does NOT include these; without this step, any"
echo "    InferenceService fails with 'No ServingRuntimes ... with the name: ...'"
kubectl apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.20.0/kserve-cluster-resources.yaml
kubectl get clusterservingruntimes

echo "==> Switching KServe's default deployment mode to RawDeployment"
echo "    (default is Serverless/Knative-based even when Knative isn't installed;"
echo "    left as default, any InferenceService sits stuck in Unknown status with"
echo "    a ServerlessModeRejected event)"
kubectl patch configmap/inferenceservice-config -n kserve --type=strategic \
  -p '{"data": {"deploy": "{\"defaultDeploymentMode\": \"RawDeployment\"}"}}'
kubectl rollout restart deployment kserve-controller-manager -n kserve
kubectl rollout status deployment kserve-controller-manager -n kserve
