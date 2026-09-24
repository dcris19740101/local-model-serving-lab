#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing KEDA (scale-to-zero without a service mesh)"
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm install keda kedacore/keda --namespace keda --create-namespace

echo "==> Installing KEDA HTTP Add-on (required for scale-FROM-zero on a request;"
echo "    plain KEDA can scale to zero but has no way to wake back up in raw/Standard"
echo "    KServe mode, confirmed by KServe's own docs, there's no Knative-style activator)"
helm install http-add-on kedacore/keda-add-ons-http --namespace keda
kubectl get pods -n keda | grep interceptor

echo "==> KEDA installed. Apply manifests/vllm-http-scaledobject.yaml after the"
echo "    InferenceService exists (do NOT hand-write a separate ScaledObject, its"
echo "    own operator creates one, correctly linked; a hand-written one fails with"
echo "    'unable to get the linked HTTPScaledObject for ScaledObject')."
echo "    Test through the interceptor's Service (kubectl get svc -n keda | grep"
echo "    interceptor), NOT the predictor's own Service, see README for the full flow."
