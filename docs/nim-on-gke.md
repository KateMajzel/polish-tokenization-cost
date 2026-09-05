# Running Nemotron NIM on Google Kubernetes Engine

These are deployment notes, kept separate from the tokenizer measurement in the main README. The Nemotron numbers there come from the published tokenizer on Hugging Face and do not depend on anything below.

## Setup

```bash
gcloud container clusters create nim-demo \
  --location=europe-west4-a --release-channel=rapid \
  --machine-type=e2-standard-4 --num-nodes=1

gcloud container node-pools create gpupool \
  --accelerator type=nvidia-l4,count=1,gpu-driver-version=latest \
  --cluster=nim-demo --location=europe-west4-a \
  --machine-type=g2-standard-16 --num-nodes=1
```

The GPU only becomes schedulable once the device plugin is running, which takes a few minutes after the node pool reports ready:

```bash
kubectl get pods -n kube-system | grep nvidia
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable."nvidia\.com/gpu"
```

Two secrets are needed — one to pull the image, one for the container to fetch weights:

```bash
kubectl create secret docker-registry ngc-secret \
  --docker-server=nvcr.io --docker-username='$oauthtoken' \
  --docker-password="$NGC_API_KEY"

kubectl create secret generic ngc-api --from-literal=NGC_API_KEY="$NGC_API_KEY"
```

Then `kubectl apply -f k8s/nim-nemotron.yaml`.

## What happened

NIM 1.12.2 started, detected exactly one compatible profile — `vllm-bf16-tp1-pp1` — and began loading. At bf16, 9B parameters occupy roughly 18 GB, leaving under 6 GB on a 24 GB L4 for activations, CUDA buffers, and KV cache. With the default 131,072-token context the engine failed during initialisation. Reducing the context to 8,192 tokens (`NIM_MAX_MODEL_LEN=8192`) and enabling `NIM_LOW_MEMORY_MODE=1` got further, but the engine process still exited with `RuntimeError: Engine process failed to start`.

The root cause of the second failure was not established. The logs captured at the time showed no explicit out-of-memory error, so memory pressure is the likely but unconfirmed explanation. Anyone reproducing this should capture the full engine log (`kubectl logs -f deploy/nemotron-nim`) before changing any setting.

One operational note: with a single GPU in the cluster, a rolling update deadlocks. The new pod stays `Pending` because the old one still holds the only accelerator. Delete the old pod by hand after changing the deployment.

## Takeaways

- Serving a 9B model at bf16 on a single 24 GB L4 leaves almost no headroom. The realistic targets are an FP8 profile, if the NIM ships one for the model, or a card with more memory — a 48 GB L40S, or A100/H100.
- Check model lifecycle dates before deploying. Nemotron Nano 9B v2 reached end of life on 2026-08-26; the container image used here predates that announcement, and the hosted endpoint for the same model now returns HTTP 410.
- Pin the image tag. Reproducibility of a failure matters as much as reproducibility of a success.
- Delete the cluster when not in use. A `g2-standard-16` node with an L4 runs at roughly USD 2/hour, which is easy to forget overnight.

```bash
gcloud container clusters delete nim-demo --location=europe-west4-a
```
