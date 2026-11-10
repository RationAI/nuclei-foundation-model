from kube_jobs import storage, submit_job


submit_job(
    job_name="nfm-token-supervised",
    username="matejpekar",
    image="cerit.io/rationai/base:2.0.6",
    cpu=48,
    memory="128Gi",
    gpu="H100",
    public=False,
    script=[
        "git clone -b token-supervised https://github.com/RationAI/nuclei-foundation-model.git workdir",
        "cd workdir",
        "uv sync --frozen",
        # Token counts vary per batch, expandable segments avoid fragmentation.
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run -m nfm +experiment=supervised",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
