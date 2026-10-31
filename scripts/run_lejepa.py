from kube_jobs import storage, submit_job


submit_job(
    job_name="nfm-lejepa",
    username="matejpekar",
    image="cerit.io/rationai/base:2.0.6",
    cpu=48,
    memory="128Gi",
    gpu="H100",
    public=False,
    script=[
        "git clone -b patch-lejepa https://github.com/RationAI/nuclei-foundation-model.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m nfm +experiment=lejepa",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
