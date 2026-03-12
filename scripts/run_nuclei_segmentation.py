from kube_jobs import storage, submit_job


submit_job(
    job_name="nfm-nuclei-segmentation",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=20,
    memory="80Gi",
    gpu="H100",
    public=False,
    script=[
        "git clone https://github.com/RationAI/nuclei-foundation-model.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.nuclei_segmentation +data=...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
