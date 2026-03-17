from kube_jobs import storage, submit_job


submit_job(
    job_name="nfm-annotation-labels",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=20,
    memory="80Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/nuclei-foundation-model.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.annotation_labels +data=...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
