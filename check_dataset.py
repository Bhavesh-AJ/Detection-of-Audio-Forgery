from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mueller91/In-The-Wild",
    repo_type="dataset",
    local_dir="In-The-Wild",
    local_dir_use_symlinks=False
)

print("Dataset downloaded successfully!")