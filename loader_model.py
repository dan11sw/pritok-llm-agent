from huggingface_hub import snapshot_download

model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
local_dir = "/home/qwen2.5-coder"

snapshot_download(repo_id=model_name, local_dir=local_dir, revision="main")




