from huggingface_hub import snapshot_download

# 下载 Qwen3-0.6B 模型到默认缓存目录
# 默认路径为: ~/.cache/huggingface/hub/
model_id = "Qwen/Qwen3-0.6B"

print(f"正在下载模型: {model_id}")
print("默认保存路径: ~/.cache/huggingface/hub/")

snapshot_download(
    repo_id=model_id,
    local_dir=None,  # 使用默认缓存目录
)

print(f"\n模型 {model_id} 下载完成!")
