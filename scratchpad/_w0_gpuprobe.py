import modal
app = modal.App("w0-gpuprobe")
img = modal.Image.debian_slim().pip_install("torch==2.2.1")
@app.function(image=img, gpu="A10G", timeout=120)
def probe():
    import torch
    return f"GPU_OK cuda={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}"
@app.local_entrypoint()
def main():
    print(probe.remote())
