import modal
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Use local src for image building
from src.experiments.backends import _modal_base_image, MODAL_VOLUME, REMOTE_ROOT

app = modal.App("crag-smoke2")
volume = modal.Volume.from_name(MODAL_VOLUME, create_if_missing=True)
image = _modal_base_image(modal).add_local_dir("src", remote_path=f"{REMOTE_ROOT}/src").add_local_dir("configs", remote_path=f"{REMOTE_ROOT}/configs").add_local_file("experiments.py", remote_path=f"{REMOTE_ROOT}/experiments.py").add_local_dir("scratchpad", remote_path=f"{REMOTE_ROOT}/scratchpad")

@app.function(image=image, gpu="A10G", volumes={f"{REMOTE_ROOT}/storage": volume}, timeout=600)
def smoke():
    import os, sys, torch
    os.chdir(REMOTE_ROOT)
    if REMOTE_ROOT not in sys.path:
        sys.path.append(REMOTE_ROOT)
    import shutil
    # symlink
    sys.path.insert(0, REMOTE_ROOT)
    from src.experiments.backends import _symlink_storage
    _symlink_storage(None)
    print("REMOTE_FUNCTION_STARTED")
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name={torch.cuda.get_device_name(0)}")
        print(f"GPU count={torch.cuda.device_count()}")
    print(f"REMOTE_ROOT={REMOTE_ROOT}")
    print(f"volume storage path={REMOTE_ROOT}/storage exists={os.path.exists(f'{REMOTE_ROOT}/storage')}")
    print(f"data exists={os.path.exists('data/ukb_storage/2wiki_clean/gte_qwen/nodes.npy')}")
    volume.commit()
    print("SMOKE PASS")

@app.local_entrypoint()
def main():
    smoke.remote()
