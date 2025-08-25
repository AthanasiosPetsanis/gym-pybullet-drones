import shutil
import os

# --- Paths to delete ---
logdir = "./ppo_drone_tensorboard"
modeldir = "./ppo_drone_vision"

def remove_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"[INFO] Removed: {path}")
    else:
        print(f"[INFO] Path does not exist: {path}")

if __name__ == "__main__":
    print("[INFO] Cleaning up old logs and models...")
    remove_dir(logdir)
    remove_dir(modeldir)
    print("[INFO] Done. Ready for new training run!")
