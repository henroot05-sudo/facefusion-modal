import modal
import subprocess

APP_NAME = "facefusion"

app = modal.App(APP_NAME)

# Persistent volumes for models and outputs
models = modal.Volume.from_name("facefusion-models", create_if_missing=True)
outputs = modal.Volume.from_name("facefusion-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "wget", "curl", "libgl1", "libglib2.0-0")
    .run_commands(
        "git clone https://github.com/facefusion/facefusion /root/facefusion",
        "cd /root/facefusion && python install.py --onnxruntime cuda",
        # Optional: pre-download common models to speed up first run
        # "cd /root/facefusion && python -c 'from facefusion.download import download; download()'"
    )
)

@app.function(
    gpu="L4",           # or A10G / H100 depending on your needs
    timeout=86400,
    image=image,
    volumes={
        "/root/.facefusion": models,
        "/root/outputs": outputs,
    },
    # Allow longer startup
    allow_background_volume_commits=True,
)
@modal.web_server(port=7860, startup_timeout=300)
def web():
    # Set Gradio to listen on all interfaces
    env = {
        "GRADIO_SERVER_NAME": "0.0.0.0",
        "GRADIO_SERVER_PORT": "7860",
    }
    
    subprocess.Popen(
        [
            "python", 
            "/root/facefusion/facefusion.py", 
            "run",
            "--execution-providers", "cuda", "cpu",
            "--listen",               # Important for Modal
        ],
        cwd="/root/facefusion",
        env={**env, **dict(os.environ)},  # merge with existing env
    )
