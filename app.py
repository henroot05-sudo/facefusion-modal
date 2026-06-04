import modal

APP_NAME = "facefusion"

app = modal.App(APP_NAME)

models = modal.Volume.from_name(
    "facefusion-models",
    create_if_missing=True
)

outputs = modal.Volume.from_name(
    "facefusion-outputs",
    create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "ffmpeg",
        "wget",
        "curl",
        "libgl1",
        "libglib2.0-0"
    )
    .run_commands(
        "git clone https://github.com/facefusion/facefusion /root/facefusion",
        "cd /root/facefusion && python install.py --onnxruntime cuda"
    )
)

@app.function(
    gpu="L4",
    timeout=86400,
    image=image,
    volumes={
        "/root/.facefusion": models,
        "/root/outputs": outputs,
    }
)
@modal.web_server(port=7860)
def web():
    import subprocess

    subprocess.Popen(
        """
        cd /root/facefusion &&
        python facefusion.py run \
            --execution-providers cuda cpu
        """,
        shell=True
    )
