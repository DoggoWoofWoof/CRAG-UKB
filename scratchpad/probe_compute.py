import modal
app = modal.App("crag-probe")
@app.function(image=modal.Image.debian_slim(python_version="3.11"), cpu=1.0, memory=1024,
              timeout=120)
def ping():
    return "ok"
@app.local_entrypoint()
def main():
    print("PROBE_RESULT=%s" % ping.remote())
