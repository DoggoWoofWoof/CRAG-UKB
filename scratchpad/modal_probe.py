# -*- coding: utf-8 -*-
"""Cheapest possible check for whether a workspace can still run compute at all.

Two of the three accounts holding a COMPLETE copy of the dense corpus turned out to be over
their spend limit, and that only surfaces when a job is submitted -- volume listing works fine
on a workspace that cannot run a container. So probe with a container that does nothing, and
pick the target account from the answer rather than from what it has on disk.
"""
import modal

app = modal.App("crag-probe")
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, cpu=0.25, memory=512, timeout=120)
def ping():
    return "ok"


@app.local_entrypoint()
def main():
    print("PROBE " + ping.remote())
