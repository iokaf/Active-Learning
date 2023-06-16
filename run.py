# Create a FAST API application with two end points.
# The first point returns the a link and the second calls a function
# that trains the model and returns the model.

# Path: interface_fast_api.py
# Compare this snippet from src/app.py:

import gc
import os
import toml 

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import threading

from src import ActiveLearning
from model import Model
if __name__ == "__main__":

    with open("config.toml", "r") as f:
        config = toml.load(f)

    active_learning = ActiveLearning(
        config=config,
        Model=Model
    )

    app = FastAPI()

    @app.get("/")
    def get_link():
        url = active_learning.ls_handler.get_ls_project_url()
        print(url)
        return {"link": url}

    @app.post("/train")
    def perform_step():
        gc.collect()
        active_learning.iterate()
    
    

    local_webserver_port = config["image-server"]["port"]
    images_server_dir = config["image-server"]["directory"]

    if not os.path.exists(images_server_dir):
        os.makedirs(images_server_dir)

    # Make app serve static files
    app.mount("/", StaticFiles(directory=images_server_dir), name="/static")
    
    ls_port = config["label-studio"]["port"]

    host = active_learning.ls_handler.ip_address    

    ls_thread = threading.Thread(
        target=os.system, args=(f"label-studio start --port {ls_port} -b",)
    )

    uvicorn_thread = threading.Thread(
        target=uvicorn.run, args=(app,), 
        kwargs={"host": "0.0.0.0", "port":local_webserver_port}
    )

    # Run both threads in parallel
    ls_thread.start()
    uvicorn_thread.start()

    # Close the threads if either of them is closed
    ls_thread.join()
    uvicorn_thread.join()


