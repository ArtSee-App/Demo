from firebase_functions import https_fn
from firebase_admin import initialize_app
from firebase_functions.options import set_global_options

set_global_options(max_instances=10)

initialize_app()

@https_fn.on_request(region="europe-west1")  # Specify the region here
def on_request_example(req: https_fn.Request) -> https_fn.Response:
    return https_fn.Response("Hello world!")
