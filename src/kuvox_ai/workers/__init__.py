"""Background workers — one process per heavy workload.

Each worker connects only the infrastructure clients it needs, declares its
queue, and binds an async ``handle_message`` callback. Workers run as
separate processes from the FastAPI app.
"""
