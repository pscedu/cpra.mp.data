from fastapi import FastAPI
from cpra.mp.data.api.routes.data import data

app = FastAPI()
app.include_router(data, prefix="/data/v1")
