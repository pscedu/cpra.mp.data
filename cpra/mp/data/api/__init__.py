from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from cpra.mp.data.api.routes.catalog import catalog
from cpra.mp.data.api.routes.data import data

app = FastAPI(
    title="CPRA Master Plan APIs",
    description="Access model data from [Louisiana's Coastal Master Plan](https://coastal.la.gov/our-plan/2023-coastal-master-plan/).",
)
app.include_router(catalog, prefix="/catalog/v1")
app.include_router(data, prefix="/data/v1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
