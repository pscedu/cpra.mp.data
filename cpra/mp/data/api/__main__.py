import uvicorn
from cpra.mp.data.api import app

uvicorn.run(app, host="0.0.0.0", port=8000)
