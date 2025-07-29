import uvicorn
from cpra.mp.data.config import DEBUG


uvicorn.run("cpra.mp.data.api:app", host="0.0.0.0", port=8000, reload=False)
