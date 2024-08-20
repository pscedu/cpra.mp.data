import anyio
import tempfile
import threading
from cpra.mp.data.api.dataset import Dataset
from starlette.responses import Response
from starlette.types import Receive, Scope, Send
from time import time


class PolarsStreamingResponse(Response):
    background = None
    chunk_size = 64 * 1024
    mime_types = {
        "arrow": "application/vnd.apache.arrow.file",
        "csv": "text/csv",
        "ndjson": "application/x-ndjson",
        "parquet": "application/vnd.apache.parquet",
    }

    def __init__(self, dataset: Dataset, format: str = "csv") -> None:
        self.dataset = dataset
        self.format = format
        self.media_type = self.mime_types[format]
        self.init_headers()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": self.raw_headers,
            }
        )
        if scope["method"].upper() == "HEAD":
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        else:
            with tempfile.NamedTemporaryFile() as fp:
                start = time()
                write_thread = threading.Thread(
                    target=stream_to_file,
                    name="stream_to_file",
                    args=[self.dataset.lf, fp.name, self.format],
                )
                write_thread.start()
                async with await anyio.open_file(fp.name, mode="rb") as file:
                    more_body = True
                    while more_body:
                        chunk = await file.read(self.chunk_size)
                        chunk_has_data = len(chunk) > 0
                        more_body = chunk_has_data or write_thread.is_alive()
                        if chunk_has_data or not more_body:
                            await send(
                                {
                                    "type": "http.response.body",
                                    "body": chunk,
                                    "more_body": more_body,
                                }
                            )
                print(f"Download took {time() - start} seconds")


def stream_to_file(lf, out_path, format):
    sink_method = {
        "arrow": "sink_ipc",
        "csv": "sink_csv",
        "ndjson": "sink_ndjson",
        "parquet": "sink_parquet",
    }[format]
    # TODO: Set parquet compression and row group parameters.
    start = time()
    getattr(lf, sink_method)(out_path)
    print(f"Save took {time() - start} seconds")
