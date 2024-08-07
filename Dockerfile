FROM python:3.12-slim
LABEL maintainer="mjyoder@psc.edu"
LABEL org.opencontainers.image.source="https://github.com/pscedu/cpra.mp.data"

WORKDIR /usr/src/app
COPY pyproject.toml \
    README.md ./

RUN mkdir -p cpra/mp/data && pip install .[api] \
    && python -m pip cache purge

COPY cpra/ ./cpra

EXPOSE 8000
CMD ["python3", "-u", "-m", "cpra.mp.data.api"]
