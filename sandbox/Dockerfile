FROM python:3.11-slim

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    flask \
    requests \
    httpx \
    pydantic \
    sqlalchemy \
    numpy \
    pandas \
    markdown \
    pytest \
    pytest-asyncio \
    pytest-json-report \
    pytest-timeout \
    anyio[trio]

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
