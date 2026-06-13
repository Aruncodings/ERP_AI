import uvicorn
from erp_backend.api.server import app


def main():
    uvicorn.run(
        "erp_backend.api.server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
