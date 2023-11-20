from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

title = 'ImageCompressorAPI'
app = FastAPI(version='1.0.0', title=title)

app.add_middleware(CORSMiddleware,
                   allow_origins=['*'],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )


@app.get("/")
async def health_check():
    return {"msg": f"{title} is up and running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
