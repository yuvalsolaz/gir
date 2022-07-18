from typing import Union, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],  # Allows all origins
                   allow_credentials=True,
                   allow_methods=["*"],  # Allows all methods
                   allow_headers=["*"])  # Allows all headers


class GeocodeResults(BaseModel):
    display_name: str
    confidance: float
    boundingbox: List[float]
    lat: float
    lon: float


@app.get("/")
def read_root():
    return {"Usage": "/geocoding/<text>"}


@app.get("/geocoding", response_model=List[GeocodeResults])
def geocoding(text: str):

    # TODO: model inference on text:
    res = GeocodeResults(display_name= 'language model',
                         confidance = 0.76,
                         boundingbox = [34.5,34.6, 31.8, 31.7],
                         lat = 34.55,
                         lon = 31.75)

    if res is None:
        raise HTTPException(status_code=404, detail="geocoding error")

    return [res]


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=5000)
