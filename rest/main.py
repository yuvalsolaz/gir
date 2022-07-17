from typing import Union, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

Point(BaseModel):
lat: float = 0.0
lon: float = 0.0


class BBoxDegrees(BaseModel):
    rect: List[Point]


app = FastAPI()


@app.get("/")
def read_root():
    return {"Usage": "/geocoding/<text>"}


@app.get("/geocoding/{text}", response_model=BBoxDegrees)
def geocoding(text: str):

    # TODO: model inference:
    bbox = [Point(34.5,31.8), Point(34.6,31.8), Point(34.6,31.7), Point(34.5,31.7),Point(34.5,31.8)]
    if bbox is None:
        raise HTTPException(status_code=404, detail="geocoding error")
    return bbox