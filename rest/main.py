from typing import Union, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class Point(BaseModel):
    lat: float = 0.0
    lon: float = 0.0


class BBoxDegrees(BaseModel):
    bbox: List[Point]


app = FastAPI()


@app.get("/")
def read_root():
    return {"Usage": "/geocoding/<text>"}


@app.get("/geocoding/{text}", response_model=BBoxDegrees)
def geocoding(text: str):
    # TODO: model inference on text:
    _bbox = [Point(lat=34.5, lon=31.8),
            Point(lat=34.6, lon=31.8),
            Point(lat=34.6, lon=31.7),
            Point(lat=34.5, lon=31.7),
            Point(lat=34.5, lon=31.8)]

    res = BBoxDegrees(bbox=_bbox)

    if res is None:
        raise HTTPException(status_code=404, detail="geocoding error")

    print(res)

    return res
