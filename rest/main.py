import sys
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from model.inference import load_model, inference
from model.geolabel import cell2geo

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],  # Allows all origins
                   allow_credentials=True,
                   allow_methods=["*"],  # Allows all methods
                   allow_headers=["*"])  # Allows all headers


@app.get("/")
def read_root():
    return {"Usage": "/geocoding/<text>"}


class GeocodeResults(BaseModel):
    display_name: str
    confidance: float
    boundingbox: List[float]

checkpoint = r'/home/yuvalso/repository/gir/seq2seq/checkpoint-2100000/'
tokenizer, model = load_model(checkpoint=checkpoint)

@app.get("/geocoding", response_model=List[GeocodeResults])
def geocoding(text: str):

    # model inference on text:
    cellid, score = inference(tokenizer=tokenizer, model=model, sentence=text)
    rect, area = cell2geo(cellid)
    if not rect:
        raise HTTPException(status_code=404, detail=f'inference error for {text}')
    ##            0   1    2     3     4     5     6     7     8     9
    ## rect = [x_lo, y_hi, x_hi, y_hi, x_hi, y_lo, x_lo, y_lo, x_lo, y_hi]    [31.81, 31.82, 35.52, 35.53])
    # bbox = y_lo,y_hi, x_lo , x_hi
    bbox = [rect[0], rect[2],rect[5], rect[1]]
    print (f'geocoding: {text} cell={cellid} level={len(cellid)} area={area:.2f} score={score:.2f}\nbbox={bbox}')
    res = GeocodeResults(display_name= text,
                         confidance = score,
                         boundingbox = bbox)

    if res is None:
        raise HTTPException(status_code=404, detail="geocoding error")

    return [res]


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)
