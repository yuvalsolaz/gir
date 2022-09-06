import sys
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from model.inference import load_model, inference
from model.geolabel import cell2geo, level2geo

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
    levels_bbox: List[List[float]]

checkpoint = r'/home/yuvalso/repository/gir/seq2seq/checkpoint-2100000/'
tokenizer, model = load_model(checkpoint=checkpoint)

@app.get("/geocoding", response_model=List[GeocodeResults])
def geocoding(text: str):

    # model inference on text:
    cellid, score = inference(tokenizer=tokenizer, model=model, sentence=text)
    rects, area = cell2geo(cellid)
    # rects, area = level2geo(min_level=2,max_level=3)
    if not rects:
        raise HTTPException(status_code=404, detail=f'inference error for {text}')
    #            0   1    2     3     4     5     6     7     8     9
    # rect = [x_lo, y_hi, x_hi, y_hi, x_hi, y_lo, x_lo, y_lo, x_lo, y_hi]    [31.81, 31.82, 35.52, 35.53])
    # bbox = y_lo,y_hi, x_lo , x_hi

    def get_bbox(r):
        ax = [r[0], r[2], r[4], r[6]]
        ay = [r[1], r[3], r[5], r[7]]
        xmax = max(ax)
        xmin = min(ax)
        ymax = max(ay)
        ymin = min(ay)
        bbox = [xmin, xmax, ymin, ymax]  # [rect[0], rect[2], rect[5], rect[1]]
        return bbox


    levels_bbox = []
    for rect in rects:
        levels_bbox.append(get_bbox(rect))
    bbox = get_bbox(rects[0])
    print (f'geocoding: {text} cell={cellid} level={len(cellid)} area={area:.2f} score={score:.2f}\nbbox={bbox}')
    res = GeocodeResults(display_name= text,
                         confidance = score,
                         boundingbox = bbox,
                         levels_bbox = levels_bbox)
    return [res]


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)
