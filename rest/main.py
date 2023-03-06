import sys
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from model.inference import load_model, inference
from model.geolabel import get_token_rects, get_token_polygon, get_token_polygons

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
    confidence: List[float]
    boundingboxes: List[List[float]]
    levels_polygons: List[List[float]]

# checkpoint = r'/home/yuvalso/repository/gir/seq2seq/checkpoint-2100000/'
checkpoint = r'/home/yuvalso/repository/gir/model/seq2seq/checkpoint-3100000/'

tokenizer, model = load_model(checkpoint=checkpoint)

@app.get("/geocoding", response_model=List[GeocodeResults])
def geocoding(text: str):

    # model inference on text :
    cellid, scores = inference(tokenizer=tokenizer, model=model, sentence=text)
    rects, area = get_token_rects(cell_id_token=cellid)
    polygons, area = get_token_polygons(cell_id_token=cellid)
    if not rects:
        raise HTTPException(status_code=404, detail=f'inference error for {text}')

    def get_bbox(r):
        ax = [r[0], r[2], r[4], r[6]]
        ay = [r[1], r[3], r[5], r[7]]
        return [min(ax),max(ax),min(ay),max(ay)]

    bboxes = [get_bbox(rect) for rect in rects]

    def flatten(polygon): # return list(sum(polygon,())) # we need it flipped
        polygon_list = []
        for p in polygon:
            polygon_list.append(p[1])
            polygon_list.append(p[0])
        return polygon_list

    levels_polygons = []
    for poly in polygons:
         levels_polygons.append(flatten(poly))

    print (f'geocoding: {text} cell={cellid} level={len(cellid)} area={area:.2f} score={scores[0]:.2f}\nbbox={bboxes[0]}')

    res = GeocodeResults(display_name= text,
                         confidence = scores,
                         boundingboxes = bboxes,
                         levels_polygons = levels_polygons)
    return [res]


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)
