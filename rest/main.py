import sys
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from model.inference import load_model, inference

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
    bbox = inference(tokenizer=tokenizer, model=model, sentence=text)
    print (f'geocoding: {text}\nbbox={bbox}')
    res = GeocodeResults(display_name= text,
                         confidance = 0.76,
                         boundingbox = bbox)

    if res is None:
        raise HTTPException(status_code=404, detail="geocoding error")

    return [res]


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=5000)
