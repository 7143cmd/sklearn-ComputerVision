from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import joblib
from generate.generate import predict_color
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, "preview.png")

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return JSONResponse({"preview_url": f"/{file_location}"})


@app.post("/censored")
async def process_result():
    try:
        model = joblib.load('color_clasifff.pkl')
            
        result = predict_color(model, path='static/uploads/preview.png')[0]
        print(f"Result: {result}")
        return JSONResponse({"result": result})
    except Exception as e:
        result: str = str(e)
        return JSONResponse({"result": result})



if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
