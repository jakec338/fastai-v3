import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from datauri import DataURI
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

#export_file_url = 'https://www.dropbox.com/s/6bgq8t6yextloqp/export.pkl?raw=1'
export_file_url = 'https://www.dropbox.com/s/gfeg4tknlagf2gi/export_v3.pkl?dl=1'
export_file_name = 'export.pkl'

#classes = ['black', 'grizzly', 'teddys']

classes = ['apples', 'bananas', 'broccoli', 'leeks', 'onions', 'oranges', 'potatos', 'tomatos']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


def get_ghg_data(class_pred):
    apple_string = "An apple a day for a year is equivalent to  driving a regular petrol car 32 miles or heating an average British home for 2 days"
    if(class_pred == "apples"):
        return apple_string
    else:
        return "balls"
    
    
def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    pred_class,pred_idx,outputs = learn.predict(img)
    print(outputs)
    class_string = pred_class.obj
    ghg_info = get_ghg_data(class_string)
    #return PlainTextResponse('Class: '+ class_string + ". \n" + ghg_info)

    return JSONResponse({ 'classification': class_string, 'ghg': ghg_info })

@app.route('/return_image', methods=['POST'])
async def return_image(request):
        res = await request.body()
        res = res.decode("utf-8")
        uri = DataURI(res)
        print(uri.data)
        return predict_image_from_bytes(uri.data)


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
