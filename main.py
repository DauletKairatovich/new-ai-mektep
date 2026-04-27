import os
import io
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
    request=request, 
    name="index.html", 
    context={}  # Если вы передавали дополнительные данные, оставьте их тут
)

@app.post("/check")
async def check_homework(task_file: UploadFile = File(...), student_file: UploadFile = File(...)):
    try:
        task_img = Image.open(io.BytesIO(await task_file.read()))
        student_img = Image.open(io.BytesIO(await student_file.read()))

        model_id = "gemini-3-flash-preview" 
        
        # Қазақ тіліндегі нұсқаулық
        prompt = """
        Сен — тәжірибелі қазақ тіліндегі репетиторсың. Саған екі сурет берілді:
        1. Тапсырманың шарты.
        2. Оқушының орындаған жұмысы.
        
        Сенің міндетің:
        - ЖАУАПТЫ ТЕК ҚАЗАҚ ТІЛІНДЕ ЖАЗ.
        - Оқушының қателерін тап және түсіндір.
        - Есептің немесе жаттығудың дұрыс шешімін көрсет.
        - Соңында баға қойып, оқушыға жылы лебіз білдір.
        
        Жауапты Markdown форматында бер (бірақ браузерде әдемі көрінуі үшін таза жаз).
        """
        
        response = client.models.generate_content(
            model=model_id,
            contents=[prompt, task_img, student_img]
        )
        
        return {"result": response.text}

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            return {"result": "Күте тұрыңыз, сұраныс тым көп. 1 минуттан соң қайталаңыз."}
        return {"result": f"Қате: {error_msg}"}

@app.post("/predict")
async def predict_performance(request: Request):
    try:
        data = await request.json()
        grades = data.get("grades", [])
        
        if not grades:
            return {"result": "Бағалар енгізілмеген."}
            
        model_id = "gemini-3-flash-preview" 
        
        prompt = f"""
        Сен — тәжірибелі қазақ тіліндегі репетиторсың. 
        Оқушының соңғы сабақтардағы бағалары (1-10 аралығында): {grades}.
        
        Сенің міндетің:
        - ЖАУАПТЫ ТЕК ҚАЗАҚ ТІЛІНДЕ ЖАЗ.
        - Осы бағалардың динамикасына (өсу немесе төмендеу тренді) қысқаша талдау жаса.
        - Оқушының алдағы үлгерімін болжап көр.
        - Оқушыға оқуын жақсарту үшін нақты әрі мотивациялық кеңес бер.
        
        Жауапты Markdown форматында бер.
        """
        
        response = client.models.generate_content(
            model=model_id,
            contents=[prompt]
        )
        
        return {"result": response.text}

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            return {"result": "Күте тұрыңыз, сұраныс тым көп. 1 минуттан соң қайталаңыз."}
        return {"result": f"Қате: {error_msg}"}

