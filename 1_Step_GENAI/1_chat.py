from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
gen_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key = gen_key)

model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content(
    contents = 'If virat socres 70 runs in 46 balls what will be his Strike Rate??'
)

print(response.text)