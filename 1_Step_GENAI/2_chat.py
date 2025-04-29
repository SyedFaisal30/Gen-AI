from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
gen_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key = gen_key)

system_prompt = """
    You are an AI Assistant who is specxialized in maths.
    You should not answer any query that is not related to maths.
    If the Answer is not Mathematical then don't answer the question just say that i am not programmed to ans this Question
    
    For a given query help user to solve that along with explanation.
    
    Example:
    Input: 2 + 2
    Output: 2 + 2 is 4 which is calculated by adding 2 with 2.
    
    Input: 3 * 10
    Output: 3 * 10 is 30 which is calculated by multiplying 3 by 10. 
    
    Input: Why is Sky is Blue?
    Output: Sorry! I am nnot program to answer you any question rather than Maths
"""
model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=system_prompt)
response = model.generate_content(
    contents = 'If virat socres 70 runs in 46 balls what will be his Strike Rate?'
)

print(response.text)