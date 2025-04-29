from dotenv import load_dotenv
import os
import google.generativeai as genai
import json

load_dotenv()

gen_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key = gen_key)

system_prompt = """
    You are an AI Assistant which solves users problem by breaking down their problem and then resolving that query.
    
    For the given user input, analyse the input and break down the problem step by step.
    
    Atleast think in 5-6 steps on how to solve the problem before solving the problem.
    
    The steps are you get user input, you analyse, you think, you again think for several times and then return and output with explanation and then finally You validate the output as well before giving final result. 
    
    Follow the steps in sequence that is 'analyse', 'think' , 'output', 'validate'  and 'result'.
    
    Instructions:
    1. Follow the Strict Json Output as per output Schema.
    2. Always perform one step at a time and wait for next input.
    3. Carefully analyse the user query.
    
    Outfut Format:
    {{  step: "String",
        content: "String"    
    }}
    
    Example:
    Input: What is 2 + 2.
    Output: {{ step: "analyse", content: "Alright! The user is intersted in maths query and he is asking a basic arthermatic operation" }}
    Output: {{ step: "think", content: "To perform the addition i must go from left to right and add all the operands" }}
    Output: {{ step: "output", content: "4" }}
    Output: {{ step: "validate", content: "seems like 4 is correct ans for 2 + 2" }}
    Output: {{ step: "result", content: "2 + 2 = 4 and that is calculated by adding all numbers" }}

"""
model = genai.GenerativeModel(
            'gemini-1.5-flash',
            system_instruction = system_prompt
        )

chat = model.start_chat()

while True:
    try:
        user_input = input(">>> ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting assistant.")
            break
        
        while True:
            response = chat.send_message(user_input)
            try:
                response_json = json.loads(response.text)
                step = response_json.get('step', '').lower()
                print(f'Step: {response_json["step"]}')
                print(f'Content: {response_json["content"]}')
                
                if step == 'result':
                    break
                
                user_input='continue'
                
            except Exception as json_err:
                print("Failed to parse JSON from model response.")
                print("Raw Response:", response.text)
                break
    except Exception as e:
        print("Error:", e)