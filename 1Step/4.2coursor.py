import os
import google.generativeai as genai
from dotenv import load_dotenv
import json

load_dotenv()

gen_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key = gen_key)

def exec_command(cmd):
    commands = cmd.split('\n')
    for command in commands:
        command = command.strip()
        if command:
            os.system(command)   
         
system_prompt = """
    You are an helpful Ai Assistant who performs command prompt execution.
    Your work in steps which are "analyse", "think", "validate", "action", "result".
    
    Rules:
    - Follow string Json Format.
    - Perform one step at a time.
    - Carefully try to understand what the user wants to do
    - Write only the commands that are safe to execute.
    - When writing code nto file, especially multi-line Python code, use the following format:
        echo line 1 > filename.py
        echo line 2 >> filename.py
        echo line 3 >> filename.py 
    - Escape double quotes inside strings using \", like: input(\"Enter Number:\")
    - Ensure correct code.
    
    Function Name:
    execute_cmd(cmd) = this function takes a Windows command prompt and execute it.
    
    Output Format:
    {
         "step": "string",
        "content": "string",
        "function": "The name of function if the step is action",
        "input": "The input parameter for the function"
    } 
    
    If the doubt is Non-coding or irrelevat to coding then 
    nput: Why is the sky blue?
    Output: { "step": "result", "content": "Bruv? I only solve programming queries!" }
"""

model = genai.GenerativeModel(
    'gemini-2.0-flash',
    system_instruction = system_prompt
)

chat = model.start_chat()

while True:
    try:
        user_prompt = input(">>> ")
        
        if user_prompt.lower() in ["exit", "quit"]:
            print("Exiting Chat Assisstant.")
            break
    
        while True:
            response = chat.send_message(user_prompt)
            
            try:
                res_text = response.candidates[0].content.parts[0].text
                res_text = res_text.strip('`').strip()
                if res_text.startswith('json'):
                    res_text = res_text[4:].strip()
                step_res = json.loads(res_text)
            except json.JSONDecodeError:
                print("Failed to Parse Json ", res_text)
                break
            
            print(f"🧠: {step_res.get('content', '')}")
            
            if step_res.get("step") == "action":
                command = step_res.get("input")
                print(f"⚒️: Executing command: {command}")
                exec_command(command)
            elif step_res.get("step") == "result":
                print(f"🤖: {step_res.get('content', '')}")
                break
            else:
                user_prompt = "Next Step please."
    except Exception as e:
        print("Error ",e)