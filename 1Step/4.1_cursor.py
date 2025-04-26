from dotenv import load_dotenv
import google.generativeai as genai
import os
import json
import requests

load_dotenv()

gen_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key = gen_key)


def get_weather (city: str):
    try:
        url = f"https://wttr.in/{city}?format=%C+%t"
        response = requests.get(url)
        print(f"API Response: {response.text}")
        if response.status_code == 200:
            return f'The weather in {city} is {response.text}.'
        return f"Error: Could not fetch weather for {city}"
    except Exception as e:
        return f"API Error: {str(e)}"
    
available_tools = {
    'get_weather': {
        'fn': get_weather,
        'description': 'Takes a city name as an input and returns the current weather for the city'
    }
}
system_prompt = """
    You are an helpfull AI Assisstant whio is specialized in resolving any users query.
    You work on start, plan, action, observe modes.
    For the given user query and available tools, plan the step by step execution, based on the planning,
    select the relevant tool from the available tool, and based on the tool selection you performs an action to call the tool.
    Wait for the Observatton and based on the observation from the tool call the resolve query.
    
    Some Rules To Follow:
    - Always PLAN first.
    - Then perform ACTION by calling the correct function.
    - Then WAIT for OBSERVATION (the output of function).
    - Only AFTER observation, you can OUTPUT the final answer.
    - You MUST NOT assume results. Always call the tool before answering.
    - Follow STRICT JSON format.
    - Never skip ACTION or OBSERVE steps.

    
    Output JSON Format:
    {{
        'step': 'string',
        'content': 'string',
        'function': 'The name of the function you called'
    }}
    
    Available_tools:
    - get_weather: Takes a city name as an input and returns the current weather for the city
    
    If a tool is available to resolve the query, it should be called directly, even if the model does not have direct access to real-time data.

"""

model = genai.GenerativeModel(
    'gemini-2.0-flash',
    system_instruction = system_prompt
)

chat = model.start_chat()

while True:
    try:
        user_input = input(">>> ")
        
        if user_input.lower() in ['exit', 'quit']:
            print('Exiting the Chat Assisstant')
            break
        
        while True:
            response = chat.send_message(user_input)
            
            try:
                response_json = json.loads(response.text)
                step = response_json.get('step', '').lower()
                print(f"step: {response_json['step']}")
                print(f"Content: {response_json['content']}")
                
                if step == 'action':
                    functaion_name = response_json['function']
                    input_value = response_json['input']
                    if functaion_name in available_tools:
                        function_call = available_tools[functaion_name]['fn']
                        observation = function_call(input_value)
                        
                        user_input = json.dumps({
                                'step': 'observe',
                                'output': observation.strip()
                            })
                        continue
                elif step == 'output':
                    print("Final Output:", response_json['content'])
                    break 
                else:
                    user_input = 'continue'            
                    
            except Exception as json_err:
                print('Failed to parse JSON from model response.')
                print('Raw Response ',response.text)
                break
            
    except Exception as e:
        print('Error ',e)