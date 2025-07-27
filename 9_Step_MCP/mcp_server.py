import json
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MCP Server!")

@mcp.tool()
def add(a:int, b:int) -> dict:
    return {"result": f"Sum id {a + b}"}

@mcp.tool()
def weather(city: str) -> dict:
    try:
        res = requests.get(f"https://wttr.in/{city}?format=%C+%t")
        if res.status_code == 200:
            return {"result": f"Weather in {city}: {res.text}"}
        else:
            return {"error": "Failed to get the weather"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run()