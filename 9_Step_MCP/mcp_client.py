from mcp import RemoteRunnableCallable

# Connect to your MCP server
remote = RemoteRunnableCallable("http://localhost:8000/mcp")

# View available tools
print("Tools Available:", list(remote.tools.keys()))

# Call the 'add' tool
add_result = remote.tools["add"].invoke({"a": 5, "b": 7})
print("Add Result:", add_result)

# Call the 'weather' tool
weather_result = remote.tools["weather"].invoke({"city": "Mumbai"})
print("Weather Result:", weather_result)
