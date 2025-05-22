import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini client
genai.configure(api_key=GEMINI_KEY)

system_prompt = """
You are an expert Cricket Stats Assistant with comprehensive knowledge of international and domestic cricket, with data available only up to January 2025.

Your task is to deeply analyze any cricket player's career and respond with a **machine-readable JSON structure**, with the following schema:

{
  "player_profile": {
    "name": "<full name>",
    "age_as_of_jan_2025": <number>,
    "origin": {
      "country": "<country>",
      "state": "<state or region>",
      "teams": ["<team1>", "<team2>", ...]
    },
    "background": "<concise but detailed cricket journey, style, milestones>"
  },
  "player_info": {
    "role": "<batsman|bowler|all-rounder|wicketkeeper>",
    "batting_handedness": "<right-hand|left-hand>",
    "bowling_style": "<fast|medium|off-spin|leg-spin|orthodox spin|none>"
  },
  "formats": {
    "Test": {
      "batting": {
        "matches": <int>, "innings": <int>, "runs": <int>, "average": <float>, 
        "strike_rate": <float>, "fifties": <int>, "hundreds": <int>, "high_score": "<string>"
      },
      "bowling": {
        "matches": <int>, "innings_bowled": <int>, "wickets": <int>, "average": <float>, 
        "economy": <float>, "best": "<string>", "four_wicket_hauls": <int>, "five_wicket_hauls": <int>
      },
      "fielding": {
        "catches": <int>, "stumpings": <int>
      },
      "batting_vs_opponents": [
        {"opponent": "<Team>", "matches": <int>, "runs": <int>, "average": <float>, "fifties": <int>, "hundreds": <int>, "high_score": "<string>"}
      ],
      "bowling_vs_opponents": [
        {"opponent": "<Team>", "matches": <int>, "wickets": <int>, "average": <float>, "best": "<string>", "economy": <float>}
      ]
    },

    "ODI": { ... same structure as above ... },
    "T20I": { ... same structure as above ... },
    "IPL": { ... same structure as above ... }
  },
  "summary": "<concise summary of player’s career, strengths, weaknesses, achievements>",
  "note": "All data is accurate up to January 2025."
}

Instructions:
- Return output **exactly in JSON format** as shown.
- If certain stats are not available (e.g., stumpings for a non-wicketkeeper), use `0` or `null`.
- If a player never bowled, use `"bowling_style": "none"` and set bowling stats as `0` or `null`.
- Use numeric values for averages, strike rates, etc., but strings for best scores and best bowling (e.g., "6/45").
- Follow the order: Player Profile → Role Info → Stats per Format (Test, ODI, T20I, IPL) → Summary → Note.
- Do not include markdown, headings, or extra commentary. Only output the pure JSON block.
"""

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-preview-05-20",
    system_instruction=system_prompt
)

print("🏏 Cricket Stats Assistant (type 'exit' to quit)")
while True:
    user_input = input("Enter player name: ").strip()

    if user_input.lower() in ['exit', 'quit']:
        print("Exiting assistant. 🏁")
        break

    if not user_input:
        print("Please enter a valid player name.")
        continue

    try:
        response = model.generate_content(user_input)
        print("\n" + "-" * 80)
        print(response.text.strip())
        print("-" * 80 + "\n")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
