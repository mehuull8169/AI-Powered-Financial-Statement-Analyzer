from google import genai

client = genai.Client(api_key="AIzaSyDBuvhFaVyP-kMQpWEpBV_YT_BuO8SSt9A")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Hello"
)

print(response.text)