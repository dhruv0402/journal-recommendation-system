from groq import Groq
import os

client = Groq(api_key="YOUR_GROQ_API_KEY")

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Say hello in JSON"}],
    temperature=0,
)

print(response.choices[0].message.content)
