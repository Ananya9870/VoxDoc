import os
from dotenv import load_dotenv
from groq import Groq
from dataclasses import dataclass

# Load .env at the very start of the file
load_dotenv()

@dataclass
class Agent:
    name: str
    instructions: str
    model: str = "llama-3.3-70b-versatile"

class Runner:
    @staticmethod
    async def run(agent: Agent, user_input: str) -> 'RunResult':
        # Ensure the key is fetched from the environment
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Check your .env file.")
            
        client = Groq(api_key=api_key)
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": agent.instructions},
                {"role": "user", "content": user_input},
            ],
            model=agent.model,
            temperature=0.5,
        )
        
        return RunResult(final_output=chat_completion.choices[0].message.content)

@dataclass
class RunResult:
    final_output: str