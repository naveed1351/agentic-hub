import asyncio  
import os  
from semantic_kernel.agents import ChatCompletionAgent  
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion  
from dotenv import load_dotenv  
  
load_dotenv()  # Load environment variables from .env file   
  
async def main():  
    service = AzureChatCompletion(  
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],               # Correct parameter  
        api_key=os.environ["AZURE_OPENAI_API_KEY"],  
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15"),  
        deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]  # Correct parameter  
    )  
  
    agent = ChatCompletionAgent(  
        service=service,  
        name="SK-Assistant",  
        instructions="You are a helpful assistant."  
    )  
  
    response = await agent.get_response(messages="Write a haiku about Semantic Kernel.")  
    print(response.content)  
  
asyncio.run(main())  