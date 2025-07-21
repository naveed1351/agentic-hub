import asyncio  
import os  
from semantic_kernel.agents import ChatCompletionAgent  
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion  
from dotenv import load_dotenv  
  
load_dotenv()  # Load environment variables from .env file   
  
async def main():  
    service = AzureChatCompletion(  
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],               # your model endpoint from .env file 
        api_key=os.environ["AZURE_OPENAI_API_KEY"],  # your Azure openAI API key from .env file
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15"),  
        deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]  # your model deployment name from .env file 
    )  
  
    agent = ChatCompletionAgent(  
        service=service,  
        name="SK-Assistant",  
        instructions="You are a helpful assistant."  
    )  
  
    response = await agent.get_response(messages="Write a haiku about Semantic Kernel.")  
    print(response.content)  
  
asyncio.run(main())  