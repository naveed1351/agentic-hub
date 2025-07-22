import asyncio  
import os  
from typing import Annotated  
from pydantic import BaseModel  
from semantic_kernel.agents import ChatCompletionAgent  
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatPromptExecutionSettings  
from semantic_kernel.functions import kernel_function, KernelArguments  
from dotenv import load_dotenv  
  
load_dotenv()  # Make sure environment variables are loaded  
  
# Plugin class  
class MenuPlugin:  
    @kernel_function(description="Provides a list of specials from the menu.")  
    def get_specials(self) -> Annotated[str, "Returns the specials from the menu."]:  
        return (  
            "Special Soup: Clam Chowder\n"  
            "Special Salad: Cobb Salad\n"  
            "Special Drink: Chai Tea"  
        )  
  
    @kernel_function(description="Provides the price of the requested menu item.")  
    def get_item_price(  
        self,  
        menu_item: Annotated[str, "The name of the menu item."]  
    ) -> Annotated[str, "Returns the price of the menu item."]:  
        # You can customize this to look up different prices  
        return "$9.99"  
  
# Structured response model  
class MenuItem(BaseModel):  
    price: float  
    name: str  
  
async def main():  
    # Configure structured output format  
    settings = OpenAIChatPromptExecutionSettings()  
    settings.response_format = MenuItem  
  
    # Provide the necessary AzureChatCompletion params  
    service = AzureChatCompletion(  
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],  
        api_key=os.environ["AZURE_OPENAI_API_KEY"],  
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15"),  
        deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]  
    )  
  
    # Create agent with plugin and settings  
    agent = ChatCompletionAgent(  
        service=service,  
        name="SK-Assistant",  
        instructions="You are a helpful assistant.",  
        plugins=[MenuPlugin()],  
        arguments=KernelArguments(settings)  
    )  
      
    response = await agent.get_response(messages="What is the price of the soup special?")  
    print(response.content)  
  
asyncio.run(main())  