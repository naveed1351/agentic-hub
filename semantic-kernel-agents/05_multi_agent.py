import os  
import asyncio  
from dotenv import load_dotenv  
  
from semantic_kernel import Kernel  
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread  
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion  
from semantic_kernel.filters import FunctionInvocationContext  
  
# 1. Load environment variables from .env  
load_dotenv()  
  
# 2. Assert required variables are present  
REQUIRED = [  
    "AZURE_OPENAI_API_KEY",  
    "AZURE_OPENAI_ENDPOINT",  
    "AZURE_OPENAI_DEPLOYMENT_NAME"  
]  
for var in REQUIRED:  
    if var not in os.environ:  
        raise EnvironmentError(f"Missing {var} in environment or .env file.")  

#  Set up Azure OpenAI service parameters  
AZURE_KEY = os.environ["AZURE_OPENAI_API_KEY"]  
AZURE_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]  
AZURE_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]  
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")  
  
# 3. Create the AzureChatCompletion service  
azure_service = AzureChatCompletion(  
    endpoint=AZURE_ENDPOINT,  
    api_key=AZURE_KEY,  
    deployment_name=AZURE_DEPLOYMENT,  
    api_version=AZURE_API_VERSION  
)  
  
# 4. Kernel and filter definition  
kernel = Kernel()  
  
async def function_invocation_filter(context: FunctionInvocationContext, next):  
    if "messages" not in context.arguments:  
        await next(context)  
        return  
    print(f"    Agent [{context.function.name}] called with messages: {context.arguments['messages']}")  
    await next(context)  
    print(f"    Response from agent [{context.function.name}]: {context.result.value}")  
  
kernel.add_filter("function_invocation", function_invocation_filter)  
  
# 5. Define the billing agents  
billing_agent = ChatCompletionAgent(  
    service=azure_service,  
    name="BillingAgent",  
    instructions=(  
        "You specialize in handling customer questions related to billing issues. "  
        "This includes clarifying invoice charges, payment methods, billing cycles, "  
        "explaining fees, addressing discrepancies in billed amounts, updating payment details, "  
        "assisting with subscription changes, and resolving payment failures. "  
        "Your goal is to clearly communicate and resolve issues specifically about payments and charges."  
    ),  
)  
  # Define the refund agent
refund_agent = ChatCompletionAgent(  
    service=azure_service,  
    name="RefundAgent",  
    instructions=(  
        "You specialize in addressing customer inquiries regarding refunds. "  
        "This includes evaluating eligibility for refunds, explaining refund policies, "  
        "processing refund requests, providing status updates on refunds, handling complaints related to refunds, "  
        "and guiding customers through the refund claim process. "  
        "Your goal is to assist users clearly and empathetically to successfully resolve their refund-related concerns."  
    ),  
)  
  # define the triage agent
triage_agent = ChatCompletionAgent(  
    service=azure_service,  
    kernel=kernel,  
    name="TriageAgent",  
    instructions=(  
        "Your role is to evaluate the user's request and forward it to the appropriate agent based on the nature of "  
        "the query. Forward requests about charges, billing cycles, payment methods, fees, or payment issues to the "  
        "BillingAgent. Forward requests concerning refunds, refund eligibility, refund policies, or the status of "  
        "refunds to the RefundAgent. Your goal is accurate identification of the appropriate specialist to ensure the "  
        "user receives targeted assistance."  
    ),  
    plugins=[billing_agent, refund_agent],  
)  
  
thread: ChatHistoryAgentThread = None  
  
# 6. User interaction/chat routine  
async def chat() -> bool:  
    try:  
        user_input = input("User:> ")  
    except (KeyboardInterrupt, EOFError):  
        print("\n\nExiting chat...")  
        return False  
  
    if user_input.lower().strip() == "exit":  
        print("\n\nExiting chat...")  
        return False  
  
    response = await triage_agent.get_response(  
        messages=user_input,  
        thread=thread,  
    )  
    if response:  
        print(f"Agent :> {response}")  
    return True  
"""
Sample Output:

User:> I was charged twice for my subscription last month, can I get one of those payments refunded?
    Agent [BillingAgent] called with messages: I was charged twice for my subscription last month.
    Agent [RefundAgent] called with messages: Can I get one of those payments refunded?
    Response from agent RefundAgent: Of course, I'll be happy to help you with your refund inquiry. Could you please 
        provide a bit more detail about the specific payment you are referring to? For instance, the item or service 
        purchased, the transaction date, and the reason why you're seeking a refund? This will help me understand your 
        situation better and provide you with accurate guidance regarding our refund policy and process.
        Response from agent BillingAgent: I'm sorry to hear about the duplicate charge. To resolve this issue, could 
        you please provide the following details:

1. The date(s) of the transaction(s).
2. The last four digits of the card used for the transaction or any other payment method details.
3. The subscription plan you are on.

Once I have this information, I can look into the charges and help facilitate a refund for the duplicate transaction. 
Let me know if you have any questions in the meantime!

Agent :> To address your concern about being charged twice and seeking a refund for one of those payments, please 
    provide the following information:

1. **Duplicate Charge Details**: Please share the date(s) of the transaction(s), the last four digits of the card used 
    or details of any other payment method, and the subscription plan you are on. This information will help us verify 
    the duplicate charge and assist you with a refund.

2. **Refund Inquiry Details**: Please specify the transaction date, the item or service related to the payment you want 
    refunded, and the reason why you're seeking a refund. This will allow us to provide accurate guidance concerning 
    our refund policy and process.

Once we have these details, we can proceed with resolving the duplicate charge and consider your refund request. If you 
have any more questions, feel free to ask!
"""
  
async def main() -> None:  
    print("Welcome to the chat bot!\n  Type 'exit' to exit.\n  Try to get some billing or refund help.")  
    chatting = True  
    while chatting:  
        chatting = await chat()  
  
if __name__ == "__main__":  
    asyncio.run(main())  

    