# Project settings  
project_endpoint = "your-project-endpoint"  
model = "your-model-name"  
deep_research_model = "o3-deep-research"  
bing_grounding_connection_name = "bing-with-grounding-search-connection-name"  

#import libraries
# 
# Note: I'm using Default Credentials, you can change that to managed identity by providing client ID or objet ID.  
from azure.identity import DefaultAzureCredential  
from azure.ai.projects import AIProjectClient  # Hypothetical import - please replace with the actual SDK if different  
from azure.ai.agents.models import DeepResearchTool, MessageRole
from azure.ai.agents.models import AgentEventHandler, MessageRole

#custom handler
class MyEventHandler(AgentEventHandler):
    def on_message_delta(self, delta):
        # Print streamed text as it's received
        if hasattr(delta.content, "text"):
            print(delta.content.text.value, end="", flush=True)

    def on_run_step(self, step):
        # Notify when a tool is invoked
        print(f"\n[Tool call: {step.type}]")
        if step.step_details and hasattr(step.step_details, "tool_calls"):
            for call in step.step_details.tool_calls:
                print(f" → {call.tool_label}: {call.status}")

    def on_done(self):
        # Final callback after the run completes
        print("\n✅ Agent run complete.")


with AIProjectClient(  
    endpoint=project_endpoint,  
    credential=DefaultAzureCredential(),  
) as project_client:  
  
    # Get properties of the Bing Grounding connection in your Foundry project  
    connection = project_client.connections.get(name=bing_grounding_connection_name)  
  
    # Initialize a Deep Research tool  
    deep_research_tool = DeepResearchTool(  
        bing_grounding_connection_id=connection.id,  
        deep_research_model=deep_research_model,  
    )  
  
    with project_client.agents as agents_client:  
  
        agent = agents_client.create_agent(  
            model=model,  
            name="my-deep-research-agent",  
            instructions="You are a helpful Agent that assists in researching scientific topics.",  
            tools=deep_research_tool.definitions,  
        )  
        print(f"Created agent, ID: {agent.id}")  
  
        # Create thread for communication  
        thread = agents_client.threads.create()  
        print(f"Created thread, ID: {thread.id}")  
  
        # Create a message in the thread  
        message = agents_client.messages.create(  
            thread_id=thread.id,  
            role="user",  
            content=(  
                "What are the latest trends and risks in enterprise adoption of generative AI in the healthcare industry?\n"  
                "Please focus on business use cases, regulatory challenges and ethical concerns.\n"  
                "Produce a detailed report with sections, headings, examples and a conclusion."  
            ),  
        )  
        print(f"Created message, ID: {message.id}")  
  
        # Process Agent run and invoke the event handler on every streamed event.  
        # It may take a few minutes for the agent to complete the run.  
        with agents_client.runs.stream(  
            thread_id=thread.id, agent_id=agent.id, event_handler=MyEventHandler()  
        ) as stream:  
            stream.until_done()  
  
        # Fetch the last message from the agent in the thread  
        response_message = agents_client.messages.get_last_message_by_role(  
            thread_id=thread.id, role=MessageRole.AGENT  
        )  
        if response_message:  
            for text_message in response_message.text_messages:  
                print(f"Agent response: {text_message.text.value}")  
                for annotation in response_message.url_citation_annotations:  
                    print(f"URL Citation: [{annotation.url_citation.title}]({annotation.url_citation.url})")  
  
        # Delete the Agent when done  
        agents_client.delete_agent(agent.id)  