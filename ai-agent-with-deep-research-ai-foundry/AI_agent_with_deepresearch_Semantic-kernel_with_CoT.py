"""
deepresearch_sk_stream_and_summarize.py

- Streams a Deep Research agent run from Azure AI Foundry (prints CoT deltas).
- Collects the final agent response.
- Uses Semantic Kernel + AzureChatCompletion to generate a structured report from the final response.

Notes:
- Requires environment variables (see top of file).
- Uses Foundry's streaming API (agents.runs.stream) which is robust for long jobs.
- After streaming completes, the final response is post-processed by Semantic Kernel.
"""

import os
import inspect
from dotenv import load_dotenv

# Azure Foundry SDK (sync pattern used below)
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import DeepResearchTool, AgentEventHandler, MessageRole

# Semantic Kernel
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

load_dotenv()

# -------------------------
# Config / env helpers
# -------------------------
def require_env(key: str) -> str:
    v = os.environ.get(key)
    if not v:
        raise EnvironmentError(f"Missing environment variable: {key}")
    return v

PROJECT_ENDPOINT = require_env("AZURE_AI_AGENT_ENDPOINT")
DEPLOYMENT_MODEL = require_env("AZURE_OPENAI_DEPLOYMENT_NAME")
DEEP_RESEARCH_MODEL = require_env("AZURE_DEEP_RESEARCH_DEPLOYMENT_NAME")
BING_CONN_NAME = require_env("AZURE_BING_CONNECTION_NAME")

# Semantic Kernel config
SK_AZURE_OPENAI_ENDPOINT = require_env("AZURE_OPENAI_ENDPOINT")
SK_AZURE_OPENAI_KEY = require_env("AZURE_OPENAI_API_KEY")
SK_AZURE_OPENAI_DEPLOYMENT = require_env("AZURE_OPENAI_DEPLOYMENT_NAME")

# -------------------------
# Event handler: prints CoT deltas
# -------------------------
class CoTEventHandler(AgentEventHandler):
    """Event handler for Foundry streaming that prints Chain-of-Thought style events."""

    def on_message_delta(self, delta):
        # delta.content may carry small text chunks as they stream
        # many SDKs use delta.content.text.value shape
        content = getattr(delta, "content", None)
        if content is not None:
            txt = None
            # try common shapes
            if hasattr(content, "text"):
                t = getattr(content, "text")
                if hasattr(t, "value"):
                    txt = t.value
                elif isinstance(t, str):
                    txt = t
            # fallback: print repr
            if txt:
                print(txt, end="", flush=True)
            else:
                print("[delta]", content)

    def on_run_step(self, step):
        # Called when a step (tool call / function) starts/updates
        print(f"\n\n[Tool step -> type: {getattr(step, 'type', '<unknown>')}]")
        step_details = getattr(step, "step_details", None)
        if step_details and hasattr(step_details, "tool_calls"):
            for call in step_details.tool_calls:
                label = getattr(call, "tool_label", getattr(call, "tool_name", "<tool>"))
                status = getattr(call, "status", "<status>")
                print(f" → {label}: {status}")

    def on_done(self):
        print("\n\n✅ Agent run complete.\n")

# -------------------------
# Helper: extract final text from a response message object
# -------------------------
def extract_final_text(response_message) -> str:
    # Many SDKs expose 'text_messages' where each has .text.value
    text_msgs = getattr(response_message, "text_messages", None)
    if text_msgs:
        parts = []
        for tm in text_msgs:
            t = getattr(tm, "text", None)
            if t is None:
                continue
            if hasattr(t, "value"):
                parts.append(str(t.value))
            elif isinstance(t, str):
                parts.append(t)
        return "\n\n".join(parts)
    # Fallback: try .content or str()
    content = getattr(response_message, "content", None)
    if content:
        return str(content)
    return str(response_message)

# -------------------------
# Main flow
# -------------------------
def main():
    cred = DefaultAzureCredential()

    # Use sync Foundry client (the streaming examples use a sync client & context managers)
    with AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=cred) as project_client:

        # 1) Resolve Bing connection by name -> get id
        print("[info] Resolving Bing connection name -> id...")
        conn = project_client.connections.get(name=BING_CONN_NAME)
        bing_conn_id = conn.id
        print(f"[info] Resolved Bing connection id: {bing_conn_id}")

        # 2) Build DeepResearchTool for the agent
        deep_tool = DeepResearchTool(
            bing_grounding_connection_id=bing_conn_id,
            deep_research_model=DEEP_RESEARCH_MODEL,
        )

        # 3) Create agent in Foundry (or reuse an existing agent)
        print("[info] Creating agent definition in Foundry...")
        agent = project_client.agents.create_agent(
            model=DEPLOYMENT_MODEL,
            name="DeepResearchAgent-SK-Streaming",
            instructions="You are a careful research assistant. Think step-by-step and show intermediate tool calls/results.",
            tools=deep_tool.definitions,
            tool_resources=deep_tool.resources,
        )
        print(f"[info] Agent id: {getattr(agent, 'id', '<no-id>')}")

        # 4) Create a thread and a message (user prompt)
        thread = project_client.agents.threads.create()
        print(f"[info] Created thread id: {thread.id}")

        message = project_client.agents.messages.create(
            thread_id=thread.id,
            role="user",
            content=(
                "Research recent breakthroughs in quantum computing and the key researchers involved. "
                "Provide short citations and a concise 3-bullet summary at the end."
            ),
        )
        print(f"[info] Created message id: {message.id}")

        # 5) Stream the agent run with CoT event handler (blocks until done)
        print("[info] Starting streaming run (this may take a little while)...\n")
        handler = CoTEventHandler()
        # the streaming context manager will call handler.on_message_delta, on_run_step, on_done as events arrive
        with project_client.agents.runs.stream(thread_id=thread.id, agent_id=agent.id, event_handler=handler) as stream:
            stream.until_done()  # waits until the run completes and streams events

        # 6) Fetch the final agent message and extract text
        response_message = project_client.agents.messages.get_last_message_by_role(thread_id=thread.id, role=MessageRole.AGENT)
        if not response_message:
            raise RuntimeError("Agent did not produce a final message in the thread.")
        final_text = extract_final_text(response_message)
        print("\n\n[info] Final agent response captured (len=%d chars)\n" % len(final_text))

        # Optional: show URL citations if available
        for ann in getattr(response_message, "url_citation_annotations", []) or []:
            uc = getattr(ann, "url_citation", None)
            if uc:
                print(f"[citation] {getattr(uc, 'title', '<title>')} -> {getattr(uc, 'url', '<url>')}")

    # -------------------------
    # 7) Post-process with Semantic Kernel (structured report)
    # -------------------------
    print("\n[info] Now invoking Semantic Kernel to create a structured report from the final agent output...\n")

    # Create Kernel and add Azure OpenAI chat completion service (AzureChatCompletion)
    kernel = Kernel()
    # Create an AzureChatCompletion instance and add to kernel
    sk_chat = AzureChatCompletion(
        service_id="foundry-sk-chat",
        api_key=SK_AZURE_OPENAI_KEY,
        deployment_name=SK_AZURE_OPENAI_DEPLOYMENT,
        endpoint=SK_AZURE_OPENAI_ENDPOINT,
    )
    kernel.add_service(sk_chat)

    # Prepare a Semantic Kernel prompt to turn the raw final_text into a structured report
    prompt_template = """
You are an expert technical writer. Given the agent's final research output below, produce:
1) A short executive summary (3 sentences).
2) Key findings (3-6 bullets).
3) Short list of the most-cited sources (if citations are inline, extract them).
4) A 2-sentence final recommendation.

=== Agent output start ===
{agent_output}
=== Agent output end ===
"""
    # Create a simple semantic function on the kernel
    from semantic_kernel.semantic_functions import SemanticFunctionConfig
    from semantic_kernel.orchestration import Plan

    # Some SK versions provide create_semantic_function; use the generic add_service + run text completion if needed.
    # For compatibility, we'll use the chat completion service directly via kernel.run
    # Build input and call the chat service through kernel
    prompt = prompt_template.format(agent_output=final_text)

    # Call the chat completion service (non-streaming) to create the structured report
    # The exact API to invoke SK's chat completion may vary by SK version; the following is compatible with current SK patterns:
    result = kernel.run(prompt, max_tokens=800)  # kernel.run uses registered chat completion service

    report_text = str(result) if result is not None else "<no-result>"
    print("\n=== Structured Report from Semantic Kernel ===\n")
    print(report_text)

    print("\nDone.")

if __name__ == "__main__":
    main()
