from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from prompt import user_goal_prompt
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional, Any, Callable
import asyncio
import json
import copy

cfg = RunnableConfig(recursion_limit=100)


def initialize_model(google_api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=google_api_key
    )


def deep_sanitize_schema(obj):
    """
    Deeply sanitize schema by fixing all array types missing 'items'.
    This function creates a deep copy and modifies it recursively.
    """
    if isinstance(obj, dict):
        # Create a new dict to avoid modification issues
        result = {}
        for key, value in obj.items():
            # First, recursively sanitize the value
            sanitized_value = deep_sanitize_schema(value)
            result[key] = sanitized_value
        
        # After processing all values, check if this dict is an array type
        if result.get("type") == "array" and "items" not in result:
            result["items"] = {"type": "string"}
        
        return result
    
    elif isinstance(obj, list):
        return [deep_sanitize_schema(item) for item in obj]
    
    else:
        # Return primitive values as-is
        return obj


def convert_tool_to_gemini_format(tool):
    """
    Convert LangChain tool to Gemini-compatible format with proper schema sanitization.
    """
    try:
        tool_name = getattr(tool, 'name', 'unknown')
        
        # Get the tool's parameters/schema
        # MCP tools have args_schema as a dict already
        if hasattr(tool, 'args_schema') and tool.args_schema is not None:
            # Check if it's already a dict or needs to be converted
            if isinstance(tool.args_schema, dict):
                schema = tool.args_schema
            else:
                # It's a Pydantic model
                schema = tool.args_schema.schema()
        else:
            schema = {"type": "object", "properties": {}}

        # Create a deep copy and sanitize
        original_schema = copy.deepcopy(schema)
        sanitized_schema = deep_sanitize_schema(original_schema)

        # Update the tool's args_schema with sanitized version
        tool.args_schema = sanitized_schema

        return tool
        
    except Exception as e:
        print(f"Error converting tool {getattr(tool, 'name', 'unknown')}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return a minimal safe tool structure
        tool.args_schema = {"type": "object", "properties": {}}
        return tool


async def setup_agent_with_tools(
    google_api_key: str,
    youtube_pipedream_url: str,
    drive_pipedream_url: Optional[str] = None,
    notion_pipedream_url: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Any:
    """
    Set up the agent with YouTube (mandatory) and optional Drive or Notion tools.
    """
    try:
        if progress_callback:
            progress_callback("Setting up agent with tools... ✅")

        tools_config = {
            "youtube": {
                "url": youtube_pipedream_url,
                "transport": "streamable_http"
            }
        }

        if drive_pipedream_url:
            tools_config["drive"] = {
                "url": drive_pipedream_url,
                "transport": "streamable_http"
            }
            if progress_callback:
                progress_callback("Added Google Drive integration... ✅")

        if notion_pipedream_url:
            tools_config["notion"] = {
                "url": notion_pipedream_url,
                "transport": "streamable_http"
            }
            if progress_callback:
                progress_callback("Added Notion integration... ✅")

        if progress_callback:
            progress_callback("Initializing MCP client... ✅")
        mcp_client = MultiServerMCPClient(tools_config)

        if progress_callback:
            progress_callback("Getting available tools... ✅")
        tools = await mcp_client.get_tools()

        # Convert and sanitize all tools
        sanitized_tools = []
        for i, tool in enumerate(tools):
            # Only print debug info for tool 12 (YOUTUBE_UPLOAD_VIDEO) which is causing the issue
            if i == 12:
                print(f"\n{'='*60}")
                print(f"Processing Tool {i}: {getattr(tool, 'name', 'unknown')}")
                print(f"{'='*60}")
                
                # Print original schema
                if hasattr(tool, 'args_schema') and tool.args_schema is not None:
                    print("BEFORE sanitization:")
                    print(json.dumps(tool.args_schema, indent=2))
            
            sanitized_tool = convert_tool_to_gemini_format(tool)
            
            # Print sanitized schema for tool 12
            if i == 12:
                if hasattr(sanitized_tool, 'args_schema') and sanitized_tool.args_schema is not None:
                    print("\nAFTER sanitization:")
                    print(json.dumps(sanitized_tool.args_schema, indent=2))
            
            sanitized_tools.append(sanitized_tool)

        # Debug: Print tool schemas to verify sanitization
        if progress_callback:
            progress_callback(f"Loaded {len(sanitized_tools)} tools... ✅")

        if progress_callback:
            progress_callback("Creating AI agent... ✅")

        mcp_orch_model = initialize_model(google_api_key)
        agent = create_react_agent(mcp_orch_model, sanitized_tools)

        if progress_callback:
            progress_callback("Setup complete! Starting to generate learning path... ✅")

        return agent

    except Exception as e:
        print(f"Error in setup_agent_with_tools: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def run_agent_sync(
    google_api_key: str,
    youtube_pipedream_url: str,
    drive_pipedream_url: Optional[str] = None,
    notion_pipedream_url: Optional[str] = None,
    user_goal: str = "",
    progress_callback: Optional[Callable[[str], None]] = None
) -> dict:
    """
    Synchronous wrapper for running the agent.
    """
    async def _run():
        try:
            agent = await setup_agent_with_tools(
                google_api_key=google_api_key,
                youtube_pipedream_url=youtube_pipedream_url,
                drive_pipedream_url=drive_pipedream_url,
                notion_pipedream_url=notion_pipedream_url,
                progress_callback=progress_callback
            )

            learning_path_prompt = "User Goal: " + user_goal + "\n" + user_goal_prompt

            if progress_callback:
                progress_callback("Generating your learning path...")

            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=learning_path_prompt)]},
                config=cfg
            )

            if progress_callback:
                progress_callback("Learning path generation complete!")

            return result
        except Exception as e:
            print(f"Error in _run: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()