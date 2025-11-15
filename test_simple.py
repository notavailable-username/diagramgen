"""Simple test script to verify Google GenAI configuration works."""

import asyncio
from google import genai
from google.genai import types
from config import settings


async def test_simple_chat():
    """Test basic chat without tools."""
    print("Testing simple chat...")

    try:
        # Initialize client
        client = genai.Client(api_key=settings.GEMINI_API_KEY).aio
        print("✓ Client initialized")

        # Create chat with minimal config
        chat = client.chats.create(
            model=settings.GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful math tutor."
            )
        )
        print("✓ Chat created")

        # Test simple message
        print("\nSending test message...")
        response = await chat.send_message_stream("What is 2 + 2?")

        print("Response: ", end="", flush=True)
        async for chunk in response:
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print("\n✓ Simple chat works!")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_chat_with_tools():
    """Test chat with function declarations."""
    print("\n" + "="*60)
    print("Testing chat with tools...")

    try:
        # Initialize client
        client = genai.Client(api_key=settings.GEMINI_API_KEY).aio
        print("✓ Client initialized")

        # Define a simple tool
        tool_declaration = {
            "name": "add_numbers",
            "description": "Add two numbers together",
            "parameters": {
                "type": "object",
                "required": ["a", "b"],
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "First number"
                    },
                    "b": {
                        "type": "integer",
                        "description": "Second number"
                    }
                }
            }
        }

        # Create chat with tools
        chat = client.chats.create(
            model=settings.GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful assistant with access to tools.",
                tools=[
                    types.Tool(
                        function_declarations=[tool_declaration]
                    )
                ],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                )
            )
        )
        print("✓ Chat with tools created")

        # Test message that might trigger tool use
        print("\nSending test message...")
        response = await chat.send_message_stream("Add 5 and 3")

        print("Response: ", end="", flush=True)
        has_function_call = False
        async for chunk in response:
            if chunk.text:
                print(chunk.text, end="", flush=True)
            if chunk.function_calls:
                has_function_call = True
                print(f"\n[Function call detected: {chunk.function_calls[0].name}]")

        print("\n✓ Chat with tools works!")
        if has_function_call:
            print("✓ Function calling works!")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("="*60)
    print("Google GenAI Configuration Test")
    print("="*60)

    # Test 1: Simple chat
    result1 = await test_simple_chat()

    # Test 2: Chat with tools
    result2 = await test_chat_with_tools()

    # Summary
    print("\n" + "="*60)
    print("Test Summary:")
    print(f"  Simple chat: {'PASS' if result1 else 'FAIL'}")
    print(f"  Chat with tools: {'PASS' if result2 else 'FAIL'}")
    print("="*60)

    if result1 and result2:
        print("\n✓ All tests passed! Configuration is correct.")
    else:
        print("\n✗ Some tests failed. Check the errors above.")


if __name__ == "__main__":
    asyncio.run(main())
