# Singapore Math Bar Model Generator

An intelligent system that generates bar model diagrams for Singapore primary school mathematics word problems using Google's Gemini 2.5 Flash LLM and Model Context Protocol (MCP).

## 📄 Research Paper

This repository contains the implementation for the paper:

**"Automatic Diagram Generation with LLMs for Visual Reasoning in Education"**

**Important**: The results presented in the paper were achieved with **thinking mode enabled** (`--thinking --show-thinking`). For reproducing paper results, use:
```bash
python main.py --thinking --show-thinking
```

## Overview

This application combines:
- **Google Gemini 2.5 Flash**: Advanced LLM for understanding and reasoning about math problems
- **Model Context Protocol (MCP)**: Standardized protocol for connecting LLMs with tools
- **Bar Model Visualization**: Automated generation of Singapore Math diagrams

Bar models are a visual problem-solving technique used extensively in Singapore primary mathematics education to help students visualize relationships in word problems.

## Important Notes

⚠️ **Output Variability**: Due to the inherent randomness in LLM responses, generated outputs may vary between runs. While the system generally produces good results, exact reproducibility is not guaranteed. If you don't get a satisfactory result on the first attempt, try running the same problem again.

Additionally, the model may occasionally exhibit unexpected behavior or fail to follow certain instructions that are clearly stated in the prompts. This is likely because Gemini 2.5 Flash, while capable, is intended to prioritise cost-effectiveness instead of achieving top benchmark performance. Hence, it is prone to making some mistakes. However, this also demonstrates that significant educational value and functionality can be achieved even with LLMs that may not be the most powerful - highlighting the potential for accessible AI-powered educational tools.

⚠️ **Thinking Mode & Error 503**: There is a suspected issue where enabling thinking mode with hidden output (`--thinking` without `--show-thinking`) may trigger server error 503 responses. This appears to be related to the Google SDK's JSON parsing behavior when `include_thoughts=False`. If you consistently encounter 503 errors with math problems:
- Try disabling thinking mode entirely
- Or enable thinking with visible output: `--thinking --show-thinking`
- See [GOOGLE_SDK_BUG.md](GOOGLE_SDK_BUG.md) for technical details

## Features

- **Automated Bar Model Generation**: Creates professional bar model diagrams automatically
- **Streaming Responses**: Real-time output with optional thinking process visibility
- **Tool Call Visibility**: Shows when tools are requested, executing, and completed
  - `[Tool Call Requested: create_bar_model]`
  - `[Executing tool...]`
  - `[Tool executed successfully]` or error details
- **Thinking Mode**:
  - Enable LLM's internal reasoning with dynamic budget (`--thinking`)
  - Uses `thinking_budget=-1` for optimal reasoning
  - Optionally display reasoning output (`--show-thinking`)
  - Toggle during chat with `/thinking` and `/show-thinking`
  - Hidden mode (thinking enabled, display off) may cause 503 errors (see Important Notes)
- **Interactive Commands**: Control settings during chat with simple commands
- **Robust Error Handling**:
  - Automatic retry for 503 (overloaded) and 429 (rate limit) errors
  - Detailed error messages showing actual API error text
  - Full logging available with `--log-level INFO`
- **Intelligent Problem Solving**: Three-step process:
  1. Algebraic analysis for verification
  2. Iterative bar model design with self-reflection
  3. Clear arithmetic explanation
- **Flexible Diagram Components**:
  - Multiple bars with variable-length segments
  - Horizontal braces for labeling differences and groups
  - Vertical braces for showing sums across bars
  - Automatic layout optimization
- **Command-Line Options**: Configure behavior via arguments or interactively

## Project Structure

```
diagramgen/
├── main.py                      # Main application entry point
├── config.py                    # Configuration management
├── requirements.txt             # Python dependencies
├── .env.example                 # Example environment configuration
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
├── USAGE.md                     # Quick usage reference
├── GOOGLE_SDK_BUG.md            # Known Google SDK bug documentation
├── test_simple.py               # Configuration test script
├── questions.txt                # All available math word problems (33 questions)
├── example_questions.txt        # Indices of questions to use as examples (1,2,3)
├── images/                      # Generated bar model images (auto-created)
└── mcp_servers/                 # MCP server implementation
    ├── main.py                  # MCP server for bar model tools
    ├── bar_model.py             # Bar model drawing library
    ├── servers_config.json      # MCP server configuration
    └── prompts/                 # System and instruction prompts
        ├── system-prompt.txt
        └── bar-model-drawing.txt
```

**Note**: `TASK.md` is not included in version control (see `.gitignore`). It's used for local task tracking only.

## Prerequisites

- Python 3.10 or higher
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

## Installation

### 1. Clone or Download

```bash
cd diagramgen
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Copy the example environment file and add your API key:

```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

Edit `.env` and set your Gemini API key:

```
GEMINI_API_KEY=your_actual_api_key_here
```

## Usage

### Test Configuration (Optional but Recommended)

Before running the main application, test that your configuration works:

```bash
python test_simple.py
```

This will verify:
- Your API key is valid
- Basic chat functionality works
- Tool/function calling works

### Run the Application

**Basic usage:**
```bash
python main.py
```

**With options:**
```bash
# Enable thinking mode (hidden by default to reduce JSON errors)
python main.py --thinking

# Enable thinking mode with visible output
python main.py --thinking --show-thinking

# Skip example questions and go straight to interactive mode
python main.py --no-examples

# Combine options
python main.py --thinking --no-examples

# Set log level for debugging
python main.py --log-level INFO
```

**Interactive commands during chat:**
- `/thinking` - Toggle thinking mode on/off
- `/show-thinking` - Toggle thinking display on/off
- `/help` - Show available commands
- `/quit` or `/exit` - Exit the application

**Note**: Toggling thinking or display settings will recreate the chat session with the new configuration.

### Example Word Problem

```
Bernard had $20 more than Rhona at first. After Bernard spent $52,
Rhona had thrice as much money as Bernard. How much money did Bernard have at first?
```

### Expected Output

1. **Thinking Process**: Shows the LLM's reasoning (if enabled with `--show-thinking`)
2. **Algebraic Solution**: Mathematical verification
3. **Tool Call Notifications**: Shows when bar model tool is requested and executing
4. **Bar Model Design**: Iterative refinement with evaluations
5. **Generated Diagram**: Saved as PNG in the `images/` folder with incremental naming (`image_000000.png`, `image_000001.png`, etc.)
6. **Explanation**: Step-by-step arithmetic solution

**Note**: Images are named sequentially starting from `image_000000.png` and incrementing by 1 for each generated diagram during the session.

## Configuration

### Environment Variables

Edit `.env` to customize:

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Your Google Gemini API key | Required |
| `GEMINI_MODEL` | Gemini model to use | `gemini-2.5-flash` |
| `MAX_RETRIES` | Max retry attempts for API calls | `15` |
| `RETRY_DELAY` | Delay between retries (seconds) | `0.3` |
| `LOG_LEVEL` | Logging level | `ERROR` |

### Adjusting Logging

For more detailed output, set `LOG_LEVEL=INFO` or `LOG_LEVEL=DEBUG` in `.env`.

### Configuring Example Questions

The application uses two configuration files for questions:

**`questions.txt`** - Contains all available math word problems (33 questions total)
- One question per line
- All questions from the research paper dataset

**`example_questions.txt`** - Specifies which questions to use as examples
- Comma-separated integers (one-indexed)
- Default: `1,2,3` (uses first 3 questions)
- Example: `1,5,10` would use questions 1, 5, and 10 from `questions.txt`

To change which questions run automatically at startup:
1. Edit `example_questions.txt`
2. Enter desired question numbers (one-indexed), separated by commas
3. Save the file

Example configurations:
```
1,2,3        # First 3 questions (default)
1,2,3,4,5    # First 5 questions
10,20,30     # Questions 10, 20, and 30
```

To skip example questions entirely, use `--no-examples` flag:
```bash
python main.py --no-examples
```

## Architecture

### Components

1. **Main Application (`main.py`)**
   - Orchestrates session management
   - Handles LLM communication
   - Manages MCP server connections
   - Processes streaming responses

2. **Configuration (`config.py`)**
   - Type-safe environment variable management
   - Path resolution utilities
   - Validation logic

3. **MCP Server (`mcp_servers/main.py`)**
   - Implements Model Context Protocol
   - Exposes bar model generation tool
   - Serves system prompts

4. **Bar Model Library (`mcp_servers/bar_model.py`)**
   - Canvas creation and management
   - Bar, segment, and brace rendering
   - Automatic layout calculations
   - Font scaling and positioning

### Data Flow

```
User Question
    ↓
Gemini LLM (via Google GenAI SDK)
    ↓
Tool Call Decision
    ↓
MCP Server (via Model Context Protocol)
    ↓
Bar Model Generator
    ↓
PNG Image (base64)
    ↓
Saved to images/ folder
    ↓
Response to User
```

## API Reference

### Bar Model Tool

The `create_bar_model` tool accepts:

```python
{
  "bars_data": [
    {
      "segments": [
        {"length": 10, "label": "A"},
        {"length": 20, "label": "B"}
      ],
      "h_braces": [
        {
          "start_segment_index": 0,
          "end_segment_index": 1,
          "label": "Total",
          "location": "top"  # or "bottom"
        }
      ],
      "label": "Person A"
    }
  ],
  "v_braces_data": [
    {
      "start_bar_index": 0,
      "end_bar_index": 1,
      "label": "Combined"
    }
  ]
}
```

### Key Concepts

- **Segments**: Represent proportional parts of a bar
- **Horizontal Braces**: Label differences or groups within a bar
- **Vertical Braces**: Show sums across multiple bars
- **Indices**: All are 0-indexed

## Troubleshooting

### "GEMINI_API_KEY cannot be empty"

Make sure you've:
1. Created a `.env` file (copy from `.env.example`)
2. Added your actual API key from Google AI Studio
3. Saved the file

### Module Import Errors

Ensure you've activated the virtual environment:

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### JSON Parsing Errors (Known Google SDK Bug)

**⚠️ This is a confirmed bug in Google's SDK ([Issue #1162](https://github.com/googleapis/python-genai/issues/1162))**

If you see `json.decoder.JSONDecodeError` errors:
- **Automatic retry**: The app automatically retries up to 3 times
- **Try disabling thinking mode**: Run without `--thinking` or use `/thinking off`
- **This is intermittent**: Often succeeds on retry
- **Not your fault**: This is Google's SDK bug, not your configuration

**What we do:**
- Automatic retry with exponential backoff
- User-friendly error messages
- Clear instructions when it fails

**For details:** See `GOOGLE_SDK_BUG.md` in the project root

### Server Overload (503 UNAVAILABLE)

When you see "Model is overloaded" errors:
- The application automatically retries (default: 15 attempts)
- You'll see retry notifications with countdown
- This is normal during high-traffic periods
- Adjust `MAX_RETRIES` and `RETRY_DELAY` in `.env` if needed

### Rate Limiting (429 errors)

The application automatically handles rate limits with exponential backoff. If you consistently hit limits, consider:
- Reducing `MAX_RETRIES` in `.env`
- Adding delays between questions
- Using `--no-examples` to control pacing

### Server Connection Issues

Check that:
- Python is in your PATH
- The `mcp_servers/main.py` path is correct
- No firewall is blocking local stdio communication

## Development

### Code Quality

The codebase follows these standards:
- **Type Hints**: Full type annotations throughout
- **Docstrings**: Comprehensive documentation for all functions/classes
- **Error Handling**: Robust exception handling with proper logging
- **Async/Await**: Modern async patterns with proper context management
- **Configuration**: Environment-based configuration with validation

### Adding New Tools

1. Define tool function in `mcp_servers/main.py`
2. Add tool schema in `list_tools()`
3. Implement handler in `call_tool()`
4. Update documentation

### Modifying Prompts

Edit files in `mcp_servers/prompts/`:
- `system-prompt.txt`: Overall system behavior
- `bar-model-drawing.txt`: Bar model creation instructions

## Examples

The application includes example questions in `main.py`. To add your own:

```python
list_qns = [
    "Your math word problem here...",
]
```

## Dependencies

- `google-genai>=1.50.1`: Google Generative AI SDK
- `mcp>=1.21.0`: Model Context Protocol
- `opencv-python>=4.10.0`: Computer vision for image generation
- `pydantic-settings>=2.6.0`: Configuration management

## License

This project is provided as-is for educational and development purposes.

## Resources

- [Google Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [Singapore Math Bar Models](https://en.wikipedia.org/wiki/Singapore_math#Bar_modeling)

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the `TASK.md` file for known issues
3. Verify your environment configuration
4. Check the logs (set `LOG_LEVEL=DEBUG`)

## Acknowledgments

Built with:
- Google Gemini 2.5 Flash
- Model Context Protocol (MCP)
- OpenCV for image generation
- Pydantic for configuration management

---

**Note**: This project is designed for educational purposes and Singapore primary school mathematics problems. Generated images are saved in the `images/` directory (auto-created on first run).
