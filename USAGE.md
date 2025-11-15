# Usage Guide

## Quick Start

```bash
# Basic usage (recommended for stability)
python main.py

# With thinking mode enabled and visible
python main.py --thinking --show-thinking

# Skip examples, go straight to interactive mode
python main.py --no-examples
```

⚠️ **Note**: Avoid using `--thinking` without `--show-thinking` as it may cause error 503 issues. See [GOOGLE_SDK_BUG.md](GOOGLE_SDK_BUG.md) for details.

## Configuring Example Questions

**To change which questions run at startup:**

1. Edit `example_questions.txt` in the project root
2. Enter question numbers (one-indexed), comma-separated
3. Example: `1,5,10` uses questions 1, 5, and 10

**Available questions**: 33 total questions in `questions.txt`

**Default**: `1,2,3` (first 3 questions)

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--thinking` | Enable thinking mode with dynamic budget (thinking_budget=-1) |
| `--show-thinking` | Display thinking output (requires `--thinking`) |
| `--no-examples` | Skip example questions, start in interactive mode |
| `--log-level LEVEL` | Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `-h, --help` | Show help message and exit |

**Technical Details**:
- `--thinking` sets `thinking_budget=-1` (dynamic reasoning) and `include_thoughts=False` (hidden)
- `--thinking --show-thinking` sets `thinking_budget=-1` and `include_thoughts=True` (visible)
- Without `--thinking`, uses `thinking_budget=0` (disabled)
- Hidden thinking helps reduce JSON parsing errors (see [GOOGLE_SDK_BUG.md](GOOGLE_SDK_BUG.md)) while maintaining quality

### Examples

```bash
# Enable thinking and debug logging (thinking hidden)
python main.py --thinking --log-level DEBUG

# Enable thinking with visible output
python main.py --thinking --show-thinking

# Interactive mode only
python main.py --no-examples

# All options combined
python main.py --thinking --show-thinking --no-examples --log-level INFO
```

## Interactive Commands

Once the application is running, you can use these commands:

| Command | Description |
|---------|-------------|
| `/thinking` | Toggle thinking mode on/off |
| `/show-thinking` | Toggle thinking display on/off |
| `/help` | Show available commands |
| `/quit` or `/exit` | Exit the application |

**Note**: Both `/thinking` and `/show-thinking` will recreate the chat session with the new configuration.

**Tip**: Use `/thinking` to enable it, then `/show-thinking` to hide the output. This enables reasoning without displaying it, which helps maintain quality while avoiding JSON parsing errors.

### Example Session

```
============================================================
Singapore Math Bar Model Generator
============================================================
Model: gemini-2.5-flash
Thinking mode: disabled
Type /help for available commands
============================================================

Question 1: Bernard had $20 more than Rhona at first...
============================================================