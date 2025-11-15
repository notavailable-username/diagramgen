# Known Google SDK Bug: JSONDecodeError During Streaming

## Overview

This document describes a **confirmed bug in Google's `google-genai` Python SDK** that causes `JSONDecodeError` exceptions during streaming responses. This bug affects our application and many others in the community.

## The Bug

**Issue:** [googleapis/python-genai#1162](https://github.com/googleapis/python-genai/issues/1162)
**Status:** Closed as NOT_PLANNED (Google will not fix this soon)
**Severity:** High - Causes crashes during normal operation

## Suspected Related Issue: Error 503 with Hidden Thinking Mode

⚠️ **New Discovery (2025-11-15)**: We have observed a pattern suggesting that the JSON parsing bug may manifest differently when using thinking mode with `include_thoughts=False`:

**Symptoms:**
- Consistently get error 503 ("Model is overloaded") when asking math questions (which trigger tool calls)
- Simple questions without tool calls work fine immediately
- Error 503 does NOT occur when thinking mode is disabled entirely
- Error 503 does NOT occur when thinking mode has visible output (`include_thoughts=True`)

**Hypothesis:**
The JSON parsing error that normally manifests as `JSONDecodeError` may be handled differently on Google's server side when `include_thoughts=False`. Instead of streaming the malformed JSON to the client (where we catch it), the server may encounter the parsing error internally and return a generic 503 error response.

**Evidence:**
1. Error only happens with tool calls (complex responses)
2. Error only happens with thinking enabled but hidden
3. Error disappears when thinking is disabled OR when output is visible
4. Pattern matches the known JSON fragmentation bug

**Workaround:**
If you encounter consistent 503 errors with math problems:
- Disable thinking mode: run without `--thinking`
- Or use visible thinking: `--thinking --show-thinking`
- Avoid hidden thinking mode: don't use `--thinking` alone

### What Happens

When using the Gemini API with streaming enabled (`send_message_stream`), error responses (especially HTTP 429 rate limit errors) arrive as **fragmented JSON chunks** instead of complete JSON objects.

**Example of fragmented response:**
```
chunk 1: '{'
chunk 2: ' "error": {'
chunk 3: ' "code": 429,'
chunk 4: ' "message": "Quota exceeded"'
...continuing line-by-line...
```

The SDK attempts to parse each chunk individually with `json.loads()`, which fails because the chunks are incomplete JSON objects.

**Error you see:**
```
json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)
```

### Where It Occurs

The bug is in the SDK itself, specifically at:
- File: `google/genai/_api_client.py`
- Line: ~259 (varies by version)
- Code: `yield json.loads(chunk)`

### When It Occurs

The bug is **intermittent** and more likely to happen with:
- ✗ Thinking mode enabled (`thinking_config`)
- ✗ Large or complex prompts
- ✗ Non-English language prompts (e.g., Vietnamese)
- ✗ Requests with images
- ✗ Code execution enabled
- ✗ High API usage (approaching rate limits)
- ✗ Server errors (429, 500, 503)

## Our Workarounds

Since we cannot modify Google's SDK, we've implemented multiple defensive strategies:

### 1. Automatic Retry with Exponential Backoff

```python
max_json_retries = 3
json_retry_delay = 2.0  # doubles each retry

try:
    async for chunk in stream:
        # process chunk
except json.JSONDecodeError:
    # Retry entire request
    await asyncio.sleep(json_retry_delay)
```

**How it works:**
- Catches `JSONDecodeError` during streaming
- Retries the entire request up to 3 times
- Uses exponential backoff (2s, 4s, 8s)
- Informs user with clear messages

### 2. User-Friendly Error Messages

When the bug occurs, users see:
```
[Encountered streaming error (Google SDK bug #1162). Retrying... (attempt 2/3)]
```

After max retries:
```
[Error: The response stream was interrupted due to a known Google SDK bug.
Please try again or rephrase your question. Tip: Try disabling thinking mode if enabled.]
```

### 3. Thinking Mode Toggle

Since thinking mode increases the likelihood of this bug:
- Users can disable it: `/thinking off`
- Or start without it: `python main.py` (without `--thinking`)

### 4. Future Enhancement: json-repair

The `json-repair` library can fix malformed JSON:
```python
import json_repair
decoded = json_repair.loads(malformed_json_string)
```

**Status:** Available but not yet integrated (commented in `requirements.txt`)

**Why not used yet:** The error occurs deep in Google's SDK before we can access the malformed chunks. Would require monkey-patching or forking the SDK.

## Recommendations for Users

### To Minimize Occurrences:

1. **Disable thinking mode** if you encounter frequent errors:
   ```bash
   python main.py  # without --thinking
   ```

2. **Keep prompts concise** and in English

3. **Avoid rapid-fire requests** that might trigger rate limits

4. **Use retry logic** (already built-in to our app)

### When It Happens:

1. **Wait for automatic retry** - The app retries up to 3 times
2. **Try again** - The bug is intermittent
3. **Disable thinking** - Use `/thinking off` command
4. **Rephrase question** - Sometimes helps avoid the trigger

## Technical Details for Developers

### Root Cause

Google's Gemini API violates its own protocol by:
1. Returning error responses as multiple incomplete chunks
2. Not maintaining chunk boundaries for JSON objects
3. Mixing success and error streaming behaviors

### Why Google Won't Fix It

From Issue #1162:
- Marked as "NOT_PLANNED"
- Issue closed as "stale"
- No timeline for fix provided

This indicates it's either:
- A low priority for Google
- A complex architectural issue
- An intentional design decision

### Proposed Solutions (Discussed but Not Implemented by Google)

1. **Server-side fix** (preferred): Return errors as single complete chunks
2. **SDK buffering**: Accumulate chunks until valid JSON before parsing
3. **Error protocol change**: Use different protocol for error vs success responses

## Monitoring and Logging

Our app logs all occurrences:

```python
logger.error(f"JSONDecodeError during streaming (known Google SDK bug): {e}")
```

Check logs at:
- Default level: `ERROR`
- Debug mode: `python main.py --log-level DEBUG`

## Related Issues

- [googleapis/python-genai#1162](https://github.com/googleapis/python-genai/issues/1162) - Main issue
- [googleapis/python-genai#1092](https://github.com/googleapis/python-genai/issues/1092) - Function call streaming
- Multiple Stack Overflow threads about "Gemini JSONDecodeError streaming"

## Community Workarounds

Other developers have tried:
- ✓ Try-except wrapping (our approach)
- ✓ Retry logic (our approach)
- ✗ Switching to non-streaming (loses real-time feedback)
- ✗ Using different SDK versions (bug persists across versions)
- ✗ Monkey-patching SDK (fragile, breaks on updates)

## Conclusion

This is a **confirmed, unfixed bug in Google's SDK**. Our application implements robust workarounds, but the bug may still occasionally cause interruptions. The best approach is:

1. Be aware it exists
2. Use our built-in retry logic (automatic)
3. Disable thinking mode if problems persist
4. Report patterns to us so we can improve workarounds

**Last Updated:** 2025-01-14
**SDK Version Affected:** All versions through 1.50.1
**Expected Fix:** None announced by Google

## References

- Google SDK Issue: https://github.com/googleapis/python-genai/issues/1162
- json-repair Library: https://github.com/mangiucugna/json_repair
- Our implementation: `main.py` lines 664-800 (send_message and _process_response_stream methods)
