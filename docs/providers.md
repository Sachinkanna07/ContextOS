# Model providers and memory privacy

ContextOS keeps one local memory store and compiles relevant facts before every model request. Native text adapters are available for Ollama, OpenAI Responses, Anthropic Messages, and Gemini Interactions. OpenAI-compatible endpoints remain configurable for local runtimes and hosted services. Provider adapters do not own a second memory store.

Remote dispatch sends the selected compiled ContextOS context, the question, and any system instruction to the selected provider. This is necessary for the remote model to use your memories. Remote providers are disabled by default. An explicit `--provider openai`, `anthropic`, `gemini`, or named remote provider requests that transmission. Automatic remote fallback needs both `--allow-fallback` and `--allow-remote`; fallback alone cannot send memory off-device.

## Local quick start

Install and run Ollama with a model such as `qwen2.5-coder:7b`, then:

```powershell
contextos start
"I am building ContextOS." | contextos memories remember
contextos models list
contextos ask "What project am I building?" --provider ollama --model qwen2.5-coder:7b
contextos stop
```

`contextos ask` prints the answer without compiled memory text. `--json` includes dispatch and token evidence but omits compiled context. Add `--show-context` only when you want that private text on your terminal. `contextos models providers` shows enabled/local/credential-presence/discovery status without displaying keys.

## Cloud configuration

Set each API key in the environment of the process that starts the daemon. The TOML file stores only the variable name. Restart the daemon after changing provider configuration.

```powershell
$env:OPENAI_API_KEY="..."
$env:ANTHROPIC_API_KEY="..."
$env:GEMINI_API_KEY="..."
```

In your ContextOS `config.toml`:

```toml
[providers.openai]
enabled = true
api_key_env = "OPENAI_API_KEY"

[providers.anthropic]
enabled = true
api_key_env = "ANTHROPIC_API_KEY"

[providers.gemini]
enabled = true
api_key_env = "GEMINI_API_KEY"
```

Choose a model from `contextos models list`, then ask explicitly:

```powershell
contextos ask "What project am I building?" --provider openai --model <discovered-model>
contextos ask "What project am I building?" --provider anthropic --model <discovered-model>
contextos ask "What project am I building?" --provider gemini --model <discovered-model>
```

The native adapters use direct HTTP through the existing `httpx` dependency. No provider SDK is required. Optional temperature is omitted unless you supply `--temperature`, because model support varies. Model discovery does not claim tools, JSON, or vision support without reliable metadata. Text only is implemented.

## Compatible endpoints

Named endpoints use safe lowercase IDs and a validated base URL. Plain HTTP is allowed only for loopback. Remote endpoints require HTTPS and an API key in an environment variable. For example:

```toml
[providers.compatible.local-vllm]
enabled = true
base_url = "http://127.0.0.1:8000/v1"

[providers.compatible.openrouter]
enabled = true
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
```

This also accommodates other OpenAI-compatible services when their `/models` and `/chat/completions` contracts match. Compatibility is adapter contract support, not proof that every service or model has been live validated.

## Validation scope

RC4 cloud adapters have protocol contract tests with mocked HTTP responses. Live OpenAI, Anthropic, and Gemini validation requires available credentials and is reported separately. Ollama remains the locally validated provider. Answer quality remains unvalidated; synthetic token reduction is not billed-provider savings. Graph augmentation remains opt-in. See the [RC4 plan](releases/v1.0.0-rc4-plan.md) for the official API sources and exact protocol fields.
