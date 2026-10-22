# Task: Implement Python vLLM Orchestrator on VM1

## Objective

Create a simple, reliable Python orchestrator on VM1 that automatically:

1. Executes an existing bash script to start VM2.
2. Retrieves VM2's ID, IP address, and vLLM port as a Python dictionary.
3. Waits until the vLLM server is ready.
4. Sends chat prompts to vLLM.
5. Validates and stores responses in a JSONL file.
6. Executes an existing bash script to stop VM2.

## Existing Setup

- VM1 already exists and runs the orchestrator.
- VM2 is created using an existing bash script.
- VM2 automatically installs its dependencies and starts vLLM.
- VM2 setup is already verified. Do not modify or revalidate it.
- VM2 takes approximately 20 minutes to boot and install packages, plus additional time to start vLLM.
- Existing VM start and stop bash scripts must be reused.

## Requirement 1 — Start VM2

Use Python subprocess to execute the existing VM startup bash script.

Inspect the existing script and implement a mechanism to obtain:

```python
{
    "vm_id": "vm-123",
    "ip": "10.0.0.25",
    "port": 8000
}
```

Prefer having the bash script print one JSON object to stdout and progress logs to stderr.

If the existing script cannot return the IP, use the existing VM management command to retrieve it.

Do not hardcode VM2's IP.

The Python function should return:

```python
def start_vm2() -> dict:
    ...
```

Validate that the returned dictionary contains a nonempty VM ID, a valid reachable IP address or hostname, and a valid port.

## Requirement 2 — Wait for vLLM

Create:

```python
def wait_for_vllm(ip, port, timeout=2700):
    ...
```

- Poll the vLLM `/health` endpoint every 15 seconds.
- Default overall readiness timeout: 45 minutes.
- Do not use a fixed 20-minute sleep.
- Handle connection errors and HTTP failures.
- Continue polling until vLLM is ready or the timeout expires.
- Log elapsed waiting time.
- Use an HTTP request timeout so individual polls cannot hang.
- After health succeeds, verify the configured model is available and perform one small chat completion request.

Return success only when the server can handle inference.

## Requirement 3 — Send Chat Prompts

Create:

```python
def run_inference(ip, port, model, prompts_file):
    ...
```

- Read test prompts from a local JSONL file.
- Use the vLLM OpenAI-compatible `/v1/chat/completions` API.
- Send prompts sequentially.
- Configure request timeouts.
- Extract the generated response.
- Measure request latency.
- Validate HTTP status and response structure.
- Record failures and continue with the next prompt.
- Do not retry failed requests indefinitely.

## Requirement 4 — Save JSONL Results

Create:

```python
def save_result(result, output_file):
    ...
```

Each result must contain:

```json
{
  "test_id": "test_001",
  "prompt": "Explain machine learning",
  "response": "Machine learning is...",
  "status": "success",
  "http_status": 200,
  "latency_ms": 1200,
  "error": null,
  "timestamp": "2026-10-22T10:00:00Z"
}
```

- Store one JSON object per line.
- Save both successful and failed requests.
- Append results after each completed request.
- Preserve results if a later request fails.
- Do not overwrite previous results accidentally.

## Requirement 5 — Stop VM2

Create:

```python
def stop_vm2(vm_id):
    ...
```

- Execute the existing VM stop bash script.
- Pass the exact VM ID returned by the startup script.
- Attempt cleanup after inference finishes.
- Attempt cleanup after errors and interruptions.
- Use try/finally in the main orchestration function.
- Do not stop an unrelated or pre-existing VM.
- Report cleanup failures clearly.

## Requirement 6 — Test Scenarios

Write automated tests for:

1. VM startup succeeds.
2. VM startup bash script fails.
3. VM startup returns invalid JSON.
4. VM startup returns no IP.
5. vLLM is immediately ready.
6. vLLM becomes ready after multiple polls.
7. vLLM never becomes ready (timeout).
8. vLLM health succeeds but inference is unavailable.
9. Chat completion succeeds.
10. Chat request times out.
11. Chat API returns HTTP error.
12. Chat API returns malformed JSON.
13. Chat API returns an empty or invalid response.
14. One prompt fails but subsequent prompts continue.
15. JSONL file is written correctly.
16. JSONL write fails.
17. VM2 is stopped after successful execution.
18. VM2 cleanup is attempted after an inference failure.
19. VM2 stop script fails.
20. Orchestrator is interrupted during execution.

Use mocks for shell commands and HTTP requests. Tests must run without creating a real VM.

## Requirement 7 — Main Orchestrator

Implement:

```python
def main():
    vm_info = start_vm2()

    try:
        wait_for_vllm(
            vm_info["ip"],
            vm_info["port"]
        )

        run_inference(
            vm_info["ip"],
            vm_info["port"],
            model=MODEL_NAME,
            prompts_file=PROMPTS_FILE
        )

    finally:
        stop_vm2(vm_info["vm_id"])
```

This is illustrative pseudocode. Implement safe handling for partially successful VM creation, process interruptions, and cleanup failures.

## Expected Files

```text
orchestrator/
    orchestrator.py
    config.yaml
    prompts.jsonl
    test_orchestrator.py
    results/
```

Keep the implementation simple. Prefer functions over unnecessary classes.

## Constraints

- Python 3.
- Use subprocess for bash execution.
- Use requests or httpx for vLLM API calls.
- Reuse existing VM start/stop scripts.
- Do not modify the verified VM2 installation process.
- Do not introduce cloud SDKs unless necessary.
- Do not introduce frameworks, databases, queues, or distributed orchestration.
- Use configurable script paths, model name, port, timeouts, and file paths.
- Do not hardcode credentials.
- Keep the implementation readable and easy to debug.

## Implementation Instructions

1. First inspect the existing repository and VM start/stop bash scripts.
2. Identify how the VM ID and IP are obtained.
3. Explain any necessary changes to return a JSON dictionary from the start script.
4. Implement the orchestrator.
5. Implement the automated tests.
6. Run the mocked test suite.
7. Report passing tests, failures, and unverified real-VM behavior.

Do not modify VM2's verified setup.

Do not claim successful end-to-end GPU inference unless it has actually been tested.

## Final Acceptance Criteria

The orchestrator must successfully execute this workflow:

**Start VM2 → Get VM2 IP → Wait for vLLM → Send prompts → Save JSONL → Stop VM2.**

The orchestrator must attempt safe VM cleanup even when inference or readiness fails.

Keep the solution minimal and focused on this workflow.

## Additional Implementation Note

If the VM creation command succeeds but fails before returning its ID, the orchestrator cannot safely clean it up. Handle that case explicitly rather than treating startup as atomic.

First demonstrate a mocked complete run showing the returned VM dictionary, readiness polling, a saved JSONL response, and the stop command targeting the correct VM ID.
