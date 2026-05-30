# Agent Observability: What to Monitor Beyond Uptime and Latency

> Originally published on [omnithium.ai](https://omnithium.ai/blog/agent-observability-beyond-uptime)

Your AI agent has 99.9% uptime. Response times are consistently under two seconds. The infrastructure team is happy, the dashboards are green, and then a customer escalates because the agent has been confidently giving wrong answers for three days.

This is the observability trap that catches most engineering teams when they first move [AI agents](https://omnithium.ai/blog/what-are-ai-agents-complete-guide.html) to production. The monitoring instincts developed for traditional services: track availability, measure latency, alert on errors: transfer poorly to systems whose primary failure mode is not crashing, but producing subtly bad outputs at scale.

A service that returns a 500 error is obviously broken. An agent that returns a plausible but incorrect recommendation, or one that has quietly started using a tool in an unintended way, or one whose output quality has degraded 15% because a prompt was updated three weeks ago: these failures are invisible to conventional infrastructure monitoring. They require a different instrumentation philosophy entirely.

This post covers the observability layer that actually matters for production AI agents: what to measure, how to capture it, and how to build a quality dashboard that gives you genuine signal rather than false confidence.

## Why Traditional Observability Falls Short

Standard observability frameworks were built around a core assumption: correctness is binary. Either the system returned the right HTTP status code, processed the record, or it didn't. Metrics like error rate, throughput, and p99 latency work well precisely because they capture deviations from a well-defined correct behavior.

AI agents violate this assumption at every level. An agent's output is not correct or incorrect in the way a database query is correct or incorrect. Quality exists on a continuum, varies by context, and often requires domain knowledge to evaluate. The agent might complete a task successfully in a narrow technical sense: it called the right tools, returned a structured response, didn't throw an exception: while producing output that is useless or actively harmful to the user.

There's also the question of behavioral stability. Traditional services have deterministic behavior given the same inputs. Agents don't. Temperature settings, model updates from providers, prompt changes, tool availability, and context window management all create surfaces where behavior can drift without any single change triggering an alert.

The gaps in conventional monitoring for agents are roughly:

- **Output quality** is not measured at all
- **Tool call behavior** is treated as a side effect rather than a primary signal
- **Token consumption patterns** are tracked for cost but not as a behavioral indicator
- **User satisfaction** is collected separately from system metrics, if at all
- **Behavioral drift** has no baseline to drift from

Filling these gaps requires instrumenting your agents differently from the start.

## Output Quality Metrics

Output quality is the hardest thing to measure and the most important. There is no perfect solution, but there are several practical approaches that, used together, give you reasonable coverage.

### LLM-as-Judge Scoring

The most scalable approach for continuous quality monitoring is using a secondary LLM to evaluate agent outputs against defined criteria. This is sometimes called LLM-as-judge or model-graded evaluation.

```python
import openai
import json
from dataclasses import dataclass
from typing import Optional

@dataclass
class QualityScore:
 task_completion: float # 0.0-1.0
 factual_accuracy: float # 0.0-1.0
 instruction_following: float # 0.0-1.0
 output_format: float # 0.0-1.0
 overall: float # weighted composite
 reasoning: str
 flags: list[str] # e.g. ["hallucination_suspected", "off_topic"]

JUDGE_PROMPT = """
You are evaluating an AI agent response. Score each dimension from 0.0 to 1.0.

Task: {task}
Agent Response: {response}
Expected Format: {expected_format}
Reference Context: {context}

Evaluate:
1. task_completion: Did the agent complete the requested task?
2. factual_accuracy: Are claims in the response accurate given the context?
3. instruction_following: Did the agent follow all instructions?
4. output_format: Does the output match the expected format?

Respond with valid JSON only:
{{
 "task_completion": <float>,
 "factual_accuracy": <float>,
 "instruction_following": <float>,
 "output_format": <float>,
 "reasoning": "<brief explanation>",
 "flags": ["<flag1>", "<flag2>"]
}}
"""

def score_agent_output(
 task: str,
 response: str,
 expected_format: str,
 context: str,
 judge_model: str = "gpt-4o-mini"
) -> Optional[QualityScore]:
 client = openai.OpenAI()

 try:
 result = client.chat.completions.create(
 model=judge_model,
 messages=[{
 "role": "user",
 "content": JUDGE_PROMPT.format(
 task=task,
 response=response,
 expected_format=expected_format,
 context=context
 )
 }],
 response_format={"type": "json_object"},
 temperature=0
 )

 data = json.loads(result.choices[0].message.content)

 weights = {
 "task_completion": 0.4,
 "factual_accuracy": 0.3,
 "instruction_following": 0.2,
 "output_format": 0.1
 }

 overall = sum(
 data[dim] * weight
 for dim, weight in weights.items()
 )

 return QualityScore(
 task_completion=data["task_completion"],
 factual_accuracy=data["factual_accuracy"],
 instruction_following=data["instruction_following"],
 output_format=data["output_format"],
 overall=overall,
 reasoning=data["reasoning"],
 flags=data.get("flags", [])
 )

 except Exception as e:
 # Log and return None; don't let eval failures break production
 logger.error(f"Quality scoring failed: {e}")
 return None
```

LLM-as-judge is not perfect: it introduces its own biases and can miss domain-specific errors. Run it on a sample of traffic (10-20% in most production systems is sufficient) rather than every call, and periodically validate judge scores against human evaluations to catch calibration drift in the judge itself.

### Structural and Semantic Validation

For agents with defined output schemas, structural validation is cheap and reliable. If your agent is supposed to return a JSON object with specific fields, validate that on every response. Structural failures that don't throw exceptions are surprisingly common, especially after prompt changes.

Beyond structure, semantic validation catches cases where outputs are structurally correct but semantically wrong: like an agent returning a valid date that is three years in the past when it should be generating a future deadline, or an email draft that references the wrong customer name because context was mishandled.

```python
from pydantic import BaseModel, validator
from datetime import date, timedelta
import re

class AgentTaskOutput(BaseModel):
 action_items: list[str]
 deadline: date
 assignee_email: str
 priority: str
 summary: str

 @validator('deadline')
 def deadline_must_be_future(cls, v):
 if v < date.today():
 raise ValueError(f"Deadline {v} is in the past")
 if v > date.today() + timedelta(days=365):
 raise ValueError(f"Deadline {v} is more than a year out: verify intent")
 return v

 @validator('assignee_email')
 def email_format(cls, v):
 if not re.match(r'^[^@]+@[^@]+\.[^@]+$', v):
 raise ValueError(f"Invalid email format: {v}")
 return v

 @validator('priority')
 def valid_priority(cls, v):
 if v.lower() not in {'low', 'medium', 'high', 'critical'}:
 raise ValueError(f"Unknown priority: {v}")
 return v

def validate_and_record(raw_output: dict, span) -> tuple[bool, list[str]]:
 errors = []
 try:
 AgentTaskOutput(**raw_output)
 span.set_attribute("output.validation.passed", True)
 return True, []
 except Exception as e:
 errors = [str(err) for err in e.errors()]
 span.set_attribute("output.validation.passed", False)
 span.set_attribute("output.validation.errors", json.dumps(errors))
 return False, errors
```

Track validation failure rates as a primary quality metric. A validation failure rate that climbs from 0.5% to 4% after a deployment is a clear signal something changed.

## Behavioral Drift Detection

Behavioral drift is the slow, often invisible change in how an agent behaves over time. It happens for several reasons: model providers update their models without notice, context accumulates in ways that shift responses, tool APIs change their return formats, or prompts are adjusted without full impact assessment.

The challenge is that drift is often gradual enough that no single response looks wrong: the pattern only becomes visible in aggregate.

### Embedding-Based Drift Detection

One practical approach is tracking the semantic distribution of outputs over time using embeddings. If you embed each agent response and track the centroid of that distribution, significant movement in the centroid indicates behavioral change.

```python
import numpy as np
from collections import deque
from openai import OpenAI

client = OpenAI()

class DriftDetector:
 def __init__(self, window_size: int = 500, alert_threshold: float = 0.15):
 self.window_size = window_size
 self.alert_threshold = alert_threshold
 self.baseline_centroid = None
 self.current_window = deque(maxlen=window_size)
 self.embedding_model = "text-embedding-3-small"

 def embed(self, text: str) -> np.ndarray:
 response = client.embeddings.create(
 input=text,
 model=self.embedding_model
 )
 return np.array(response.data[0].embedding)

 def update_baseline(self, texts: list[str]):
 embeddings = [self.embed(t) for t in texts]
 self.baseline_centroid = np.mean(embeddings, axis=0)

 def observe(self, output_text: str) -> dict:
 embedding = self.embed(output_text)
 self.current_window.append(embedding)

 if len(self.current_window) < 50:
 return {"drift_score": None, "alert": False}

 current_centroid = np.mean(list(self.current_window), axis=0)

 # Cosine distance from baseline
 similarity = np.dot(self.baseline_centroid, current_centroid) / (
 np.linalg.norm(self.baseline_centroid) *
 np.linalg.norm(current_centroid)
 )
 drift_score = 1.0 - similarity

 return {
 "drift_score": float(drift_score),
 "alert": drift_score > self.alert_threshold,
 "window_size": len(self.current_window)
 }
```

Set your baseline using a known-good period: typically the first week after a validated deployment. A drift score above 0.10-0.15 warrants investigation; above 0.20 often indicates a meaningful behavioral change.

### Prompt Fingerprinting

Beyond output-level drift, track [prompt-level stability](https://omnithium.ai/blog/prompt-versioning-regression-testing.html). Store a hash of the effective prompt (system prompt + any dynamic injections) for every agent run. When that hash changes, you have a change event you can correlate against quality metric shifts.

```python
import hashlib

def compute_prompt_fingerprint(system_prompt: str, injected_context: dict) -> str:
 canonical = json.dumps({
 "system": system_prompt,
 "context_keys": sorted(injected_context.keys()),
 "context_hash": hashlib.md5(
 json.dumps(injected_context, sort_keys=True).encode()
 ).hexdigest()
 }, sort_keys=True)
 return hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

Record this fingerprint as a span attribute on every trace. When quality metrics dip, the first question you want to answer is whether a prompt changed: and this makes that query instantaneous.

## Tool Call Success Rates

For [agentic workflows](https://omnithium.ai/blog/enterprise-ai-agent-orchestration-patterns.html), tool calls are first-class operations that deserve their own monitoring. Tracking whether tool calls succeed or fail at the HTTP level is a start, but it misses most of what matters.

The signals to capture per tool call:

- **Call frequency**: How often is each tool being called per task? An agent that has started calling a search tool 8 times per task when it used to call it 3 times is doing something different.
- **Selection accuracy**: Is the agent choosing the right tool for the job? For workflows where the correct tool is deterministic, track tool selection accuracy explicitly.
- **Argument quality**: Are the arguments passed to tools valid and semantically appropriate? A tool call that succeeds at the API level but passes a malformed query is still a failure from the agent's perspective.
- **Retry rate**: How often is the agent retrying tool calls? High retry rates indicate either flaky tools or the agent generating bad inputs.
- **Error recovery**: When a tool fails, does the agent recover gracefully or does it hallucinate an answer?

```python
import time
from functools import wraps
from opentelemetry import trace

tracer = trace.get_tracer("omnithium.tools")

def instrument_tool(tool_name: str):
 def decorator(func):
 @wraps(func)
 def wrapper(*args, **kwargs):
 with tracer.start_as_current_span(f"tool.{tool_name}") as span:
 span.set_attribute("tool.name", tool_name)
 span.set_attribute("tool.args", json.dumps(kwargs, default=str))

 start = time.perf_counter()
 attempt = 0

 while attempt < 3:
 attempt += 1
 try:
 result = func(*args, **kwargs)
 duration_ms = (time.perf_counter() - start) * 1000

 span.set_attribute("tool.success", True)
 span.set_attribute("tool.attempts", attempt)
 span.set_attribute("tool.duration_ms", duration_ms)
 span.set_attribute("tool.result_size_chars",
 len(str(result)))

 return result

 except Exception as e:
 if attempt == 3:
 span.set_attribute("tool.success", False)
 span.set_attribute("tool.error", str(e))
 span.set_attribute("tool.attempts", attempt)
 raise
 time.sleep(0.5 * attempt)

 return wrapper
 return decorator

# Usage
@instrument_tool("jira_create_issue")
def create_jira_issue(project_key: str, summary: str, description: str,
 issue_type: str = "Task"):
 # ... implementation
 pass
```

Aggregate tool call metrics by agent, by workflow, and by time window. A tool success rate dashboard that shows each tool's call volume, success rate, p95 latency, and retry rate gives you immediate visibility into tool-layer health separate from model-layer health.

## Token Efficiency

Token consumption is tracked by nearly every team, but usually only for [cost optimization](https://omnithium.ai/blog/llm-cost-optimization-agents.html). Treated as a behavioral signal, token patterns tell you a great deal more.

**Input token growth**: If input tokens for a given task type are increasing over time, your [context management](https://omnithium.ai/blog/memory-context-management-agents.html) may be leaking: accumulating history or retrieved context beyond what's needed. An agent whose average input tokens grew from 2,400 to 6,800 over two months without any deliberate change is running on context it doesn't need.

**Output token variance**: High variance in output token counts for nominally similar tasks suggests inconsistent behavior. A customer-facing summary agent that sometimes returns 150 tokens and sometimes returns 2,400 tokens for equivalent inputs is not behaving stably.

**Thinking token consumption**: For reasoning models that expose chain-of-thought token usage, track thinking tokens as a proxy for task difficulty. Systematic increases in thinking token usage for previously simple tasks can indicate prompt degradation or context quality issues.

**Token efficiency ratio**: Define this as the ratio of useful output content to total tokens consumed. A rough proxy: for structured output tasks, compare output token count to the size of the validated structured output. Agents that produce valid 500-character JSON summaries while consuming 8,000 output tokens are doing something inefficient.

```python
from dataclasses import dataclass

@dataclass
class TokenMetrics:
 input_tokens: int
 output_tokens: int
 thinking_tokens: int
 cached_tokens: int
 model: str
 task_type: str

 @property
 def total_tokens(self) -> int:
 return self.input_tokens + self.output_tokens

 @property
 def cache_hit_rate(self) -> float:
 if self.input_tokens == 0:
 return 0.0
 return self.cached_tokens / self.input_tokens

 @property
 def output_density(self) -> float:
 """Ratio of output to total: low values suggest verbose reasoning."""
 if self.total_tokens == 0:
 return 0.0
 return self.output_tokens / self.total_tokens

def record_token_metrics(metrics: TokenMetrics, span):
 span.set_attribute("llm.input_tokens", metrics.input_tokens)
 span.set_attribute("llm.output_tokens", metrics.output_tokens)
 span.set_attribute("llm.thinking_tokens", metrics.thinking_tokens)
 span.set_attribute("llm.cached_tokens", metrics.cached_tokens)
 span.set_attribute("llm.total_tokens", metrics.total_tokens)
 span.set_attribute("llm.cache_hit_rate", metrics.cache_hit_rate)
 span.set_attribute("llm.output_density", metrics.output_density)
 span.set_attribute("llm.model", metrics.model)
 span.set_attribute("llm.task_type", metrics.task_type)
```

Alert on statistical anomalies: not just absolute thresholds. An agent workflow whose input token count is 3 standard deviations above its 30-day mean is worth investigating regardless of whether it crossed a cost threshold.

## User Satisfaction Signals

Implicit and explicit user feedback is the ground truth for agent quality, and it is frequently collected in isolation from system metrics. Connecting user signals to specific agent runs, tool calls, and model parameters is what turns feedback into actionable observability.

**Explicit feedback**: Thumbs up/down, star ratings, or correction submissions. These are low-volume but high-signal. Tag each feedback event with the agent run ID, so you can pull the full trace for any rejected response.

**Implicit signals**: Task abandonment (user started a workflow and didn't complete it), immediate re-requests (user submitted almost the same task again within two minutes, suggesting the first answer was inadequate), [escalation to human agents](https://omnithium.ai/blog/human-in-the-loop-patterns.html), and edit distance (for draft generation tasks, how much did the user modify the output before using it).

```python
class SatisfactionTracker:
 def __init__(self, telemetry_client):
 self.client = telemetry_client

 def record_explicit_feedback(
 self,
 run_id: str,
 rating: int, # 1-5
 feedback_text: str,
 user_id: str
 ):
 self.client.record_event("agent.feedback.explicit", {
 "run_id": run_id,
 "rating": rating,
 "has_text": bool(feedback_text),
 "user_id": user_id,
 "timestamp": time.time()
 })

 def record_abandonment(self, run_id: str, step_reached: str):
 self.client.record_event("agent.feedback.abandonment", {
 "run_id": run_id,
 "step_reached": step_reached,
 "timestamp": time.time()
 })

 def record_re_request(
 self,
 original_run_id: str,
 new_run_id: str,
 similarity_score: float,
 gap_seconds: int
 ):
 self.client.record_event("agent.feedback.re_request", {
 "original_run_id": original_run_id,
 "new_run_id": new_run_id,
 "similarity_score": similarity_score,
 "gap_seconds": gap_seconds,
 "implicit_failure": gap_seconds < 120 and similarity_score > 0.85
 })

 def record_edit_distance(
 self,
 run_id: str,
 original_length: int,
 final_length: int,
 edit_distance: int
 ):
 edit_ratio = edit_distance / max(original_length, 1)
 self.client.record_event("agent.feedback.edit", {
 "run_id": run_id,
 "edit_distance": edit_distance,
 "edit_ratio": edit_ratio,
 "substantial_edit": edit_ratio > 0.3
 })
```

A task abandonment rate above 15% or an implicit re-request rate above 8% on a specific workflow are strong signals that something is wrong, even if all your technical metrics look healthy.

## Building a Quality Dashboard

Raw metrics are not observability. Observability is the ability to ask questions about your system's behavior and get answers. A quality dashboard for AI agents should be structured to answer the questions you'll actually ask during an incident or a deployment review.

### Layer 1: Health at a Glance

The top level of your dashboard should answer: "Is anything critically wrong right now?"

- Composite quality score (weighted average of LLM-judge scores, validation pass rate, and user satisfaction) with a 24-hour trend line
- Tool call success rate across all tools, with per-tool breakdown available on click
- Active behavioral drift alerts
- Explicit feedback summary: ratio of positive to negative in the last 24 hours vs. 7-day baseline

### Layer 2: Task-Type Breakdown

One level deeper, break down quality metrics by task type or workflow. An agent that handles both document summarization and action item extraction should have separate quality tracking for each, because the failure modes are different and the acceptable quality thresholds may differ.

```yaml
# Example: Omnithium quality dashboard configuration
dashboards:
 agent_quality:
 refresh_interval: 60s
 task_views:
 - task_type: "document_summarization"
 metrics:
 - name: "output_quality_score"
 source: "llm_judge"
 threshold_warn: 0.75
 threshold_critical: 0.60
 - name: "validation_pass_rate"
 threshold_warn: 0.97
 threshold_critical: 0.92
 - name: "avg_edit_ratio"
 threshold_warn: 0.25
 threshold_critical: 0.40

 - task_type: "action_item_extraction"
 metrics:
 - name: "output_quality_score"
 source: "llm_judge"
 threshold_warn: 0.80
 threshold_critical: 0.70
 - name: "schema_validation_rate"
 threshold_warn: 0.99
 threshold_critical: 0.95
 - name: "tool_selection_accuracy"
 threshold_warn: 0.92
 threshold_critical: 0.85
```

### Layer 3: Trace-Level Investigation

When a metric degrades, you need to get from "quality dropped 12% on document summarization over the last 6 hours" to "here are the 47 runs that scored below 0.65, with full traces" in one or two clicks. This requires that your quality scores are stored as attributes on your distributed traces, not in a separate system.

Every quality score, validation result, token metric, tool call outcome, and user feedback event should carry the run ID, span ID, task type, model, and prompt fingerprint. With that, any quality metric becomes a filter on your trace data.

### Alerting Philosophy

Alert on trends, not just thresholds. A quality score of 0.72 may be fine if your baseline is 0.70, and alarming if your baseline is 0.88. Use rolling z-score alerts:

```python
def should_alert(
 current_value: float,
 historical_values: list[float],
 z_threshold: float = 2.5
) -> tuple[bool, float]:
 if len(historical_values) < 30:
 return False, 0.0

 mean = np.mean(historical_values)
 std = np.std(historical_values)

 if std < 1e-6:
 return False, 0.0

 z_score = (current_value - mean) / std

 # Negative z-score = below mean = quality degradation
 return z_score < -z_threshold, z_score
```

This catches gradual degradation that never crosses an absolute threshold, which is the failure mode that absolute threshold alerting misses entirely.

## Instrumentation Overhead

A practical concern: this is a lot of instrumentation. For high-throughput agent systems, running LLM-as-judge on every call is not feasible. A few principles for managing overhead:

**Sample intelligently**: Run full LLM-judge evaluation on 10-20% of traffic normally, increase to 50-100% during the first 48 hours after any prompt change or model update, and on 100% of cases that triggered validation failures or produced implicit failure signals.

**Separate evaluation from production path**: Quality scoring should be asynchronous. Write the raw output to an evaluation queue and score it out-of-band. Never let evaluation latency affect user-facing response times.

**Embed cheaply**: If you're doing drift detection via embeddings, use the smallest embedding model that gives adequate separation for your task domain. `text-embedding-3-small` costs roughly 10x less than `text-embedding-3-large` and is sufficient for most drift detection use cases.

**Validate everything, judge a sample**: Structural and semantic validation is cheap: run it on 100% of responses. LLM-judge is expensive: sample it. Use validation failures as triggers for elevated sampling rates.

## What Good Looks Like

For a production agent workflow that has been running for 60 days on mature prompts with a stable model version, reasonable baselines to target:

| Metric | Healthy Range |
| --------------------------- | ------------- |
| LLM-judge composite score | ≥ 0.80 |
| Schema validation pass rate | ≥ 98.5% |
| Tool call success rate | ≥ 97% |
| Implicit re-request rate | ≤ 5% |
| Task abandonment rate | ≤ 10% |
| Behavioral drift score | ≤ 0.10 |
| Output token variance (CV) | ≤ 0.35 |

These are starting points, not universal standards. Calibrate against your own baselines: a customer service agent and a code review agent have very different quality profiles. What matters is establishing a stable baseline and alerting on deviation from it.

## Conclusion

The gap between "the agent is running" and "the agent is working" is where production AI failures live. Uptime and latency tell you that your infrastructure is functioning. They tell you almost nothing about whether your agents are producing good outputs, behaving consistently, using tools effectively, or satisfying users.

Building real agent observability means instrumenting across five dimensions: output quality via LLM-judge scoring and structural validation, behavioral drift via embedding-based distribution tracking and prompt fingerprinting, tool call health beyond simple success/failure rates, token efficiency as a behavioral signal rather than just a cost signal, and user satisfaction connected to individual runs rather than collected in isolation.

Start with structural validation and tool call metrics: they are cheap, reliable, and catch a wide class of failures. Add LLM-judge scoring on a sample, connect user feedback to trace IDs, and establish drift baselines from your first stable deployment. Build toward a dashboard that lets you move from a degrading metric to the specific traces that explain it in under two minutes.

The teams that catch agent quality problems before their users do are not the ones with the most sophisticated infrastructure. They are the ones who decided early that output quality is a first-class operational metric, instrumented accordingly, and built the discipline to review it alongside latency and error rates. That decision is available to any team. It just requires treating what your agent says as seriously as whether it responds.

[Omnithium](https://omnithium.ai) is built to give enterprise teams exactly that: first-class observability, quality scoring, and end-to-end traceability from agent runs to user feedback. [Explore pricing](https://omnithium.ai/pricing) or visit our [resource library](https://omnithium.ai/resources) to learn more.