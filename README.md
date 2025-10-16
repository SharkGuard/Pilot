# ProofOfThought

LLM-based reasoning using Z3 theorem proving.

## Quick Start

```python
from openai import OpenAI
from z3dsl.reasoning import ProofOfThought

client = OpenAI(api_key="...")
pot = ProofOfThought(llm_client=client)

result = pot.query("Would Nancy Pelosi publicly denounce abortion?")
print(result.answer)  # False
```

## Batch Evaluation

```python
from z3dsl.reasoning import EvaluationPipeline

evaluator = EvaluationPipeline(pot, output_dir="results/")
result = evaluator.evaluate(
    dataset="strategyqa_train.json",
    max_samples=10
)
print(f"Accuracy: {result.metrics.accuracy:.2%}")
```

## Installation

```bash
pip install z3-solver openai scikit-learn numpy
```

## Architecture

The system has two layers:

1. **High-level API** (`z3dsl.reasoning`) - Simple Python interface for reasoning tasks
2. **Low-level DSL** (`z3dsl`) - JSON-based Z3 theorem prover interface

Most users should use the high-level API.

## Examples

See `examples/` directory for complete examples including Azure OpenAI support.
