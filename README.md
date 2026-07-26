# Vehicle Fault Diagnosis System

A diagnostic assistant that combines a knowledge graph of 99 vehicle fault records with statistical sensor validation to diagnose vehicle faults.

## Architecture

The system follows a pipeline architecture:

```
Query Input → Preprocessing → Hybrid Retrieval → Fault Mapping
                                                    ↓
                                          Sensor Analysis (optional)
                                                    ↓
                                        Evidence Fusion → Reasoning Engine
                                                            ↓
                                              LLM Explanation → Streamlit UI
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `query_preprocessor.py` | Input normalization and expansion |
| `hybrid_retrieval.py` | KG + BM25 + Symptom embedding retrieval |
| `fault_mapper.py` | Candidate scoring and ranking |
| `sensor_analysis.py` | Statistical sensor validation |
| `evidence_fusion.py` | KG + sensor evidence fusion |
| `reasoning_engine.py` | Mode determination + reasoning chain |
| `llm_provider.py` | LLM abstraction (Ollama) |
| `explanation_generator.py` | Natural language explanation |

## Three Diagnostic Modes

1. **AMBIGUARY** - Needs more information from technician
2. **INFERRED** - Best guess based on evidence, needs confirmation
3. **EXTRACTED** - High confidence diagnosis

## Running

### Streamlit App
```bash
streamlit run app.py
```

### CLI
```bash
python num_pipeline/scripts/run_pipeline.py "brake pedal feels spongy"
python num_pipeline/scripts/run_pipeline.py --symptoms "ABS warning light" "brake pedal pulsation"
python num_pipeline/scripts/run_pipeline.py --skip-sensor "engine overheating"
```

### With Verbose Output
```bash
python num_pipeline/scripts/run_pipeline.py -v "steering pulls to the left"
```

### JSON Output
```bash
python num_pipeline/scripts/run_pipeline.py --json "check engine light, rough idle"
```

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- `streamlit` - Web UI
- `sentence-transformers` - Symptom embeddings
- `rank-bm25` - BM25 search
- `neo4j` - Knowledge graph database
- `requests` - LLM API calls (optional)

## Configuration

- Ollama model: `llama3.1:8b` (configured in `.opencode/opencode.json`)
- Sensor profiles: `num_pipeline/outputs/profiles/`
- Fault data: `num_pipeline/data/processed/sensor_mapping.json`
- Knowledge graph: `vehicle-fault-kg/`

## Design Principles

1. Knowledge Graph is the primary reasoning engine
2. LLM is used ONLY for explanation/presentation, never diagnosis
3. Missing sensor data never suppresses graph-derived diagnoses
4. Internal confidence scores are not exposed in the UI
