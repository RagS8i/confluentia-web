# Careflow OCR

OCR pipeline for CareFlow document intake, backed by **GLM-OCR** served by a
local [Ollama](https://ollama.com/) instance. `model.py` extracts text from each
page; `organizer.py` performs the clinical-relevance pass afterwards.

CareFlow's document intake runs both steps in-process from
`app/intake.py`. There is no OCR HTTP service.

## Install

Install the repository dependencies, then install Ollama and pull the OCR model:

```sh
pip install -r requirements.txt
ollama pull glm-ocr
ollama serve
```

The default Ollama endpoint is `http://127.0.0.1:11434`. It can be changed with:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CAREFLOW_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama base URL |
| `CAREFLOW_OLLAMA_MODEL` | `qwen3:8b` | Ollama text model used by the organizer |
| `CAREFLOW_OCR_MODEL` | `glm-ocr` | Ollama vision model used for OCR |
| `CAREFLOW_OCR_TIMEOUT_S` | `300` | Per-page OCR request timeout |

The model is loaded lazily by Ollama on the first page. No separate OCR model
weights are downloaded by this repository.

## Usage

```python
from OCR1.model import extract_document
from OCR1.organizer import organize_ocr_result

ocr_output = extract_document("scan.pdf")
result = organize_ocr_result(ocr_output)
```

### Extraction output

```json
{
  "file_name": "scan.pdf",
  "ocr_model": "glm-ocr",
  "page_count": 1,
  "pages": [
    {
      "page_number": 1,
      "blocks": [
        {"label": "text", "text": "Hemoglobin: 12.5 g/dL"}
      ]
    }
  ]
}
```

GLM-OCR returns text but no pixel-accurate bounding boxes or confidence scores,
so blocks contain only a label and text.

### Organizer output

```json
{
  "file_name": "scan.pdf",
  "ocr_model": "glm-ocr",
  "is_medical": true,
  "sources": [
    {
      "block_id": "p1_b1",
      "text": "Hemoglobin: 12.5 g/dL",
      "page": 1,
      "unreadable": false
    }
  ]
}
```

The organizer uses a compact exclusion gate: it returns numeric ID ranges for
non-clinical blocks and flags unreadable kept blocks. The caller keeps every
remaining OCR block and rebuilds each source's text and page from the extraction
output, so the model cannot change provenance. Keeping clinical content by
default also makes the organizer response much shorter and faster.

Supported extensions: `.pdf .png .jpg .jpeg .webp .bmp .tif .tiff`
