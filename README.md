# Zero-Shot Image Classification with CLIP and Few-Shot Learning

A local educational image-classification app. It uses the pretrained Hugging Face model `openai/clip-vit-base-patch32` to compare an uploaded image with text labels (zero-shot) or with class prototypes made from example images (few-shot). The existing Virtual Vision dashboard is the front end; Flask serves it and the JSON API from one local origin.

## Requirements

- Windows 10/11
- Python 3.12 (64-bit) available from the Python launcher as `py -3.12`
- Internet access for the first model download; later runs use the Hugging Face cache
- Several gigabytes of free disk space are recommended for Python packages and cached model files
- CPU inference is supported. CUDA is used only if the installed PyTorch build and machine support it.

The model weights are downloaded lazily the first time a classification request is made, not while the server starts or when the health check runs. That first request can take several minutes. The server runs locally and does not require API keys.

## Setup on Windows

Open this project folder in VS Code, then open a PowerShell terminal in the folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, either allow scripts for the current user with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or skip activation and run `.\.venv\Scripts\python.exe` in place of `python` in the commands below. Installing PyTorch and Transformers downloads sizable packages; the first classification also downloads model weights.

## Run

Start Flask in the project folder:

```powershell
python app.py
```

Open <http://127.0.0.1:5000/> in a browser. Keep the terminal open while using the app. Flask serves the page, styles, JavaScript, and API together so the browser can call the backend without a separate CORS server. The dashboard's model status comes from `/api/health`.

## Use the dashboard

1. Upload or drop a JPG, JPEG, PNG, or WEBP image (10 MB maximum per image).
2. In Zero-shot mode, enter candidate labels separated by commas or new lines. Results only rank the labels you supplied.
3. Select a prompt template. Its `{label}` text is sent to CLIP, so wording can change the ranking.
4. Set the score threshold. Results below it are hidden; the displayed score is a softmax ranking score over the current candidates, not calibrated confidence.
5. In Few-shot mode, add up to eight example images with class labels, then classify a separate query image. Examples stay in this browser tab's session workflow and are sent for that request; they are not saved as model training or fine-tuning.
6. Export the actual prediction history as CSV or JSON. The history is held in this browser tab's session storage. No accuracy is reported because the app has no independent ground-truth labels.

The initial model name and all backend responses use `openai/clip-vit-base-patch32`; the interface does not claim ViT-L/14.

## API

All endpoints are served by the Flask app on `127.0.0.1:5000`.

- `GET /api/health` returns `status`, `model_id`, `model_loaded`, and `device`. It never triggers model loading.
- `POST /api/classify/zero-shot` accepts multipart fields `image`, `labels` (a JSON array), `prompt_template` (exactly one `{label}`), and optional `threshold` (0 to 1).
- `POST /api/classify/few-shot` accepts multipart fields `image` (query), repeated `examples`, matching repeated `example_labels`, and optional `threshold`.

Successful classifications return ranked `results` and unfiltered `all_results`. Each result includes cosine `similarity`, learned-scale `logit`, and `score` (softmax-normalized ranking). Errors have the shape `{"error":{"message":"..."}}`. Uploads are read in memory, their image data is decoded and checked, and client filenames are never used as filesystem paths.

## Tests and manual checklist

Run backend API tests without loading/downloading CLIP:

```powershell
python -m unittest discover -s tests -v
```

Manual UI checks:

- Confirm the model status says `openai/clip-vit-base-patch32` and starts as not loaded.
- Try an unsupported file and a file larger than 10 MB; confirm a clear message appears.
- Try a valid image, vary candidate labels and prompt templates, and verify results come from the API.
- Raise the threshold and confirm lower-scoring results are filtered without their scores being relabeled as probabilities.
- Add examples from at least two classes, classify a different query image, remove an example, and try an empty class label.
- Export JSON and CSV after predictions; verify each contains the actual prediction and model/mode information.
- Stop Flask and confirm the page surfaces a backend/network error rather than inventing results.

## Reproducibility and limitations

The model ID is fixed in `app.py` as `openai/clip-vit-base-patch32`; package versions are pinned in `requirements.txt`. Inference uses the Hugging Face `CLIPProcessor`, `model.eval()`, and `torch.inference_mode()`. It normalizes CLIP embeddings, computes cosine similarity, multiplies by the model's learned exponential logit scale, and applies softmax over the current candidate set for ranking. Softmax values are not calibrated real-world probabilities and change when candidate labels change. Prompt choice and candidate wording affect results.

Few-shot mode averages normalized image embeddings into class prototypes. It does not fine-tune CLIP, and its rankings can be weak when examples are few, unrepresentative, or visually varied. A score threshold filters the ranking results; it is not a correctness threshold. There is no labeled evaluation set, so no accuracy is estimated. CPU inference may be slow. The app keeps prediction history in the current browser tab's session storage and sends the chosen query and examples to the local backend for inference.

If model loading fails, check internet access, available disk space, and the terminal output. To verify basic backend availability without loading the model, open <http://127.0.0.1:5000/api/health>.
