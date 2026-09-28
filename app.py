"""Local Flask API for CLIP zero-shot and prototype classification."""

from __future__ import annotations

import io
import json
import os
import re
import threading
from collections import defaultdict
from pathlib import Path
from string import Formatter
from typing import Any

from flask import Flask, jsonify, request, send_from_directory, Response
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import HTTPException


BASE_DIR = Path(__file__).resolve().parent
MODEL_ID = "openai/clip-vit-base-patch32"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_LABELS = 30
MAX_LABEL_LENGTH = 80
MAX_EXAMPLES = 8
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = (MAX_EXAMPLES + 1) * MAX_IMAGE_BYTES + 1024 * 1024

_model_lock = threading.Lock()
_model_bundle: tuple[Any, Any, Any, str] | None = None


class APIError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def get_model() -> tuple[Any, Any, Any, str]:
    """Load the shared model lazily, once per server process."""
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle

    with _model_lock:
        if _model_bundle is None:
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor
            except ImportError as error:
                raise APIError(
                    "Model dependencies are missing. Install requirements.txt, then restart the server.",
                    503,
                ) from error

            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                processor = CLIPProcessor.from_pretrained(MODEL_ID, use_fast=False)
                model = CLIPModel.from_pretrained(MODEL_ID)
                model.to(device)
                model.eval()
            except Exception as error:
                raise APIError(
                    "CLIP could not be loaded. Check the network connection and model cache, then restart.",
                    503,
                ) from error
            _model_bundle = (torch, model, processor, device)
    return _model_bundle


def parse_labels(raw_labels: str) -> list[str]:
    try:
        labels = json.loads(raw_labels)
    except (TypeError, json.JSONDecodeError) as error:
        raise APIError("Labels must be sent as a JSON array of strings.") from error

    if not isinstance(labels, list) or not 1 <= len(labels) <= MAX_LABELS:
        raise APIError(f"Provide between 1 and {MAX_LABELS} candidate labels.")

    cleaned_labels = []
    seen = set()
    for label in labels:
        if not isinstance(label, str):
            raise APIError("Each candidate label must be text.")
        label = re.sub(r"\s+", " ", label).strip()
        if not label or len(label) > MAX_LABEL_LENGTH:
            raise APIError(f"Labels must contain 1 to {MAX_LABEL_LENGTH} characters.")
        if label.casefold() in seen:
            raise APIError("Candidate labels must be unique.")
        seen.add(label.casefold())
        cleaned_labels.append(label)
    return cleaned_labels


def read_image(upload: Any) -> Image.Image:
    if upload is None or not upload.filename:
        raise APIError("Choose an image to classify.")

    extension = Path(upload.filename).suffix.lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS or upload.mimetype not in ALLOWED_MIME_TYPES:
        raise APIError("Use a JPG, JPEG, PNG, or WEBP image.")

    image_bytes = upload.stream.read(MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise APIError("Each image must be 10 MB or smaller.", 413)
    if not image_bytes:
        raise APIError("The uploaded image is empty.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise APIError("Image dimensions must be 25 megapixels or smaller.")
            image.verify()
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGBA")
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                return background
            return image.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as error:
        raise APIError("The uploaded file is not a valid image.") from error


def parse_threshold() -> float:
    try:
        threshold = float(request.form.get("threshold", "0"))
    except ValueError as error:
        raise APIError("Threshold must be a number from 0 to 1.") from error
    if not 0 <= threshold <= 1:
        raise APIError("Threshold must be between 0 and 1.")
    return threshold


def parse_template() -> str:
    template = request.form.get("prompt_template", "a photo of {label}").strip()
    try:
        fields = list(Formatter().parse(template))
    except ValueError as error:
        raise APIError("Prompt template must contain {label} exactly once.") from error
    placeholders = [(field, spec, conversion) for _, field, spec, conversion in fields if field is not None]
    if (
        len(template) > 160
        or len(placeholders) != 1
        or placeholders[0] != ("label", "", None)
    ):
        raise APIError("Prompt template must contain {label} exactly once.")
    return template


def rank_embeddings(
    torch: Any,
    model: Any,
    image_embedding: Any,
    text_embeddings: Any,
    labels: list[str],
    threshold: float,
) -> dict[str, Any]:
    image_embedding = torch.nn.functional.normalize(image_embedding, dim=-1)
    text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
    similarities = image_embedding @ text_embeddings.T
    logits = similarities * model.logit_scale.exp()
    scores = torch.softmax(logits, dim=-1)

    results = [
        {
            "label": label,
            "similarity": float(similarities[0, index].item()),
            "logit": float(logits[0, index].item()),
            "score": float(scores[0, index].item()),
        }
        for index, label in enumerate(labels)
    ]
    results.sort(key=lambda item: item["score"], reverse=True)
    filtered_results = [item for item in results if item["score"] >= threshold]
    return {
        "results": filtered_results,
        "all_results": results,
        "threshold": threshold,
        "filtered_count": len(results) - len(filtered_results),
        "score_note": "ranking score, not calibrated probability",
    }


def zero_shot_prediction(image: Image.Image, labels: list[str], template: str, threshold: float) -> dict[str, Any]:
    torch, model, processor, device = get_model()
    prompts = [template.format(label=label) for label in labels]
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs)
    ranked = rank_embeddings(torch, model, output.image_embeds, output.text_embeds, labels, threshold)
    return {**ranked, "model_id": MODEL_ID, "mode": "zero-shot", "prompt_template": template}


def few_shot_prediction(
    query_image: Image.Image,
    examples: list[tuple[Image.Image, str]],
    threshold: float,
) -> dict[str, Any]:
    torch, model, processor, device = get_model()
    class_images: dict[str, list[Image.Image]] = defaultdict(list)
    display_labels: dict[str, str] = {}
    for image, label in examples:
        class_key = label.casefold()
        display_labels.setdefault(class_key, label)
        class_images[class_key].append(image)

    labels = list(display_labels.values())
    if len(labels) > MAX_LABELS:
        raise APIError(f"Use no more than {MAX_LABELS} distinct example labels.")

    example_inputs = processor(images=[image for image, _ in examples], return_tensors="pt")
    query_inputs = processor(images=query_image, return_tensors="pt")
    example_inputs = {name: value.to(device) for name, value in example_inputs.items()}
    query_inputs = {name: value.to(device) for name, value in query_inputs.items()}
    with torch.inference_mode():
        example_embeddings = model.get_image_features(**example_inputs)
        query_embedding = model.get_image_features(**query_inputs)

    example_embeddings = torch.nn.functional.normalize(example_embeddings, dim=-1)
    prototypes = []
    for label in labels:
        indices = [index for index, (_, example_label) in enumerate(examples) if example_label.casefold() == label.casefold()]
        prototype = example_embeddings[indices].mean(dim=0, keepdim=True)
        prototypes.append(torch.nn.functional.normalize(prototype, dim=-1))
    prototype_embeddings = torch.cat(prototypes, dim=0)
    ranked = rank_embeddings(torch, model, query_embedding, prototype_embeddings, labels, threshold)
    return {
        **ranked,
        "model_id": MODEL_ID,
        "mode": "few-shot-prototypes",
        "examples_per_class": {label: len(class_images[label.casefold()]) for label in labels},
        "method_note": "Class prototypes are averages of normalized CLIP image embeddings; the model is not fine-tuned.",
    }


@app.errorhandler(APIError)
def handle_api_error(error: APIError):
    return jsonify({"error": {"message": error.message}}), error.status_code


@app.errorhandler(413)
def handle_request_too_large(_error: Any):
    return jsonify({"error": {"message": "The total upload is too large. Reduce image sizes or upload fewer examples."}}), 413


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception):
    if isinstance(error, HTTPException):
        return jsonify({"error": {"message": error.description}}), error.code
    app.logger.exception("Unhandled request failure")
    return jsonify({"error": {"message": "An internal server error occurred. Check the local Flask terminal."}}), 500


LOCAL_ORIGIN_REGEX = re.compile(
    r"^(https?://(127\.0\.0\.1|localhost|0\.0\.0\.0|\[::1\]|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+)(:\d+)?|null)$",
    re.IGNORECASE,
)


def is_allowed_origin(origin: str | None) -> bool:
    if not origin:
        return True
    clean_origin = origin.rstrip("/")
    if clean_origin == request.host_url.rstrip("/"):
        return True
    return bool(LOCAL_ORIGIN_REGEX.match(clean_origin))


@app.before_request
def reject_cross_origin_api_requests():
    if request.method == "OPTIONS":
        return None
    if request.path.startswith("/api/") and request.method == "POST":
        origin = request.headers.get("Origin")
        if origin and not is_allowed_origin(origin):
            raise APIError("Cross-origin API requests are not allowed.", 403)
        if request.headers.get("Sec-Fetch-Site") == "cross-site" and not is_allowed_origin(origin):
            raise APIError("Cross-origin API requests are not allowed.", 403)


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin and is_allowed_origin(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
        response.headers["Access-Control-Max-Age"] = "86400"
    elif not origin and request.path.startswith("/api/"):
        response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.route("/api/<path:_path>", methods=["OPTIONS"])
def api_options(_path: str):
    response = app.make_default_options_response()
    origin = request.headers.get("Origin")
    if origin and is_allowed_origin(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response



@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/styles.css")
def styles():
    return send_from_directory(BASE_DIR, "styles.css")


@app.get("/button.js")
@app.get("/app.js")
def javascript():
    return send_from_directory(BASE_DIR, "button.js")


@app.get("/favicon.ico")
def favicon():
    svg_icon = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
        <rect width="32" height="32" rx="8" fill="#0d9488"/>
        <circle cx="16" cy="16" r="7" fill="#67e8f9"/>
        <circle cx="16" cy="16" r="3" fill="#0f172a"/>
    </svg>"""
    return Response(svg_icon, mimetype="image/svg+xml")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_id": MODEL_ID,
        "model_loaded": _model_bundle is not None,
        "device": _model_bundle[3] if _model_bundle is not None else None,
    })


@app.post("/api/classify/zero-shot")
def classify_zero_shot():
    image = read_image(request.files.get("image"))
    labels = parse_labels(request.form.get("labels", ""))
    template = parse_template()
    threshold = parse_threshold()
    return jsonify(zero_shot_prediction(image, labels, template, threshold))


@app.post("/api/classify/few-shot")
def classify_few_shot():
    query_image = read_image(request.files.get("image"))
    uploads = request.files.getlist("examples")
    labels = request.form.getlist("example_labels")
    if not uploads or len(uploads) != len(labels) or len(uploads) > MAX_EXAMPLES:
        raise APIError(f"Upload 1 to {MAX_EXAMPLES} examples, each with a class label.")

    examples = []
    for upload, label in zip(uploads, labels):
        cleaned_label = re.sub(r"\s+", " ", label).strip()
        if not cleaned_label or len(cleaned_label) > MAX_LABEL_LENGTH:
            raise APIError(f"Example labels must contain 1 to {MAX_LABEL_LENGTH} characters.")
        examples.append((read_image(upload), cleaned_label))

    threshold = parse_threshold()
    return jsonify(few_shot_prediction(query_image, examples, threshold))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
