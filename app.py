import json
import os
import tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import cgi


HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8000))
MODEL_PATH = "cnn_bilstm_full.keras"
SPOOF_THRESHOLD = 0.5

_model = None


def _json_response(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(body)


def _load_model():
    global _model
    if _model is not None:
        return _model

    import tensorflow as tf

    _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model


def _predict_audio(audio_path):
    import numpy as np
    from feature_extraction import extract_mfcc_for_dl

    model = _load_model()
    mfcc = extract_mfcc_for_dl(audio_path)
    features = np.expand_dims(mfcc.astype("float32"), axis=0)
    prediction = model.predict(features, verbose=0)

    print("\n===== DEBUG =====")
    print("Prediction Raw:", prediction)
    print("Shape:", prediction.shape)
    print("=================\n")

    spoof_probability = float(prediction[0][0])

    print("\n===== PROBABILITY =====")
    print("Probability:", spoof_probability)
    print("Type:", type(spoof_probability))
    print("=======================\n")

    label = "fake" if spoof_probability >= SPOOF_THRESHOLD else "real"
    confidence = spoof_probability if label == "fake" else 1.0 - spoof_probability
    return {
        "label": label,
        "spoof_probability": spoof_probability,
        "confidence": confidence,
        "threshold": SPOOF_THRESHOLD,
        "frames": int(mfcc.shape[0]),
        "features": int(mfcc.shape[1]),
    }


class AppHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            payload = {
                "ok": True,
                "model_path": MODEL_PATH,
                "model_exists": os.path.exists(MODEL_PATH),
            }
            _json_response(self, 200, payload)
            return

        if parsed.path == "/":
            self.path = "/index.html"

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/predict":
            _json_response(self, 404, {"error": "Unknown endpoint"})
            return

        if not os.path.exists(MODEL_PATH):
            _json_response(self, 500, {"error": f"Model file not found: {MODEL_PATH}"})
            return

        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                },
            )

            if "audio" not in form:
                _json_response(self, 400, {"error": "Missing audio file field"})
                return

            audio_item = form["audio"]
            filename = os.path.basename(audio_item.filename or "sample.wav")
            _, ext = os.path.splitext(filename)
            if not ext:
                ext = ".wav"

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_audio:
                temp_audio.write(audio_item.file.read())
                temp_path = temp_audio.name

            try:
                result = _predict_audio(temp_path)
            finally:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

            _json_response(self, 200, result)

        except ModuleNotFoundError as exc:
            _json_response(
                self,
                500,
                {
                    "error": f"Missing Python dependency: {exc.name}",
                    "hint": "Install project dependencies, then restart the server.",
                },
            )
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})


def main():
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"Audio Forgery Detection server running at http://{HOST}:{PORT}")
    print("Open the web page from this URL so upload and recording can call the model.")
    server.serve_forever()


if __name__ == "__main__":
    main()
