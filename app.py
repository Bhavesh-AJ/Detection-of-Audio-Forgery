import json
import os
import tempfile
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import cgi

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8000))
MODEL_PATH = "cnn_bilstm_full.keras"
RF_MODEL_PATH = "random_forest.pkl"
SPOOF_THRESHOLD = 0.5

_model = None
_rf_model = None


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


# ─── CNN-BiLSTM ───────────────────────────────────────────────────────────────

def _load_model():
    global _model

    if _model is not None:
        return _model

    import tensorflow as tf

    print("Loading CNN-BiLSTM model:", MODEL_PATH)

    _model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("CNN-BiLSTM model loaded successfully")

    return _model


def _predict_audio(audio_path):

    import numpy as np
    from feature_extraction import extract_mfcc_for_dl

    print("\n========== AUDIO ==========")
    print("File:", audio_path)

    model = _load_model()

    mfcc = extract_mfcc_for_dl(audio_path)

    print("MFCC Shape:", mfcc.shape)

    features = np.expand_dims(
        mfcc.astype("float32"),
        axis=0
    )

    print("Input Shape:", features.shape)

    prediction = model.predict(
        features,
        verbose=0
    )

    print("\n===== DEBUG =====")
    print("Prediction Raw:", prediction)
    print("Prediction Shape:", prediction.shape)
    print("=================\n")

    spoof_probability = float(
        prediction[0][0]
    )

    print("\n===== PROBABILITY =====")
    print("Probability:", spoof_probability)
    print("Type:", type(spoof_probability))
    print("=======================\n")

    label = (
        "fake"
        if spoof_probability >= SPOOF_THRESHOLD
        else "real"
    )

    confidence = (
        spoof_probability
        if label == "fake"
        else 1.0 - spoof_probability
    )

    return {
        "label": label,
        "spoof_probability": spoof_probability,
        "confidence": confidence,
        "threshold": SPOOF_THRESHOLD,
        "frames": int(mfcc.shape[0]),
        "features": int(mfcc.shape[1]),
        "model": "cnn_bilstm",
    }


# ─── Random Forest ────────────────────────────────────────────────────────────

def _load_rf_model():
    global _rf_model

    if _rf_model is not None:
        return _rf_model

    import joblib

    print("Loading Random Forest model:", RF_MODEL_PATH)

    _rf_model = joblib.load(RF_MODEL_PATH)

    print("Random Forest model loaded successfully")

    return _rf_model


def _predict_audio_rf(audio_path):

    import numpy as np
    import librosa

    print("\n========== RF AUDIO ==========")
    print("File:", audio_path)

    model = _load_rf_model()

    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # Extract chroma features (same as training baseline)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features = np.mean(chroma, axis=1).reshape(1, -1)

    print("Feature Shape:", features.shape)

    proba = model.predict_proba(features)[0]

    # class order: 0 = real (bona-fide), 1 = fake (spoof)
    spoof_probability = float(proba[1]) if len(proba) > 1 else float(proba[0])

    label = (
        "fake"
        if spoof_probability >= SPOOF_THRESHOLD
        else "real"
    )

    confidence = (
        spoof_probability
        if label == "fake"
        else 1.0 - spoof_probability
    )

    print("RF Spoof Probability:", spoof_probability)

    return {
        "label": label,
        "spoof_probability": spoof_probability,
        "confidence": confidence,
        "threshold": SPOOF_THRESHOLD,
        "frames": len(y),
        "features": int(features.shape[1]),
        "model": "random_forest",
    }


# ─── HTTP Handler ─────────────────────────────────────────────────────────────

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
                "cnn_bilstm_model_path": MODEL_PATH,
                "cnn_bilstm_model_exists": os.path.exists(MODEL_PATH),
                "rf_model_path": RF_MODEL_PATH,
                "rf_model_exists": os.path.exists(RF_MODEL_PATH),
            }

            _json_response(
                self,
                200,
                payload
            )

            return

        if parsed.path == "/":
            self.path = "/index.html"

        return super().do_GET()

    def do_POST(self):

        parsed = urlparse(self.path)

        if parsed.path == "/predict":
            self._handle_predict(model_type="cnn_bilstm")
        elif parsed.path == "/predict_rf":
            self._handle_predict(model_type="random_forest")
        else:
            _json_response(
                self,
                404,
                {"error": "Unknown endpoint"}
            )

    def _handle_predict(self, model_type="cnn_bilstm"):

        if model_type == "cnn_bilstm" and not os.path.exists(MODEL_PATH):
            _json_response(
                self,
                500,
                {
                    "error": f"CNN-BiLSTM model file not found: {MODEL_PATH}"
                }
            )
            return

        if model_type == "random_forest" and not os.path.exists(RF_MODEL_PATH):
            _json_response(
                self,
                500,
                {
                    "error": f"Random Forest model file not found: {RF_MODEL_PATH}"
                }
            )
            return

        try:

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get(
                        "Content-Type",
                        ""
                    ),
                    "CONTENT_LENGTH": self.headers.get(
                        "Content-Length",
                        "0"
                    ),
                },
            )

            if "audio" not in form:

                _json_response(
                    self,
                    400,
                    {
                        "error": "Missing audio file field"
                    }
                )

                return

            audio_item = form["audio"]

            filename = os.path.basename(
                audio_item.filename or "sample.wav"
            )

            _, ext = os.path.splitext(
                filename
            )

            if not ext:
                ext = ".wav"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=ext
            ) as temp_audio:

                temp_audio.write(
                    audio_item.file.read()
                )

                temp_path = temp_audio.name

            try:

                if model_type == "random_forest":
                    result = _predict_audio_rf(temp_path)
                else:
                    result = _predict_audio(temp_path)

            finally:

                try:
                    os.remove(temp_path)

                except OSError:
                    pass

            _json_response(
                self,
                200,
                result
            )

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

            print("\n====================================")
            print("PREDICTION ERROR")
            print("====================================")

            traceback.print_exc()

            print("====================================\n")

            _json_response(
                self,
                500,
                {
                    "error": str(exc)
                }
            )


def main():

    print("Preloading CNN-BiLSTM model...")
    _load_model()
    print("CNN-BiLSTM model ready.")

    server = ThreadingHTTPServer(
        (HOST, PORT),
        AppHandler
    )

    print(f"Server running at http://127.0.0.1:{PORT}")

    server.serve_forever()


if __name__ == "__main__":
    main()