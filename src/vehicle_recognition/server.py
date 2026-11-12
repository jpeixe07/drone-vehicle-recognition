"""Local HTTP interface sharing the offline prediction pipeline."""
import argparse
import io
import json
from http.server import SimpleHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageOps, UnidentifiedImageError
from .configuration import configure_environment, load_settings

MAX_UPLOAD = 20 * 1024 * 1024


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent / "web"), **kwargs)

    def reply(self, status, payload):
        body = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/predict":
            return self.reply(404, {"error": "Rota desconhecida."})
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if not 0 < size <= MAX_UPLOAD:
            return self.reply(413, {"error": "Envie uma imagem de até 20 MB."})
        try:
            with Image.open(io.BytesIO(self.rfile.read(size))) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return self.reply(400, {"error": "Arquivo de imagem inválido."})
        try:
            import torch
            threads = int(self.server.settings.config["recognition"].get("cpu_threads", 4))
            if threads > 0:
                torch.set_num_threads(threads)
            from .pipeline import run_predictions
            with TemporaryDirectory(prefix="sobrevoo-") as directory:
                root = Path(directory)
                image.save(root / "image.png")
                output = root / "results"
                run_predictions(self.server.settings, root / "image.png", output=output)
                result = json.loads((output / "predictions.json").read_text())
                if result["images"][0].get("error"):
                    raise RuntimeError("Falha na leitura da imagem.")
                result["acceptance_threshold"] = self.server.settings.config["recognition"]["acceptance_threshold"]
            self.reply(200, result)
        except Exception:
            import logging
            logging.exception("Prediction failed")
            self.reply(503, {"error": "Falha na análise. Confira os pesos, o índice e o log do servidor."})


def main():
    parser = argparse.ArgumentParser(description="Sobrevoo local web interface")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    configure_environment(args.root.resolve())
    settings = load_settings(args.root, args.config)
    from dotenv import load_dotenv
    load_dotenv(settings.root / ".env", override=False)
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    server.settings = settings
    print(f"Sobrevoo: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
