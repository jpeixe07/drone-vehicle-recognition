"""HTTP validation and packaged UI regression tests without model downloads."""
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import HTTPServer

from vehicle_recognition.server import Handler


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_serves_frontend(self):
        with urlopen(self.url) as response:
            self.assertIn(b"/api/predict", response.read())

    def test_rejects_invalid_image_before_inference(self):
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.url + "/api/predict", data=b"invalid"))
        self.assertEqual(caught.exception.code, 400)

    def test_rejects_empty_upload(self):
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.url + "/api/predict", data=b""))
        self.assertEqual(caught.exception.code, 413)

    def test_unknown_route(self):
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.url + "/api/other", data=b"image"))
        self.assertEqual(caught.exception.code, 404)
