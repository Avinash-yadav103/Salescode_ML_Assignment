"""
app.py
======
Minimal Flask web demo + JSON API for the recapture detector.

    python app.py            # then open http://127.0.0.1:5000

* GET  /            -> a one-page uploader / camera demo
* POST /predict     -> multipart 'image' field, returns {"score": 0.93, ...}

Pure CPU, offline. Handy for a quick "show it working" without Streamlit.
"""

import io
import os
import time

from flask import Flask, request, jsonify, Response
from PIL import Image

from predict import predict

app = Flask(__name__)

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>Spot the Fake Photo</title>
<style>
 body{font-family:system-ui;max-width:640px;margin:40px auto;padding:0 16px}
 #score{font-size:42px;font-weight:700;margin:8px 0}
 .real{color:#138000}.fake{color:#c0392b}
 img{max-width:100%;border-radius:8px;margin-top:12px}
 input{margin:8px 0}
</style></head><body>
<h1>🕵️ Spot the Fake Photo</h1>
<p>Real photo vs. photo-of-a-screen. <b>0 = real, 1 = screen.</b></p>
<input type=file accept="image/*" capture="environment" id=f>
<div id=out></div>
<script>
const f=document.getElementById('f'),out=document.getElementById('out');
f.onchange=async()=>{
  if(!f.files[0])return;
  const fd=new FormData();fd.append('image',f.files[0]);
  out.innerHTML='scoring...';
  const r=await fetch('/predict',{method:'POST',body:fd});
  const j=await r.json();
  const fake=j.score>=0.5, cls=fake?'fake':'real';
  const label=fake?'PHOTO OF A SCREEN':'REAL photo';
  const conf=Math.round((fake?j.score:1-j.score)*100);
  out.innerHTML=`<div id=score class=${cls}>${j.score.toFixed(3)}</div>
    <b class=${cls}>${label}</b> · ${conf}% confidence · ${j.ms} ms
    <img src="${URL.createObjectURL(f.files[0])}">`;
};
</script></body></html>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify(error="no image"), 400
    raw = request.files["image"].read()
    tmp = "._api_tmp.jpg"
    Image.open(io.BytesIO(raw)).convert("RGB").save(tmp, quality=95)
    t0 = time.perf_counter()
    score = predict(tmp)
    ms = round((time.perf_counter() - t0) * 1000)
    os.remove(tmp)
    return jsonify(score=score, label="screen" if score >= 0.5 else "real", ms=ms)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
