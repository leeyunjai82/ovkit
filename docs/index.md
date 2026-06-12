# ovkit

**A simple Python inference API for OpenVINO.** One import, one `Model` class, a
callable object, and clean `Results` — with OpenVINO's strengths layered on top:
`AUTO`/`NPU` devices, async throughput, and INT8 quantization.

```python
from ovkit import Model

model = Model("rtdetr_r50")            # name -> auto download / convert / cache
results = model("img.jpg", conf=0.25)  # __call__ == predict

for r in results:
    print(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)
    r.save("out.jpg")
```

ovkit is **Apache-2.0** and stays license-clean: it never bundles or downloads
AGPL-licensed model stacks or non-commercial weights (InsightFace
pretrained). See the [license policy](usage.md#license-policy).

```{toctree}
:maxdepth: 2
:caption: Contents

usage
api
```

## Indices

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
