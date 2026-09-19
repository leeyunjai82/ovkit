# Contributing

Bug reports, models and capabilities are all welcome. This page is short on
purpose — here is what is actually different about this repository.

## Run what CI runs

```bash
git clone https://github.com/leeyunjai82/ovkit.git && cd ovkit
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python scripts/check.py          # ruff + black + pytest, the same three CI runs
python scripts/check.py --fix    # and fix what can be fixed automatically
```

`scripts/check.py` exists because "it passed locally" and "it passes in CI" came
apart three times. It is the same commands with the same exit codes; if it is
green, CI is green.

Tests never touch the network — `conftest.py` sets `OVKIT_OFFLINE=1` for every
one of them. A test that needs a model is a test that is really an integration
check, and those live in `scripts/verify_capabilities.py`.

## Adding a model is a YAML edit

Models are **data, not code**. One entry in `src/ovkit/manifests/`:

```yaml
my_model:
  src: hf
  repo: leeyunjai/ovkit-models
  filename: detect/my_model/model.xml
  task: detect
  description: Shown by `ovkit list`.
  license: apache-2.0
```

Two rules the code enforces, so you cannot forget them:

- **The licence must be one ovkit can pass on.** Permissive loads; OpenRAIL-M
  loads only with a `license_url` to point at; AGPL and CC-BY-NC are refused at
  load time. See the License section of the README for why.
- **The weights go on the mirror.** Everything ovkit serves comes from
  [`leeyunjai/ovkit-models`](https://huggingface.co/leeyunjai/ovkit-models), so
  one offline site can take one copy. Run the **Sync the model mirror**
  workflow, or `python scripts/sync_mirror.py --upload` with an `HF_TOKEN`.

## Adding a capability

A capability is a `Pipeline` subclass in `src/ovkit/pipelines/`, registered in
`PIPELINES`. It should answer a question a person would ask, not expose a
tensor — `print(r)` has to read like a sentence.

Give it a Korean name in `core/i18n.py` too. Someone who types `Model("얼굴분석")`
should not have to learn `face_analyze` first.

Inside a pipeline, call sub-models with **`.predict(...)`**, never `()`:

```python
out = self.model("face_detection").predict(image, conf=conf)   # always a list
```

`model(x)` shapes its answer to the input — one `Results` for a photo, a
stream for a camera — which is right for a person and wrong for plumbing that
wants the same shape every time. A test fails on the short form in `src/`.

## Reporting a bug

The useful bug report has the command you ran and the output you got, verbatim.
`python scripts/selfcheck.py` prints the versions and devices in one block and
saves a round trip.

If a model answers "nothing found" on a picture that plainly contains the thing,
say so — that class of bug hid ten broken capabilities behind a plausible
answer, and it is exactly what `scripts/verify_capabilities.py` was written to
catch.

## The spirit of the thing

ovkit goes into classrooms. That shapes more decisions than it looks:

- **An answer beats a tensor.** A student three weeks into Python has not met
  list indexing; a photo returns one result, not a list of one.
- **A wrong answer is worse than an error.** If the model found nothing, say
  nothing was found — do not return an empty box list and let it read as
  success.
- **No licence conversations.** A teacher should be able to use this without
  asking a lawyer first.
