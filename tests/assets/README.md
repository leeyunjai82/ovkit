# Test assets

`sample.png` is a 96x64 image with four flat quadrants and a thin border.

**It is generated, not photographed.** `scripts/make_sample.py` writes it, and
that matters for what it can be used to test:

- **Good for:** file I/O — that a file this repository ships decodes, that a
  name in any script round-trips, that channels are not swapped and the image
  is not half-read. Each quadrant is a known BGR value, so a test can assert on
  a pixel and a person can see a wrong decode at a glance.
- **Not good for:** anything that needs a real scene. A detector finds no people
  in four rectangles. Capability checks need real photographs, and
  `scripts/verify_capabilities.py` downloads those at run time rather than
  committing megabytes of pictures whose licence would then be ovkit's problem.
