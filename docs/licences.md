# Licence audit

Date: 2026-09-28. Audited: every package in the three lock files (`uv.lock`, `workers/qwen3tts/uv.lock`,
`workers/qa/uv.lock`), every model the service pins, and `THIRD_PARTY_NOTICES` against the code that is
actually vendored. Evidence is labelled as the design labels it: **KNOW** (read here from the source named),
**BELIEVE** (from a secondary source, named), **ASSUME** (a placeholder).

## The policy

From plan.md §1.4 and AGENTS.md §1 rule 9. This project's code is PolyForm Noncommercial 1.0.0, and its prose
is CC BY-NC 4.0.

- Permissive licences (MIT, BSD, Apache-2.0, ISC, PSF and the like) are fine.
- GPL and AGPL are not allowed anywhere, tests included.
- Nothing non-commercial or "research only" is allowed.
- LGPL needs the lead's OK.
- MPL-2.0 is not named in the policy. It is weak, file-level copyleft: it binds changes to the MPL-covered
  files themselves, not the code that uses them. It is listed under Findings so the lead can confirm it.

## How the tables were made

1. Each lock file was parsed (`tomllib`), one row per `[[package]]` entry, transitive dependencies included.
   The project's own packages (`source = { editable = … }`) are marked "this project".
2. For each package from a registry, the PyPI JSON API was read for the exact locked version
   (`https://pypi.org/pypi/<name>/<version>/json`), taking `info.license_expression`, then `info.license`, then
   the `License ::` classifiers. torch and torchaudio are locked from the `pytorch-cu128` index; the PyPI
   metadata of the same version (2.11.0) was used for them.
3. Where that metadata was empty, a whole licence text, or disagreed with itself, the wheel's own
   `*.dist-info/METADATA` and licence files were read. Large wheels (torch, the nvidia-* CUDA wheels) were not
   downloaded: their `METADATA` was read through PyPI's PEP 658 metadata files, and their licence files through
   HTTP range requests on the wheel's zip directory. The one package with no wheel (`sox`) was read from its
   sdist.
4. For the server project, the result was cross-checked against `importlib.metadata` in the installed
   environment: the two agree for every installed package. The worker environments were not installed for
   this audit.
5. Dependency chains (Findings) come from the lock files' own dependency edges.

The "source" column says where each licence was read: `license_expression` (the SPDX field), `license field`,
`classifier`, or a licence file in the wheel. "dev only" marks packages reached only through the `dev`
dependency group, which a user's install does not need.

All the licences in these tables are **KNOW** in the sense above (read from the package's own metadata or
files), except where a note says otherwise.

## Server project (`uv.lock`, 52 packages)

| package | version | licence | source | notes |
|---|---|---|---|---|
| annotated-types | 0.8.0 | MIT | license_expression |  |
| anyio | 4.15.1 | MIT | license_expression |  |
| attrs | 26.1.0 | MIT | license_expression |  |
| basedpyright | 1.40.1 | MIT | classifier; LICENSE.txt in wheel | dev only |
| cffi | 2.1.1 | MIT-0 | license_expression |  |
| click | 8.5.0 | BSD-3-Clause | license_expression |  |
| colorama | 0.4.6 | BSD-3-Clause | classifier; LICENSE.txt in wheel | dev only; Windows only |
| cryptography | 50.0.1 | Apache-2.0 OR BSD-3-Clause | license_expression | dual: Apache-2.0 OR BSD-3-Clause |
| h11 | 0.16.0 | MIT | license field; classifier |  |
| httpcore2 | 2.13.1 | BSD-3-Clause | license_expression |  |
| httpx2 | 2.13.1 | BSD-3-Clause | license_expression |  |
| httpx2-jsfetch | 1.0 | BSD-3-Clause | license_expression | Emscripten/Pyodide only; not installed on Windows or Linux |
| idna | 3.20 | BSD-3-Clause | license_expression |  |
| iniconfig | 2.3.0 | MIT | license_expression | dev only |
| jiwer | 4.0.0 | Apache-2.0 | license_expression |  |
| jsonschema | 4.26.0 | MIT | license_expression |  |
| jsonschema-specifications | 2025.9.1 | MIT | license_expression |  |
| mcp | 2.2.0 | MIT | license field; classifier |  |
| mcp-types | 2.2.0 | MIT | license field; classifier |  |
| narration | 0.1.0.dev0 | this project (PolyForm-Noncommercial-1.0.0) |  | workspace member (editable) |
| narration-worker | 0.1.0 | this project (PolyForm-Noncommercial-1.0.0) |  | workspace member (editable) |
| nodejs-wheel-binaries | 24.19.0 | MIT (wrapper) | classifier; LICENSE in wheel | dev only (basedpyright's runtime); ships a Node.js binary, whose own licence is MIT with bundled third-party notices |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | license_expression | wheels also bundle OpenBLAS (BSD-3-Clause) and the GCC runtime (GPL-3.0-or-later WITH GCC-exception-3.1), listed in LICENSE.txt |
| nvidia-ml-py | 13.615.71 | BSD (variant not stated) | license field; classifier | NVML bindings (pynvml) |
| opentelemetry-api | 1.45.0 | Apache-2.0 | license_expression |  |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | license_expression | dev only; dual: Apache-2.0 OR BSD-2-Clause |
| pluggy | 1.6.0 | MIT | license field; classifier | dev only |
| psutil | 7.2.2 | BSD-3-Clause | license field; LICENSE in wheel |  |
| pycparser | 3.0 | BSD-3-Clause | license_expression |  |
| pydantic | 2.13.5 | MIT | license_expression |  |
| pydantic-core | 2.46.5 | MIT | license_expression |  |
| pygments | 2.21.0 | BSD-2-Clause | license_expression | dev only |
| pyjwt | 2.15.0 | MIT | license_expression |  |
| pyloudnorm | 0.2.0 | MIT | license_expression |  |
| pytest | 9.1.1 | MIT | license_expression | dev only |
| pytest-timeout | 2.4.0 | MIT | license field; classifier | dev only |
| python-multipart | 0.0.32 | Apache-2.0 | license_expression |  |
| pywin32 | 312 | PSF | license field; licence files in wheel | Windows only (via mcp). The wheel's `adodbapi` sub-package is LGPL-2.1; this project never imports it |
| rapidfuzz | 3.14.6 | MIT | license_expression |  |
| referencing | 0.37.0 | MIT | license_expression |  |
| regex | 2026.9.10 | Apache-2.0 AND CNRI-Python | license_expression | CNRI-Python is a permissive, OSI-approved licence |
| rfc8785 | 0.1.4 | Apache-2.0 | classifier; LICENSE in wheel |  |
| rpds-py | 2026.6.3 | MIT | license_expression |  |
| ruff | 0.16.9 | MIT | license_expression | dev only |
| scipy | 1.18.1 | BSD-3-Clause | classifier; LICENSE.txt in wheel | license field holds the whole text; wheels bundle OpenBLAS (BSD-3-Clause) and the GCC runtime (GPL-3.0-or-later WITH GCC-exception-3.1) |
| soundfile | 0.14.0 | BSD-3-Clause | license field; LICENSE in wheel | platform wheels bundle libsndfile (LGPL-2.1) with LAME (LGPL-2+) and mpg123 (LGPL-2.1). See Findings |
| sse-starlette | 3.4.11 | BSD-3-Clause | license_expression |  |
| starlette | 1.7.0 | BSD-3-Clause | license_expression |  |
| truststore | 0.10.4 | MIT | license_expression |  |
| typing-extensions | 4.16.0 | PSF-2.0 | license_expression |  |
| typing-inspection | 0.4.4 | MIT | license_expression |  |
| uvicorn | 0.54.0 | BSD-3-Clause | license_expression |  |

## Qwen3-TTS worker (`workers/qwen3tts/uv.lock`, 114 packages)

| package | version | licence | source | notes |
|---|---|---|---|---|
| accelerate | 1.12.0 | Apache-2.0 | license field; classifier |  |
| annotated-doc | 0.0.5 | MIT | license_expression |  |
| annotated-types | 0.8.0 | MIT | license_expression |  |
| anyio | 4.15.1 | MIT | license_expression |  |
| brotli | 1.2.0 | MIT | license field |  |
| certifi | 2026.7.22 | MPL-2.0 | license field; LICENSE in wheel | weak copyleft (file level). See Findings |
| cffi | 2.1.1 | MIT-0 | license_expression |  |
| charset-normalizer | 3.5.1 | MIT | license field |  |
| click | 8.5.0 | BSD-3-Clause | license_expression |  |
| cloudpickle | 3.1.2 | BSD-3-Clause | license field; classifier |  |
| colorama | 0.4.6 | BSD-3-Clause | classifier; LICENSE.txt in wheel | Windows only |
| cuda-bindings | 12.9.9 | Apache-2.0 | license_expression |  |
| cuda-pathfinder | 1.8.2 | Apache-2.0 | license_expression |  |
| cuda-toolkit | 12.8.1 | none stated | wheel METADATA (no licence field, classifier or file) | an empty NVIDIA meta-package that selects the nvidia-* wheels below. See Findings |
| decorator | 5.3.1 | BSD-2-Clause | license field |  |
| einops | 0.8.2 | MIT | license field; classifier |  |
| fastapi | 0.141.1 | MIT | license_expression |  |
| filelock | 4.0.4 | MIT | license_expression |  |
| flatbuffers | 25.12.19 | Apache-2.0 | license field; classifier |  |
| fsspec | 2026.9.0 | BSD-3-Clause | license_expression |  |
| gradio | 6.17.3 | Apache-2.0 | license_expression | pulled in by qwen-tts for its demo UI; the worker does not use it |
| gradio-client | 2.5.0 | Apache-2.0 | license_expression |  |
| groovy | 0.1.2 | MIT | classifier |  |
| h11 | 0.16.0 | MIT | license field; classifier |  |
| hf-gradio | 0.4.1 | MIT | license_expression |  |
| hf-xet | 1.6.0 | Apache-2.0 | license_expression |  |
| httpcore | 1.0.9 | BSD-3-Clause | license_expression |  |
| httpx | 0.28.1 | BSD-3-Clause | license field; classifier |  |
| huggingface-hub | 0.36.2 | Apache-2.0 | license field; classifier |  |
| idna | 3.20 | BSD-3-Clause | license_expression |  |
| iniconfig | 2.3.0 | MIT | license_expression | dev only |
| jinja2 | 3.1.6 | BSD-3-Clause | classifier; LICENSE.txt in wheel |  |
| joblib | 1.6.0 | BSD-3-Clause | license_expression |  |
| lazy-loader | 0.6 | BSD-3-Clause | license_expression |  |
| librosa | 1.0.0 | ISC | license field; classifier |  |
| llvmlite | 0.49.0 | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | license_expression |  |
| markdown-it-py | 4.2.0 | MIT | classifier; LICENSE in wheel |  |
| markupsafe | 3.0.3 | BSD-3-Clause | license_expression |  |
| mdurl | 0.1.2 | MIT | classifier; LICENSE in wheel |  |
| mpmath | 1.3.0 | BSD (variant not stated) | license field; classifier |  |
| msgpack | 1.2.2 | Apache-2.0 | license_expression |  |
| narration-worker | 0.1.0 | this project (PolyForm-Noncommercial-1.0.0) |  | workspace member (editable) |
| narration-worker-qwen3tts | 0.1.0 | this project (PolyForm-Noncommercial-1.0.0) |  | workspace member (editable) |
| narwhals | 2.26.0 | MIT | license_expression |  |
| networkx | 3.7 | BSD-3-Clause | license_expression |  |
| numba | 0.67.0 | BSD (variant not stated) | license field; classifier |  |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | license_expression | wheels also bundle OpenBLAS (BSD-3-Clause) and the GCC runtime (GPL-3.0-or-later WITH GCC-exception-3.1), listed in LICENSE.txt |
| nvidia-cublas-cu12 | 12.8.4.1 | NVIDIA Proprietary Software | license field; classifier Other/Proprietary | CUDA runtime library. See Findings |
| nvidia-cuda-cupti-cu12 | 12.8.90 | NVIDIA Proprietary Software | license field; classifier Other/Proprietary | CUDA runtime library. See Findings |
| nvidia-cuda-nvrtc-cu12 | 12.8.93 | NVIDIA Proprietary Software | license field; classifier Other/Proprietary | CUDA runtime library. See Findings |
| nvidia-cuda-runtime-cu12 | 12.8.90 | NVIDIA Proprietary Software | license field; classifier Other/Proprietary | CUDA runtime library. See Findings |
| nvidia-cudnn-cu12 | 9.19.0.56 | LicenseRef-NVIDIA-Proprietary | wheel METADATA License-Expression; License.txt | NVIDIA SDK licence agreement. See Findings |
| nvidia-cufft-cu12 | 11.3.3.83 | NVIDIA Proprietary Software | license field; classifier Other/Proprietary | CUDA runtime library. See Findings |
| nvidia-cufile-cu12 | 1.13.1.3 | NVIDIA Proprietary Software | license field; classifier Other/Proprietary | CUDA runtime library. See Findings |
| nvidia-curand-cu12 | 10.3.9.90 | NVIDIA Proprietary Software | license field; classifier Other/Proprietary | CUDA runtime library. See Findings |
| nvidia-cusolver-cu12 | 11.7.3.90 | NVIDIA Proprietary Software | license field; classifier Other/Proprietary | CUDA runtime library. See Findings |
| nvidia-cusparse-cu12 | 12.5.8.93 | NVIDIA Proprietary Software | license field; classifier Other/Proprietary | CUDA runtime library. See Findings |
| nvidia-cusparselt-cu12 | 0.7.1 | NVIDIA Proprietary Software | license field | See Findings |
| nvidia-ml-py | 13.615.71 | BSD (variant not stated) | license field; classifier | NVML bindings (pynvml) |
| nvidia-nccl-cu12 | 2.28.9 | LicenseRef-NVIDIA-Proprietary | wheel METADATA License-Expression | the bundled License.txt is a BSD-3-Clause text; metadata and file disagree. See Findings |
| nvidia-nvjitlink-cu12 | 12.8.93 | NVIDIA Proprietary Software | license field; classifier |  |
| nvidia-nvshmem-cu12 | 3.4.5 | LicenseRef-NVIDIA-Proprietary | wheel METADATA License-Expression; License.txt | NVIDIA licence agreement. See Findings |
| nvidia-nvtx-cu12 | 12.8.90 | Apache-2.0 | License.txt in wheel (license field "Apache 2.0") | its classifier says Other/Proprietary; the licence file is Apache-2.0 |
| onnxruntime | 1.30.0 | MIT | license field; classifier |  |
| orjson | 3.12.0 | MPL-2.0 AND (Apache-2.0 OR MIT) | license_expression | weak copyleft part. See Findings |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | license_expression | dual: Apache-2.0 OR BSD-2-Clause |
| pandas | 3.0.6 | BSD-3-Clause | classifier; LICENSE in wheel | license field holds the whole text |
| pillow | 12.3.0 | MIT-CMU | license_expression | MIT-CMU (HPND), permissive |
| platformdirs | 4.12.0 | MIT | license_expression |  |
| pluggy | 1.6.0 | MIT | license field; classifier | dev only |
| pooch | 1.9.0 | BSD-3-Clause | license_expression |  |
| protobuf | 7.36.2 | BSD-3-Clause | license field |  |
| psutil | 7.2.2 | BSD-3-Clause | license field; LICENSE in wheel |  |
| pycparser | 3.0 | BSD-3-Clause | license_expression |  |
| pydantic | 2.13.5 | MIT | license_expression |  |
| pydantic-core | 2.46.5 | MIT | license_expression |  |
| pydub | 0.25.1 | MIT | license field; classifier | runs ffmpeg if one is installed; the lock installs none |
| pygments | 2.21.0 | BSD-2-Clause | license_expression |  |
| pytest | 9.1.1 | MIT | license_expression | dev only |
| python-dateutil | 2.9.0.post0 | Apache-2.0 AND BSD-3-Clause | LICENSE in wheel (license field: "Dual License") | Apache-2.0 for contributions after 2017-12-01, BSD-3-Clause for earlier ones |
| python-multipart | 0.0.32 | Apache-2.0 | license_expression |  |
| pytz | 2026.4 | MIT | license field; classifier |  |
| pyyaml | 6.0.3 | MIT | license field; classifier |  |
| qwen-tts | 0.1.1 | Apache-2.0 | license field |  |
| regex | 2026.9.10 | Apache-2.0 AND CNRI-Python | license_expression | CNRI-Python is a permissive, OSI-approved licence |
| requests | 2.34.2 | Apache-2.0 | license field; classifier |  |
| rich | 15.0.0 | MIT | license field; classifier |  |
| safehttpx | 0.1.7 | MIT | classifier |  |
| safetensors | 0.8.0 | Apache-2.0 | classifier; LICENSE in wheel |  |
| scikit-learn | 1.9.1 | BSD-3-Clause | license_expression |  |
| scipy | 1.18.1 | BSD-3-Clause | classifier; LICENSE.txt in wheel | license field holds the whole text; wheels bundle OpenBLAS (BSD-3-Clause) and the GCC runtime (GPL-3.0-or-later WITH GCC-exception-3.1) |
| semantic-version | 2.10.0 | BSD (variant not stated) | license field; classifier |  |
| setuptools | 81.0.0 | MIT | license_expression |  |
| shellingham | 1.5.4 | ISC | license field; classifier |  |
| six | 1.17.0 | MIT | license field; classifier |  |
| soundfile | 0.14.0 | BSD-3-Clause | license field; LICENSE in wheel | platform wheels bundle libsndfile (LGPL-2.1) with LAME (LGPL-2+) and mpg123 (LGPL-2.1). See Findings |
| sox | 1.5.0 | BSD-3-Clause | license field; LICENSE in sdist (no wheel is published) | pysox, a wrapper that runs the SoX command-line program if one is installed; the lock installs no SoX binary |
| soxr | 1.1.0 | LGPL-2.1-or-later | license_expression; COPYING.LGPL in wheel | bundles libsoxr (LGPL-2.1-or-later) and PFFFT (BSD-like). See Findings |
| starlette | 1.7.0 | BSD-3-Clause | license_expression |  |
| sympy | 1.14.0 | BSD (variant not stated) | license field; classifier |  |
| threadpoolctl | 3.7.0 | BSD-3-Clause | license_expression |  |
| tokenizers | 0.22.2 | Apache-2.0 | classifier only | the wheel carries no licence file |
| tomlkit | 0.14.0 | MIT | license field; classifier |  |
| torch | 2.11.0+cu128 | BSD-3-Clause | license field (PyPI JSON of 2.11.0; locked from the pytorch-cu128 index) | the wheel's LICENSE and NOTICE list bundled third-party code (not reviewed here); on Windows the wheel also carries the CUDA libraries (Finding 5) |
| torchaudio | 2.11.0+cu128 | BSD-2-Clause | classifier; LICENSE in wheel (PyPI 2.11.0; locked from the pytorch-cu128 index) |  |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | license field | weak copyleft part. See Findings |
| transformers | 4.57.3 | Apache-2.0 | license field; classifier |  |
| triton | 3.6.0 | MIT | classifier; LICENSE in wheel |  |
| typer | 0.27.2 | MIT | license_expression |  |
| typing-extensions | 4.16.0 | PSF-2.0 | license_expression |  |
| typing-inspection | 0.4.4 | MIT | license_expression |  |
| tzdata | 2026.4 | Apache-2.0 | license field |  |
| urllib3 | 2.8.0 | MIT | license_expression |  |
| uvicorn | 0.54.0 | BSD-3-Clause | license_expression |  |

## QA worker (`workers/qa/uv.lock`, 93 packages)

| package | version | licence | source | notes |
|---|---|---|---|---|
| annotated-doc | 0.0.5 | MIT | license_expression |  |
| anyio | 4.15.1 | MIT | license_expression |  |
| certifi | 2026.7.22 | MPL-2.0 | license field; LICENSE in wheel | weak copyleft (file level). See Findings |
| cffi | 2.1.1 | MIT-0 | license_expression |  |
| charset-normalizer | 3.5.1 | MIT | license field |  |
| click | 8.5.0 | BSD-3-Clause | license_expression |  |
| cloudpickle | 3.1.2 | BSD-3-Clause | license field; classifier |  |
| colorama | 0.4.6 | BSD-3-Clause | classifier; LICENSE.txt in wheel | Windows only |
| contourpy | 1.4.0 | BSD-3-Clause | license_expression |  |
| cuda-bindings | 12.9.9 | Apache-2.0 | license_expression |  |
| cuda-pathfinder | 1.8.2 | Apache-2.0 | license_expression |  |
| cuda-toolkit | 12.8.1 | none stated | wheel METADATA (no licence field, classifier or file) | an empty NVIDIA meta-package that selects the nvidia-* wheels below. See Findings |
| cycler | 0.12.1 | BSD-3-Clause | classifier; LICENSE in wheel | license field holds the whole text |
| decorator | 5.3.1 | BSD-2-Clause | license field |  |
| filelock | 4.0.4 | MIT | license_expression |  |
| fonttools | 4.66.0 | MIT | license field |  |
| fsspec | 2026.9.0 | BSD-3-Clause | license_expression |  |
| h11 | 0.16.0 | MIT | license field; classifier |  |
| hf-xet | 1.6.0 | Apache-2.0 | license_expression |  |
| httpcore | 1.0.9 | BSD-3-Clause | license_expression |  |
| httpx | 0.28.1 | BSD-3-Clause | license field; classifier |  |
| huggingface-hub | 1.33.0 | Apache-2.0 | license field; classifier |  |
| idna | 3.20 | BSD-3-Clause | license_expression |  |
| iniconfig | 2.3.0 | MIT | license_expression | dev only |
| jinja2 | 3.1.6 | BSD-3-Clause | classifier; LICENSE.txt in wheel |  |
| joblib | 1.6.0 | BSD-3-Clause | license_expression |  |
| kiwisolver | 1.5.1 | BSD-3-Clause | classifier; LICENSE in wheel | license field holds the whole text |
| lazy-loader | 0.6 | BSD-3-Clause | license_expression |  |
| librosa | 1.0.0 | ISC | license field; classifier |  |
| llvmlite | 0.49.0 | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | license_expression |  |
| markdown-it-py | 4.2.0 | MIT | classifier; LICENSE in wheel |  |
| markupsafe | 3.0.3 | BSD-3-Clause | license_expression |  |
| matplotlib | 3.11.2 | PSF-based (matplotlib licence) | classifier; LICENSE in wheel | license field holds the whole text |
| mdurl | 0.1.2 | MIT | classifier; LICENSE in wheel |  |
| mpmath | 1.3.0 | BSD (variant not stated) | license field; classifier |  |
| msgpack | 1.2.2 | Apache-2.0 | license_expression |  |
| narration-worker | 0.1.0 | this project (PolyForm-Noncommercial-1.0.0) |  | workspace member (editable) |
| narration-worker-qa | 0.1.0 | this project (PolyForm-Noncommercial-1.0.0) |  | workspace member (editable) |
| narwhals | 2.26.0 | MIT | license_expression |  |
| networkx | 3.7 | BSD-3-Clause | license_expression |  |
| numba | 0.67.0 | BSD (variant not stated) | license field; classifier |  |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | license_expression | wheels also bundle OpenBLAS (BSD-3-Clause) and the GCC runtime (GPL-3.0-or-later WITH GCC-exception-3.1), listed in LICENSE.txt |
| nvidia-cublas-cu12 | 12.8.4.1 | NVIDIA Proprietary Software | license field; classifier Other/Proprietary | CUDA runtime library. See Findings |
| nvidia-cuda-cupti-cu12 | 12.8.90 | NVIDIA Proprietary Software | license field; classifier Other/Proprietary | CUDA runtime library. See Findings |
| nvidia-cuda-nvrtc-cu12 | 12.8.93 | NVIDIA Proprietary Software | license field; classifier Other/Proprietary | CUDA runtime library. See Findings |
| nvidia-cuda-runtime-cu12 | 12.8.90 | NVIDIA Proprietary Software | license field; classifier Other/Proprietary | CUDA runtime library. See Findings |
| nvidia-cudnn-cu12 | 9.19.0.56 | LicenseRef-NVIDIA-Proprietary | wheel METADATA License-Expression; License.txt | NVIDIA SDK licence agreement. See Findings |
| nvidia-cufft-cu12 | 11.3.3.83 | NVIDIA Proprietary Software | license field; classifier Other/Proprietary | CUDA runtime library. See Findings |
| nvidia-cufile-cu12 | 1.13.1.3 | NVIDIA Proprietary Software | license field; classifier Other/Proprietary | CUDA runtime library. See Findings |
| nvidia-curand-cu12 | 10.3.9.90 | NVIDIA Proprietary Software | license field; classifier Other/Proprietary | CUDA runtime library. See Findings |
| nvidia-cusolver-cu12 | 11.7.3.90 | NVIDIA Proprietary Software | license field; classifier Other/Proprietary | CUDA runtime library. See Findings |
| nvidia-cusparse-cu12 | 12.5.8.93 | NVIDIA Proprietary Software | license field; classifier Other/Proprietary | CUDA runtime library. See Findings |
| nvidia-cusparselt-cu12 | 0.7.1 | NVIDIA Proprietary Software | license field | See Findings |
| nvidia-ml-py | 13.615.71 | BSD (variant not stated) | license field; classifier | NVML bindings (pynvml) |
| nvidia-nccl-cu12 | 2.28.9 | LicenseRef-NVIDIA-Proprietary | wheel METADATA License-Expression | the bundled License.txt is a BSD-3-Clause text; metadata and file disagree. See Findings |
| nvidia-nvjitlink-cu12 | 12.8.93 | NVIDIA Proprietary Software | license field; classifier |  |
| nvidia-nvshmem-cu12 | 3.4.5 | LicenseRef-NVIDIA-Proprietary | wheel METADATA License-Expression; License.txt | NVIDIA licence agreement. See Findings |
| nvidia-nvtx-cu12 | 12.8.90 | Apache-2.0 | License.txt in wheel (license field "Apache 2.0") | its classifier says Other/Proprietary; the licence file is Apache-2.0 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | license_expression | dual: Apache-2.0 OR BSD-2-Clause |
| pillow | 12.3.0 | MIT-CMU | license_expression | MIT-CMU (HPND), permissive |
| platformdirs | 4.12.0 | MIT | license_expression |  |
| pluggy | 1.6.0 | MIT | license field; classifier | dev only |
| pooch | 1.9.0 | BSD-3-Clause | license_expression |  |
| psutil | 7.2.2 | BSD-3-Clause | license field; LICENSE in wheel | dev only |
| pycparser | 3.0 | BSD-3-Clause | license_expression |  |
| pygments | 2.21.0 | BSD-2-Clause | license_expression |  |
| pyloudnorm | 0.2.0 | MIT | license_expression |  |
| pyparsing | 3.3.3 | MIT | license_expression |  |
| pytest | 9.1.1 | MIT | license_expression | dev only |
| python-dateutil | 2.9.0.post0 | Apache-2.0 AND BSD-3-Clause | LICENSE in wheel (license field: "Dual License") | Apache-2.0 for contributions after 2017-12-01, BSD-3-Clause for earlier ones |
| pyyaml | 6.0.3 | MIT | license field; classifier |  |
| regex | 2026.9.10 | Apache-2.0 AND CNRI-Python | license_expression | CNRI-Python is a permissive, OSI-approved licence |
| requests | 2.34.2 | Apache-2.0 | license field; classifier |  |
| rich | 15.0.0 | MIT | license field; classifier |  |
| safetensors | 0.8.0 | Apache-2.0 | classifier; LICENSE in wheel |  |
| scikit-learn | 1.9.1 | BSD-3-Clause | license_expression |  |
| scipy | 1.18.1 | BSD-3-Clause | classifier; LICENSE.txt in wheel | license field holds the whole text; wheels bundle OpenBLAS (BSD-3-Clause) and the GCC runtime (GPL-3.0-or-later WITH GCC-exception-3.1) |
| setuptools | 81.0.0 | MIT | license_expression |  |
| shellingham | 1.5.4 | ISC | license field; classifier |  |
| six | 1.17.0 | MIT | license field; classifier |  |
| soundfile | 0.14.0 | BSD-3-Clause | license field; LICENSE in wheel | platform wheels bundle libsndfile (LGPL-2.1) with LAME (LGPL-2+) and mpg123 (LGPL-2.1). See Findings |
| soxr | 1.1.0 | LGPL-2.1-or-later | license_expression; COPYING.LGPL in wheel | bundles libsoxr (LGPL-2.1-or-later) and PFFFT (BSD-like). See Findings |
| sympy | 1.14.0 | BSD (variant not stated) | license field; classifier |  |
| threadpoolctl | 3.7.0 | BSD-3-Clause | license_expression |  |
| tokenizers | 0.23.2 | Apache-2.0 | classifier only | the wheel carries no licence file |
| torch | 2.11.0+cu128 | BSD-3-Clause | license field (PyPI JSON of 2.11.0; locked from the pytorch-cu128 index) | the wheel's LICENSE and NOTICE list bundled third-party code (not reviewed here); on Windows the wheel also carries the CUDA libraries (Finding 5) |
| torchaudio | 2.11.0+cu128 | BSD-2-Clause | classifier; LICENSE in wheel (PyPI 2.11.0; locked from the pytorch-cu128 index) |  |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | license field | weak copyleft part. See Findings |
| transformers | 5.17.0 | Apache-2.0 | license field |  |
| triton | 3.6.0 | MIT | classifier; LICENSE in wheel |  |
| typer | 0.27.2 | MIT | license_expression |  |
| typing-extensions | 4.16.0 | PSF-2.0 | license_expression |  |
| urllib3 | 2.8.0 | MIT | license_expression |  |

## Models

The service pins five models (`PINNED` in `src/narration/engine/models.py`) and installs them with
`narration-admin install` (`src/narration/admin/install.py`, which reads that table through
`src/narration/admin/models.py:74`). None of them is distributed with this repository.

huggingface.co, and the mirror hf-mirror.com, could not be reached from the machine this audit ran on, so no
model card was read for this audit. Every model licence below is therefore **BELIEVE**, and the source is
named. `narration-admin install` should read each card's `license` at the pinned revision and record it (as
design §18 already asks).

| repo | pinned revision | used by | licence | evidence |
|---|---|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `fd4b254389122332181a7c3db7f27e918eec64e3` | `src/narration/engine/models.py:48` (`QWEN_BASE`); loaded by `workers/qwen3tts/src/narration_qwen3tts/engine.py:234` | Apache-2.0 | BELIEVE: design §18 and the pin's own `licence` field say Apache-2.0; the QwenLM/Qwen3-TTS GitHub repository's `LICENSE` is Apache-2.0 (read 2026-09-28); the `qwen-tts` 0.1.1 package's metadata says Apache-2.0 |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `5ecdb67327fd37bb2e042aab12ff7391903235d3` | `src/narration/engine/models.py:51` (`QWEN_DESIGN`); same loader | Apache-2.0 | BELIEVE: as for Base |
| `openai/whisper-large-v3` | `06f233fe06e710322aca913c1bc4249a0d71fce1` | `src/narration/engine/models.py:54` (`WHISPER`), `src/narration/engine/qa.py:63`; loaded by `workers/qa/src/narration_worker_qa/asr.py:158` | Apache-2.0 (weights, per the Hugging Face card); code MIT | BELIEVE: the card's `license: apache-2.0` is from memory of the card, not read here. The openai/whisper GitHub `LICENSE` is MIT (read 2026-09-28); its `model-card.md` states no licence and no use restriction. The pin records "per model card" |
| `microsoft/wavlm-base-plus-sv` | `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e` | `src/narration/engine/models.py:57` (`WAVLM_SV`), `src/narration/engine/qa.py:69`; loaded by `workers/qa/src/narration_worker_qa/sv.py:125` | **unconfirmed; probably MIT** | BELIEVE, weakly: the card is understood to have no `license:` field and to point to the microsoft/unilm repository's `LICENSE` ("the official license can be found here", seen only as a search-result snippet). That `LICENSE` is MIT (read 2026-09-28), and `unilm/wavlm/README.md` says the project is under it. See Findings |
| `facebook/wav2vec2-large-960h-lv60-self` | `54074b1c16f4de6a5ad59affb4caa8f2ea03a119` | `src/narration/engine/models.py:62` (`CTC_ALIGNER`); loaded by `workers/qa/src/narration_worker_qa/align.py:438` | Apache-2.0 | BELIEVE: design §4 and §18 record `apache-2.0` "checked on the Hugging Face API"; that check's output was not saved. The fairseq code it was trained with is MIT (facebookresearch/fairseq `LICENSE`, read 2026-09-28) |
| `Qwen/Qwen3-ForcedAligner-0.6B` | not pinned | named only, as `MODEL_ALIGNER_ALTERNATIVE` in `src/narration/contracts/names.py:120` | Apache-2.0 | BELIEVE: design §18 ("model card, read 2026-09-26; verify at install"). Not installed or loaded by any code |

**The speech tokenizer.** Qwen3-TTS needs its 12 Hz speech tokenizer. It is not pinned separately: `qwen-tts`
0.1.1 loads it from the model's own snapshot (KNOW: `Qwen3TTSForConditionalGeneration.from_pretrained` in
`qwen_tts/core/models/modeling_qwen3_tts.py`, around line 1900, resolves it with `cached_file` against the
model's path and then calls `load_speech_tokenizer`). It is covered by the model repository's licence. The
standalone `Qwen/Qwen3-TTS-Tokenizer-12Hz` repository is not used.

**Training data.** No card read for this audit states a restriction on use that comes from the training data.
WavLM Base+ was pre-trained on Libri-Light, GigaSpeech and VoxPopuli, and the `-sv` model was fine-tuned on
VoxCeleb (KNOW: `unilm/wavlm/README.md`). Whether any of those terms reach the fine-tuned weights is not
stated in any source read here: unknown.

**Excluded.** torchaudio's `MMS_FA` bundle and `facebook/mms-300m` are CC-BY-NC-4.0 and stay excluded (design
§18). Neither appears in the code.

## THIRD_PARTY_NOTICES check

`THIRD_PARTY_NOTICES` has one entry: Whisper's English text normaliser and its spelling map, in
`src/narration/qa/normaliser/_whisper/`, from openai/whisper, tag v20250625, commit
`31243bad24cc746f07d4c8bfdd2d974872cb1803`, MIT.

Checked (KNOW, 2026-09-28):

- The tag `v20250625` resolves to commit `31243bad24cc746f07d4c8bfdd2d974872cb1803` (`git ls-remote`).
- `english.json` and `LICENSE` are byte-identical to that commit's `whisper/normalizers/english.json` and
  `LICENSE`.
- The upstream sha256 in each `.py` file's provenance header matches that commit's file.
- Diffed against upstream, the only changes are the ones `THIRD_PARTY_NOTICES` lists: the provenance header
  and pragmas in each `.py` file; in `english.py`, a local `windowed` in place of `more_itertools.windowed`, and
  the spelling map opened as UTF-8 in a `with` block.
- `tests/qa/test_normaliser.py` pins the upstream commit and checks the headers and the map.

Searched for anything else from a third party: licence and notice files outside the root, "vendored",
"ported from", "Origin:", "Copyright", "SPDX", "adapted from", "copied" and "verbatim", across `src/`,
`workers/`, `tools/`, `tests/`, `spikes/` and `material/`. Found:

- **Nothing else vendored.** The other "ported from" notes (for example
  `workers/qa/src/narration_worker_qa/align.py:197`, `spikes/acceptance-wp20/README.md`) name the bake-off,
  the owner's own earlier code, not a third party.
- **`material/`**: every manifest says PolyForm Noncommercial 1.0.0, and `material/README.md` says it was all
  written for the service. No third-party text, font or audio is tracked. The canary clip is made on the
  installing machine by the pinned VoiceDesign model (Apache-2.0, BELIEVE) and is not in git.
- **`CODE_OF_CONDUCT.md`** is adapted from the Contributor Covenant (CC BY 4.0) and says so in its own text.
  That attribution is enough for CC BY 4.0. It is not code, so it is not a gap in `THIRD_PARTY_NOTICES`, but
  the lead may want to list it there for completeness.
- **`spikes/k-job-escape/node_client.js`** is the project's own script.

No gaps: `THIRD_PARTY_NOTICES` matches what is vendored.

## Findings

Nothing in any lock file is GPL, AGPL, non-commercial or "research only" as a Python package. The items below
are the ones that need a decision, a note, or no action but a record.

### 1. LGPL: `soundfile`'s bundled libsndfile (server and both workers)

- **What.** `soundfile` 0.14.0 is BSD-3-Clause, but its Windows, macOS and Linux wheels bundle a libsndfile
  binary (`_soundfile_data/libsndfile_x64.dll` on Windows) that is LGPL-2.1, built with LAME (LGPL-2+) and
  mpg123 (LGPL-2.1) (KNOW: `licensing/license_notes.md` and `_soundfile_data/COPYING` in the win_amd64 wheel).
- **Chain.** server: `narration -> soundfile` (direct), used in `narration.admin.voices`, `narration.backend.clips`,
  `narration.design.handler`, `narration.jobs.stages` and `narration.post.pcm`. qwen3tts worker:
  `narration-worker-qwen3tts -> soundfile`. QA worker: `narration-worker-qa -> soundfile`, and
  `librosa -> soundfile`.
- **Risk: low.** The library is loaded at run time through cffi, unmodified, from a package the user installs
  from PyPI. This project ships only source and does not redistribute the binary. The LGPL's duties (source
  offer, replaceability) fall on whoever redistributes the binary, and the user can replace the DLL.
- **Action.** Lead's OK, recorded here. AGENTS.md §6 already mandates `soundfile` for audio I/O. If the
  service only ever reads and writes PCM WAV, the standard library's `wave` plus numpy would remove it from the
  server, but that is a design change and not needed for licence reasons.

### 2. LGPL: `soxr` (both workers)

- **What.** `soxr` 1.1.0: `LGPL-2.1-or-later` (KNOW: `license_expression`; the wheel bundles libsoxr, LGPL, and
  PFFFT, BSD-like).
- **Chain.** qwen3tts worker: `narration-worker-qwen3tts -> qwen-tts -> librosa -> soxr`. QA worker:
  `narration-worker-qa -> librosa -> soxr`.
- **Risk: low.** Workers only; a separately installed, replaceable package, used unmodified.
- **Action.** None new. plan.md DC-1 (applied) already records it as "acceptable as a separately installed,
  replaceable package; it goes in the licence audit". This is that record.

### 3. LGPL sub-component: `pywin32`'s `adodbapi` (server, Windows only)

- **What.** `pywin32` 312 is PSF-licensed, but its wheel includes the `adodbapi` sub-package under LGPL-2.1
  (KNOW: `licenses/adodbapi/license.txt` in the wheel).
- **Chain.** `narration -> mcp -> pywin32` (Windows only).
- **Risk: none in practice.** Nothing in this project imports `adodbapi`.
- **Action.** None; recorded for completeness.

### 4. MPL-2.0 (weak copyleft)

| package | licence | where | chain |
|---|---|---|---|
| `certifi` 2026.7.22 | MPL-2.0 | both workers | qwen3tts: `qwen-tts -> gradio -> httpx -> certifi` (also via `requests`); QA: `transformers -> huggingface-hub -> httpx -> certifi`, `librosa -> pooch -> requests -> certifi` |
| `tqdm` 4.70.1 | MPL-2.0 AND MIT | both workers | `transformers -> tqdm`, `huggingface-hub -> tqdm` |
| `orjson` 3.12.0 | MPL-2.0 AND (Apache-2.0 OR MIT) | qwen3tts worker | `qwen-tts -> gradio -> orjson` |

- **Risk: low.** MPL-2.0 applies to its own files; using them unmodified puts no terms on this project's code.
- **Action.** Lead to confirm MPL-2.0 is acceptable (the policy does not name it). No replacement is practical:
  all three come in through `transformers`, `huggingface-hub` and `gradio`.

### 5. NVIDIA proprietary CUDA libraries (both workers)

- **What.** On Linux x86_64, torch pulls in NVIDIA's CUDA runtime wheels. Their licence is NVIDIA's own
  (KNOW, from each wheel's metadata and `License.txt`):
  - `nvidia-cublas-cu12`, `-cuda-cupti-cu12`, `-cuda-nvrtc-cu12`, `-cuda-runtime-cu12`, `-cufft-cu12`,
    `-cufile-cu12`, `-curand-cu12`, `-cusolver-cu12`, `-cusparse-cu12`, `-nvjitlink-cu12`: "NVIDIA Proprietary
    Software", classifier Other/Proprietary; the file is NVIDIA's CUDA Toolkit End User License Agreement;
  - `nvidia-cusparselt-cu12`: "NVIDIA Proprietary Software";
  - `nvidia-cudnn-cu12`, `nvidia-nvshmem-cu12`: `LicenseRef-NVIDIA-Proprietary`, with NVIDIA's SDK licence
    agreement;
  - `nvidia-nccl-cu12`: metadata says `LicenseRef-NVIDIA-Proprietary`, but its bundled `License.txt` is a
    BSD-3-Clause text. Metadata and file disagree; treat it as proprietary to be safe;
  - `nvidia-nvtx-cu12`: classifier Other/Proprietary, but its `License.txt` is Apache-2.0. Treated as
    Apache-2.0.
- **Chain.** `narration-worker-qwen3tts -> torch -> cuda-toolkit[cublas, cudart, …] -> nvidia-*`, and
  `torch -> nvidia-cudnn-cu12 / nvidia-nccl-cu12 / nvidia-nvshmem-cu12 / nvidia-cusparselt-cu12`; the same from
  `narration-worker-qa -> torch`. All are marked `platform_machine == 'x86_64' and sys_platform == 'linux'` in
  the locks.
- **On Windows**, the first platform, these wheels are not installed. BELIEVE: the Windows `+cu128` torch wheel
  carries the same CUDA and cuDNN DLLs inside it (as PyTorch's Windows CUDA wheels have done so far), under the
  same NVIDIA terms. Not checked here: `download.pytorch.org` could not be reached from this machine.
- **Does it conflict with PolyForm Noncommercial? No.** These are runtime libraries that the user's package
  manager installs from NVIDIA's or PyTorch's index. This project does not vendor, modify or redistribute
  them, and its own code does not link to them (only torch does, in the worker processes). NVIDIA's terms
  bind the user who installs them, not this project's licence, and they do not forbid commercial use. Every
  user of CUDA PyTorch already accepts them.
- **Action.** Say it in the README: using the GPU workers means accepting NVIDIA's CUDA and cuDNN licence
  terms. Do not bundle these libraries in any future installer or image without reading the EULA's
  distribution terms (its Attachment A lists what may be redistributed, and its §1.1.2 sets conditions).

### 6. No licence stated: `cuda-toolkit` 12.8.1 (both workers, Linux only)

- **What.** An NVIDIA meta-package with no licence field, classifier or licence file (KNOW: the wheel holds
  only `METADATA`, `WHEEL` and `RECORD`). It contains no code; it only selects the nvidia-* wheels above
  through its extras.
- **Chain.** `narration-worker-* -> torch -> cuda-toolkit` (Linux x86_64 only).
- **Risk: none beyond Finding 5.** There is nothing in it to license.
- **Action.** None; recorded as "unknown" for completeness.

### 7. Wrappers of external programs that would be GPL if present (qwen3tts worker)

- `sox` 1.5.0 (pysox, BSD-3-Clause; KNOW from its sdist) runs the SoX program, which is GPL-2.0-or-later
  (BELIEVE), when one is on `PATH`. Chain: `narration-worker-qwen3tts -> qwen-tts -> sox`. `qwen-tts` imports
  it in `qwen_tts/core/tokenizer_25hz/vq/speech_vq.py`, a 25 Hz tokenizer this service does not use.
- `pydub` 0.25.1 (MIT) runs ffmpeg when one is installed. Chain: `qwen-tts -> gradio -> pydub`.
- **Risk: none as locked.** No SoX or ffmpeg binary is in any lock file, and a separate program run as a
  subprocess does not combine with this project's code.
- **Action.** Don't add SoX or ffmpeg as a dependency, and don't tell users to install them.

### 8. GPL runtime code with the GCC exception: numpy and scipy wheels (all three)

- **What.** The numpy 2.5.3 and scipy 1.18.1 wheels bundle OpenBLAS (BSD-3-Clause) and the GCC runtime
  (libgfortran and others), whose licence is `GPL-3.0-or-later WITH GCC-exception-3.1` (KNOW: `LICENSE.txt` in
  the win_amd64 wheels).
- **Risk: none.** The GCC Runtime Library Exception exists so that non-GPL programs can use these
  libraries. This is the standard scientific-Python stack.
- **Action.** None.

### 9. Dual-licensed packages

Where a user may choose, both choices are permissive:

- `cryptography` 50.0.1: Apache-2.0 OR BSD-3-Clause (server, via `mcp -> pyjwt[crypto]`);
- `packaging` 26.3: Apache-2.0 OR BSD-2-Clause;
- `orjson` 3.12.0: MPL-2.0 AND (Apache-2.0 OR MIT). The MPL part is not optional (Finding 4).

`python-dateutil` 2.9.0.post0 reports "Dual License", but its `LICENSE` applies Apache-2.0 to contributions
after 2017-12-01 and BSD-3-Clause to earlier ones: both apply, and both are permissive.

### 10. Models whose licence is unconfirmed

- **`microsoft/wavlm-base-plus-sv`: the most uncertain.** No licence could be confirmed from the model card.
  The best available evidence is that the card points to microsoft/unilm's MIT `LICENSE`, which covers "this
  project" (code and released models). The code records it as "per model card" (`src/narration/engine/models.py:57`).
  - **Action:** read the card at revision `feb593a6…` before release. If it has no `license:` field and only
    points to unilm, record "MIT (via microsoft/unilm LICENSE, linked from the card)" and say in the README that
    the card itself sets no licence. If the card names anything non-commercial, replace the model. An
    alternative speaker-verification model with an explicit permissive card would be a design change (§11).
- **`openai/whisper-large-v3`.** Apache-2.0 is expected (BELIEVE). Read and record it at install. The pin still
  says "per model card" (`src/narration/engine/models.py:54`).
- **Qwen3-TTS Base and VoiceDesign, wav2vec2-large-960h-lv60-self.** Apache-2.0, BELIEVE from the design's own
  earlier checks and the upstream code repositories. Re-read the cards at the pinned revisions before release
  (WP44).

### Not verified, and why

- **Every model card.** huggingface.co and hf-mirror.com are blocked from the machine this audit ran on
  (HTTP 403 from its network proxy).
- **The Windows torch wheel's bundled CUDA DLLs** (Finding 5). download.pytorch.org is blocked from the same
  machine. The PyPI metadata of torch 2.11.0 was used instead.
- **The worker environments' installed metadata.** They were not installed, by design (torch is several GB).
  The workers' rows come from PyPI and from the wheels themselves, not from an installed environment.
