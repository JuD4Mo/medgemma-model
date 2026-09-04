# MedGemma 1.5 4B — análisis radiológico local con llama.cpp + GGUF

Proyecto para ejecutar `google/medgemma-1.5-4b-it` localmente en una
**NVIDIA GeForce RTX 2060 Laptop (6 GB, Turing sm_75)** mediante
**llama.cpp + GGUF**, sin Ollama y sin APIs externas.

Transformers + BitsAndBytes NO pudo ejecutar el modelo correctamente en
esta GPU (FP16 → overflow/NaN; FP32 → no cabe y el CPU offload deja pesos
en `meta`). La solución definitiva es llama.cpp con el mismo MedGemma en
formato GGUF, servido por `llama-server` en `127.0.0.1`.

## Contenido

| Archivo | Descripción |
| --- | --- |
| `radiologia_medgemma_llamacpp.ipynb` | **Solución activa**: pipeline completo con motor llama.cpp. |
| `requirements_llamacpp.txt` | Dependencias pip del entorno `medgemma-t5`. |
| `modelos_gguf/` | GGUF del modelo + mmproj (NO versionado; descargar, ver abajo). |

## Requisitos

- GPU NVIDIA RTX 2060 (6 GB) con driver compatible con CUDA 12+.
- Anaconda / Miniconda.
- ~5 GB libres para los archivos GGUF.

## 1. Crear el entorno

```bash
conda create -n medgemma-t5 python=3.10 -y
conda activate medgemma-t5
pip install -r requirements_llamacpp.txt

# nvcc + toolkit CUDA (necesario para compilar llama.cpp con soporte CUDA)
conda install -n medgemma-t5 -c nvidia cuda-toolkit -y
```

## 2. Descargar el modelo GGUF y el proyector de visión (mmproj)

```bash
mkdir -p modelos_gguf
python - <<'PY'
from huggingface_hub import hf_hub_download
repo = "unsloth/medgemma-1.5-4b-it-GGUF"
hf_hub_download(repo, "medgemma-1.5-4b-it-Q4_K_M.gguf", local_dir="modelos_gguf")
hf_hub_download(repo, "mmproj-F16.gguf", local_dir="modelos_gguf")
PY
```

## 3. Compilar `llama-server` con CUDA

La rueda precompilada de `llama-cpp-python` falla en esta CPU (requiere
AVX512). Se compila desde fuente:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j8 --target llama-server
```

`CMAKE_CUDA_ARCHITECTURES=75` corresponde a la RTX 2060 (Turing).

## 4. Levantar el servidor

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

/path/to/llama.cpp/build/bin/llama-server \
  --model modelos_gguf/medgemma-1.5-4b-it-Q4_K_M.gguf \
  --mmproj modelos_gguf/mmproj-F16.gguf \
  --n-gpu-layers 999 \
  --host 127.0.0.1 --port 8081
```

Verificar:

```bash
curl http://127.0.0.1:8081/health   # -> {"status":"ok"}
```

El servidor solo escucha en `127.0.0.1` (no expuesto). Mantenerlo corriendo;
el notebook se conecta a él y el modelo no se recarga por imagen.

## 5. Ejecutar el notebook

```bash
python -m ipykernel install --user --name medgemma-t5 --display-name "Python (MedGemma T5)"
```

Abrir `radiologia_medgemma_llamacpp.ipynb` con el kernel **medgemma-t5** y
ejecutar en orden:

1. Celdas 1–4: entorno, imports, configuración del proyecto.
2. Celda 13: configuración llama.cpp (rutas del GGUF/mmproj, wrapper `llama_chat`).
3. Celda 14: verifica la salud del servidor.
4. Celda 16: prompt radiológico.
5. Celda 17: `infer_radiology` (extrae secciones, categoría, confianza).
6. Celda 18 en adelante: corrida de 10 imágenes, métricas, persistencia, Gradio, TTS.

> Nota: MedGemma 1.5 razona antes de responder; el wrapper `strip_thinking`
> elimina ese bloque y conserva solo el reporte estructurado.

## 6. Configuración `.env`

```bash
cp .env.template .env
```

Completar `HF_TOKEN` (descarga de modelos) y `GEMINI_API_KEY` (solo TTS de
un reporte; la radiografía la interpreta MedGemma).

## Resultados observados (RTX 2060, 6 GB)

| Métrica | Valor |
| --- | --- |
| Velocidad | ~75 tokens/s |
| 1 imagen + prompt corto (64 tok) | ~2.5 s |
| Prompt completo (~1000 tok) | ~14.7 s |
| VRAM | ~4.9–5.15 GiB (con `-ngl 999`) |
| Cuantización | Q4_K_M |
| Inferencia multimodal | válida (reporte en español, categoría/confianza extraídos) |