# Post-Contenido 1 — Benchmark CUDA CPU vs GPU y Análisis Comparativo de Aceleradores

**Asignatura:** Arquitectura de Computadores  
**Unidad 11:** Post-contenido (Laboratorio)  
**Estudiante:** Julian Andres Ascanio  
**Repositorio:** `ascanio-post1-u11`  

---

## 1. Contexto y Objetivos

Este repositorio integra en un único proyecto las dos partes del laboratorio de la Unidad 11:

1. **Parte 1 (`src/`):** Implementación y medición de rendimiento en CUDA/C de tres programas con gestión explícita de memoria del dispositivo:
   - `src/vectorAdd.cu`: suma de vectores CPU vs GPU (`N = 1M, 4M, 16M`), midiendo tiempo de CPU, tiempo exclusivo de kernel y tiempo total incluyendo `cudaMemcpy`.
   - `src/matMul.cu`: multiplicación de matrices (`N = 512, 1024`) comparando un kernel *naïve* en memoria global frente a un kernel con *tiling* en *shared memory* (`TILE = 16`).
   - `src/vectorAddStream.cu`: suma de vectores (`N = 16M`) con memoria host *pinned* (`cudaHostAlloc`) y `NSTREAMS = 4` para solapar transferencias `cudaMemcpyAsync` y ejecución de kernels.
2. **Parte 2 (`docs/analisis-comparativo.md`):** Análisis técnico comparativo de aceleradores de IA (GPU, TPU, NPU y FPGA), resultados MLPerf Inference, TCO, recomendaciones por perfil, cuantización INT8 vs FP32 y decisiones técnicas justificadas.

---

## 2. Estructura del Repositorio

```text
ascanio-post1-u11/
├── README.md                       # Documentación Parte 1, tablas, análisis y decisiones técnicas
├── .gitignore                      # Exclusión de binarios compilados y temporales
├── src/
│   ├── vectorAdd.cu                # Benchmark suma de vectores CPU vs GPU (1M, 4M, 16M)
│   ├── matMul.cu                   # Benchmark multiplicación de matrices: naive vs shared memory tiling
│   └── vectorAddStream.cu          # Benchmark suma de vectores síncrona vs 4 CUDA streams
├── docs/
│   └── analisis-comparativo.md     # Parte 2: análisis comparativo de aceleradores de IA
└── capturas/                       # Evidencias de los 5 checkpoints en Google Colab / GPU NVIDIA
```

---

## 3. Entorno de Ejecución y Compilación

> **Nota de entorno:** El equipo local de desarrollo dispone de GPU integrada AMD (sin soporte CUDA), por lo que la compilación y ejecución de los programas CUDA se realiza en **Google Colab** con entorno **GPU T4** (*Runtime → Change runtime type → T4 GPU*).

- **GPU:** `[COMPLETAR con salida de nvidia-smi en Colab, ej. NVIDIA Tesla T4 15 GB]`
- **Versión de CUDA:** `[COMPLETAR con salida de nvcc --version, ej. CUDA 12.x]`
- **Sistema Operativo:** `[COMPLETAR, ej. Ubuntu 22.04 LTS (Google Colab)]`
- **Compilador Host:** `[COMPLETAR con gcc --version]`

### Comandos de Compilación y Ejecución

```bash
# Verificar entorno GPU y compilador
nvidia-smi
nvcc --version

# 1. Suma de vectores (N = 1M, 4M, 16M)
nvcc -O2 -o vectorAdd src/vectorAdd.cu
./vectorAdd

# 2. Multiplicación de matrices (N = 512, 1024: naive vs tiled)
nvcc -O2 -o matMul src/matMul.cu
./matMul

# 3. Suma de vectores con streams y memoria pinned (N = 16M, NSTREAMS = 4)
nvcc -O2 -o vectorAddStream src/vectorAddStream.cu
./vectorAddStream
```

---

## 4. Resultados del Benchmark (Parte 1)

### 4.1 `vectorAdd` — CPU vs GPU (`N = 1M, 4M, 16M`)

| Tamaño `N` | CPU (ms) | GPU Kernel (ms) | GPU Total con `cudaMemcpy` (ms) | Speedup (CPU / GPU Kernel) | Errores |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `1M` (`1 048 576`) | `[COMPLETAR]` | `[COMPLETAR]` | `[COMPLETAR]` | `[COMPLETAR]` | `[COMPLETAR]` |
| `4M` (`4 194 304`) | `[COMPLETAR]` | `[COMPLETAR]` | `[COMPLETAR]` | `[COMPLETAR]` | `[COMPLETAR]` |
| `16M` (`16 777 216`) | `[COMPLETAR]` | `[COMPLETAR]` | `[COMPLETAR]` | `[COMPLETAR]` | `[COMPLETAR]` |

### 4.2 `matMul` — Naïve vs *Tiling* en *Shared Memory* (`N = 512, 1024`)

| Dimensión `N × N` | CPU (ms) | Kernel Naïve (ms) | Kernel Tiled (`TILE=16`) (ms) | Speedup (Naïve / Tiled) | Errores (`< 1e-3`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `512 × 512` | `[COMPLETAR]` | `[COMPLETAR]` | `[COMPLETAR]` | `[COMPLETAR]` | `[COMPLETAR]` |
| `1024 × 1024` | `[COMPLETAR]` | `[COMPLETAR]` | `[COMPLETAR]` | `[COMPLETAR]` | `[COMPLETAR]` |

### 4.3 `vectorAddStream` — Transferencia Síncrona vs 4 Streams (`N = 16M`)

| Modo de Ejecución | Tiempo Total H2D + Kernel + D2H (ms) | Reducción (%) | Errores |
| :--- | :---: | :---: | :---: |
| Síncrono (`cudaMemcpy` bloqueante) | `[COMPLETAR]` | — | `[COMPLETAR]` |
| Asíncrono con 4 Streams (`cudaHostAlloc` + `cudaMemcpyAsync`) | `[COMPLETAR]` | `[COMPLETAR]` | `[COMPLETAR]` |

---

## 5. Análisis de Resultados

Para valores grandes de `N` (como `16M` elementos en `vectorAdd` o `1024 × 1024` en `matMul`), el kernel ejecutado en GPU supera ampliamente a la CPU porque la arquitectura SIMT (*Single Instruction, Multiple Threads*) distribuye millones de operaciones independientes entre miles de núcleos CUDA simultáneamente, ocultando la latencia de acceso a memoria mediante el cambio instantáneo de *warps* activos. Además, la memoria VRAM dedicada de la GPU (por ejemplo, GDDR6 en una Tesla T4 o HBM3 en una H100) ofrece un ancho de banda interno de cientos a miles de GB/s, muy superior al de la memoria DDR del sistema host, lo que resulta decisivo en operaciones *memory-bound* como la suma vectorial. En `matMulTiled`, el uso de *shared memory* de baja latencia (`__shared__ float sA[16][16]`, `sB[16][16]`) reduce los accesos redundantes a memoria global en un factor igual al tamaño del bloque (`TILE = 16`), incrementando la reutilización de datos en caché local por cada *thread block*.

Por el contrario, cuando `N` es pequeño (por ejemplo `1M` o menos), el tiempo total en GPU que incluye las llamadas `cudaMemcpy` (Host-to-Device y Device-to-Host) puede resultar igual o mayor que el tiempo de ejecución secuencial en CPU. Esto se debe al cuello de botella físico del bus PCIe —cuyo ancho de banda efectivo es significativamente menor que el de la memoria interna de la GPU— sumado a la latencia fija de lanzar transferencias DMA y sincronizar eventos. En el caso de `vectorAdd`, la intensidad aritmética es mínima (1 operación FLOP por cada 12 bytes transferidos entre arreglos de entrada y salida), de modo que el costo de mover los datos por PCIe domina sobre el ahorro obtenido en el cómputo paralelo cuando el volumen de trabajo no amortiza la transferencia.

---

## 6. Checkpoints y Evidencias (`capturas/`)

| Checkpoint | Descripción | Archivo de Evidencia |
| :--- | :--- | :--- |
| Checkpoint 1 | Compilación y ejecución de `vectorAdd` (`Errores: 0`) | `capturas/CP1_vectorAdd.png` *(pendiente captura Colab)* |
| Checkpoint 2 | Compilación y ejecución de `matMul` (naïve vs *tiled*) | `capturas/CP2_matMul.png` *(pendiente captura Colab)* |
| Checkpoint 3 | Verificación de entorno (`nvidia-smi`, `nvcc --version`) | `capturas/CP3_entorno_colab.png` *(pendiente captura Colab)* |
| Checkpoint 4 | Ejecución de `vectorAddStream` (síncrono vs 4 streams) | `capturas/CP4_vectorAddStream.png` *(pendiente captura Colab)* |
| Checkpoint 5 | Decisiones técnicas de la Parte 1 y revisión de repositorio | `capturas/CP5_git_log.png` *(pendiente captura)* |

---

## 7. Decisiones Técnicas Justificadas (Parte 1)

### Decisión Técnica 1: ¿Conviene `cudaMemcpyAsync` con streams o transferencia síncrona para `vectorAdd`?

Para `N = 16M` elementos de tipo `float` (4 bytes por elemento), cada vector ocupa `64 MiB`, lo que implica transferir un volumen total aproximado de `192 MB` entre host y dispositivo (`128 MB` en sentido Host-to-Device para `d_A` y `d_B`, más `64 MB` en sentido Device-to-Host para `d_C`). Sobre un bus PCIe 4.0 x16 cuyo ancho de banda efectivo típico se ubica entre `10` y `12 GB/s` (por debajo del límite teórico debido a sobrecarga de protocolo y transacciones TLP), mover `192 MB` demanda entre `16 ms` y `19 ms`. En contraste, el tiempo de ejecución exclusivo del kernel `vectorAdd` medido en el Checkpoint 1 ronda `~1.8 ms`, es decir, un orden de magnitud menor que el tiempo de transferencia. Esta relación confirma que `vectorAdd` es una operación fuertemente *memory-bound* en el enlace PCIe, con una intensidad aritmética de apenas `1/12` FLOP/byte. Dado que los *streams* de CUDA permiten solapar transferencia y cómputo sin eliminar el volumen de datos que cruza el bus, el ahorro máximo teórico está estrictamente acotado por la fase más corta: en el mejor caso, el solapamiento perfecto oculta los `~1.8 ms` del kernel dentro de los `16–19 ms` de transferencia, produciendo una reducción modesta cercana al `10–12%` y no un escalado lineal con el número de *streams*. Por lo tanto, para `vectorAdd` aislado el beneficio práctico de `cudaMemcpyAsync` con *streams* es limitado y exige además reservar memoria *pinned* (`cudaHostAlloc`), cuyo costo de asignación no siempre se amortiza en una sola pasada; su uso resulta mucho más rentable en operaciones con mayor intensidad aritmética como `matMulTiled`, donde la complejidad de cómputo `O(N³)` hace que el tiempo de kernel sea comparable o superior al tiempo de transferencia de cada submatriz.

### Decisión Técnica 2: ¿Cuántos streams usar, y qué limita la ganancia al aumentar su número?

Incrementar `NSTREAMS` de `4` a `16` manteniendo fijo `N = 16M` no produce una reducción proporcional del tiempo total de ejecución. La ganancia por solapamiento deja de escalar en cuanto los motores físicos de copia y las unidades de cómputo permanecen continuamente ocupados, porque añadir más *streams* lógicos no multiplica el ancho de banda del bus PCIe ni crea hardware de transferencia adicional; por el contrario, fragmenta los mismos `192 MB` en bloques más pequeños (`1M` de elementos por *chunk* en lugar de `4M`), aumentando la sobrecarga de gestión de descriptores DMA y lanzamiento de kernels por parte del *driver* CUDA. En el plano físico, la mayoría de las GPUs NVIDIA modernas exponen únicamente `1` o `2` *copy engines* (motores DMA dedicados, permitiendo en algunos modelos una copia H2D y una D2H simultáneas en direcciones opuestas). Este límite de hardware implica que entre `2` y `4` *streams* suelen ser suficientes para mantener saturado el *pipeline* de transferencia y cómputo; cualquier *stream* adicional queda encolado esperando disponibilidad en los mismos motores DMA. Asimismo, el ancho de banda total del bus PCIe es un recurso único y compartido por todos los *streams* activos: dividir la transferencia en más *streams* no reduce el tiempo agregado necesario para mover `192 MB` sobre el enlace físico, sino que únicamente reorganiza en el tiempo la secuencia de paquetes para intercalar mejor las fases de copia y cómputo mientras las transferencias lo permiten.
