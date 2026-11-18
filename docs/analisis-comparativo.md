# Análisis Comparativo Técnico de Aceleradores de Hardware para Cargas de Trabajo de Inteligencia Artificial

**Asignatura:** Arquitectura de Computadores — Unidad 11  
**Estudiante:** Julian Andres Ascanio  
**Repositorio:** `ascanio-post1-u11`  

---

## Introducción

El crecimiento exponencial en el tamaño y la complejidad computacional de las redes neuronales profundas —desde modelos convolucionales de visión por computador hasta transformadores de lenguaje con decenas de miles de millones de parámetros— ha superado la capacidad de escalado de las unidades centrales de procesamiento (CPU) de propósito general. En respuesta a esta brecha, la industria ha consolidado arquitecturas especializadas de aceleración de hardware diseñadas para maximizar la densidad de operaciones matriciales y tensoriales por vatio y por dólar.

El presente documento desarrolla un análisis técnico comparativo de cuatro familias representativas de aceleradores para inteligencia artificial (IA): unidades de procesamiento gráfico (**GPU**), unidades de procesamiento tensorial (**TPU**), unidades de procesamiento neuronal (**NPU**) y matrices de compuertas programables en campo (**FPGA**). El estudio evalúa sus diferencias arquitectónicas internas, su desempeño verificado en el estándar **MLPerf Inference v4.1**, su costo total de propiedad (**TCO**) a tres años, el impacto de la cuantización **INT8** frente a **FP32** sobre el *throughput* y la exactitud, y concluye con recomendaciones fundamentadas y decisiones de ingeniería para escenarios productivos concretos.

---

## 1. Caracterización Arquitectónica de los Aceleradores

Cada familia de aceleradores responde a un paradigma de cómputo distinto que determina su eficiencia según el patrón de acceso a memoria, el tamaño de lote (*batch size*) y las restricciones térmicas del entorno de despliegue.

### 1.1 GPU — NVIDIA H100 SXM (Arquitectura Hopper)

La GPU NVIDIA H100 en factor de forma SXM5 opera bajo el paradigma **SIMT masivo** (*Single Instruction, Multiple Threads*). Incorpora `132` *Streaming Multiprocessors* (SMs) que integran `16 896` núcleos CUDA para cómputo vectorial general y `528` *Tensor Cores* de cuarta generación especializados en multiplicación-acumulación de matrices (NVIDIA Corporation, 2023). Soporta nativamente precisiones **FP64** (`67 TFLOPS` en Tensor Core), **FP32** (`67 TFLOPS`), **TF32** (`989 TFLOPS` con esparsidad 2:4), **FP16/BF16** (`1 979 TFLOPS` con esparsidad; `989 TFLOPS` denso) y **FP8/INT8** (`3 958 TOPS` con esparsidad; `1 979 TOPS` denso) mediante su *Transformer Engine* dinámico.

Su jerarquía de memoria combina registros y memoria compartida (*shared memory* / caché L1 configurable por SM), una caché L2 unificada de gran capacidad y `80 GB` de memoria **HBM3** con un ancho de banda de `3.35 TB/s`, interconectada mediante NVLink de cuarta generación (`900 GB/s` bidireccionales) con un TDP máximo de `700 W` (NVIDIA Corporation, 2023). El modelo de programación se apoya en **CUDA**, bibliotecas optimizadas como **cuDNN** y el compilador de inferencia **TensorRT**, lo que la convierte en el estándar dominante para entrenamiento de redes profundas y cómputo de alto rendimiento (HPC).

### 1.2 TPU — Google Cloud TPU v5e y v5p

Las Unidades de Procesamiento Tensorial de Google adoptan una arquitectura basada en **arreglos sistólicos** (*Systolic Arrays*) orientada exclusivamente al álgebra lineal densa. En la generación **TPU v5p**, cada chip integra `2` *TensorCores* (con `4` Unidades de Multiplicación Matricial o **MXU** de `128 × 128` por *TensorCore*, es decir, `8` MXUs en total) junto con `4` *SparseCores* de segunda generación para tablas de *embeddings* (Google Cloud, 2024a). Un chip TPU v5p alcanza `459 TFLOPS` en **BF16/FP8** y `918 TOPS` en **INT8**, respaldado por `95 GiB` útiles (`96 GiB` físicos) de memoria HBM2e a `2 765 GB/s` e interconexión entre chips (ICI) de `1 200 GB/s` en topología toro 3D escalable hasta `8 960` chips (Google Cloud, 2024a).

Por su parte, la variante **TPU v5e**, optimizada para inferencia costo-eficiente y entrenamiento de mediana escala, integra `1` *TensorCore* con `4` MXUs, entrega `197 TFLOPS` en **BF16** y `393 TOPS` en **INT8**, y cuenta con `16 GiB` de HBM2e a `819 GB/s` en topología toro 2D de hasta `256` chips (Google Cloud, 2024b). A diferencia de las GPUs, el flujo de datos sistólico minimiza los accesos a registros intermedios reutilizando operandos directamente entre elementos de procesamiento adyacentes. Su programación se realiza exclusivamente mediante compilación estática **XLA** desde **JAX**, **PyTorch/XLA** o **TensorFlow**.

### 1.3 NPU — Apple M3 Ultra (Neural Engine)

Las Unidades de Procesamiento Neuronal representan aceleradores de **flujo de datos fijo** (*fixed-function dataflow*) integrados en SoCs para cómputo en el borde o estaciones de trabajo compactas. El SoC **Apple M3 Ultra** une dos matrices M3 Max mediante interconexión *UltraFusion* e integra un **Neural Engine de 32 núcleos** capaz de entregar `72 TOPS` en inferencia neuronal cuantizada (Apple Inc., 2025).

Su característica arquitectónica distintiva es la **memoria unificada** LPDDR5 de hasta `512 GB` compartida de forma coherente entre CPU, GPU y Neural Engine con un ancho de banda de `819 GB/s` (Apple Inc., 2025), eliminando por completo las copias explícitas por bus PCIe. Opera con un consumo del paquete sustancialmente menor que el de un acelerador de centro de datos (en el rango de `~125–200 W` bajo carga combinada en Mac Studio, frente a `< 5 W` en variantes móviles de bajo consumo como las NPU de teléfonos inteligentes). Se programa mediante los *frameworks* **Core ML** y **BNNS** (*Basic Neural Network Subroutines*).

### 1.4 FPGA — AMD/Xilinx Alveo U55C

Una FPGA ofrece hardware **reconfigurable a nivel de circuito**, permitiendo sintetizar tuberías (*pipelines*) de procesamiento a la medida del modelo y del flujo de entrada/salida. La tarjeta aceleradora **AMD Alveo U55C** (basada en arquitectura Virtex UltraScale+ XCU55) dispone de `1 304K` tablas de búsqueda (**LUT**), `2 607K` registros, `9 024` bloques **DSP** para aritmética de punto fijo/flotante, `70.9 Mb` de Block RAM (**BRAM**) y `270 Mb` de UltraRAM (**URAM**) en chip, complementados con `16 GB` de memoria **HBM2** a `460 GB/s` y un TDP típico de `115 W` (`150 W` máximo) (AMD Inc., 2022).

Al carecer de sistema operativo o planificador de hilos interno entre la entrada de red (`2 × 100 GbE` QSFP28) y la lógica de cómputo, una FPGA puede ejecutar inferencia determinística con latencias inferiores a `1 ms`. Su modelo de programación emplea síntesis de alto nivel **HLS** con **Vitis AI** o lenguajes de descripción de hardware (**VHDL/Verilog**), lo que incrementa la complejidad de desarrollo frente a CUDA pero garantiza latencia acotada y predecible.

### 1.5 Tabla Comparativa de Caracterización

| Acelerador | Paradigma | Unidad de Cómputo Principal | Precisiones Soportadas | Memoria y Ancho de Banda | TDP / Consumo | Modelo de Programación | Caso de Uso Primario |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GPU (NVIDIA H100 SXM)** | SIMT masivo | 16 896 CUDA Cores + 528 Tensor Cores | FP64, FP32, TF32, FP16, BF16, FP8, INT8 | 80 GB HBM3 (`3.35 TB/s`) | 700 W | CUDA / cuDNN / TensorRT | Entrenamiento DNN/LLM y HPC |
| **TPU v5p / v5e (Google)** | *Systolic Array* | MXU `128×128` (8 en v5p; 4 en v5e) | BF16, FP8 (v5p), INT8 | 95 GiB (`2.76 TB/s`) / 16 GiB (`819 GB/s`) | N/D (Cloud Pod) | XLA / JAX / TensorFlow | Entrenamiento e inferencia LLM a gran escala |
| **NPU (Apple M3 Ultra)** | *Dataflow* fijo | Neural Engine 32-core (`72 TOPS`) | FP16, INT8 | Hasta 512 GB Unificada (`819 GB/s`) | ~125–200 W (SoC) | Core ML / BNNS | Inferencia local de baja latencia y alto tamaño en RAM |
| **FPGA (AMD Alveo U55C)** | Reconfigurable | 1 304K LUT + 9 024 DSP + BRAM/URAM | INT4, INT8, INT16, FP16/FP32 | 16 GB HBM2 (`460 GB/s`) + SRAM on-chip | 115 W típico (150 W máx.) | Vitis HLS / VHDL / Verilog | Inferencia determinística `< 1 ms` y redes 100GbE |

---

## 2. Desempeño en Benchmarks MLPerf Inference v4.1

El consorcio **MLCommons** publica resultados auditados bajo el estándar **MLPerf Inference v4.1** (agosto de 2024), diferenciando dos escenarios principales: **Offline** (máximo *throughput* en muestras/segundo sin restricción de latencia por lote) y **Server** (llegada estocástica de consultas bajo un límite estricto de latencia $p99$) (MLCommons, 2024).

En la división *Datacenter* (nodos de 8 aceleradores) y *Edge*, los resultados públicos reportados para los tres modelos de referencia muestran las siguientes magnitudes:

| Modelo de Referencia | Métrica y Escenario | Sistema 8× NVIDIA H100 SXM (TensorRT) | Sistema Edge NVIDIA L4 (72 W) | Google Cloud TPU v5e / v5p | Líder en *Throughput* | Líder en Eficiencia Energética |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **ResNet-50 v1.5** (`224×224`, ImageNet) | *Offline* (`muestras/s`) | `~650 000 – 700 000+` (8× H100) | `~11 000 – 14 000` (1× L4, *Edge*) | No enviado en v4.1 (enfoque en LLM; TPU v3 32-chip alcanzó `1 000 000` en rondas previas) | **NVIDIA H100** | **NVIDIA L4 / NPUs Edge** (en `muestras/s/W` para baja potencia) |
| **BERT-Large** (SQuAD v1.1, 99% / 99.9%) | *Offline* (`muestras/s`) y *Server* ($p99$ SLA) | `~50 000 – 75 000+` *Offline*; cumple SLA $p99$ (`≤ 130 ms` Server) | Cumple SLA con menor QPS por nodo | No enviado en v4.1 (prioridad a GPT-J 6B y Llama 2) | **NVIDIA H100** | **NVIDIA H100 (FP8)** en centro de datos por densidad |
| **Stable Diffusion XL** (COCO, texto a imagen) | *Offline* (`muestras/s`) | `~11 – 15+` (nodo 8× H100; `~1.5–2` img/s por GPU) | No competitivo para alta resolución | Enfoque en servicio JetStream/MaxText | **NVIDIA H100** | **NVIDIA H100** (mejora `~1.25×` vs v4.0 por TensorRT) |

*Nota metodológica:* En las rondas MLPerf Inference v3.1 y v4.1, Google orientó sus envíos oficiales de Cloud TPU v5e hacia modelos generativos de lenguaje (como GPT-J 6B) para demostrar mejoras de `2.7×` en costo-rendimiento respecto a TPU v4, omitiendo envíos de modelos heredados como ResNet-50 v1.5 o BERT-Large (Google Cloud, 2023; MLCommons, 2024).

### Análisis Arquitectónico del Liderazgo

1. **Liderazgo en *Throughput* Bruto (NVIDIA H100 SXM):** La H100 lidera en los tres modelos dentro de la categoría *Datacenter* gracias a la combinación de `528` *Tensor Cores* con soporte **FP8/INT8** y `3.35 TB/s` de ancho de banda HBM3. En ResNet-50 v1.5, la ejecución en lotes grandes satura las unidades matriciales; en BERT-Large y Stable Diffusion XL, el *Transformer Engine* reduce a la mitad la huella de memoria de activaciones y pesos frente a FP16, permitiendo mayor tamaño de lote efectivo dentro del presupuesto de latencia $p99$.
2. **Liderazgo en Eficiencia Energética (`muestras/s/W`):** En cargas convolucionales compactas como ResNet-50, las arquitecturas de menor TDP (como la GPU NVIDIA L4 de `72 W` o aceleradores NPU dedicados de `< 15 W` en el borde) superan a un nodo H100 de `700 W` por GPU en `muestras/s/W`, debido a que la energía estática del *host* PCIe y de los enlaces NVLink de alta velocidad penaliza la eficiencia cuando la intensidad aritmética por consulta es moderada. En contraste, para modelos generativos pesados como Stable Diffusion XL o LLMs, la H100 recupera eficiencia energética al evitar desbordamientos de memoria hacia la RAM del host.
