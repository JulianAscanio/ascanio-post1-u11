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

---

## 3. Análisis Costo-Rendimiento y Costo Total de Propiedad (TCO a 3 Años)

La selección de un acelerador en producción no puede basarse únicamente en los TOPS nominales; requiere evaluar el costo de adquisición (CapEx), el consumo energético operativo (OpEx) y la complejidad de desarrollo de software. A continuación se resumen los costos de referencia de mercado para las plataformas analizadas:

- **NVIDIA H100 SXM (80 GB):** Precio de adquisición aproximado de `$30 000 USD` por GPU (sin incluir chasis servidor HGX/DGX), con un TDP de `700 W` (NVIDIA Corporation, 2023).
- **Google Cloud TPU (v4 Pod / v5e):** Disponible exclusivamente bajo modelo en la nube (OpEx); una porción de TPU v4 Pod tiene una tarifa de referencia cercana a `$32.00 USD/hora` bajo demanda, mientras que un chip TPU v5e ronda `$1.20 USD/hora` (Google Cloud, 2024b).
- **FPGA AMD Alveo U55C:** Precio de lista directo de `$4 747 USD` y entre `$5 240–$8 000 USD` en distribuidores e integraciones empresariales, con TDP típico de `115 W` (AMD Inc., 2022).
- **Apple M3 Ultra Mac Studio:** Equipo completo desde `~$4 000 USD` (configuración base) con consumo típico bajo carga de `~125–200 W` y sin costo adicional de servidor host (Apple Inc., 2025).

### 3.1 Estimación Simplificada de TCO a 3 Años (`26 280 horas`)

Para el cálculo energético se asume una tarifa eléctrica industrial promedio de `$0.12 USD/kWh` y un factor de efectividad de uso energético (PUE) de `1.3` en centro de datos.

#### Escenario A: Startup de IA con Inferencia en la Nube a Escala Media (`< 100 solicitudes/s`)

| Alternativa | Costo de Adquisición / Renta (3 años) | Energía + Enfriamiento (3 años) | Costo de Ingeniería Software | TCO Estimado (3 años) |
| :--- | :---: | :---: | :---: | :---: |
| **1× H100 SXM (On-Premise)** | `$30 000` (GPU) + `~$12 000` (host) | `0.7 kW × 26 280h × 1.3 × $0.12 ≈ $2 870` | Bajo (CUDA/TensorRT estándar) | **`~$44 870 USD`** |
| **Cloud GPU / TPU v5e (24/7 bajo demanda)** | `~1.20–2.50 USD/h × 26 280h ≈ $31 536–$65 700` | Incluido en tarifa cloud | Bajo–Medio (XLA / TensorRT) | **`~$31 500–$65 700 USD`** |
| **TPU v4 Pod (8 chips, `$32/h` continuo)** | `$32 × 26 280h = $840 960` | Incluido en tarifa cloud | Medio (JAX/XLA) | **`~$840 960 USD`** *(sobredimensionado)* |

**Argumentación Escenario A:** Para una startup con tráfico inferior a `100 req/s`, adquirir hardware propio de gama alta (`H100`) o rentar un *Pod* completo de TPU v4 resulta económicamente ineficiente por baja utilización. Una instancia elástica en la nube con **TPU v5e** o una GPU de inferencia de gama media (**NVIDIA L4**), combinada con autoescalado y reservas anuales, minimiza el CapEx inicial y alinea el gasto con la demanda real.

#### Escenario B: Inferencia en el Borde / Dispositivo (Cámara Industrial, Vehículo Autónomo, Smartphone)

| Alternativa (por unidad/nodo de borde) | Costo Hardware (CapEx) | Energía (3 años, `PUE = 1.0`) | Costo de Integración | Evaluación Técnico-Económica |
| :--- | :---: | :---: | :---: | :--- |
| **NPU integrada (SoC móvil / Apple M3 Ultra local)** | `$50–$4 000` según factor de forma | `< 5 W` (`~$16`) a `150 W` (`~$473`) | Bajo (Core ML / NNAPI) | **Óptimo para consumo `< 15 W`** y bajo costo unitario |
| **FPGA Alveo U55C (borde industrial)** | `~$5 500–$8 000` | `0.115 kW × 26 280h × $0.12 ≈ $363` | Alto (HLS / RTL) | **Óptimo si exige latencia determinística `< 1 ms`** |
| **GPU H100 SXM** | `$30 000+` | `~$2 200+` (inviable en borde por `700 W`) | Bajo | **Inviable** por peso, disipación térmica y costo |

**Argumentación Escenario B:** En el borde físico, la restricción térmica y de espacio descarta por completo a la GPU H100. Si el dispositivo es autónomo o portátil (smartphone, cámara inteligente básica), una **NPU integrada** maximiza los `TOPS/W` a menor costo. Si la aplicación industrial requiere conectividad directa de sensores a `100 GbE` y latencia estrictamente acotada por debajo del milisegundo, la **FPGA** justifica su mayor costo de ingeniería.

---

## 4. Recomendación Fundamentada por Perfil Organizacional

### Perfil A: Laboratorio de Investigación Universitario (Presupuesto `$10 000 USD`, Visión y NLP hasta `7B` parámetros)

- **Acelerador óptimo:** **Estación de trabajo con GPU NVIDIA RTX 4090 / L40S o Apple M3 Ultra Mac Studio (`96–192 GB` RAM unificada)**; dentro del presupuesto estricto de `$10 000 USD`, una **estación Linux con GPU NVIDIA RTX 4090 (`24 GB` VRAM) + CPU/RAM del sistema**, o alternativamente un **Mac Studio M3 Ultra** si la prioridad es ejecutar modelos `7B–70B` cuantizados en memoria unificada. Para experimentación académica general en visión y NLP (`≤ 7B` parámetros, entrenamiento *fine-tuning* LoRA/QLoRA), la **GPU NVIDIA con CUDA (`~$4 000–$7 000 USD` equipo completo)** resulta óptima.
- **Justificación técnica (3 criterios):**
  1. **Ecosistema de software abierto:** Compatibilidad nativa e inmediata con PyTorch, Hugging Face, CUDA y TensorRT sin requerir reescritura en XLA ni síntesis HLS.
  2. **Capacidad de entrenamiento y *fine-tuning*:** A diferencia de una NPU orientada solo a inferencia, la GPU permite retropropagación (*backpropagation*) en FP16/BF16 con *Tensor Cores* para modelos de hasta `7B` parámetros mediante QLoRA.
  3. **Ajuste presupuestal:** Permite adquirir el servidor completo (GPU, CPU, `128 GB` RAM, almacenamiento NVMe) por debajo del tope de `$10 000 USD` sin depender de tarifas recurrentes en la nube.
- **Limitaciones:** La memoria VRAM limitada (`24 GB` en una GPU de consumo o `48 GB` en gama profesional dentro del presupuesto) restringe el tamaño máximo de lote y exige cuantización de `4 bits` o *gradient checkpointing* al entrenar modelos de `7B` parámetros.

### Perfil B: Empresa de Manufactura (Inspección Visual en Tiempo Real `< 5 ms`, `50` Líneas de Producción)

- **Acelerador óptimo:** **FPGA distribuida (1 unidad compacta/industrial por línea o grupo de cámaras)** cuando existe requisito de seguridad industrial y latencia determinística garantizada; o **servidor centralizado con GPU NVIDIA H100/L4 + TensorRT con nodo de respaldo** si prima el menor costo de capital.
- **Justificación técnica (3 criterios):**
  1. **Latencia determinística (`< 1 ms` en FPGA):** El procesamiento en hardware reconfigurable elimina la variabilidad introducida por planificadores de sistema operativo y colas de *driver*, garantizando el cumplimiento estricto de la ventana de `< 5 ms` en cada ciclo de inspección (AMD Inc., 2022).
  2. **Escalabilidad de E/S directa:** Las FPGAs pueden recibir flujos de cámaras industriales directamente mediante interfaces Ethernet sin pasar por la pila TCP/IP del kernel host.
  3. **Aislamiento de fallas:** Un fallo físico en una unidad FPGA afecta únicamente a `1` de las `50` líneas de producción, evitando la detención simultánea de toda la planta.
- **Limitaciones:** El despliegue de `50` tarjetas FPGA de gama alta (como Alveo U55C a `~$8 000 USD`) eleva el CapEx a `~$400 000 USD` y demanda especialistas en Vitis HLS/VHDL para actualizar modelos neuronales, por lo que en planta suelen emplearse FPGAs embebidas de menor costo (como la familia Kria) o evaluarse una GPU centralizada redundante.

### Perfil C: Proveedor Cloud Entrenando LLM de `70B` Parámetros en Clúster de `512` Aceleradores

- **Acelerador óptimo:** **Clúster de GPUs NVIDIA H100 SXM (con NVLink + InfiniBand NDR)** o **Google Cloud TPU v5p Pod (`512` chips en toro 3D)**.
- **Justificación técnica (3 criterios):**
  1. **Ancho de banda de comunicación colectiva (`All-Reduce` / `All-Gather`):** El entrenamiento distribuido de `70B` parámetros con paralelismo tensorial y de datos exige interconexiones dedicadas (`900 GB/s` NVLink en H100 o `1 200 GB/s` ICI en toro 3D en TPU v5p) para evitar que la sincronización de gradientes sature el tiempo de paso (Google Cloud, 2024a; NVIDIA Corporation, 2023).
  2. **Capacidad y ancho de banda HBM:** Con `80 GB` HBM3 (`3.35 TB/s`) en H100 o `95 GiB` HBM2e (`2.76 TB/s`) en TPU v5p, los estados del optimizador Adam, pesos y activaciones de un modelo `70B` caben en memoria de alto ancho de banda mediante paralelismo *pipeline* y FSDP/ZeRO.
  3. **Aceleración nativa BF16/FP8:** Tanto el *Transformer Engine* de H100 (`989–1 979 TFLOPS` BF16/FP8 densos) como los `8` MXUs de TPU v5p (`459 TFLOPS` BF16/FP8) reducen semanas de entrenamiento frente a arquitecturas sin soporte nativo de baja precisión.
- **Limitaciones:** Elevadísimo costo de capital e infraestructura eléctrica/refrigeración líquida (`700 W` por GPU en H100 implica `> 358 kW` solo en aceleradores para `512` unidades), además de alta dependencia del proveedor en el caso de TPU.

---

## 5. Cuantización INT8 vs FP32: Efecto en *Throughput* y Precisión

Las cifras de rendimiento pico en TOPS/TFLOPS publicadas por los fabricantes dependen directamente de la precisión numérica empleada. Al cuantizar pesos y activaciones desde punto flotante de 32 bits (**FP32**) hacia enteros de 8 bits (**INT8**), las unidades matriciales (*Tensor Cores* en GPU o *MXU* en TPU) empaquetan más operaciones por ciclo de reloj y reducen en un factor de `4×` el tráfico de memoria, duplicando aproximadamente el *throughput* efectivo respecto a **FP16/BF16** dentro de una misma generación de GPU NVIDIA (por ejemplo, en T4 y A100) (NVIDIA Corporation, 2022).

| Precisión | *Throughput* Relativo (Tensor Core) | Modelo de Referencia | Exactitud Reportada | Restricción de Exactitud MLPerf |
| :--- | :---: | :--- | :--- | :--- |
| **FP32 (referencia)** | `1×` (línea base, sin Tensor Core) | ResNet-50 v1.5 | `76.46%` *top-1* (ImageNet) | Referencia de exactitud (`100%`) |
| **FP16 / BF16 (Tensor Core)** | `~1×` frente a referencia interna Tensor Core | ResNet-50 v1.5 | `≈ 76.4%` *top-1* (pérdida despreciable) | Debe alcanzar `≥ 99%` de la exactitud FP32 |
| **INT8 (Tensor Core, calibrado)** | `~2×` frente a FP16/BF16 (misma generación GPU) | ResNet-50 v1.5 | `≈ 75.7–76.1%` *top-1* (con calibración PTQ) | Debe alcanzar `≥ 99%` de FP32 (`≥ 75.70%`) |
| **INT8 (Tensor Core, calibrado)** | `~2×` frente a FP16/BF16 (misma generación GPU) | BERT-Large (SQuAD v1.1) | `F1 ≈ 90.0–90.8` (vs. `≈ 90.87` en FP32) | Debe alcanzar `≥ 99.9%` de la exactitud FP32 |

### Necesidad de Calibración y Reglas de MLPerf

Convertir pesos y activaciones de FP32 a INT8 mediante un *cast* directo sin calibración degrada severamente la exactitud del modelo, debido a que el rango dinámico no uniforme de los tensores en punto flotante no se distribuye linealmente sobre los `256` niveles discretos de INT8, provocando saturación de valores atípicos (*outliers*) o pérdida de resolución en valores cercanos a cero. Para alcanzar calidad de producción, TensorRT y marcos equivalentes aplican **cuantización post-entrenamiento (PTQ) con calibración** (por ejemplo, minimización de divergencia KL/entropía sobre un conjunto representativo de validación) o **entrenamiento consciente de cuantización (QAT)** para ajustar los factores de escala por capa (NVIDIA Corporation, 2022). Adicionalmente, las reglas de la división *Closed* de MLPerf Inference exigen verificar sobre el conjunto completo de validación que el modelo cuantizado conserve al menos el `99%` de la exactitud FP32 en visión (como ResNet-50) y el `99.9%` en modelos lingüísticos más sensibles como BERT-Large (MLCommons, 2024).

---

## 6. Decisiones Técnicas Justificadas (Parte 2)

### Decisión Técnica 1: ¿GPU centralizada o FPGA distribuida para el Perfil B según latencia y TCO?

En el Perfil B (`50` líneas de producción con `10` imágenes/segundo por línea), la carga agregada asciende a `500` imágenes/segundo. Con TensorRT y un modelo compacto similar a ResNet-18 en `batch = 1`, una sola GPU NVIDIA H100 logra latencias por imagen de `1–2 ms`, por lo que su *throughput* teórico supera holgadamente las `500` imágenes/segundo requeridas y permitiría, en promedio, cumplir el presupuesto de `< 5 ms` incluyendo el encolamiento de red. Desde el punto de vista estrictamente financiero de adquisición (CapEx), centralizar las `50` líneas en un servidor con una GPU H100 (`~$30 000 USD` la tarjeta) resulta sustancialmente más económico que adquirir `50` tarjetas FPGA Alveo U55C (`50 × ~$8 000 USD = ~$400 000 USD`). Sin embargo, la arquitectura centralizada introduce un **punto único de falla**: una caída del servidor, de la GPU o del enlace de red detiene simultáneamente la inspección de las `50` líneas, además de exponer la latencia de cola (*tail latency*) a *jitter* del sistema operativo, contención en el planificador CUDA y ráfagas simultáneas de tramas. Por consiguiente, si el sistema de rechazo de piezas defectuosas tiene implicaciones de **seguridad industrial** o exige una garantía dura de latencia máxima determinística en el peor caso, debe preferirse la **FPGA distribuida** pese a su mayor costo; si en cambio se trata de control de calidad estadístico no crítico para la seguridad y el presupuesto domina la decisión, la **GPU centralizada con un nodo secundario en redundancia activa** ofrece un balance TCO superior.

### Decisión Técnica 2: ¿Conviene cuantizar el modelo de inspección visual del Perfil B a INT8?

Aunque el modelo de detección de defectos ya satisfaga el límite de `< 5 ms` por imagen ejecutándose en FP16, cuantizarlo a INT8 aporta un beneficio operacional directo más allá de reducir la latencia individual: aproximadamente **duplica el *throughput* disponible** sobre los mismos *Tensor Cores*, liberando margen de capacidad (*headroom*) que permite absorber picos de tráfico, ejecutar modelos de respaldo o escalar futuras líneas de producción sobre el mismo hardware sin degradar el tiempo de respuesta. No obstante, en inspección industrial existe una asimetría crítica de costos: un **falso negativo** (pieza defectuosa aprobada como válida) genera devoluciones, fallas en campo o riesgos de seguridad mucho más costosos que un **falso positivo** (pieza válida enviada a revisión manual). Dado que la calibración INT8 puede reducir la exactitud *top-1* promedio en menos de `1` punto porcentual (como se observa en ResNet-50 dentro de MLPerf), validar únicamente con una métrica agregada global resulta insuficiente, pues la clase minoritaria correspondiente a "defecto" suele ser más sensible al ruido de cuantización que las clases mayoritarias normales. En conclusión, se recomienda **cuantizar el modelo a INT8 únicamente después de ejecutar una validación específica de precisión y *recall* sobre la clase "defecto"** en el conjunto de prueba industrial: si el *recall* de defectos se mantiene intacto (o dentro de la tolerancia aceptable mediante QAT), el *headroom* ganado justifica plenamente la cuantización; de lo contrario, debe mantenerse FP16 para proteger la tasa de detección.

---

## Conclusiones

1. Ninguna arquitectura de aceleración domina todos los escenarios: la **GPU (H100)** lidera en flexibilidad y entrenamiento/inferencia de alto *throughput* en centro de datos; la **TPU (v5e/v5p)** optimiza el escalado de arreglos sistólicos e interconexión 2D/3D para grandes modelos en la nube; la **NPU (M3 Ultra / borde)** maximiza la eficiencia energética y la capacidad de memoria unificada sin copias PCIe; y la **FPGA (Alveo U55C)** garantiza latencia determinística sub-milisegundo mediante lógica fija en hardware.
2. Las métricas de *benchmark* verificadas en **MLPerf Inference v4.1** demuestran que el ancho de banda de memoria (HBM3) y el soporte nativo de baja precisión (FP8/INT8) determinan el liderazgo en modelos generativos y transformadores, mientras que el TDP y el costo por vatio dominan en cargas de borde.
3. La cuantización **INT8** calibrada (PTQ/QAT) duplica de manera consistente el *throughput* en *Tensor Cores* frente a FP16/BF16, pero exige validar métricas específicas del dominio (como el *recall* en detección de defectos) y no solo promedios agregados como *top-1*, para evitar costos ocultos por falsos negativos.

---

## Referencias

AMD Inc. (2022). *Alveo U55C High Performance Compute Card Data Sheet (DS978)*. Advanced Micro Devices, Inc. https://docs.amd.com/r/en-US/ds978-u55c

Apple Inc. (2025). *Mac Studio with M3 Ultra Technical Specifications*. Apple Inc. https://www.apple.com/mac-studio/specs/

Google Cloud. (2023). *Google Cloud TPU v5e delivers 2.7x higher inference performance per dollar on MLPerf v3.1*. Google Cloud Blog. https://cloud.google.com/blog/products/compute/announcing-cloud-tpu-v5e-and-a3-gpus-in-ga

Google Cloud. (2024a). *Cloud TPU v5p Architecture and System Specifications*. Google Cloud Documentation. https://cloud.google.com/tpu/docs/v5p

Google Cloud. (2024b). *Cloud TPU v5e Architecture and Pricing*. Google Cloud Documentation. https://cloud.google.com/tpu/docs/v5e

MLCommons. (2024). *MLPerf Inference v4.1 Benchmark Results*. MLCommons Association. https://mlcommons.org/benchmarks/inference-datacenter/

NVIDIA Corporation. (2022). *NVIDIA TensorRT Developer Guide: Working with INT8 Quantization and Calibration*. NVIDIA Documentation. https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html

NVIDIA Corporation. (2023). *NVIDIA H100 Tensor Core GPU Architecture & Datasheet*. NVIDIA Corporation. https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
