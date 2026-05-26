# Fractal de Mandelbrot 8K + Convolución 2D (versión secuencial)

Generación de una imagen ultra-alta-resolución del **conjunto de Mandelbrot**
seguida de **procesamiento por convolución 2D pesada** (desenfoque Gaussiano
de radio amplio + filtro Sobel), implementada en **C++17 puro y 100 %
secuencial** (sin hilos, sin OpenMP, sin MPI, sin CUDA).

El objetivo no es solamente producir las imágenes, sino servir como **línea
base para un proyecto de paralelización**: medir el tiempo en un solo hilo,
identificar los cuellos de botella, y discutir cómo se aceleraría con
OpenMP / pthreads / MPI / CUDA.

---

## Índice

1. [Explicación teórica](#1-explicación-teórica)
2. [Diseño del algoritmo](#2-diseño-del-algoritmo)
3. [Archivos del proyecto](#3-archivos-del-proyecto)
4. [Instrucciones de compilación](#4-instrucciones-de-compilación)
5. [Ejemplo de ejecución y salidas](#5-ejemplo-de-ejecución-y-salidas)
6. [Análisis de complejidad](#6-análisis-de-complejidad)
7. [Cuellos de botella](#7-cuellos-de-botella)
8. [Posibles optimizaciones secuenciales](#8-posibles-optimizaciones-secuenciales)
9. [Estrategias de paralelización recomendadas](#9-estrategias-de-paralelización-recomendadas)

---

## 1. Explicación teórica

### 1.1 El conjunto de Mandelbrot

El conjunto de Mandelbrot **M** es el subconjunto del plano complejo formado
por los números **c ∈ ℂ** para los cuales la sucesión

> z₀ = 0,    zₙ₊₁ = zₙ² + c

permanece **acotada** cuando n → ∞.

Un teorema clásico afirma que si en algún momento `|zₙ| > 2`, la sucesión
diverge necesariamente a infinito. Eso da el algoritmo de **"escape time"**:

- Para cada píxel (que representa un número complejo `c`), iteramos.
- Si `|z|² > 4` en alguna iteración `k`, decimos que `c ∉ M` y coloreamos
  según `k` (cuantos más rápido escape, más "lejos" de M).
- Si tras `MAX_ITER` iteraciones no escapa, asumimos `c ∈ M` y pintamos
  negro.

Cuando `z = x + iy` y `c = a + ib`:

```
z² + c = (x² - y²) + a   +   i (2xy + b)
```

Esto se calcula con **3 multiplicaciones** y unas sumas por iteración,
manteniendo `x²` e `y²` precomputados.

### 1.2 La convolución 2D

Dada una imagen `I` y un kernel `K` de tamaño `(2r+1) × (2r+1)`, la
convolución 2D produce una imagen `O` definida píxel a píxel como:

```
O(x, y)  =  Σⱼ Σᵢ  K(i, j) · I(x + i, y + j)         con i, j ∈ [-r, r]
```

La operación es **local**: cada píxel de salida depende solo de un vecindario
del píxel de entrada. Por tanto **no hay dependencias entre píxeles de
salida**, lo que la convierte en un problema **embarazosamente paralelizable**.

- **Desenfoque Gaussiano**: el kernel es una Gaussiana 2D normalizada.
  Suaviza la imagen y reduce el ruido. Aquí usamos `r = 15` (kernel 31×31)
  para que el filtro sea **pesado**.
- **Sobel**: dos kernels 3×3 que aproximan ∂I/∂x y ∂I/∂y. La magnitud del
  gradiente `√(gx² + gy²)` resalta los bordes.

---

## 2. Diseño del algoritmo

```
                   ┌─────────────────────────────┐
                   │  Configuración (CLI args)   │
                   └─────────────┬───────────────┘
                                 │
                                 ▼
   ┌─────────────────────────────────────────────────────────┐
   │  FASE 1 — Generar Mandelbrot                            │
   │  ─────────────────────────────                          │
   │  for y in 0..H:                                         │
   │     for x in 0..W:                                      │
   │         iter = escape_time(mapear(x,y))                 │
   │         img[x,y] = paleta(iter)                         │
   │  → guardar fractal.ppm                                  │
   └─────────────────────────────────────────────────────────┘
                                 │
                                 ▼
   ┌─────────────────────────────────────────────────────────┐
   │  FASE 2 — Convolución 2D (Gaussiana, r = 15)            │
   │  ─────────────────────────────────────────              │
   │  kernel = gaussian_kernel(r, σ = r/2)                   │
   │  for y in 0..H:                                         │
   │     for x in 0..W:                                      │
   │         out[x,y] = Σ Σ kernel[i,j] · fractal[x+i, y+j]  │
   │  → guardar blurred.ppm                                  │
   └─────────────────────────────────────────────────────────┘
                                 │
                                 ▼
   ┌─────────────────────────────────────────────────────────┐
   │  FASE 3 — Sobel (sobre el fractal original)             │
   │  ────────────────────────────────────────               │
   │  for y in 0..H:                                         │
   │     for x in 0..W:                                      │
   │         gx, gy = convolución 3×3                        │
   │         out[x,y] = √(gx² + gy²)                         │
   │  → guardar edges.ppm                                    │
   └─────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                ┌───────────────────────────────┐
                │  Reporte de tiempos (chrono)  │
                └───────────────────────────────┘
```

Decisiones de diseño relevantes:

- **Formato de salida PPM (P6 binario)**: sin dependencias externas, cabecera
  ASCII trivial + datos crudos. Cualquier visor (GIMP, IrfanView, feh, etc.)
  los abre, y `convert` (ImageMagick) los pasa a PNG.
- **`std::vector<uint8_t>` contiguo, row-major**: máxima localidad de caché
  para el filtro y el guardado.
- **`std::chrono::high_resolution_clock`**: precisión sub-milisegundo para
  medir cada fase por separado y la suma total.
- **Argumentos CLI**: permiten probar resoluciones bajas (Full HD, 4K) antes
  de lanzar el 8K final, que puede tardar varios minutos.

---

## 3. Archivos del proyecto

| Archivo     | Descripción                                            |
| ----------- | ------------------------------------------------------ |
| `main.cpp`  | Implementación completa con comentarios detallados.    |
| `README.md` | Este documento.                                        |

Solo un archivo de código, sin makefile, sin dependencias.

---

## 4. Instrucciones de compilación

### Linux (recomendado: GCC)

```bash
g++ -O3 -std=c++17 -march=native main.cpp -o fractal
```

- `-O3` activa todas las optimizaciones (inlining agresivo, auto-vectorización,
  desenrollado de bucles).
- `-march=native` permite usar el set de instrucciones específico de tu CPU
  (AVX2, AVX-512 si está disponible). Mejora notable en la convolución.
- `-std=c++17` por `std::clamp` y el manejo limpio de lambdas.

Para depurar:

```bash
g++ -O0 -g -std=c++17 -Wall -Wextra main.cpp -o fractal_debug
```

### Linux (con Clang)

```bash
clang++ -O3 -std=c++17 -march=native main.cpp -o fractal
```

### Windows con MinGW-w64

```cmd
g++ -O3 -std=c++17 -march=native main.cpp -o fractal.exe
```

### Windows con MSVC (Visual Studio Developer Command Prompt)

```cmd
cl /O2 /std:c++17 /EHsc main.cpp /Fe:fractal.exe
```

> `/O2` es el equivalente más alto de optimización en MSVC.  
> `/EHsc` habilita el manejo estándar de excepciones de C++.

---

## 5. Ejemplo de ejecución y salidas

### Opciones de línea de comandos

```text
Uso: ./fractal [opciones]
  -w, --width  N     ancho de la imagen          (default 7680)
  -h, --height N     alto  de la imagen          (default 4320)
  -i, --iter   N     iteraciones máximas         (default 1000)
  -r, --radius N     radio del kernel Gaussiano  (default 15)
      --no-blur      omite el desenfoque Gaussiano
      --no-sobel     omite el filtro Sobel
      --help         muestra esta ayuda
```

### Ejecución de prueba en 800×450 (validación rápida)

```text
$ ./fractal -w 800 -h 450 -i 300 -r 7
============================================================
 Fractal de Mandelbrot + Convolucion 2D (SECUENCIAL)
============================================================
 Resolucion        : 800 x 450
 Iter. max         : 300
 Radio Gaussiano   : 7 (kernel 15x15)
 Aplicar Gaussiana : si
 Aplicar Sobel     : si
------------------------------------------------------------

[1/3] Generando fractal de Mandelbrot...
  Mandelbrot: 100%
      Guardando fractal.ppm ...

[2/3] Aplicando desenfoque Gaussiano (r=7)...
  Gaussiana: 100%
      Guardando blurred.ppm ...

[3/3] Aplicando filtro Sobel...
  Sobel: 100%
      Guardando edges.ppm ...

============================================================
 Tiempos (milisegundos)
------------------------------------------------------------
  Mandelbrot          : 125 ms
  Guardar fractal     : 0 ms
  Desenfoque Gauss.   : 190 ms
  Guardar blurred     : 0 ms
  Sobel               : 10 ms
  Guardar edges       : 0 ms
------------------------------------------------------------
  TOTAL               : 325 ms  (0.325 s)
============================================================
```

### Ejecución final 8K

```bash
./fractal                       # 7680x4320, iter=1000, r=15
```

Tiempos esperados aproximados en una CPU desktop moderna (Ryzen 5/Intel i5 reciente,
un solo núcleo) compilado con `-O3 -march=native`:

| Fase                          | Tiempo aproximado |
| ----------------------------- | ----------------- |
| Mandelbrot (1000 iter)        | 30 – 90 s         |
| Gaussiana (kernel 31×31)      | 60 – 180 s        |
| Sobel                         | 1 – 3 s           |
| Guardar 3 PPM (~95 MiB c/u)   | 1 – 3 s           |
| **TOTAL**                     | **~2 – 5 min**    |

(El rango es amplio porque depende fuertemente de la frecuencia turbo de
la CPU, la caché disponible y el ancho de banda a memoria.)

### Visualizar / convertir las salidas

Los archivos `fractal.ppm`, `blurred.ppm` y `edges.ppm` pesan unos **95 MiB**
cada uno a 8K. Para convertirlos a PNG (mucho más pequeño y compatible):

```bash
# Linux / Mac (ImageMagick)
convert fractal.ppm fractal.png
convert blurred.ppm blurred.png
convert edges.ppm   edges.png

# o todos a la vez
for f in *.ppm; do convert "$f" "${f%.ppm}.png"; done
```

En Windows: GIMP y IrfanView abren PPM directamente.

---

## 6. Análisis de complejidad

Notación:

- **W**, **H**: ancho y alto en píxeles. **N = W·H**.
- **MAX_ITER**: tope de iteraciones del Mandelbrot.
- **r**: radio del kernel Gaussiano. **k = (2r+1)²** elementos del kernel.
- **C**: canales (aquí 3, RGB).

### 6.1 Mandelbrot

- **Peor caso por píxel**: `O(MAX_ITER)`.
- **Total peor caso**: `O(N · MAX_ITER)` operaciones.
- **Caso medio**: mucho menor, porque la mayoría de píxeles del exterior
  escapan rápido (en pocas decenas de iteraciones). Pero los píxeles
  cercanos al borde del conjunto y los del interior consumen el máximo.
  → **Carga muy desigual entre regiones de la imagen** (esto importa
  para la paralelización).

Para 7680×4320 con MAX_ITER=1000:

> Peor caso ≈ 33.2 M × 1000 = **3.3 × 10¹⁰ iteraciones**.  
> Caso real ≈ 5–15 × 10⁹ iteraciones (depende de la región mostrada).

### 6.2 Convolución Gaussiana 2D (no separable)

- **Por píxel**: `k · C` multiply-add = `(2r+1)² · 3` operaciones.
- **Total**: `O(N · k · C)`.

Para 7680×4320 con r=15:

> N = 33 177 600 píxeles  
> k = 31² = 961 elementos  
> C = 3 canales  
> Total ≈ **9.6 × 10¹⁰ multiply-add**.

### 6.3 Sobel

- **Por píxel**: 9 multiply-add por gradiente × 2 (gx y gy) + sqrt.
- **Total**: `O(N · 18)` ≈ **6 × 10⁸** operaciones para 8K. Comparativamente
  barato.

### 6.4 Memoria

- Cada imagen 8K RGB ocupa `7680 · 4320 · 3 = 95 432 400 bytes ≈ 91 MiB`.
- Mantenemos 3 imágenes en memoria simultáneamente (original, blurred,
  edges) ⇒ **~273 MiB**. Razonable en cualquier máquina moderna; nada
  comparado al GiB que ocuparían en otros formatos.

---

## 7. Cuellos de botella

### 7.1 Cuello de botella **computacional**

La **convolución Gaussiana 2D no separable** domina el tiempo total: ~10¹¹
operaciones contra ~10¹⁰ del Mandelbrot. Es **compute-bound** porque la
densidad aritmética por byte leído es alta.

### 7.2 Cuello de botella **de memoria**

La convolución accede a una **ventana 2D** del tamaño del kernel. Para r=15:

- Cada píxel de salida lee 31 píxeles de **31 filas distintas**.
- Una fila de 8K ocupa `7680 · 3 = 22.5 KiB`.
- 31 filas ≈ **700 KiB**, **mucho mayor que la caché L1 (32–48 KiB)** y
  comparable o mayor a L2 (256 KiB – 1 MiB).

Resultado: muchos fallos de L1/L2 al moverse en `y`. La velocidad real
queda lejos del pico teórico de la CPU.

### 7.3 Cuello de botella **algorítmico**

- En el Mandelbrot, los **píxeles del interior del conjunto siempre
  alcanzan MAX_ITER**: una franja densa de la imagen es 10–100× más cara
  que el resto. Esto no afecta al programa secuencial (lo paga igual al
  final), pero **complica la paralelización** (balanceo).
- En el Sobel, la **raíz cuadrada** por píxel es ~10× más cara que una
  multiplicación; en presupuesto total no domina, pero limita el ILP.

### 7.4 Cuello de botella **de E/S**

- Cada PPM 8K son ~91 MiB; tres archivos ⇒ ~273 MiB en disco. En SSD NVMe
  es ≤ 1 s, en HDD puede ser 5–10 s. Para evaluación de rendimiento puro,
  conviene escribir a `/tmp` (tmpfs) o usar las flags `--no-blur` /
  `--no-sobel` mientras se mide solo el cómputo.

---

## 8. Posibles optimizaciones secuenciales

Incluso sin paralelizar, hay margen considerable:

1. **Separar la Gaussiana en dos pasadas 1D** (`(2r+1) · 2` vs `(2r+1)²`).
   Para r=15, eso son **62 mul/píxel en lugar de 961**: ~**15× más rápido**
   sin perder calidad.
2. **Auto-vectorización SIMD**: con `-O3 -march=native` el compilador genera
   AVX2/AVX-512 para los bucles internos. Mejora 2–8× según CPU.
3. **Trabajar en `float` en vez de `double`** en la Gaussiana (mitad de
   memoria + doble densidad SIMD).
4. **Tiled / blocked convolution**: procesar la imagen en bloques que quepan
   en L2 mejora drásticamente la tasa de aciertos de caché.
5. **Pre-escapado del Mandelbrot**: detectar componentes cardioide y bulbo
   principal por fórmula cerrada (puntos que son sabidamente del interior),
   ahorrando hasta el 30 % de las iteraciones.
6. **"Periodicity checking"**: si `zₙ` vuelve a aparecer durante la
   iteración, sabemos que el punto está en M y podemos cortar.
7. **Smooth coloring** con `iter - log₂(log₂|z|²)` para una transición
   continua de colores (mejora visual sin coste relevante).

---

## 9. Estrategias de paralelización recomendadas

Tanto el Mandelbrot como la convolución 2D son **trivialmente paralelos
a nivel de píxel**: ningún píxel de salida depende de otro píxel de
salida. Esto los hace candidatos ideales para multi-threading, SIMD y GPU.

### 9.1 OpenMP (la primera opción razonable)

Una sola directiva paraleliza el bucle externo del Mandelbrot, la
convolución y el Sobel:

```cpp
#pragma omp parallel for schedule(dynamic, 16)
for (int y = 0; y < img.height; ++y) {
    for (int x = 0; x < img.width; ++x) {
        // ... cómputo por píxel ...
    }
}
```

- **`schedule(dynamic, 16)`** es **crítico para el Mandelbrot**: las filas
  no son iguales en coste (las que cruzan el conjunto son caras). Con
  `static` algunos hilos terminan al 20 % y otros al 100 %. Con `dynamic`,
  cada hilo toma trabajo cuando lo necesita.
- Para la **convolución** el coste por fila es constante; `schedule(static)`
  funciona perfectamente.
- **Speedup esperado**: cercano al número de núcleos físicos (8 núcleos →
  ~7×) en convolución. En Mandelbrot suele ser algo menor por el
  desequilibrio residual.
- **Esfuerzo de implementación**: minutos.

### 9.2 Hilos POSIX (`pthreads`) o `std::thread`

Equivalente a OpenMP pero a mano: división explícita de filas entre hilos,
join al final.

```cpp
auto worker = [&](int yStart, int yEnd) {
    for (int y = yStart; y < yEnd; ++y) { /* ... */ }
};
std::vector<std::thread> th;
for (int t = 0; t < nThreads; ++t)
    th.emplace_back(worker, t * H / nThreads, (t + 1) * H / nThreads);
for (auto& t : th) t.join();
```

- **Cuándo elegirlo en lugar de OpenMP**: si necesitas balanceo dinámico
  más sofisticado (cola de tareas, work stealing), o si trabajas en un
  entorno sin OpenMP.
- **Speedup esperado**: parecido a OpenMP. **Esfuerzo**: medio.

### 9.3 SIMD explícito (AVX2 / AVX-512)

Procesar **4–8 píxeles del Mandelbrot en paralelo** dentro de un solo
núcleo mediante intrínsecos:

```cpp
__m256d cr = _mm256_loadu_pd(...);   // 4 c's reales a la vez
__m256d zr = _mm256_setzero_pd();
// ... iterar 4 píxeles simultáneamente ...
```

La complicación: cuando algunos píxeles del vector ya escaparon y otros
no, hay que enmascarar con `_mm256_blendv_pd`. Aún así, **2–6×** sobre la
versión escalar es típico.

### 9.4 MPI (memoria distribuida)

Para clústers o el caso en que una sola máquina no basta:

- **Partir la imagen en bandas** (e.g. el proceso `i` calcula las filas
  `[i·H/P, (i+1)·H/P)`).
- Cada rango se calcula completamente local (Mandelbrot no requiere
  datos de otros procesos).
- Para la **convolución**, cada proceso necesita además **`r` filas extra**
  por encima y por debajo de su banda (ghost cells / halo) — un único
  intercambio `MPI_Sendrecv` con los vecinos.
- Recolección final con `MPI_Gatherv` al rango 0 para guardar el PPM.
- **Cuándo usarlo**: imágenes gigantes (16K, 32K), zoom profundo,
  renderizado animado en clúster.
- **Esfuerzo**: alto, pero el patrón "banda + halo" es el ABC del HPC.

### 9.5 CUDA / GPU

El candidato ideal: **decenas de miles de hilos**, latencia oculta por la
GPU, alta densidad aritmética.

- **Mandelbrot**: un hilo por píxel (`gridDim × blockDim = W × H`).
  Speedups de **50–200×** sobre la versión CPU secuencial son rutinarios.
- **Convolución 2D**: usar memoria shared para cachear el tile + halo, un
  hilo por píxel de salida. Especialmente eficiente cuando el kernel es
  pequeño o se hace separable. **50–500×** sobre CPU secuencial según
  GPU.
- Frameworks alternativos: OpenCL, SYCL, HIP (AMD), Vulkan compute. El
  patrón es el mismo: grid 2D de hilos, una pasada por imagen.

### 9.6 Resumen comparativo

| Tecnología   | Esfuerzo | Speedup típico (8 núcleos CPU / GPU) | Cuándo elegirla                                                |
| ------------ | -------- | ------------------------------------ | -------------------------------------------------------------- |
| OpenMP       | Bajo     | 6 – 8×                               | Primer paso; CPU multinúcleo de una sola máquina               |
| `std::thread`/pthreads | Medio  | 6 – 8×                | Cuando quieres control fino (work stealing, tareas asíncronas) |
| SIMD AVX2/512 | Medio-alto | 2 – 6× (combinable con MT)        | Aprovechar mejor cada núcleo, sobre todo en Mandelbrot         |
| MPI          | Alto     | Lineal en número de nodos            | Clústers; imágenes que no caben en una máquina                 |
| CUDA / OpenCL | Alto    | 50 – 500×                            | Resolución muy alta o renderizado animado en tiempo real       |

**Ruta recomendada**: primero medir con esta versión secuencial → añadir
`#pragma omp parallel for` (un par de líneas, gran salto) → si aún se
necesita más, vectorizar a mano con AVX2 o portar la convolución a CUDA.
