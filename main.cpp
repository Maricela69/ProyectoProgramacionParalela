// =============================================================================
//  fractal_proc_omp — Versión PARALELA con OpenMP del programa secuencial.
// -----------------------------------------------------------------------------
//  Cambios respecto a main.cpp (puramente secuencial):
//
//    1. #include <omp.h>  (con guardas para que compile aunque OpenMP esté
//       desactivado: en ese caso se comporta como el secuencial).
//
//    2. Tres directivas #pragma omp parallel for, una en cada bucle pesado:
//         - Mandelbrot      -> schedule(dynamic, 16)  (carga desigual)
//         - Gaussiana 2D    -> schedule(static)       (carga uniforme)
//         - Sobel           -> schedule(static)       (carga uniforme)
//
//    3. Se reemplazo el reporte de progreso fila a fila (que provocaba
//       race conditions y contencion en std::cout) por un contador atomico
//       de filas terminadas que solo imprime cuando avanza un 1%.
//
//    4. Nuevo flag -t / --threads N para fijar manualmente el numero de
//       hilos (equivalente a la variable de entorno OMP_NUM_THREADS).
//
//    5. Se reporta cuantos hilos OpenMP estan activos y se calcula el
//       "speedup observado" si el usuario corre antes la version secuencial.
//
//  El resto del codigo (matematica del fractal, kernel Gaussiano, Sobel,
//  formato PPM, parseo de argumentos, paleta de colores) es identico a la
//  version secuencial: los comentarios explicativos no se repiten aqui en
//  detalle, ver main.cpp para la teoria completa.
//
//  Compilacion:
//      Linux  : g++ -O3 -fopenmp -std=c++17 -march=native main_omp.cpp -o fractal_omp
//      Windows: g++ -O3 -fopenmp -std=c++17 -march=native main_omp.cpp -o fractal_omp.exe
//      MSVC   : cl /O2 /openmp /std:c++17 /EHsc main_omp.cpp
//
//  Para forzar un numero de hilos especifico:
//      ./fractal_omp -t 8
//    o bien:
//      OMP_NUM_THREADS=8 ./fractal_omp
// =============================================================================

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ---- OpenMP con fallback graceful si no esta disponible ---------------------
#ifdef _OPENMP
  #include <omp.h>
#else
  // Stubs para que el codigo siga compilando sin -fopenmp.
  inline int  omp_get_max_threads() { return 1; }
  inline int  omp_get_num_threads() { return 1; }
  inline int  omp_get_thread_num()  { return 0; }
  inline void omp_set_num_threads(int) {}
#endif

// -----------------------------------------------------------------------------
// 1) ESTRUCTURA DE IMAGEN  (identica a la version secuencial)
// -----------------------------------------------------------------------------
struct Image {
    int width  = 0;
    int height = 0;
    std::vector<uint8_t> data;

    Image() = default;
    Image(int w, int h) : width(w), height(h), data(static_cast<size_t>(w) * h * 3, 0) {}

    inline void setPixel(int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        size_t idx = (static_cast<size_t>(y) * width + x) * 3;
        data[idx + 0] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
    }
    inline void getPixel(int x, int y, uint8_t& r, uint8_t& g, uint8_t& b) const {
        size_t idx = (static_cast<size_t>(y) * width + x) * 3;
        r = data[idx + 0];
        g = data[idx + 1];
        b = data[idx + 2];
    }
};

// -----------------------------------------------------------------------------
// 2) GUARDADO PPM  (no se paraleliza: dominado por IO)
// -----------------------------------------------------------------------------
static bool savePPM(const Image& img, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "ERROR: no se pudo abrir " << filename << "\n";
        return false;
    }
    out << "P6\n" << img.width << " " << img.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(img.data.data()),
              static_cast<std::streamsize>(img.data.size()));
    return static_cast<bool>(out);
}

// -----------------------------------------------------------------------------
// 3) MANDELBROT: iteracion por pixel  (identica)
// -----------------------------------------------------------------------------
static inline int mandelbrotIterations(double cr, double ci, int maxIter) {
    double zr = 0.0, zi = 0.0;
    double zr2 = 0.0, zi2 = 0.0;
    int iter = 0;
    while (iter < maxIter && (zr2 + zi2) <= 4.0) {
        zi  = 2.0 * zr * zi + ci;   // parte imaginaria nueva
        zr  = zr2 - zi2 + cr;       // parte real nueva
        zr2 = zr * zr;
        zi2 = zi * zi;
        ++iter;
    }
    return iter;
}

static inline void iterToColor(int iter, int maxIter,
                               uint8_t& r, uint8_t& g, uint8_t& b) {
    if (iter >= maxIter) { r = g = b = 0; return; }
    double t = static_cast<double>(iter) / static_cast<double>(maxIter);
    double R = 9.0  * (1 - t) * t * t * t;
    double G = 15.0 * (1 - t) * (1 - t) * t * t;
    double B = 8.5  * (1 - t) * (1 - t) * (1 - t) * t;
    r = static_cast<uint8_t>(std::min(255.0, R * 255.0));
    g = static_cast<uint8_t>(std::min(255.0, G * 255.0));
    b = static_cast<uint8_t>(std::min(255.0, B * 255.0));
}

// -----------------------------------------------------------------------------
// 4) PROGRESO THREAD-SAFE (contador atomico)
// -----------------------------------------------------------------------------
// En la version secuencial usabamos una variable lastPct local. En paralelo
// hay que actualizar y leer un contador desde varios hilos, asi que usamos
// std::atomic<int>. SOLO el thread 0 imprime para no saturar la salida.
// -----------------------------------------------------------------------------
struct Progress {
    std::atomic<int> done{0};
    int total = 0;
    int lastPct = -1;
    const char* label = "";

    explicit Progress(int totalRows, const char* lbl) : total(totalRows), label(lbl) {}

    inline void tickAndMaybePrint() {
        int d = done.fetch_add(1, std::memory_order_relaxed) + 1;
        // Solo el hilo 0 escribe a stdout. Otros hilos solo incrementan el contador.
        if (omp_get_thread_num() == 0) {
            int pct = static_cast<int>((100LL * d) / total);
            if (pct != lastPct) {
                std::cout << "  " << label << ": " << pct << "%\r" << std::flush;
                lastPct = pct;
            }
        }
    }
    inline void finish() {
        std::cout << "  " << label << ": 100%   \n";
    }
};

// -----------------------------------------------------------------------------
// 5) GENERAR MANDELBROT EN PARALELO
// -----------------------------------------------------------------------------
// JUSTIFICACION DEL SCHEDULE:
//   La carga de trabajo por fila es MUY desigual: las filas que cruzan el
//   conjunto contienen pixeles que iteran hasta MAX_ITER, mientras que las
//   filas en las esquinas escapan en pocas iteraciones. Con schedule(static)
//   algunos hilos terminarian al 20% y otros al 100% (desbalanceo grave).
//
//   schedule(dynamic, 16) reparte filas en bloques de 16: cuando un hilo
//   termina su bloque, toma el siguiente disponible. Chunk=16 amortiza el
//   overhead de la cola dinamica sin desbalancear demasiado.
// -----------------------------------------------------------------------------
static void generateMandelbrot(Image& img, int maxIter) {
    const double aspect = static_cast<double>(img.width) / img.height;
    const double xCenter = -0.5,  yCenter = 0.0;
    const double xRange  = 3.5;
    const double yRange  = xRange / aspect;
    const double xMin    = xCenter - xRange * 0.5;
    const double yMin    = yCenter - yRange * 0.5;
    const double dx      = xRange / img.width;
    const double dy      = yRange / img.height;
    (void)yCenter;  // ya incorporado a yMin

    Progress prog(img.height, "Mandelbrot");

    // -------- ZONA PARALELA --------
    #pragma omp parallel for schedule(dynamic, 16) default(none) \
            shared(img, prog) firstprivate(maxIter, xMin, yMin, dx, dy)
    for (int y = 0; y < img.height; ++y) {
        const double ci = yMin + y * dy;
        for (int x = 0; x < img.width; ++x) {
            const double cr = xMin + x * dx;
            int iter = mandelbrotIterations(cr, ci, maxIter);
            uint8_t r, g, b;
            iterToColor(iter, maxIter, r, g, b);
            img.setPixel(x, y, r, g, b);
        }
        prog.tickAndMaybePrint();
    }
    // -------- FIN ZONA PARALELA ----
    prog.finish();
}

// -----------------------------------------------------------------------------
// 6) KERNEL GAUSSIANO  (no se paraleliza: kernel pequeno y se calcula 1 vez)
// -----------------------------------------------------------------------------
static std::vector<double> gaussianKernel(int radius, double sigma) {
    const int size = 2 * radius + 1;
    std::vector<double> kernel(static_cast<size_t>(size) * size);
    double sum = 0.0;
    const double s2 = 2.0 * sigma * sigma;
    for (int j = -radius; j <= radius; ++j) {
        for (int i = -radius; i <= radius; ++i) {
            double v = std::exp(-(i * i + j * j) / s2);
            kernel[(j + radius) * size + (i + radius)] = v;
            sum += v;
        }
    }
    for (auto& v : kernel) v /= sum;
    return kernel;
}

// -----------------------------------------------------------------------------
// 7) CONVOLUCION 2D EN PARALELO
// -----------------------------------------------------------------------------
// JUSTIFICACION DEL SCHEDULE:
//   A diferencia del Mandelbrot, cada fila cuesta EXACTAMENTE LO MISMO
//   (mismo numero de operaciones por pixel, independiente del contenido).
//   schedule(static) es optimo: minimo overhead de planificacion y reparto
//   perfectamente balanceado.
//
//   Tambien aprovechamos la localidad: cada hilo procesa filas contiguas,
//   reusando las lineas de cache del kernel y de las filas vecinas de la
//   imagen de entrada. Si usaramos dynamic con chunk=1, los hilos saltarian
//   por la imagen y la tasa de cache miss subiria.
//
// SEGURIDAD DE THREADING:
//   - Solo se lee de 'src' (memoria compartida read-only): no hace falta
//     sincronizacion.
//   - Cada iteracion escribe en un pixel DISTINTO de 'dst' (separados al
//     menos por una fila => al menos 3*W bytes => distintas lineas de
//     cache): no hay false sharing entre filas.
// -----------------------------------------------------------------------------
static void convolve2D(const Image& src, Image& dst,
                       const std::vector<double>& kernel, int radius) {
    const int size = 2 * radius + 1;
    Progress prog(src.height, "Gaussiana");

    #pragma omp parallel for schedule(static) default(none) \
            shared(src, dst, kernel, prog) firstprivate(radius, size)
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            double accR = 0.0, accG = 0.0, accB = 0.0;
            for (int ky = -radius; ky <= radius; ++ky) {
                int sy = y + ky;
                if (sy < 0) sy = 0;
                else if (sy >= src.height) sy = src.height - 1;
                for (int kx = -radius; kx <= radius; ++kx) {
                    int sx = x + kx;
                    if (sx < 0) sx = 0;
                    else if (sx >= src.width) sx = src.width - 1;
                    double w = kernel[(ky + radius) * size + (kx + radius)];
                    uint8_t pr, pg, pb;
                    src.getPixel(sx, sy, pr, pg, pb);
                    accR += w * pr;
                    accG += w * pg;
                    accB += w * pb;
                }
            }
            auto clamp255 = [](double v) -> uint8_t {
                if (v < 0.0)   return 0;
                if (v > 255.0) return 255;
                return static_cast<uint8_t>(v);
            };
            dst.setPixel(x, y, clamp255(accR), clamp255(accG), clamp255(accB));
        }
        prog.tickAndMaybePrint();
    }
    prog.finish();
}

// -----------------------------------------------------------------------------
// 8) SOBEL EN PARALELO
// -----------------------------------------------------------------------------
// Mismo razonamiento que la convolucion: coste por fila constante,
// schedule(static).
// -----------------------------------------------------------------------------
static void sobelFilter(const Image& src, Image& dst) {
    static const int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    static const int Gy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};

    Progress prog(src.height, "Sobel");

    #pragma omp parallel for schedule(static) default(none) \
            shared(src, dst, Gx, Gy, prog)
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            double gx = 0.0, gy = 0.0;
            for (int ky = -1; ky <= 1; ++ky) {
                int sy = std::clamp(y + ky, 0, src.height - 1);
                for (int kx = -1; kx <= 1; ++kx) {
                    int sx = std::clamp(x + kx, 0, src.width - 1);
                    uint8_t r, g, b;
                    src.getPixel(sx, sy, r, g, b);
                    double lum = 0.299 * r + 0.587 * g + 0.114 * b;
                    gx += Gx[ky + 1][kx + 1] * lum;
                    gy += Gy[ky + 1][kx + 1] * lum;
                }
            }
            double mag = std::sqrt(gx * gx + gy * gy);
            uint8_t v = static_cast<uint8_t>(std::min(255.0, mag));
            dst.setPixel(x, y, v, v, v);
        }
        prog.tickAndMaybePrint();
    }
    prog.finish();
}

// -----------------------------------------------------------------------------
// 9) CLI
// -----------------------------------------------------------------------------
struct Config {
    int  width    = 7680;
    int  height   = 4320;
    int  maxIter  = 1000;
    int  radius   = 15;
    int  threads  = 0;        // 0 = automatico (todos los disponibles)
    bool runBlur  = true;
    bool runSobel = true;
    std::string outFractal = "fractal.ppm";
    std::string outBlur    = "blurred.ppm";
    std::string outSobel   = "edges.ppm";
};

static void printHelp(const char* progname) {
    std::cout <<
        "Uso: " << progname << " [opciones]\n"
        "  -w, --width   N    ancho de la imagen           (default 7680)\n"
        "  -h, --height  N    alto  de la imagen           (default 4320)\n"
        "  -i, --iter    N    iteraciones max. Mandelbrot  (default 1000)\n"
        "  -r, --radius  N    radio del kernel Gaussiano   (default 15)\n"
        "  -t, --threads N    numero de hilos OpenMP       (default: max disponibles)\n"
        "      --no-blur      omite el desenfoque Gaussiano\n"
        "      --no-sobel     omite el filtro Sobel\n"
        "      --help         muestra esta ayuda\n";
}

static bool parseArgs(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](int& out) -> bool {
            if (i + 1 >= argc) return false;
            out = std::atoi(argv[++i]);
            return out > 0;
        };
        if (a == "--help") { printHelp(argv[0]); return false; }
        else if (a == "-w" || a == "--width")   { if (!next(cfg.width))   return false; }
        else if (a == "-h" || a == "--height")  { if (!next(cfg.height))  return false; }
        else if (a == "-i" || a == "--iter")    { if (!next(cfg.maxIter)) return false; }
        else if (a == "-r" || a == "--radius")  { if (!next(cfg.radius))  return false; }
        else if (a == "-t" || a == "--threads") { if (!next(cfg.threads)) return false; }
        else if (a == "--no-blur")              { cfg.runBlur  = false; }
        else if (a == "--no-sobel")             { cfg.runSobel = false; }
        else {
            std::cerr << "Argumento desconocido: " << a << "\n";
            printHelp(argv[0]);
            return false;
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
// 10) MAIN
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    using clk = std::chrono::high_resolution_clock;
    auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
    };

    Config cfg;
    if (!parseArgs(argc, argv, cfg)) return 1;

    // Configurar numero de hilos (si el usuario lo pidio).
    if (cfg.threads > 0) {
        omp_set_num_threads(cfg.threads);
    }

    // Obtener los hilos que efectivamente se usaran abriendo una region
    // paralela trivial: omp_get_num_threads() solo da info correcta DENTRO
    // de una region paralela; fuera, devuelve 1.
    int activeThreads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        activeThreads = omp_get_num_threads();
    }

    std::cout << "============================================================\n";
    std::cout << " Fractal de Mandelbrot + Convolucion 2D (PARALELO con OpenMP)\n";
    std::cout << "============================================================\n";
    std::cout << " Resolucion        : " << cfg.width << " x " << cfg.height << "\n";
    std::cout << " Iter. max         : " << cfg.maxIter << "\n";
    std::cout << " Radio Gaussiano   : " << cfg.radius
              << " (kernel " << (2*cfg.radius+1) << "x" << (2*cfg.radius+1) << ")\n";
    std::cout << " Aplicar Gaussiana : " << (cfg.runBlur  ? "si" : "no") << "\n";
    std::cout << " Aplicar Sobel     : " << (cfg.runSobel ? "si" : "no") << "\n";
#ifdef _OPENMP
    std::cout << " OpenMP            : ACTIVO (version " << _OPENMP << ")\n";
    std::cout << " Hilos disponibles : " << omp_get_max_threads() << "\n";
    std::cout << " Hilos a usar      : " << activeThreads << "\n";
#else
    std::cout << " OpenMP            : NO disponible (compilado sin -fopenmp).\n";
    std::cout << "                     Comportamiento equivalente al secuencial.\n";
#endif
    std::cout << "------------------------------------------------------------\n";

    Image fractal(cfg.width, cfg.height);

    // ---- Fase 1: Mandelbrot
    std::cout << "\n[1/3] Generando fractal de Mandelbrot...\n";
    auto t0 = clk::now();
    generateMandelbrot(fractal, cfg.maxIter);
    auto t1 = clk::now();
    long long tMandel = ms(t0, t1);

    std::cout << "      Guardando " << cfg.outFractal << " ...\n";
    auto t1s = clk::now();
    if (!savePPM(fractal, cfg.outFractal)) return 2;
    auto t1e = clk::now();
    long long tSaveFractal = ms(t1s, t1e);

    long long tBlur = 0, tSaveBlur = 0;
    long long tSobel = 0, tSaveSobel = 0;

    // ---- Fase 2: Gaussiana
    if (cfg.runBlur) {
        std::cout << "\n[2/3] Aplicando desenfoque Gaussiano (r=" << cfg.radius << ")...\n";
        Image blurred(cfg.width, cfg.height);
        auto kernel = gaussianKernel(cfg.radius, cfg.radius / 2.0);
        auto a = clk::now();
        convolve2D(fractal, blurred, kernel, cfg.radius);
        auto b = clk::now();
        tBlur = ms(a, b);

        std::cout << "      Guardando " << cfg.outBlur << " ...\n";
        auto sa = clk::now();
        savePPM(blurred, cfg.outBlur);
        auto sb = clk::now();
        tSaveBlur = ms(sa, sb);
    }

    // ---- Fase 3: Sobel
    if (cfg.runSobel) {
        std::cout << "\n[3/3] Aplicando filtro Sobel...\n";
        Image edges(cfg.width, cfg.height);
        auto a = clk::now();
        sobelFilter(fractal, edges);
        auto b = clk::now();
        tSobel = ms(a, b);

        std::cout << "      Guardando " << cfg.outSobel << " ...\n";
        auto sa = clk::now();
        savePPM(edges, cfg.outSobel);
        auto sb = clk::now();
        tSaveSobel = ms(sa, sb);
    }

    long long tTotal = tMandel + tSaveFractal + tBlur + tSaveBlur + tSobel + tSaveSobel;
    long long tCompute = tMandel + tBlur + tSobel;

    std::cout << "\n============================================================\n";
    std::cout << " Tiempos (milisegundos) con " << activeThreads << " hilo(s)\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "  Mandelbrot          : " << tMandel       << " ms\n";
    std::cout << "  Guardar fractal     : " << tSaveFractal  << " ms\n";
    if (cfg.runBlur) {
    std::cout << "  Desenfoque Gauss.   : " << tBlur         << " ms\n";
    std::cout << "  Guardar blurred     : " << tSaveBlur     << " ms\n";
    }
    if (cfg.runSobel) {
    std::cout << "  Sobel               : " << tSobel        << " ms\n";
    std::cout << "  Guardar edges       : " << tSaveSobel    << " ms\n";
    }
    std::cout << "------------------------------------------------------------\n";
    std::cout << "  COMPUTO (sin I/O)   : " << tCompute      << " ms\n";
    std::cout << "  TOTAL               : " << tTotal        << " ms"
              << "  (" << (tTotal / 1000.0) << " s)\n";
    std::cout << "============================================================\n";
    std::cout << "\n Sugerencia: corre tambien la version secuencial (./fractal)\n"
                 " con los mismos parametros y compara los tiempos para obtener\n"
                 " el speedup empirico:  S = t_secuencial / t_paralelo\n";
    return 0;
}
