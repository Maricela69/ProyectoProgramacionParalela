// =============================================================================
//  main_omp_simd.cpp — Version VECTORIZADA (SPMD) de main_omp.cpp
// -----------------------------------------------------------------------------
//  Objetivo 2 reescrito para FORZAR la vectorizacion del bucle mas interno
//  de los filtros de convolucion usando estructuras SPMD de OpenMP:
//
//      #pragma omp simd          (cada lane SIMD = un pixel de salida)
//
//  combinado con el paralelismo de hilos ya existente:
//
//      #pragma omp parallel for  (cada hilo = un bloque de filas)
//
//  -> Modelo hibrido: hilos en el eje Y, lanes SIMD en el eje X.
//
// -----------------------------------------------------------------------------
//  POR QUE EL CODIGO ORIGINAL NO VECTORIZABA BIEN
//  -----------------------------------------------
//  El convolve2D original tenia 3 obstaculos en el bucle interno:
//    1) clamp-to-edge con ramas  (if sx<0 ... else if ...) -> control de flujo
//       dependiente de datos, que impide al vectorizador.
//    2) acceso RGB intercalado (stride 3) -> cargas tipo "gather", no contiguas.
//    3) reduccion sobre la ventana con direccionamiento irregular.
//
//  SOLUCION (estructura SPMD-friendly):
//    A) Separar canales en planos contiguos (planar R, G, B) -> cargas
//       vectoriales unitarias (stride 1).
//    B) Pre-rellenar la imagen con un borde de tamano 'radius' replicando el
//       borde (padding). Asi el bucle interno NO tiene ninguna rama de limites.
//    C) Reordenar el bucle: para cada peso del kernel (j,i), acumular sobre
//       TODA la fila de salida:  acc[x] += w * P[(y+j)*PW + (x+i)].
//       El acceso P[... + x + i] es CONTIGUO en x -> AXPY perfecta para SIMD.
//
//  Resultado: el bucle interno es un multiply-add contiguo sin ramas, que el
//  compilador convierte en instrucciones FMA de AVX2/AVX-512.
//
// -----------------------------------------------------------------------------
//  COMPILACION (con verificacion de vectorizacion) — ver SIMD.md para detalle.
//
//    GCC  (recomendado):
//      g++ -O3 -fopenmp -march=native -std=c++17 \
//          -fopt-info-vec-optimized=vec.txt main_omp_simd.cpp -o fractal_simd
//      # luego:  grep "convolve\|sobel\|simd" vec.txt   (o revisar vec.txt)
//
//    Clang:
//      clang++ -O3 -fopenmp -march=native -std=c++17 \
//          -Rpass=loop-vectorize -Rpass-missed=loop-vectorize \
//          main_omp_simd.cpp -o fractal_simd 2> vec.txt
//
//    MSVC:
//      cl /O2 /openmp:experimental /arch:AVX2 /Qvec-report:2 /std:c++17 main_omp_simd.cpp
//
//  BANDERAS CLAVE:
//    -O3            : habilita el auto-vectorizador (imprescindible).
//    -march=native  : permite usar AVX2/AVX-512/FMA de TU CPU.
//    -fopenmp       : activa #pragma omp parallel for Y #pragma omp simd.
//    (-fopenmp-simd : si solo quieres los pragmas simd sin el runtime de hilos)
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

#ifdef _OPENMP
  #include <omp.h>
#else
  inline int  omp_get_max_threads() { return 1; }
  inline int  omp_get_num_threads() { return 1; }
  inline int  omp_get_thread_num()  { return 0; }
  inline void omp_set_num_threads(int) {}
#endif

// -----------------------------------------------------------------------------
// 1) IMAGEN (RGB intercalado, igual que antes — es el formato de E/S)
// -----------------------------------------------------------------------------
struct Image {
    int width = 0, height = 0;
    std::vector<uint8_t> data;
    Image() = default;
    Image(int w, int h) : width(w), height(h),
                          data(static_cast<size_t>(w) * h * 3, 0) {}
    inline void setPixel(int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        size_t idx = (static_cast<size_t>(y) * width + x) * 3;
        data[idx + 0] = r; data[idx + 1] = g; data[idx + 2] = b;
    }
    inline void getPixel(int x, int y, uint8_t& r, uint8_t& g, uint8_t& b) const {
        size_t idx = (static_cast<size_t>(y) * width + x) * 3;
        r = data[idx + 0]; g = data[idx + 1]; b = data[idx + 2];
    }
};

static bool savePPM(const Image& img, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) { std::cerr << "ERROR abriendo " << filename << "\n"; return false; }
    out << "P6\n" << img.width << " " << img.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(img.data.data()),
              static_cast<std::streamsize>(img.data.size()));
    return static_cast<bool>(out);
}

// -----------------------------------------------------------------------------
// 2) MANDELBROT (sin cambios respecto a main_omp.cpp)
// -----------------------------------------------------------------------------
static inline int mandelbrotIterations(double cr, double ci, int maxIter) {
    double zr = 0.0, zi = 0.0, zr2 = 0.0, zi2 = 0.0;
    int iter = 0;
    while (iter < maxIter && (zr2 + zi2) <= 4.0) {
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
        zr2 = zr * zr; zi2 = zi * zi;
        ++iter;
    }
    return iter;
}
static inline void iterToColor(int iter, int maxIter,
                               uint8_t& r, uint8_t& g, uint8_t& b) {
    if (iter >= maxIter) { r = g = b = 0; return; }
    double t = static_cast<double>(iter) / maxIter;
    double R = 9.0  * (1 - t) * t * t * t;
    double G = 15.0 * (1 - t) * (1 - t) * t * t;
    double B = 8.5  * (1 - t) * (1 - t) * (1 - t) * t;
    r = static_cast<uint8_t>(std::min(255.0, R * 255.0));
    g = static_cast<uint8_t>(std::min(255.0, G * 255.0));
    b = static_cast<uint8_t>(std::min(255.0, B * 255.0));
}

static void generateMandelbrot(Image& img, int maxIter) {
    const double aspect = static_cast<double>(img.width) / img.height;
    const double xRange = 3.5, xCenter = -0.5;
    const double yRange = xRange / aspect;
    const double xMin = xCenter - xRange * 0.5;
    const double yMin = -yRange * 0.5;
    const double dx = xRange / img.width;
    const double dy = yRange / img.height;

    #pragma omp parallel for schedule(dynamic, 16) default(none) \
            shared(img) firstprivate(maxIter, xMin, yMin, dx, dy)
    for (int y = 0; y < img.height; ++y) {
        const double ci = yMin + y * dy;
        for (int x = 0; x < img.width; ++x) {
            int iter = mandelbrotIterations(xMin + x * dx, ci, maxIter);
            uint8_t r, g, b;
            iterToColor(iter, maxIter, r, g, b);
            img.setPixel(x, y, r, g, b);
        }
    }
}

// -----------------------------------------------------------------------------
// 3) PLANOS PLANARES CON PADDING (la clave de la vectorizacion)
// -----------------------------------------------------------------------------
// Convierte la imagen RGB intercalada en 3 planos float contiguos (R, G, B),
// cada uno rodeado por un borde de 'radius' pixeles replicando el borde
// (clamp-to-edge). Asi el bucle de convolucion NO necesita comprobar limites.
//
//   Dimensiones del plano con padding:
//     PW = width  + 2*radius
//     PH = height + 2*radius
//   El pixel de imagen (x,y) vive en el plano en (x + radius, y + radius).
// -----------------------------------------------------------------------------
struct PaddedPlanesF {
    int W, H, r, PW, PH;
    std::vector<float> R, G, B;

    PaddedPlanesF(const Image& img, int radius)
        : W(img.width), H(img.height), r(radius),
          PW(img.width + 2 * radius), PH(img.height + 2 * radius),
          R(static_cast<size_t>(PW) * PH),
          G(static_cast<size_t>(PW) * PH),
          B(static_cast<size_t>(PW) * PH)
    {
        // Rellenar TODOS los pixeles del plano (incluido el borde) con el
        // valor de imagen mas cercano (clamp). El clamp se hace AQUI, una sola
        // vez, fuera del hot-loop.
        #pragma omp parallel for schedule(static) default(none) shared(img)
        for (int py = 0; py < PH; ++py) {
            int sy = py - r;
            if (sy < 0) sy = 0; else if (sy >= H) sy = H - 1;
            for (int px = 0; px < PW; ++px) {
                int sx = px - r;
                if (sx < 0) sx = 0; else if (sx >= W) sx = W - 1;
                uint8_t cr, cg, cb;
                img.getPixel(sx, sy, cr, cg, cb);
                size_t idx = static_cast<size_t>(py) * PW + px;
                R[idx] = cr; G[idx] = cg; B[idx] = cb;
            }
        }
    }
};

// -----------------------------------------------------------------------------
// 4) KERNEL GAUSSIANO (float, para coincidir con los planos)
// -----------------------------------------------------------------------------
static std::vector<float> gaussianKernelF(int radius, double sigma) {
    const int size = 2 * radius + 1;
    std::vector<float> kernel(static_cast<size_t>(size) * size);
    double sum = 0.0;
    const double s2 = 2.0 * sigma * sigma;
    for (int j = -radius; j <= radius; ++j)
        for (int i = -radius; i <= radius; ++i) {
            double v = std::exp(-(i * i + j * j) / s2);
            kernel[(j + radius) * size + (i + radius)] = static_cast<float>(v);
            sum += v;
        }
    for (auto& v : kernel) v = static_cast<float>(v / sum);
    return kernel;
}

// -----------------------------------------------------------------------------
// 5) CONVOLUCION 2D VECTORIZADA (SPMD)
// -----------------------------------------------------------------------------
// Estructura de bucles (de fuera a dentro):
//   parallel for (y)        -> hilos OpenMP, una banda de filas por hilo
//     for (j) for (i)       -> recorrido de los (2r+1)^2 pesos del kernel
//       omp simd (x)        -> LANES SIMD: acc[x] += w * P[(y+j)*PW + x+i]
//
// El bucle 'x' es el que se vectoriza:
//   * acc[x] es contiguo (stride 1).
//   * srcRow[x + i] es contiguo en x (i es constante dentro del simd).
//   * No hay ramas: el padding garantiza que x+i siempre esta en rango.
//   => El compilador emite FMA vectoriales (vfmadd...ps en AVX).
//
// Los acumuladores accR/accG/accB son por-hilo (declarados dentro de la region
// parallel), evitando false sharing y reasignaciones por fila.
// -----------------------------------------------------------------------------
static void convolve2D_simd(const PaddedPlanesF& P, Image& dst,
                            const std::vector<float>& kernel) {
    const int W = P.W, H = P.H, r = P.r, PW = P.PW;
    const int ksize = 2 * r + 1;

    #pragma omp parallel default(none) shared(P, dst, kernel) \
            firstprivate(W, H, r, PW, ksize)
    {
        // Acumuladores por hilo (una sola asignacion por hilo).
        std::vector<float> accR(W), accG(W), accB(W);

        #pragma omp for schedule(static)
        for (int y = 0; y < H; ++y) {
            // Reiniciar acumuladores de esta fila.
            std::fill(accR.begin(), accR.end(), 0.0f);
            std::fill(accG.begin(), accG.end(), 0.0f);
            std::fill(accB.begin(), accB.end(), 0.0f);

            // Recorrer los pesos del kernel.
            for (int j = 0; j < ksize; ++j) {
                const float* rowR = &P.R[static_cast<size_t>(y + j) * PW];
                const float* rowG = &P.G[static_cast<size_t>(y + j) * PW];
                const float* rowB = &P.B[static_cast<size_t>(y + j) * PW];
                const float* krow = &kernel[static_cast<size_t>(j) * ksize];
                for (int i = 0; i < ksize; ++i) {
                    const float w = krow[i];
                    const float* pr = rowR + i;   // P.R[(y+j)*PW + x + i] al indexar [x]
                    const float* pg = rowG + i;
                    const float* pb = rowB + i;
                    // ----- BUCLE VECTORIZADO (SPMD) -----
                    #pragma omp simd
                    for (int x = 0; x < W; ++x) {
                        accR[x] += w * pr[x];
                        accG[x] += w * pg[x];
                        accB[x] += w * pb[x];
                    }
                    // ------------------------------------
                }
            }

            // Volcar la fila a la imagen de salida con clamp a [0,255].
            for (int x = 0; x < W; ++x) {
                auto c = [](float v) -> uint8_t {
                    if (v < 0.0f) return 0;
                    if (v > 255.0f) return 255;
                    return static_cast<uint8_t>(v + 0.5f);
                };
                dst.setPixel(x, y, c(accR[x]), c(accG[x]), c(accB[x]));
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 6) SOBEL VECTORIZADO (SPMD)
// -----------------------------------------------------------------------------
// Se precomputa un plano de LUMINANCIA con padding=1 (float). Luego cada fila
// de salida se calcula con un bucle 'omp simd' sobre x, leyendo las 3 filas
// vecinas del plano. Todas las cargas son contiguas y sin ramas.
//
//   Gx = [-1 0 1; -2 0 2; -1 0 1]      Gy = [-1 -2 -1; 0 0 0; 1 2 1]
//   mag = sqrt(gx^2 + gy^2)            (la sqrt tambien se vectoriza)
// -----------------------------------------------------------------------------
static void sobel_simd(const Image& src, Image& dst) {
    const int W = src.width, H = src.height;
    const int PW = W + 2;            // padding = 1
    const int PH = H + 2;

    // Plano de luminancia con padding (clamp-to-edge), una sola vez.
    std::vector<float> L(static_cast<size_t>(PW) * PH);
    #pragma omp parallel for schedule(static) default(none) \
            shared(src, L) firstprivate(W, H, PW, PH)
    for (int py = 0; py < PH; ++py) {
        int sy = py - 1; if (sy < 0) sy = 0; else if (sy >= H) sy = H - 1;
        for (int px = 0; px < PW; ++px) {
            int sx = px - 1; if (sx < 0) sx = 0; else if (sx >= W) sx = W - 1;
            uint8_t r, g, b; src.getPixel(sx, sy, r, g, b);
            L[static_cast<size_t>(py) * PW + px] =
                0.299f * r + 0.587f * g + 0.114f * b;
        }
    }

    #pragma omp parallel for schedule(static) default(none) \
            shared(L, dst) firstprivate(W, H, PW)
    for (int y = 0; y < H; ++y) {
        const float* r0 = &L[static_cast<size_t>(y + 0) * PW];  // fila superior
        const float* r1 = &L[static_cast<size_t>(y + 1) * PW];  // fila central
        const float* r2 = &L[static_cast<size_t>(y + 2) * PW];  // fila inferior
        // ----- BUCLE VECTORIZADO (SPMD) -----
        #pragma omp simd
        for (int x = 0; x < W; ++x) {
            // ventana 3x3: columnas x, x+1, x+2 del plano con padding
            float gx = -r0[x] + r0[x + 2]
                     - 2.0f * r1[x] + 2.0f * r1[x + 2]
                     - r2[x] + r2[x + 2];
            float gy = -r0[x] - 2.0f * r0[x + 1] - r0[x + 2]
                     + r2[x] + 2.0f * r2[x + 1] + r2[x + 2];
            float mag = std::sqrt(gx * gx + gy * gy);
            if (mag > 255.0f) mag = 255.0f;
            uint8_t v = static_cast<uint8_t>(mag);
            dst.setPixel(x, y, v, v, v);
        }
        // ------------------------------------
    }
}

// -----------------------------------------------------------------------------
// 7) CLI (igual que main_omp.cpp + nada nuevo)
// -----------------------------------------------------------------------------
struct Config {
    int width = 7680, height = 4320, maxIter = 1000, radius = 15, threads = 0;
    bool runBlur = true, runSobel = true;
    std::string outFractal = "fractal.ppm", outBlur = "blurred.ppm", outSobel = "edges.ppm";
};

static void printHelp(const char* p) {
    std::cout <<
        "Uso: " << p << " [opciones]\n"
        "  -w N  ancho (7680)   -h N alto (4320)   -i N iter (1000)\n"
        "  -r N  radio Gauss (15)   -t N hilos (max)\n"
        "  --no-blur   --no-sobel   --help\n";
}
static bool parseArgs(int argc, char** argv, Config& c) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&](int& o){ if (i+1>=argc) return false; o = std::atoi(argv[++i]); return o>0; };
        if      (a=="--help"){ printHelp(argv[0]); return false; }
        else if (a=="-w"){ if(!nx(c.width))return false; }
        else if (a=="-h"){ if(!nx(c.height))return false; }
        else if (a=="-i"){ if(!nx(c.maxIter))return false; }
        else if (a=="-r"){ if(!nx(c.radius))return false; }
        else if (a=="-t"){ if(!nx(c.threads))return false; }
        else if (a=="--no-blur"){ c.runBlur=false; }
        else if (a=="--no-sobel"){ c.runSobel=false; }
        else { std::cerr<<"Arg desconocido: "<<a<<"\n"; printHelp(argv[0]); return false; }
    }
    return true;
}

// -----------------------------------------------------------------------------
// 8) MAIN
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    using clk = std::chrono::high_resolution_clock;
    auto ms = [](auto a, auto b){ return std::chrono::duration_cast<std::chrono::milliseconds>(b-a).count(); };

    Config cfg;
    if (!parseArgs(argc, argv, cfg)) return 1;
    if (cfg.threads > 0) omp_set_num_threads(cfg.threads);

    int activeThreads = 1;
    #pragma omp parallel
    { 
        #pragma omp single
        activeThreads = omp_get_num_threads(); 
    }

    std::cout << "============================================================\n";
    std::cout << " Mandelbrot + Convolucion 2D (OpenMP + SIMD / SPMD)\n";
    std::cout << "============================================================\n";
    std::cout << " Resolucion      : " << cfg.width << " x " << cfg.height << "\n";
    std::cout << " Radio Gaussiano : " << cfg.radius
              << " (kernel " << (2*cfg.radius+1) << "x" << (2*cfg.radius+1) << ")\n";
    std::cout << " Hilos a usar    : " << activeThreads << "\n";
#ifdef __AVX512F__
    std::cout << " SIMD compilado  : AVX-512 disponible\n";
#elif defined(__AVX2__)
    std::cout << " SIMD compilado  : AVX2 disponible\n";
#elif defined(__SSE2__)
    std::cout << " SIMD compilado  : SSE2 disponible\n";
#else
    std::cout << " SIMD compilado  : (sin macro SIMD; revisa -march)\n";
#endif
    std::cout << "------------------------------------------------------------\n";

    Image fractal(cfg.width, cfg.height);

    std::cout << "\n[1/3] Mandelbrot...\n";
    auto t0 = clk::now();
    generateMandelbrot(fractal, cfg.maxIter);
    auto t1 = clk::now();
    long long tMandel = ms(t0, t1);
    savePPM(fractal, cfg.outFractal);

    long long tBlur = 0, tSobel = 0, tPad = 0;

    if (cfg.runBlur) {
        std::cout << "[2/3] Gaussiana vectorizada (r=" << cfg.radius << ")...\n";
        Image blurred(cfg.width, cfg.height);
        auto kernel = gaussianKernelF(cfg.radius, cfg.radius / 2.0);
        auto pa = clk::now();
        PaddedPlanesF planes(fractal, cfg.radius);   // conversion planar + padding
        auto pb = clk::now();
        tPad = ms(pa, pb);
        auto a = clk::now();
        convolve2D_simd(planes, blurred, kernel);
        auto b = clk::now();
        tBlur = ms(a, b);
        savePPM(blurred, cfg.outBlur);
    }

    if (cfg.runSobel) {
        std::cout << "[3/3] Sobel vectorizado...\n";
        Image edges(cfg.width, cfg.height);
        auto a = clk::now();
        sobel_simd(fractal, edges);
        auto b = clk::now();
        tSobel = ms(a, b);
        savePPM(edges, cfg.outSobel);
    }

    std::cout << "\n============================================================\n";
    std::cout << " Tiempos (ms) con " << activeThreads << " hilo(s)\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "  Mandelbrot              : " << tMandel << " ms\n";
    if (cfg.runBlur) {
    std::cout << "  Conversion planar+pad   : " << tPad   << " ms\n";
    std::cout << "  Gaussiana (SIMD)        : " << tBlur  << " ms\n";
    }
    if (cfg.runSobel)
    std::cout << "  Sobel (SIMD)            : " << tSobel << " ms\n";
    std::cout << "============================================================\n";
    return 0;
}
