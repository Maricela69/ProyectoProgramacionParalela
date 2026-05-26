// =============================================================================
//  fractal_proc — Generación secuencial de un fractal de Mandelbrot (hasta 8K)
//  y procesamiento mediante convolución 2D (Gaussiana de radio amplio + Sobel)
// -----------------------------------------------------------------------------
//  Cumple los requisitos del enunciado:
//    * 100% secuencial (sin hilos, sin OpenMP, sin MPI, sin CUDA).
//    * Solo C++17 estándar + cabeceras estándar (sin librerías externas).
//    * Guarda imágenes en formato PPM binario (P6), sin dependencias.
//    * Mide tiempos con std::chrono.
//    * Compila en Linux y Windows con g++/clang++/MSVC.
//
//  Compilación recomendada:
//      Linux  : g++ -O3 -std=c++17 -march=native main.cpp -o fractal
//      Windows: g++ -O3 -std=c++17 -march=native main.cpp -o fractal.exe
//                 (con MinGW-w64; con MSVC: cl /O2 /std:c++17 /EHsc main.cpp)
//
//  Uso:
//      ./fractal                       (por defecto: 7680x4320, iter=1000, r=15)
//      ./fractal -w 1920 -h 1080 -i 500 -r 7
//      ./fractal --help
// =============================================================================

#include <algorithm>
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

// -----------------------------------------------------------------------------
// 1) ESTRUCTURA DE IMAGEN
// -----------------------------------------------------------------------------
// Almacenamos los píxeles como un vector contiguo de bytes RGB intercalados
// (row-major). Esto favorece el acceso secuencial y la localidad de caché.
// Para una imagen 7680x4320, el buffer ocupa ~95 MiB (3 bytes por píxel).
// -----------------------------------------------------------------------------
struct Image {
    int width  = 0;
    int height = 0;
    std::vector<uint8_t> data;   // tamaño = width * height * 3

    Image() = default;
    Image(int w, int h) : width(w), height(h), data(static_cast<size_t>(w) * h * 3, 0) {}

    // Acceso rápido inlined. No comprobamos límites en el hot-path por rendimiento.
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
// 2) GUARDADO EN PPM BINARIO (P6)
// -----------------------------------------------------------------------------
// PPM es el formato más simple posible: una cabecera ASCII + bytes RGB en crudo.
// No requiere ninguna librería. Se puede convertir a PNG con ImageMagick:
//     convert imagen.ppm imagen.png
// o visualizarse con GIMP, IrfanView, feh, etc.
// -----------------------------------------------------------------------------
static bool savePPM(const Image& img, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "ERROR: no se pudo abrir " << filename << " para escritura\n";
        return false;
    }
    out << "P6\n" << img.width << " " << img.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(img.data.data()),
              static_cast<std::streamsize>(img.data.size()));
    return static_cast<bool>(out);
}

// -----------------------------------------------------------------------------
// 3) NÚCLEO DEL FRACTAL DE MANDELBROT
// -----------------------------------------------------------------------------
// El conjunto de Mandelbrot M está definido como el conjunto de números
// complejos c para los cuales la iteración
//
//            z_{n+1} = z_n^2 + c,   z_0 = 0
//
// permanece acotada. Se demuestra que si en algún momento |z_n| > 2, la
// sucesión diverge a infinito. Por tanto el algoritmo de "escape time" es:
//
//   1. Para cada punto c del plano complejo (un píxel), iniciamos z = 0.
//   2. Iteramos hasta MAX_ITER veces calculando z = z^2 + c.
//   3. Si |z|^2 supera 4, declaramos que c NO pertenece al conjunto y
//      registramos en qué iteración escapó (eso da el color).
//   4. Si nunca escapa en MAX_ITER iteraciones, lo pintamos negro
//      (asumimos que pertenece al conjunto).
//
// OPTIMIZACIÓN: en lugar de calcular módulos con sqrt(), comparamos
// zr^2 + zi^2 > 4. Además mantenemos zr2 = zr*zr y zi2 = zi*zi para no
// repetir multiplicaciones (cada iteración hace solo 3 multiplicaciones
// y unas sumas).
//
// COMPLEJIDAD POR PÍXEL: O(k), donde k es el número de iteraciones
// realmente ejecutadas (entre 1 y MAX_ITER). Los píxeles del interior
// del conjunto son los más caros (siempre llegan a MAX_ITER), por lo
// que la carga es muy desigual entre regiones de la imagen: este es
// uno de los motivos por los que paralelizar requiere un balanceo
// dinámico (más sobre esto en el README).
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

// -----------------------------------------------------------------------------
// 4) MAPEO DE ITERACIONES A COLOR (paleta tipo "Bernstein")
// -----------------------------------------------------------------------------
// Para que la imagen luzca bien, los píxeles del exterior del conjunto se
// colorean en función de cuán rápido escapan. Usamos un polinomio de
// Bernstein de bajo grado: produce un degradado suave y multicolor.
// -----------------------------------------------------------------------------
static inline void iterToColor(int iter, int maxIter,
                               uint8_t& r, uint8_t& g, uint8_t& b) {
    if (iter >= maxIter) {
        // Punto que (probablemente) pertenece al conjunto -> negro
        r = g = b = 0;
        return;
    }
    // t en [0,1)
    double t = static_cast<double>(iter) / static_cast<double>(maxIter);
    // Polinomios de Bernstein: dan un tránsito suave entre azul, magenta,
    // amarillo y blanco; son los típicos colores de muchas visualizaciones
    // clásicas del fractal.
    double R = 9.0  * (1 - t) * t * t * t;
    double G = 15.0 * (1 - t) * (1 - t) * t * t;
    double B = 8.5  * (1 - t) * (1 - t) * (1 - t) * t;
    r = static_cast<uint8_t>(std::min(255.0, R * 255.0));
    g = static_cast<uint8_t>(std::min(255.0, G * 255.0));
    b = static_cast<uint8_t>(std::min(255.0, B * 255.0));
}

// -----------------------------------------------------------------------------
// 5) GENERACIÓN DE LA IMAGEN DEL FRACTAL
// -----------------------------------------------------------------------------
// Mapeo de coordenadas del píxel (x,y) al plano complejo. Centramos la
// vista clásica del conjunto (-0.5, 0) y elegimos un ancho de 3.5 unidades.
// El alto se deriva conservando la relación de aspecto de la imagen para
// no deformar el fractal.
//
// El bucle es estructuralmente trivial: dos for anidados, sin dependencias
// entre píxeles -> "embarrassingly parallel".
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

    int lastPct = -1;
    for (int y = 0; y < img.height; ++y) {
        const double ci = yMin + y * dy;
        for (int x = 0; x < img.width; ++x) {
            const double cr = xMin + x * dx;
            int iter = mandelbrotIterations(cr, ci, maxIter);
            uint8_t r, g, b;
            iterToColor(iter, maxIter, r, g, b);
            img.setPixel(x, y, r, g, b);
        }
        // Progreso (no afecta al rendimiento global)
        int pct = (100 * (y + 1)) / img.height;
        if (pct != lastPct) {
            std::cout << "  Mandelbrot: " << pct << "%\r" << std::flush;
            lastPct = pct;
        }
    }
    std::cout << "  Mandelbrot: 100%   \n";
}

// -----------------------------------------------------------------------------
// 6) CONSTRUCCIÓN DEL KERNEL GAUSSIANO 2D
// -----------------------------------------------------------------------------
// Un kernel Gaussiano 2D se define como
//
//        G(x,y) = (1 / (2*pi*sigma^2)) * exp( -(x^2 + y^2) / (2*sigma^2) )
//
// Lo discretizamos en una matriz cuadrada de tamaño (2r+1)x(2r+1) y luego
// normalizamos para que la suma valga 1 (así el desenfoque preserva la
// luminancia media). Para r=15, sigma ~= r/2 = 7.5 produce un desenfoque
// muy notable.
//
// NOTA TEÓRICA: el kernel Gaussiano es SEPARABLE en producto de dos kernels
// 1D, lo que reduce el costo de O((2r+1)^2) a O(2(2r+1)) por píxel. Aquí
// hacemos la versión NO separable (matriz completa) PORQUE EL ENUNCIADO
// PIDE UN FILTRO PESADO, y porque queremos que el resultado se beneficie
// mucho de la paralelización (ver README). En "Posibles optimizaciones"
// explicamos la versión separable.
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
    // Normalización: dividimos por la suma para que el filtro sea de media 1.
    for (auto& v : kernel) v /= sum;
    return kernel;
}

// -----------------------------------------------------------------------------
// 7) CONVOLUCIÓN 2D COMPLETA (NO SEPARABLE)
// -----------------------------------------------------------------------------
// Definición matemática:
//
//   O(x,y) = sum_{j=-r}^{r} sum_{i=-r}^{r}  K(i,j) * I(x+i, y+j)
//
// donde K es el kernel y r su radio. Manejamos los bordes con la estrategia
// "clamp-to-edge" (replicación del borde), que es la más simple y suficiente
// para nuestro caso de uso.
//
// COMPLEJIDAD: para una imagen de N píxeles y un kernel de tamaño k=(2r+1)^2,
// el costo es O(N * k * C), con C = 3 canales. Para 7680x4320 y r=15,
// k = 31*31 = 961, lo que da aprox. 33.2M * 961 * 3 ≈ 9.6 * 10^10 operaciones
// multiplicar-acumular: ¡un trabajo enorme para un solo hilo!
//
// CUELLOS DE BOTELLA:
//   - Densidad aritmética alta (muchos productos por píxel).
//   - Acceso a memoria con patrón "ventana 2D" -> ineficiente respecto a la
//     línea de caché si la imagen no cabe completa en L2/L3.
//   - Saltos en y (rows) saltan ancho*3 bytes => fallos de caché si la
//     ventana es muy grande.
// -----------------------------------------------------------------------------
static void convolve2D(const Image& src, Image& dst,
                       const std::vector<double>& kernel, int radius) {
    const int size = 2 * radius + 1;
    int lastPct = -1;
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            double accR = 0.0, accG = 0.0, accB = 0.0;
            for (int ky = -radius; ky <= radius; ++ky) {
                int sy = y + ky;
                // clamp-to-edge
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
                if (v < 0.0) return 0;
                if (v > 255.0) return 255;
                return static_cast<uint8_t>(v);
            };
            dst.setPixel(x, y, clamp255(accR), clamp255(accG), clamp255(accB));
        }
        int pct = (100 * (y + 1)) / src.height;
        if (pct != lastPct) {
            std::cout << "  Gaussiana: " << pct << "%\r" << std::flush;
            lastPct = pct;
        }
    }
    std::cout << "  Gaussiana: 100%   \n";
}

// -----------------------------------------------------------------------------
// 8) FILTRO SOBEL (DETECCIÓN DE BORDES)
// -----------------------------------------------------------------------------
// Sobel aplica dos kernels 3x3, Gx y Gy, que aproximan las derivadas
// parciales de la imagen (en luminancia) respecto a x e y:
//
//          [-1  0  1]                [-1 -2 -1]
//   Gx =   [-2  0  2]      Gy =      [ 0  0  0]
//          [-1  0  1]                [ 1  2  1]
//
// Para cada píxel:
//   1) Convertimos a luminancia con los coeficientes BT.601:
//          L = 0.299*R + 0.587*G + 0.114*B
//   2) Calculamos las dos respuestas: gx = Gx ⊛ L, gy = Gy ⊛ L
//   3) La magnitud del gradiente es G = sqrt(gx^2 + gy^2)
//
// COMPLEJIDAD: O(N * 9), mucho más barato que la Gaussiana de radio amplio.
// Pero el patrón de cómputo es esencialmente el MISMO: convolución 2D, y por
// tanto se paraleliza con las mismas técnicas.
// -----------------------------------------------------------------------------
static void sobelFilter(const Image& src, Image& dst) {
    static const int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    static const int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    int lastPct = -1;
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
        int pct = (100 * (y + 1)) / src.height;
        if (pct != lastPct) {
            std::cout << "  Sobel: " << pct << "%\r" << std::flush;
            lastPct = pct;
        }
    }
    std::cout << "  Sobel: 100%   \n";
}

// -----------------------------------------------------------------------------
// 9) UTILIDADES: PARSEO DE ARGUMENTOS DE LÍNEA DE COMANDOS
// -----------------------------------------------------------------------------
struct Config {
    int  width    = 7680;     // ancho por defecto: 8K UHD
    int  height   = 4320;     // alto por defecto:  8K UHD
    int  maxIter  = 1000;     // iteraciones máximas Mandelbrot
    int  radius   = 15;       // radio del kernel Gaussiano (31x31)
    bool runBlur  = true;
    bool runSobel = true;
    std::string outFractal = "fractal.ppm";
    std::string outBlur    = "blurred.ppm";
    std::string outSobel   = "edges.ppm";
};

static void printHelp(const char* progname) {
    std::cout <<
        "Uso: " << progname << " [opciones]\n"
        "  -w, --width  N     ancho de la imagen (default 7680)\n"
        "  -h, --height N     alto  de la imagen (default 4320)\n"
        "  -i, --iter   N     iteraciones máximas Mandelbrot (default 1000)\n"
        "  -r, --radius N     radio del kernel Gaussiano (default 15)\n"
        "      --no-blur      omite el desenfoque Gaussiano\n"
        "      --no-sobel     omite el filtro Sobel\n"
        "      --help         muestra esta ayuda\n"
        "\nEjemplos:\n"
        "  " << progname << "                          # 8K completo (lento)\n"
        "  " << progname << " -w 1920 -h 1080          # prueba rápida en Full HD\n"
        "  " << progname << " -w 3840 -h 2160 -r 10    # 4K, blur radio 10\n";
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
        else if (a == "-w" || a == "--width")  { if (!next(cfg.width))   return false; }
        else if (a == "-h" || a == "--height") { if (!next(cfg.height))  return false; }
        else if (a == "-i" || a == "--iter")   { if (!next(cfg.maxIter)) return false; }
        else if (a == "-r" || a == "--radius") { if (!next(cfg.radius))  return false; }
        else if (a == "--no-blur")             { cfg.runBlur  = false; }
        else if (a == "--no-sobel")            { cfg.runSobel = false; }
        else {
            std::cerr << "Argumento desconocido: " << a << "\n";
            printHelp(argv[0]);
            return false;
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
// 10) MAIN: ORQUESTACIÓN + MEDICIÓN DE TIEMPOS CON std::chrono
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    using clk = std::chrono::high_resolution_clock;
    auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
    };

    Config cfg;
    if (!parseArgs(argc, argv, cfg)) return 1;

    std::cout << "============================================================\n";
    std::cout << " Fractal de Mandelbrot + Convolucion 2D (SECUENCIAL)\n";
    std::cout << "============================================================\n";
    std::cout << " Resolucion        : " << cfg.width << " x " << cfg.height << "\n";
    std::cout << " Iter. max         : " << cfg.maxIter << "\n";
    std::cout << " Radio Gaussiano   : " << cfg.radius
              << " (kernel " << (2*cfg.radius+1) << "x" << (2*cfg.radius+1) << ")\n";
    std::cout << " Aplicar Gaussiana : " << (cfg.runBlur  ? "si" : "no") << "\n";
    std::cout << " Aplicar Sobel     : " << (cfg.runSobel ? "si" : "no") << "\n";
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

    // ---- Fase 2: Desenfoque Gaussiano
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

    // ---- Resumen
    std::cout << "\n============================================================\n";
    std::cout << " Tiempos (milisegundos)\n";
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
    std::cout << "  TOTAL               : " << tTotal        << " ms"
              << "  (" << (tTotal / 1000.0) << " s)\n";
    std::cout << "============================================================\n";
    return 0;
}
