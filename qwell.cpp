#include <cmath>       // for std::sin, std::sqrt
#include <cstdint>     // for std::int32_t
#include <format>      // for std::format
#include <fstream>     // for std::ofstream
#include <functional>  // for std::function
#include <iostream>    // for std::cout, std::endl
#include <numbers>     // for std::numbers::pi
#include <omp.h>       // for #pragma omp parallel
#include <vector>      // for std::vector

// ---- Boost 関連 (Jacobi elliptic, ellint_1, Gauss–Legendre) ----
#include <boost/math/quadrature/gauss.hpp>
// #include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/jacobi_elliptic.hpp>

// ---- Eigen 関連 (generalized eigenvalue problem) ----
#include <Eigen/Dense>

//============================================================
// 物理定数やパラメータ (ここでは簡単のため \hbar=1, 質量=1)
//============================================================
static auto constexpr N           = 250;  // 基底数
static auto constexpr GAUSS_ORDER = 4 * N;
static auto constexpr HBAR        = 1.0;
static auto constexpr MASS        = 1.0;
static auto constexpr OUTPUTNUM   = 1000;

//============================================================
// Gauss–Legendre 積分 (Boost) を使った数値積分
//============================================================
double numeric_integral(std::function<double(double)> const & func, double x_min, double x_max)
{
    boost::math::quadrature::gauss<double, GAUSS_ORDER> integrator;
    return integrator.integrate(func, x_min, x_max);
}

//============================================================
// Boost: Jacobi elliptic sn, cn, dn をまとめて返す構造体
//============================================================
struct JacobiElliptic
{
    double sn;
    double cn;
    double dn;
};

//============================================================
// jacobi_elliptic(u, m):
//   Boost の jacobi_sn(u, k), jacobi_cn(u, k), jacobi_dn(u, k)
//   ただし k^2 = m => k = sqrt(m).
//============================================================
JacobiElliptic jacobi_elliptic(double u, double m)
{
    auto const     k = std::sqrt(m);
    JacobiElliptic res;
    res.sn = boost::math::jacobi_sn(k, u);
    res.cn = boost::math::jacobi_cn(k, u);
    res.dn = boost::math::jacobi_dn(k, u);
    return res;
}

//============================================================
// \omega_n = 2*(n+1)*K(m)
//   (n は 0-based,  n+1 が本来の "モード番号")
//============================================================
double omega_n(std::int32_t n, double m)
{
    auto const k    = std::sqrt(m);
    //auto const Kval = boost::math::ellint_1(k);  // 完全楕円積分 K(m)
    auto const Kval = std::comp_ellint_1(k);  // 完全楕円積分 K(m)
    // n=0 -> 2*1*K(m) = 2K(m)
    // n=1 -> 2*2*K(m) = 4K(m), など
    return 2.0 * static_cast<double>(n + 1) * Kval;
}

//============================================================
// phi_n_raw(x, n, m):
//   sn( \omega_n * x, m )    [正規化定数は掛けない]
//============================================================
double phi_n_raw(double x, std::int32_t n, double m)
{
    auto const w  = omega_n(n, m);
    auto const je = jacobi_elliptic(w * x, m);
    return je.sn;  // sn(...)
}

//============================================================
// 正規化係数 N_n を計算
//   N_n = 1 / sqrt( \int_0^1 sn^2(\omega_n x, m) dx )
//============================================================
double normalization_factor(std::int32_t n, double m)
{
    auto const integrand = [&](auto xx) {
        auto const val = phi_n_raw(xx, n, m);
        return val * val;
    };
    auto const I = numeric_integral(integrand, 0.0, 1.0);
    return 1.0 / std::sqrt(I);
}

//============================================================
// (正規化込み) \phi_n(x) = N_n * sn(\omega_n x, m)
//============================================================
double phi_n(double x, std::int32_t n, double m, double Nn)
{
    return Nn * phi_n_raw(x, n, m);
}

//============================================================
// S_{ij} = \int_0^1 \phi_i(x) \phi_j(x) dx
//============================================================
double overlap_Sij(std::int32_t i, std::int32_t j, double m, std::vector<double> const & Norm,
                   std::int32_t gauss_order = 32)
{
    auto const integrand = [&](auto xx) {
        auto const pi_val = phi_n_raw(xx, i, m) * Norm[i];
        auto const pj_val = phi_n_raw(xx, j, m) * Norm[j];
        return pi_val * pj_val;
    };
    return numeric_integral(integrand, 0.0, 1.0);
}

//============================================================
// H_{ij} = (hbar^2 / 2m) \int_0^1 \phi_i'(x)* \phi_j'(x) dx
//   ( 無限井戸内 V=0 => 運動エネルギー項のみ )
//============================================================
double hamiltonian_Hij(std::int32_t i, std::int32_t j, double m, std::vector<double> const & Norm)
{
    auto const phi_n_prime = [&](auto x, auto nIndex) {
        // d/dx [ N_n * sn(\omega_n x, m ) ]
        //   = N_n * [ \omega_n * cn(...) * dn(...) ]
        auto const w     = omega_n(nIndex, m);
        auto const je    = jacobi_elliptic(w * x, m);
        auto const deriv = w * je.cn * je.dn;
        return Norm[nIndex] * deriv;
    };

    auto integrand = [&](auto const xx) {
        auto const dpi = phi_n_prime(xx, i);
        auto const dpj = phi_n_prime(xx, j);
        return dpi * dpj;
    };

    auto const val    = numeric_integral(integrand, 0.0, 1.0);
    auto const factor = (HBAR * HBAR) / (2.0 * MASS);
    return factor * val;
}

//============================================================
// 行列 S, H を構築し、一般化固有値問題を解いて
// 固有値と固有ベクトルを得る。
//============================================================
struct DiagResult
{
    std::vector<double> norms;  // 各基底の正規化定数 [n=0..N-1]
    Eigen::VectorXd     eigenvalues;
    Eigen::MatrixXd     eigenvectors;  // 各列が固有ベクトル
};

DiagResult solve_eigen_problem(std::int32_t N, double m)
{
    DiagResult res;
    res.norms.resize(N);

    // 1) 各基底の正規化定数
    for (auto n = 0; n < N; ++n) {
        res.norms[n] = normalization_factor(n, m);
    }

    // 2) 行列 S, H
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(N, N);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(N, N);

#pragma omp parallel for
    for (auto i = 0; i < N; i++) {
        for (auto j = 0; j < N; j++) {
            auto const sij = overlap_Sij(i, j, m, res.norms);
            auto const hij = hamiltonian_Hij(i, j, m, res.norms);
            S(i, j)        = sij;
            H(i, j)        = hij;
        }
    }

    // 3) 一般化固有値問題 H c = E S c を解く
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(H, S);
    res.eigenvalues  = solver.eigenvalues();   // 昇順に並ぶ
    res.eigenvectors = solver.eigenvectors();  // (N x N)行列。各列が固有ベクトル

    return res;
}

//============================================================
// ヤコビ楕円基底で得た k番目(0-based) の固有状態の波動関数 \psi_k(x)
//   psi_k(x) = sum_{n=0..N-1} c_{n,k} * [ N_n * sn(\omega_n x, m) ]
//   c_{n,k} = evecs(n, k)  (行: n, 列: k)
//============================================================
double wavefunc_k(double x, std::int32_t k, const Eigen::MatrixXd & evecs, std::vector<double> const & norms, double m,
                  std::int32_t N)
{
    auto sum = 0.0;
    for (auto n = 0; n < N; n++) {
        auto const c_nk = evecs(n, k);
        // phi_n(x) = norms[n] * sn(omega_n*x, m)
        auto const phi_val = phi_n_raw(x, n, m) * norms[n];
        sum += c_nk * phi_val;
    }
    return sum;
}

//============================================================
// 無限井戸の厳密解 (節の数=ell) に対応する関数
//   node = 0 -> \ell=1
//   node = 1 -> \ell=2
//   ...
//   \psi_exact_node(x) = sqrt(2)*sin( (node+1)*pi*x )
//============================================================
double wavefunc_exact(std::int32_t node, double x)
{
    // node=0 => l=1
    // node=1 => l=2
    // ...
    auto const l = node + 1;
    return std::sqrt(2.0) * std::sin(l * std::numbers::pi * x);
}

//============================================================
// メイン
//============================================================
std::int32_t main()
{
    // ---------------------------
    // 1) パラメータ設定
    // ---------------------------
    auto constexpr m                     = 0.5;  // モジュラス m (固定)
    auto constexpr num_states_to_compare = 5;    // 基底状態(節0)～第4励起(節4)

    std::cout << std::format("=== Jacobi-sn basis (m={:.3f}, N={}) ===\n", m, N);

    // ---------------------------
    // 2) 一般化固有値問題を解く
    // ---------------------------
    auto const diag = solve_eigen_problem(N, m);

    // 固有値, 固有ベクトルを取得
    auto const & Evals = diag.eigenvalues;
    auto const & Evecs = diag.eigenvectors;
    // Evals[k] が k番目(0-based)の固有値 (昇順)
    // Evecs.col(k) が k番目の固有ベクトル

    // ---------------------------
    // 3) 固有値を表示 (0～4まで)
    // ---------------------------
    std::cout << "\nEigenvalues (lowest 5):\n";
    for (auto k = 0; k < num_states_to_compare; k++) {
        auto const E_exact = std::numbers::pi * std::numbers::pi * 0.5 * static_cast<double>((k + 1) * (k + 1));
        std::cout << std::format("E[{:>1}]        = {:>10.6f}\nE(exact)[{:>1}] = {:>10.6f}\n\n", k, Evals[k], k, E_exact);
    }

    // ---------------------------
    // 4) 波動関数をファイルに出力して比較
    //    x=0.0～1.0(0.01刻み) で
    //    (a) 数値解 psi_k(x) (k=0..4)
    //    (b) 厳密解
    //       node=0..4 => psi_exact(x) = sqrt{2} sin((node+1)*pi*x)
    // ---------------------------

#ifdef FILE_OUTPUT
    std::ofstream ofs("wavefunctions_compare.txt");
    if (!ofs) {
        std::cerr << "Cannot open wavefunctions_compare.txt\n";
        return 1;
    }

    ofs << "# x  psi_num[0..4]  psi_exact[0..4]\n";

    for (auto i = 0; i <= OUTPUTNUM; i++) {
        auto const x = static_cast<double>(i) / static_cast<double>(OUTPUTNUM);
        ofs << std::format("{:.3f}", x);
        // 数値解
        for (auto k = 0; k < num_states_to_compare; k++) {
            auto const val_num = wavefunc_k(x, k, Evecs, diag.norms, m, N);
            ofs << std::format(" {:.6f}", val_num);
        }
        // 厳密解
        for (auto node = 0; node < num_states_to_compare; node++) {
            auto const val_exact = wavefunc_exact(node, x);
            ofs << std::format(" {:.6f}", val_exact);
        }
        ofs << std::endl;
    }

    std::cout << std::format("\nWavefunctions (k=0..4) and exact solutions have been saved to {}\n",
                             "wavefunctions_compare.txt");
    std::cout << "Done.\n";
#endif

    return 0;
}
