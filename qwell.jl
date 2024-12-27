#!/usr/bin/env julia

using LinearAlgebra
using FastGaussQuadrature
using Elliptic
using Base.Threads  # スレッド並列用

# ============================================
# 定数
# ============================================
const N           = 250   # 基底数
const GAUSS_ORDER = 4 * N # Gauss–Legendreの次数
const HBAR        = 1.0
const MASS        = 1.0
const OUTPUTNUM   = 1000  # 波動関数出力用の分割数 (0～1を何等分するか)

# ============================================
# Gauss–Legendre積分
#  [x_min, x_max] 上で関数func(x)を積分
# ============================================
function numeric_integral(func::Function, x_min::Float64, x_max::Float64;
                          order::Int = GAUSS_ORDER)
    # FastGaussQuadrature.jl の gausslegendre(order) は
    # [-1, +1] 区間の節点 x[] と重み w[] を返す
    x, w = gausslegendre(order)
    # [-1, +1] を [x_min, x_max] に写像
    #   t = ((x_max - x_min)/2)*x + (x_max + x_min)/2
    #   dt/dx = (x_max - x_min)/2
    half_length = 0.5 * (x_max - x_min)
    mid_point   = 0.5 * (x_max + x_min)
    result = 0.0
    for i in eachindex(x)
        t = half_length * x[i] + mid_point
        result += w[i] * func(t)
    end
    return half_length * result
end

# ============================================
# Jacobi elliptic sn, cn, dn をまとめて返す
#
# Elliptic.jl の ellipj(u, m) は
#   sn, cn, dn, φ = ellipj(u, m)
# の形で返してくれる。
# ここではsn, cn, dn のみ使用。
# ============================================
struct JacobiElliptic
    sn::Float64
    cn::Float64
    dn::Float64
end

function jacobi_elliptic(u::Float64, m::Float64)
    sn_, cn_, dn_ = ellipj(u, m)  # Elliptic.jl
    return JacobiElliptic(sn_, cn_, dn_)
end

# ============================================
# \omega_n = 2*(n+1)*K(m)
#   (n は 0-based, n+1 が本来の"モード番号")
# ============================================
function omega_n(n::Int, m::Float64)
    # 完全楕円積分K(m): Ellipticのellipk(m) で計算
    Kval = Elliptic.K(m)
    return 2.0 * (n+1) * Kval
end

# ============================================
# phi_n_raw(x, n, m):
#   sn( ωₙ * x, m )  [正規化定数はまだ掛けない版]
# ============================================
function phi_n_raw(x::Float64, n::Int, m::Float64)
    w = omega_n(n, m)
    je = jacobi_elliptic(w*x, m)
    return je.sn  # sn(...)
end

# ============================================
# 正規化係数 Nₙ = 1 / sqrt( ∫₀¹ sn²(ωₙ x, m) dx )
# ============================================
function normalization_factor(n::Int, m::Float64)
    integrand(x) = phi_n_raw(x, n, m)^2
    I = numeric_integral(integrand, 0.0, 1.0)
    return 1.0 / sqrt(I)
end

# ============================================
# (正規化込み) φₙ(x) = Nₙ * sn(ωₙ x, m)
# ============================================
function phi_n(x::Float64, n::Int, m::Float64, Nn::Float64)
    return Nn * phi_n_raw(x, n, m)
end

# ============================================
# オーバーラップ行列 Sᵢⱼ = ∫₀¹ φᵢ(x) φⱼ(x) dx
# ============================================
function overlap_Sij(i::Int, j::Int, m::Float64, Norms::Vector{Float64})
    integrand(x) = (phi_n_raw(x, i, m) * Norms[i+1]) *
                   (phi_n_raw(x, j, m) * Norms[j+1])
    return numeric_integral(integrand, 0.0, 1.0)
end

# ============================================
# ハミルトニアン行列 Hᵢⱼ = (ℏ² / 2m) ∫₀¹ φᵢ'(x)* φⱼ'(x) dx
#
#   φₙ'(x) = Nₙ * d/dx[ sn(ωₙ x, m) ]
#           = Nₙ * [ ωₙ * cn(ωₙ x, m) * dn(ωₙ x, m) ]
# ============================================
function hamiltonian_Hij(i::Int, j::Int, m::Float64, Norms::Vector{Float64})
    # φₙ'(x) の無名関数を定義
    phi_n_prime(x, nIndex) = begin
        w  = omega_n(nIndex, m)
        je = jacobi_elliptic(w*x, m)
        return Norms[nIndex+1] * (w * je.cn * je.dn)
    end

    integrand(x) = phi_n_prime(x, i) * phi_n_prime(x, j)
    val = numeric_integral(integrand, 0.0, 1.0)
    factor = (HBAR^2) / (2.0 * MASS)
    return factor * val
end

# ============================================
# 一般化固有値問題 Hc = E S c を解いて固有値・固有ベクトルを返す
# ============================================
mutable struct DiagResult
    norms::Vector{Float64}
    eigenvalues::Vector{Float64}
    eigenvectors::Matrix{Float64}
end

function solve_eigen_problem(N::Int, m::Float64)
    # 1) 各基底の正規化定数
    norms = Vector{Float64}(undef, N)
    for n in 0:(N-1)
        norms[n+1] = normalization_factor(n, m)
    end

    # 2) 行列 S, H を構築
    S = Matrix{Float64}(I, N, N)  # とりあえず零行列で初期化
    H = Matrix{Float64}(I, N, N)

    # スレッド並列
    @threads for i in 0:(N-1)
        for j in 0:(N-1)
            S[i+1, j+1] = overlap_Sij(i, j, m, norms)
            H[i+1, j+1] = hamiltonian_Hij(i, j, m, norms)
        end
    end

    # 3) 一般化固有値問題を解く:  H v = λ S v
    # Juliaでは eigvalsや eigen が使える (LinearAlgebra.standard).
    # ただしH, Sは対称(エルミート)とみなせるので `eigen(Symmetric(H), Symmetric(S))`
    # のようにSymmetric()でラップして高速化可能 (ただし厳密にはSに正定値性が要る)
    F = eigen(Symmetric(H), Symmetric(S))

    # 固有値は F.values, 固有ベクトルは F.vectors
    eigenvals = F.values
    eigenvecs = F.vectors

    return DiagResult(norms, eigenvals, eigenvecs)
end

# ============================================
# ヤコビ楕円基底で得た k番目(0-based) の固有状態の波動関数 ψₖ(x)
#   ψₖ(x) = Σₙ cₙₖ * [ Nₙ * sn(ωₙ x, m) ]
# ============================================
function wavefunc_k(x::Float64, k::Int,
                    evecs::Matrix{Float64},
                    norms::Vector{Float64}, m::Float64, N::Int)
    s = 0.0
    for n in 0:(N-1)
        c_nk = evecs[n+1, k+1]
        # φₙ(x) = norms[n+1]*sn(ωₙ*x,m)
        s += c_nk * (phi_n_raw(x, n, m) * norms[n+1])
    end
    return s
end

# ============================================
# 無限井戸の厳密解 (節の数=node) に対応する関数
#   node = 0 -> ℓ=1
#   node = 1 -> ℓ=2
#   ...
#   ψ_exact_node(x) = √2 * sin((node+1)*π*x)
# ============================================
function wavefunc_exact(node::Int, x::Float64)
    l = node + 1
    return sqrt(2.0) * sin(l * π * x)
end

# ============================================
# メイン部分
# ============================================
function main()
    # 1) パラメータ (ここでは m=0.5 とし、基底数=250 は定数として上部で定義)
    m = 0.5
    num_states_to_compare = 5

    println("=== Jacobi-sn basis (m=$(round(m, digits=3)), N=$N) ===")

    # 2) 一般化固有値問題を解く
    diag = solve_eigen_problem(N, m)
    Evals = diag.eigenvalues
    Evecs = diag.eigenvectors

    # 3) 固有値を表示 (0～4まで)
    println("\nEigenvalues (lowest $num_states_to_compare):")
    for k in 0:(num_states_to_compare-1)
        # 無限井戸厳密解: E_exact = (π²/2) * (k+1)²
        E_exact = (π^2 * 0.5) * (k+1)^2
        println("E[$k]        = $(round(Evals[k+1], digits=6))")
        println("E(exact)[$k] = $(round(E_exact, digits=6))")
        println()
    end

    # 4) 波動関数をファイルに出力して比較 (コメントアウトでOFF)
    # --------------------------------------------------
    # using Printf
    # open("wavefunctions_compare.txt", "w") do io
    #     println(io, "# x   ψ_num[0..4]              ψ_exact[0..4]")
    #     for i in 0:OUTPUTNUM
    #         x = i / OUTPUTNUM
    #         # 数値解
    #         num_vals = [wavefunc_k(x, k, Evecs, diag.norms, m, N) for k in 0:(num_states_to_compare-1)]
    #         # 厳密解
    #         exact_vals = [wavefunc_exact(node, x) for node in 0:(num_states_to_compare-1)]
    #
    #         @printf(io, "%.3f", x)
    #         for val in num_vals
    #             @printf(io, " %.6f", val)
    #         end
    #         for val in exact_vals
    #             @printf(io, " %.6f", val)
    #         end
    #         println(io)
    #     end
    # end
    # println("\nWavefunctions (k=0..4) and exact solutions have been saved to wavefunctions_compare.txt")
    # println("Done.")

    return
end

# スクリプトとして実行された場合のみ main() を呼ぶ
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
