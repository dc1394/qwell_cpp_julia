cmake_minimum_required(VERSION 3.13)
project(qwell LANGUAGES CXX)

# C++20 を使用する
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------------
# Boost の場所を明示的に指定
# ----------------------------------------------------------
# システム全体の Boost ではなく、ローカルの boost_1_87_0 ディレクトリを使用
set(Boost_NO_SYSTEM_PATHS ON)
# BOOST_ROOT を指定(ヘッダ + ライブラリの両方をここに置いている想定)
set(BOOST_ROOT "${CMAKE_CURRENT_LIST_DIR}/boost_1_87_0")

# Boost 1.87.0 を探す
# もし特定のライブラリ（例: filesystem, program_options など）が必要なら
# find_package(Boost 1.87.0 EXACT REQUIRED COMPONENTS filesystem program_options ...)
find_package(Boost 1.87.0 EXACT REQUIRED)

# 必要に応じて、ビルド済みのライブラリディレクトリを指定
# Boost を b2/bjam などでビルドしている場合の例:
# set(BOOST_LIBRARYDIR "${CMAKE_CURRENT_LIST_DIR}/boost_1_87_0/stage/lib")

# ----------------------------------------------------------
# Eigen のヘッダを設定 (ヘッダオンリー想定)
# ----------------------------------------------------------
# Eigen3::Eigen というターゲットを使いたい場合は
# find_package(Eigen3 REQUIRED NO_DEFAULT_PATH HINTS "${CMAKE_CURRENT_LIST_DIR}/eigen-3.4.0")
# のように書けることもありますが、ここでは直接 include_directories() します。
set(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/eigen-3.4.0")

# ----------------------------------------------------------
# OpenMP の自動検出
# ----------------------------------------------------------
find_package(OpenMP REQUIRED)

# 実行ファイルの生成
add_executable(qwell qwell.cpp)

# Boost のインクルードディレクトリを追加
target_include_directories(qwell PRIVATE
    ${Boost_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIR}
)

# コンパイルオプションを追加 (共通部分)
target_compile_options(qwell PRIVATE
    -O2
    -mtune=native
    -march=native
)

# OpenMP が見つかった場合にリンクを追加
if(OpenMP_CXX_FOUND)
    # コンパイラフラグがあれば付与 (clang などで -fopenmp を付ける必要がある場合)
    target_compile_options(qwell PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(qwell PRIVATE OpenMP::OpenMP_CXX)
endif()

# Boost のライブラリをリンク
# (Boost::boost はヘッダオンリーのメタターゲット、他に必要なものがあれば追加)
target_link_libraries(qwell PRIVATE
    ${Boost_LIBRARIES}
)