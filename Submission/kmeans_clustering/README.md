# ノートブックの概要
## 1-clustering_train_3_7_mouse
- パラメータ(sigma_min=3,sigma_max=7)でmouse画像をkmeansで学習してモデルを保存
- (保存先：config["common"]["MODEL_DIR"])

### 1-clustering_train_6_20_mouse
- パラメータ(sigma_min=6,sigma_max=20)でmouse画像をkmeansで学習してモデルを保存
- (保存先：config["common"]["MODEL_DIR"])

### 1-clustering_train_allpairs_mouse
- 複数のパラメータの組[[3, 10],[3, 15],[4, 7],[4, 10],[4, 15],[5, 7],[5, 10],[5, 15]]
でmouse画像をkmeansで学習してモデルを保存
- (保存先：config["common"]["MODEL_DIR"])

### 2-clustering_predict_3_7_human
- パラメータ(sigma_min=3,sigma_max=3)で学習したモデルでhuman画像のクラスターを予測
- (保存先：config["common"]["OUTPUT_HUMAN_DIR"])

### 2-clustering_predict_3_7_mouse
- パラメータ(sigma_min=3,sigma_max=7)で学習したモデルでmouse画像のクラスターを予測
- (保存先：config["common"]["OUTPUT_DIR"])

### 2-clustering_predict_6_20
- パラメータ(sigma_min=6,sigma_max=20)で学習したモデルでmouse画像のクラスターを予測
- (保存先：config["common"]["OUTPUT_DIR"])

### 2-clustering_predict_allpairs_human
- 複数のパラメータの組[[3, 10],[3, 15],[4, 7],[4, 10],[4, 15],[5, 7],[5, 10],[5, 15]]
で学習したモデルでhuman画像のクラスターを予測
- (保存先：config["common"]["OUTPUT_DIR"])

### 2-clustering_predict_allpairs_mouse
- 複数のパラメータの組[[3, 10],[3, 15],[4, 7],[4, 10],[4, 15],[5, 7],[5, 10],[5, 15]]
で学習したモデルでmouse画像のクラスターを予測
- (保存先：config["common"]["OUTPUT_DIR"])

### 3-make_highlight_klabels
- 画像をハイライト画像に変換

### 4-make_3_7_plot_all_clusters
- humanの各tissue毎にパラメータ(sigma_min=3,sigma_max=7)でクラスタリングした結果（クラスター数が30なので画像は30枚）を1つの画像にまとめて保存
- (保存先：config["common"]["OUTPUT_HUMAN_DIR"]のフォルダー"all_clusters_plot")

### 4-make_6_20_plot_all_clusters
- humanの各tissue毎にパラメータ(sigma_min=6,sigma_max=20)でクラスタリングした結果（クラスター数が30なので画像は30枚）を1つの画像にまとめて保存
- (保存先：config["common"]["OUTPUT_HUMAN_DIR"]のフォルダー"all_clusters_plot")

### 5-make_3_7_plot_all_clusters_mouse
- mouseの各tissue毎にパラメータ(sigma_min=3,sigma_max=7)でクラスタリングした結果（クラスター数が30なので画像は30枚）を1つの画像にまとめて保存
- (保存先：config["common"]["OUTPUT_DIR"]のフォルダー"all_clusters_plot")

### 5-make_all_plot_all_clusters_mouse
- mouseの各tissue毎に複数のパラメータの組[[3, 10],[3, 15],[4, 7],[4, 10],[4, 15],[5, 7],[5, 10],[5, 15]]でクラスタリングした結果（クラスター数が30なので画像は30枚）を1つの画像にまとめて保存
- (保存先：config["common"]["OUTPUT_DIR"]のフォルダー"all_clusters_plot")
