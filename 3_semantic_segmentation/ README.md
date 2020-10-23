# 3-1 セマンティックセグメンテーションとは
- ピクセルレベルで物体名をラベルづけ 
- ex. 製造業の製品の傷部位検出、医療画像診断の病変部分の検出、自動運転の周囲の環境把握
1. セマンティックセグメンテーションのインプット、アウトプットを理解
    - インプット：色付き画像　ex. 300(縦)×500(横)×3(RGB)
    - アウトプット ：画像(値:予測したクラス) ex. 300(縦)×500(横)
2. カラーパレット形式の画像データを理解
    - 物体ラベルとRGBを対応させる
      ex. 
      背景：物体ラベル=0, RGB=(0, 0, 0) 黒
      飛行機：物体ラベル=1, RGB=(128, 0, 0) 赤
3. PSPNetによるセマンティックセグメンテーションの4stepの流れを理解
    - Step1. 画像をPSPNet用にリサイズ
    - Step2. PSPNetに入れる　→出力例：21(クラス数)×475(縦)×475(横)
    - Step3. 確信度が高いクラスを抽出　→出力例：475(縦)×475(横)　
    - Step4. 画像を元の大きさに戻す
その他：本章でもPASCAL VOC2012のデータを使用　→訓練データ1464枚、検証データ1449枚、クラス数21

# 3-2 DatasetとDataLoaderの実装
- 二章とほぼ同じ
1. クラスDataset, DataLoaderを作成できるようになる
    - Dataset
        - DataTransformクラス：前処理
          Compose内に前処理内容を記述
        - VOCDatasetクラス：データセットを作成
    - DataLoader(一章と同じ)
2. PSPNetの前処理、データオーギュメンテーションを理解
    - train, validに共通の前処理
        - リサイズ
        - 色情報の標準化とテンソル化
    - trainだけの前処理
        - 画像の拡大
        - 回転
        - ランダムミラー

# 3-3 PSPNetのネットワーク構成と実装
1. ネットワーク構造をモジュール単位で理解
    - P143 図3.3.1 参照
2. 各モジュールの役割を理解
    - PSPNetは4つのモジュールから構成
        - Feature: 入力画像の特徴を捉える →3-4節
        - Pyramid Pooling: 周辺情報から特定のピクセルのクラスを予測する(PSPNetオリジナル) →3-5節 
        - Decoder: クラス分類 →3-6節
        - AuxLoss: 損失関数計算の補助、学習精度の向上。推論時は使わない →3-6節
3. ネットワーククラスの実装を理解
    - P143 図3.3.1の構成となるように実装
    - 順伝搬は以下のとおり
        - Feature
            - feature_conv
            - feature_res_1
            - feature_res_2
            - feature_dilated_res_1
                - AuxLoss
            - feature_dilated_res_2
        - Pyramid Pooling
        - Decoder

# 3-4 Featureモジュールの解説と実装
1. FEATUREモジュールのサブネットワーク構成を理解
 - 5つのサブネットワークで構成 (P149 図3.4.1 参照)
    - FeatureMap_convolution
    - residualBlockPSP
    - residualBlockPSP
    - residualBlockPSP(dilated)
    - residualBlockPSP(dilated)
2. サブネットワークFeatureMap_convolutionを実装
    - P150 図3.4.2 参照
    - 単純に畳み込み、特徴量を抽出
    - nn.Relu(inplace=True) にするとメモリ効率向上
3. Residual Blockを理解
    - 構成：
        - bottleNeckPSP ×1  →スキップ結合に畳み込み層が入る
        - bottleNeckIdentifyPSP ×複数   →スキップ結合に畳み込み層が入らない
    - ResNetで使用されているResidual Blockという構造を使用
    - スキップ結合：劣化問題(ネットワークが深くなると訓練データの誤差が大きくなる)を防ぐため
    - residual bolckの特徴：残差F(x)=y-xを学ばせる。y: 理想状態、x: 入力。
4. Dilated Convolutionを理解
5. サブネットワークbottleNeckPSPとbottleNeckIdentifyPSPを実装
6. Featureモジュールを実装
→resnetの解説

# 3-5 Pyramid Poolingモジュールの解説と実装
1. Pyramid Poolingモジュールのサブネットワーク構成を理解
    - PSPNetのオリジナルなモジュール
    - 構成：入力後、5つに分岐する
        - 5つの分岐のうち4つは、Adaptive Average Pooling層 → conv2DBatchNormRelu → UpSample層、と通過する
            - Adaptive Average Pooling層の出力は、6×6, 3×3, 2×2, 1×1 とサイズが異なる
            - UpSample層: 入力時のサイズまで単純な拡大処理
        - 5つの分岐のうち1つは、入力をそのまま出力に送る
        - 5つの分岐は、最後に結合して4096(5×512)×60×60となる
2. Pyramid Poolingモジュールのマルチスケール処理の実現方法を理解
    - 異なるサイズの特徴量マップを取得することで、あるピクセルの予測にさまざまなスケールの周囲情報を取り込むことができる
3. Pyramid Poolingモジュールを実装
    - Adaptive Average Pooling層: nn.AdaptiveAvgPool2dクラスを使用
    - UpSample層：F.interpolateによる演算?
    - 分岐の結合：tourch.cat(output, dim=1)

# 3-6 Decoder, AuxLossモジュールの解説と実装
1. Decoderモジュールのサブネットワーク構成を理解
    - Pyramid Poolingモジュールの出力をDecode(読み出し)し、ピクセルごとにクラス分類し、画像をもとのサイズに戻す
    - P163 図3.6.1
2. Decoderモジュールを実装
    - 実装クラス名：AuxiliaryPSPlayers
3. AuxLossモジュールのサブネットワーク構成を理解
    - Featureモジュールの出力をDecode(読み出し)し、ピクセルごとにクラス分類し、画像をもとのサイズに戻す(Decoderモジュールと同じ処理, 入力だけ違う)
    - P163 図3.6.1
4. AuxLossモジュールを実装
    - いずれも出力が21×475×475(クラス数×高さ×幅)となるように最後の畳み込み層を工夫(pointwise convolution　→5章のGANで詳細)

# 3-7 ファインチューニングによる学習と検証の実施
1. PSPNetの学習と検証の実装
    - 学習済みモデル「pspnet50_ADE20K.pth」を使用。caffeモデルを筆者が編集したもの。
    - 出力を21クラスに対応させるため畳み込み層を付け替える。また、Xevier(ゼイビアー、ザビエル)」で初期化
    - 初期値は活性化関数によって使い分ける
        - ReLU：「Heの初期値」を使用(2章)
        - シグモイド：「Xavierの初期値」を使用(3章)
    - 損失関数
        - クロスエントロピー誤差関数を使用
        - トータルの損失 = メインの損失 + AuxLossの損失×0.4
    - 学習率
        - 入力に近いモジュール：学習率を小さく
        - 出力に近いモジュール(Decoder, AuxLoss)：学習率を大きく 
        - スケジューラ
            - epochに応じて学習率を変化
            - 実装：
                - lambda_epoch(epoch): ...
                - scheduler = optim.lr_scheduler.LambdaR(optimizer, lr_lambda=lambda_epoch)
            - 関数lambda_epochの内容に従いインスタンスoptimizerの学習率を変化
            - 関数lambda_epochは、epochを経るごとに学習率が小さくなるようになっている
    - 学習・検証の関数を実装(train_model)
        - ほぼ第2章と同じ
        - 違いは2つ
            - スケジューラの存在。スケジューラ更新：scheduler.step()
            - multiple minibatchの使用。複数個のミニバッチで勾配を学習し、複数回に一度パラメータ更新。今回は3回
2. ファインチューニングの理解
3. スケジューラでepochごとに学習率を変化させる手法の実装

# 3-8 セマンティックセグメンテーションの推論
1. 推論を実装
    - ダミーデータとしてアノテーション画像が1枚必要。理由は2つ
        - アノテーション画像がないと前処理クラスの関数がうまく動作しないため
        - カラーパレットの情報が必要なため
    - 背景は透過、それ以外は色を付ける

# 全体構成(PSPNetによるセマンティックセグメンテーションのまとめ)
- Step1. 画像をPSPNet用にリサイズ
- Step2. PSPNet (P143 図3.3.1)
    - Feature: 入力画像の特徴を捉える (P149 図3.4.1)
        - FeatureMap_convolution: 単純に畳み込み、特徴量を抽出 (P150 図3.4.2)
            - conv2DBatchNormRelu
            - conv2DBatchNormRelu
            - conv2DBatchNormRelu
            - MaxPooling
        - residualBlockPSP：役割は不明...ResNetのresidualBlockを使用。 (P152 図3.4.3)
            - bottleNeckPSP: スキップ結合に畳み込み層が入る (P154 図3.4.4)
            - bottleNeckIdentifyPSP ×複数 : スキップ結合に畳み込み層が入らない (P154 図3.4.4)
        - residualBlockPSP: 上に同じ
        - residualBlockPSP(dilated):より大きな特徴量を抽出 (P155 図3.4.5)
          →AuxLossモジュールへ
        - residualBlockPSP(dilated): 上に同じ
    - Pyramid Pooling: 周辺情報から特定のピクセルのクラスを予測する。PSPNetのオリジナル
        - bottleNeckPSP (P158 図3.5.1)
    - Decoder: クラス分類 (P163 図3.6.1)　
        - conv2DBatchNormRelu
        - DO, conv2D
        - UP
    - AuxLoss: 損失関数計算の補助、学習精度の向上。推論時は使わない。 (P163 図3.6.1)
        - conv2DBatchNormRelu
        - DO, conv2D
        - UP
- Step3. 確信度が高いクラスを抽出　
- Step4. 画像を元の大きさに戻す
