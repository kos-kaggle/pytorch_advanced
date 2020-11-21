## 8.1 BERTのメカニズム

- BERT (Bidiractional Encoder Representations from Transformers) のモデル構造の概要を理解する
  - 文章の単語ID列 (bs x seq_len(512))
  - Embeddingsモジュール (seq_len x hidden(768))
  - BertLayerモジュール x 12 (seq_len x hidden(768); Transformer)★
  - BertPoolerモジュール (1 x hidden; 冒頭の\<cls>の特徴量を使用)
- BERTが事前学習する2種類の言語タスクを理解する
  - Masked Language Model
    - 概要
      - CBOWの拡張．文章の適当な複数の単語をマスクし，マスク以外の単語からマスクされた単語を予測する
    - 実装
      - BertLayerモジュール後に接続
      - softmax関数で，(seq_len x vocab_len)を予測
  - Next Sentence Prediction
    - 概要
      - 二つの文章を入力し，二つの文章に意味的なつながりがあるか否かを予測する
    - 実装
      - BertPoolerモジュール後に接続
      - 全結合想で，つながりがあるか否かを予測
- BERTの3つの特徴を理解する
  - 単語毎特徴量の性能向上：文脈に依存した特徴量化
    - 概要
      - Embeddings時には同じ分散表現であっても，BertLayerにて文脈を加味した特徴量にする
    - 工夫点
      - Masked Language Modelによる事前学習
      - Self-Attentionによる周辺語を加味した特徴量化
  - 自然言語処理でのファインチューニングを可能にする
    - 概要
      - 画像処理におけるVGGのように，様々なタスクに転用できる基本的なモデルを構築できる
    - 工夫点
      - Masked Language ModelとNext Sentence Predictionによる事前学習
  - Attentionにより，説明性と可視化を簡単にする
    - Transformerと同様

## 8.2 BERTの実装

- BERTのEmbeddingsモジュールの動作を理解し，実装できる
  - TransformerのEmbeddingとの違い
    - Positoinal Embeddingを学習させる
    - Sentence Embeddingの追加 (二文章入力用)
  - 実装
    - embeddings = words_embeddings + position_embeddings + token_type_embeddings
      - words_embeddings : 単語id毎分散表現化
      - position_embeddings : 語順番号毎分散表現化
      - token_type_embeddings : 文章番号毎分散表現化(Sentence Embedding)
- BERTのSelf-Attentionを活用したTransformer部分であるBertLayerモジュールの動作を理解し，実装できる
  - ネットワーク構成
    - BertAttention 
      - BertSelfAttention
      - BertSelfOutput (seq_len x hidden; 全結合層)
    - BertIntermediate (seq_len x intermediate_size(3072); 全結合層)
    - BertOutput (BertAttention出力とBertIntermediate出力の足し算)
      - BertIntermediate出力を，全結合層により(seq_len x hidden)に変換後，BertAttention出力と加算
  - ポイント  
    - BertIntermediateの全結合層後に，GELU(Gaussian Error Linear Unit)を使用
      - ReLUの非連続部分を，なめらかにしたもの
- BERTのPoolerモジュールの動作を理解し，実装できる
  - 活性化関数として，Tanhを使う

## 8.3 BERTを用いたベクトル表現の比較(bank: 銀行と土手)

- BERTの学習済みモデルを自分の実装モデルにロードできる
- BERT用の単語分割クラスなど，言語データの前処理部分を実装できる
- BERTで単語ベクトルを取り出して確認する内容を実装できる

## 8.4 BERTの学習・推論，判定根拠の可視化を実装

- BERTのボキャブラリーをtorchtextで使用する実装方法を理解する
- BERTに分類タスク用のアダプタモジュールを追加し，感情分析を実施するモデルを実装できる
- BERTをファインチューニングして，モデルを学習できる
- BERTのSelf-Attentionの重みを可視化し，推論の説明を試みられる

