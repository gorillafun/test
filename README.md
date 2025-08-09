# compare_pdf_rag

このリポジトリは、PDF 文書を異なる方法で取り込んだ場合の Retrieval Augmented Generation (RAG) の性能を比較するための軽量なスクリプトを提供します。

## インストール

依存パッケージをインストールします:

```bash
pip install -r requirements.txt
```

## 使い方

準備するもの:

* 解析対象の PDF ファイル
* 1 行に 1 問を記載した `questions.txt`
* 正解となる回答を記載した `truths.txt`

スクリプトの実行:

```bash
python compare_pdf_rag.py <PDF> <questions.txt> <truths.txt> [--method metadata|mistral]
```

デフォルトの `metadata` メソッドは PDF から直接テキストを抽出します。`mistral` メソッドは `unstructured` で PDF を分割し、Mistral モデルでテキストを整形します。

スクリプトは生成された回答に対する ragas 指標を出力します。Azure OpenAI 用の必要な環境変数が設定されていることを確認してください。`mistral` メソッドを使用する場合は `MISTRAL_API_KEY` も必要です。

## ノートブック

`compare_pdf_rag.ipynb` には同じパイプラインを体験できるインタラクティブなノートブックが含まれています。

