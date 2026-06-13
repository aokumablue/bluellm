class UnsupportedContentError(ValueError):
    """Anthropic の content block を OpenAI リクエスト形式で表現できない場合に発生する。

    プロキシは変換できないコンテンツをサイレントに削除してはならない
    （``file``/file_id 参照などの未知の image/document source type、
    またはペイロードのない base64/url source）。そうすると、モデルはそのブロックが
    送信されなかったかのように回答してしまう。この例外により、サーバーは問題を
    400 ``invalid_request_error`` として表面化できる。
    無効なリクエスト値を示すため ``ValueError`` のサブクラスとする。
    """
