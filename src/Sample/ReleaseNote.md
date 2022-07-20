# TL10向け　DS5 upsampling
## Version 1.0
* 日付 22/7/12
* 変更点
  * API 
    * 関数名変更：run_pc() -> run()
    * パラメータ変更：set_upsampling_parameters()のパラメータをSpotとFloodそれぞれ２つに設定
    * filter_by_conf()消し
  * 機能
    * spotのみの対応
    * spotとfloodの合わせの対応
    * 処理時間の短縮（flood単体約 10 ms、spot+flood約 24 ms 測定環境AMD 3700X, Windows 10)
    * NaNの出力
    * 信頼度：仮に0か1の出力、次回変更。

* 使用
  * sample.cppをご参照ください。