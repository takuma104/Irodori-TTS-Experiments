# Irodori TTS アーキテクチャ解析

**User:** Anonymous  
**Created:** 9/26/2026 20:05:47  
**Updated:** 9/28/2026 21:23:52  
**Exported:** 9/28/2026 21:51:06  
**Link:** [https://chatgpt.com/c/6ab7a70a-a6e0-83ee-9c60-6ef3d7f39381](https://chatgpt.com/c/6ab7a70a-a6e0-83ee-9c60-6ef3d7f39381)  

## Prompt:
2026/9/28 21:23:14

ありがとう。

https://zenn.dev/parakeet_tech/articles/936532be817118 によると、IrodoriTTSはかなり漢字の読みの正解率が高い方なんですが、まだ改善の余地はありそうです。既存のcheckpointのFTで改善させるとしたらどの様なトレーニングが考えられるでしょう？理想的にはテキストデータのみでBERT部とprojectorのみに絞れるとよいんですが、可能性ありそうでしょうか？

## Response:
2026/9/28 21:23:16

> **8件のウェブサイトを検索しました**
>
> 
>
> 数秒間考えました

可能性はありますが、**BERT + projectorだけを、読みの教師データだけで直接FTするには「TTSが期待する512次元条件空間」と「読み」を結ぶ補助タスクを設計する必要があります**。単純なModernBERTのG2P SFTだけだと、BERTの表現が改善しても既存DiTがその変化を正しく利用できる保証がありません。

特に興味深いのは、v4→v4.1で生成本体はほぼ同じなのに、JKYB-ParakeetではIrodori v4.1が約93%で、G2Pを明示的に持つ上位系が95%前後まで行っている点です。熟字訓がno-G2P TTSの弱点という傾向も出ています。([Zenn](https://zenn.dev/parakeet_tech/articles/936532be817118?utm_source=chatgpt.com)) 一方、Irodori v4でModernBERT導入自体が「難しい漢字の読み」を改善する目的だったことからも、**text encoder側にはまだ改善余地がある**と考えるのはかなり自然です。([GitHub](https://github.com/Aratako/Irodori-TTS/blob/main/docs/parameters.md?utm_source=chatgpt.com))

### 私ならまず試す構成

一番面白いのは、既存の

```text
text
  ↓
ModernBERT
  ↓
Irodori projector
  ↓
[B,L,512]
  ↓
frozen DiT
```

を維持したまま、学習時だけ小さな**reading head**を付ける方法です。

```text
漢字仮名混じり文
        │
        ▼
  ModernBERT-ja
        │ [B,L,768]
        ▼
 Irodori Projector
        │ [B,L,512]
        ├──────────────────→ frozen Irodori DiT
        │
        ▼
  Reading Decoder
  (training only)
        │
        ▼
カナ読み / phoneme列
```

ポイントは、**reading headをBERTの768次元に付けず、必ずIrodori projector後の512次元に付ける**ことです。

これなら学習目標が、

> 「DiTが実際に見る512次元表現に、正しい読みを復元できる情報を持たせる」

になります。

推論時にはreading decoderを捨てられるので、**Irodoriのアーキテクチャも推論速度も一切変更不要**です。

---

### ただし、普通のtoken-wise分類にはしない方がいい

日本語G2Pは入力tokenと読みtokenが1:1ではありません。

例えば、

```text
今日      → キョウ
一日      → ツイタチ / イチニチ
人気      → ニンキ / ヒトケ
五月雨    → サミダレ
```

なので、

```text
ModernBERT token
     ↓
各tokenについて読みを分類
```

だと扱いづらいです。

私なら**小型AR decoderによるseq2seq**にします。

```text
projected state
[B,L,512]
      │
      │ cross attention
      ▼
2～4 layer Transformer decoder
      │
      ▼
<bos> キ ョ ウ ワ ... <eos>
```

reading decoderは学習時にしか使わないので、多少重くても推論コストは0です。

教師データも、

```json
{
  "text": "今日の人気商品です",
  "reading": "キョウノニンキショウヒンデス"
}
```

だけで済みます。つまりご希望の**「音声なし、テキストデータだけ」**が実現できます。

---

## ただ、ここに1つ大きな問題があります

この方法だけで、

```text
reading loss ↓
       ⇒
TTS読み間違い ↓
```

になるとは限りません。

既存DiTは現在のprojector出力分布

$$
z_\text{old}\in\mathbb{R}^{512}
$$

を前提に学習済みです。

G2PだけでBERT/projectorを強く更新すると、

$$
z_\text{new}
$$

が読みを非常によく表現していても、DiTから見るとOODなcondition embeddingになってしまう可能性があります。

なので、**元checkpointをteacherとしてrepresentation distillationを同時に入れる**のがかなり重要だと思います。

---

# 提案するloss

元のIrodori text encoderを凍結してteacherとして保持します。

```text
             ┌→ frozen BERT + projector → z_teacher
text ────────┤
             └→ trainable BERT + projector → z_student
                                           │
                                           └→ Reading Decoder
```

そして、

$$
L =
L_\mathrm{reading}
+\lambda_\mathrm{distill}L_\mathrm{repr}
$$

とします。

単純には

$$
L_\mathrm{repr}
=
\frac{1}{N}
\sum_i
\|z_i^\mathrm{student}
-z_i^\mathrm{teacher}\|_2^2
$$

です。

ただ、これだと「読みを改善するために必要な変化」まで抑えてしまいます。

そこで面白いのが、**通常文と難読文でλを変える**方法です。

```text
普通の文章：
    λdistill = 大
    → 元Irodoriのcondition空間をほぼ維持

読み曖昧・難読文章：
    λdistill = 小
    → G2P lossが表現を変更できる
```

これはかなり試す価値があると思います。

---

# さらに良さそうなのが「誤読箇所だけ表現を変える」

今回の目的なら、私はむしろこちらを本命にします。

既存Irodoriに大量の文章を読ませ、

```text
正しく読めた文
誤読した文
```

を分けます。

例えば、

```text
人気のない公園だった
    Irodori: ニンキノナイ...
    正解:    ヒトケノナイ...
```

だったとします。

reading lossは全文に掛けますが、distillationは

```text
人気
^^^^
```

周辺だけ弱める。

概念的には、

$$
L_\mathrm{repr}
=
\sum_i w_i
\|z_i^S-z_i^T\|^2
$$

として、

```text
通常token       w = 1.0
難読語周辺       w = 0.05～0.2
```

など。

こうすると、

> **現在うまく動いているIrodoriのcondition manifoldを最大限保存しながら、読み間違いに関係する部分だけ動かす**

というFTになります。

これはBERT+projectorだけを触る方式と非常に相性がいいです。

---

## データについても、音声よりテキストの方がむしろ作りやすい

この実験なら数百万～数千万文をかなり安価に作れます。

例えば、

```text
Wikipedia
青空文庫
ニュース
固有名詞辞書
Wikipediaタイトル
住所
駅名
人名
企業名
Wikipediaの曖昧語
JKYB型synthetic sentence
```

から文章を集め、

```text
text → 高精度G2P → kana
```

とします。

ただし、ここでOpenJTalkだけを教師にすると、OpenJTalkの誤りをそのまま蒸留することになります。今回の記事でも高性能なreading modelと古典的形態素解析器には差があります。([Zenn](https://zenn.dev/parakeet_tech/articles/936532be817118?utm_source=chatgpt.com))

なので、

```text
通常文      → Sudachi/OpenJTalk等
難読文      → 強いreading model / LLM
辞書語      → 辞書由来ground truth
```

のように教師を分ける方がよいでしょう。

特に**同じ表記で読みが変わるcontrastive pair**を大量に入れたいです。

```text
人気の商品だ
→ ニンキ

人気のない山道だ
→ ヒトケ

上手に演奏する
→ ジョウズ

上手から登場する
→ カミテ

明日会社へ行く
→ アシタ

明日は明後日の前日だ
→ アス（文脈次第）
```

こういうデータはModernBERTを使う意味が非常に大きいところです。

---

# もう一段よい案：reading CE + contrastive loss

G2P decoderだけではなく、**同じ漢字表記で異なる読みをする文章のprojector表現を明示的に離す**のも面白いです。

例えば、

```text
A: 人気商品です       → ニンキ
B: 人気のない場所です → ヒトケ
```

に対して、対象語付近のprojected representationを、

$$
z_A,\quad z_B
$$

として、

```text
同じ読み → embeddingを近づける
違う読み → embeddingを離す
```

というSupervised Contrastive Lossを加える。

最終的には、

$$
\boxed{
L =
L_\mathrm{G2P}
+\lambda_1 L_\mathrm{repr-distill}
+\lambda_2 L_\mathrm{contrastive}
}
$$

という構成です。

これはかなり研究としても面白いと思います。

---

## BERTとProjectorのどこをFTするか

最初から全部動かすより、私は段階的に実験します。

| 実験 | Trainable | 意味 |
|---|---|---|
| A | projectorのみ | 最小変更baseline |
| B | BERT上位4～6層 + projector | **本命** |
| C | BERT全層 + projector | 最大自由度 |
| D | BERT LoRA + projector | parameter-efficient |
| E | BERT上位層 + projector + DiT text K/V LoRA | 音声FTありの上限比較 |

特に**B**が良さそうです。

ModernBERT下位層には一般的な字面・構文情報を残し、上位層だけをpronunciation-sensitiveにするイメージです。

Projectorだけだと、

> BERT内部では「人気=ニンキ」と「人気=ヒトケ」を十分区別できているが、512次元へ落とす過程で失っている

ケースには効きます。

逆にBERT自体が区別できていなければprojectorだけでは原理的に厳しい。その切り分けも実験A/Bでできます。

---

# 実は最初にやると面白い診断実験があります

FTする前に、**現在のv4.1 checkpointの各層から読みをprobeする**ことを強く勧めます。

```text
ModernBERT layer 1
          ↓
         ...
ModernBERT layer 25 ──→ probe → reading

projector出力 512 ─────→ probe → reading
```

各層を完全freezeして、軽量なreading decoderだけ学習します。

もし、

```text
BERT layer 25:
    JKYB 98%

projector後:
    JKYB 94%
```

なら、

> **projectorが読み情報を捨てている**

というかなり明確な結果になります。

逆に、

```text
BERT layer 25:
    94%

projector後:
    94%
```

ならBERT自体のG2P能力がボトルネック。

さらに、

```text
probeでは98%
実TTSでは93%
```

なら、**情報はtext conditionに存在するがDiTが利用できていない**ので、BERT/projectorだけをFTする方向自体が本質的でない可能性が高いです。

これは今回の研究を進める上でかなり重要な切り分けになると思います。

---

## なので、実験順序としては

1. **既存checkpointを完全freezeしてreading probeを学習**
2. ModernBERT各層とprojector後512次元のreading accuracyを比較
3. projectorのみを`G2P + distillation`でFT
4. BERT上位4～6層までunfreeze
5. contrastive homographデータを追加
6. JKYBだけでなく、**実際にTTS生成してKana-CERを評価**
7. 最後に比較としてDiTの`wk_text/wv_text`だけLoRA

がかなり綺麗です。

特に**3と7の比較**が面白いです。

```text
Text only
BERT + projector FT
       vs
Audio supervised
BERT + projector + DiT text-KV LoRA
```

とすれば、「**音声データなしでTTSのG2Pをどこまで改善できるか**」というかなり明確なテーマになります。

そして最初のprobeで、**ModernBERTの768次元では正しい読みがrecoverできるのにprojector後512次元で落ちている**ことが確認できれば、今回おっしゃっている「BERT/projectorだけをテキストでFT」はかなり勝算のある方向になります。逆にprojector後にも十分情報が残っているなら、次に疑うべきはDiT側のtext conditioning利用です。

[Irodori-TTS GitHub](https://github.com/Aratako/Irodori-TTS?utm_source=chatgpt.com)  
[JKYB-ParakeetによるTTS比較記事](https://zenn.dev/parakeet_tech/articles/936532be817118?utm_source=chatgpt.com)

---

**Sources:**

- [AIは漢字をどこまで読める？LLM・音声合成69モデルを徹底比較してみた（常用漢字読みベンチマークParakeet Editionの紹介）](https://zenn.dev/parakeet_tech/articles/936532be817118?utm_source=chatgpt.com)
- [Irodori-TTS/docs/parameters.md at main · Aratako/Irodori-TTS · GitHub](https://github.com/Aratako/Irodori-TTS/blob/main/docs/parameters.md?utm_source=chatgpt.com)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)