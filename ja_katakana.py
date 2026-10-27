"""
日语 → 片假名读音转换（处理助词特殊读音，适配 CosyVoice3）
"""
from sudachipy import dictionary

_tokenizer = dictionary.Dictionary().create()

# 常见口语化修正（词典默认正式读音 → 日常口语读音）
_READING_FIX = {
    '私': 'ワタシ',       # ワタクシ → ワタシ
    '僕': 'ボク',
    '俺': 'オレ',
}

def ja_to_katakana(text: str) -> str:
    """日文 → 片假名读音（正确处理助词「は/へ/を」）"""
    parts = []
    for m in _tokenizer.tokenize(text):
        surface = m.surface()
        reading = m.reading_form()
        pos = m.part_of_speech()

        # 助词特殊读音
        if pos[0] == '助詞':
            if surface == 'は':
                reading = 'ワ'
            elif surface == 'へ':
                reading = 'エ'
            elif surface == 'を':
                reading = 'オ'
        # 固定表达
        elif surface == 'こんにちは':
            reading = 'コンニチワ'
        elif surface == 'こんばんは':
            reading = 'コンバンワ'
        # 口语化修正
        elif surface in _READING_FIX:
            reading = _READING_FIX[surface]

        parts.append(reading)
    return ' '.join(parts)


if __name__ == '__main__':
    tests = [
        'こんにちは、私は安和昴です。',
        '今日は、私の声を聞いてくれて、ありがとう。',
        'ドラムを叩くのは、すごく楽しいです。',
    ]
    for t in tests:
        print('原文:', t)
        print('读音:', ja_to_katakana(t))
        print()
