import re

def word2features(sent, i):
    word = sent[i][0]
    
    # Türkçe için kritik ekler
    relcl_suffixes = ('an', 'en', 'dik', 'dık', 'ecek', 'acak', 'miş', 'mış', 'ki')
    case_suffixes = ('da', 'de', 'dan', 'den', 'ya', 'ye', 'ı', 'i', 'u', 'ü', 'la', 'le')
    # Yaygın bağlaçlar (Öbek sonlandırıcılar)
    conjunctions = ('ve', 'ile', 'veya', 'ama', 'fakat', 'ancak', 'da', 'de', 'ise')
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[:3]': word[:3],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'len(word)': len(word),
        # Gelişmiş dilbilgisi ipuçları
        'is_relcl_suffix': any(word.lower().endswith(s) for s in relcl_suffixes),
        'is_case_suffix': any(word.lower().endswith(s) for s in case_suffixes),
        'is_conjunction': word.lower() in conjunctions,
        'is_punctuation': bool(re.search(r'^[^\w\s]+$', word)),
    }
    
    # Bağlam (Context) özelliklerini daha da derinleştiriyoruz
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:is_conjunction': word1.lower() in conjunctions,
            '-1:is_relcl': any(word1.lower().endswith(s) for s in relcl_suffixes),
            '-1:suffix': word1[-2:],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:is_conjunction': word1.lower() in conjunctions,
            '+1:suffix': word1[-2:],
        })
    else:
        features['EOS'] = True
                
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent, label_idx):
    return [word[label_idx] for word in sent]

def load_conll(filename):
    sentences = []
    current_sent = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if current_sent:
                    sentences.append(current_sent)
                    current_sent = []
                continue
            parts = line.split('\t')
            if len(parts) >= 5:
                # FORM, OUTER, INNER, CLAUSE
                current_sent.append((parts[1], parts[2], parts[3], parts[4]))
            elif len(parts) >= 2:
                # Handle cases where some columns might be missing in synthesized data
                # Fill missing with 'O' or '_'
                word = parts[1]
                outer = parts[2] if len(parts) > 2 else 'O'
                inner = parts[3] if len(parts) > 3 else '_'
                clause = parts[4] if len(parts) > 4 else 'O'
                current_sent.append((word, outer, inner, clause))
        if current_sent:
            sentences.append(current_sent)
    return sentences
