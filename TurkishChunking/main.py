import argparse
import os
from src.features import load_conll, sent2features, sent2labels
from src.trainer import ChunkingTrainer
from src.evaluator import evaluate_and_plot

def main(args):
    trainer = ChunkingTrainer()
    
    if args.mode == "train":
        print("Starting training phase...")
        trainer.train(args.train_file)
        trainer.save_models(args.model_dir)
        print(f"Models saved to {args.model_dir}")
        
        # Self-evaluation on training data
        test_sentences = load_conll(args.train_file)
        X_test = [sent2features(s) for s in test_sentences]
        predictions = trainer.predict(X_test)
        
        for idx, target in enumerate(trainer.targets):
            y_true = [sent2labels(s, idx + 1) for s in test_sentences]
            evaluate_and_plot(y_true, predictions[target], target, args.results_dir)

    elif args.mode == "test":
        print("Starting testing phase...")
        trainer.load_models(args.model_dir)
        test_sentences = load_conll(args.test_file)
        X_test = [sent2features(s) for s in test_sentences]
        
        predictions = trainer.predict(X_test)
        
        for idx, target in enumerate(trainer.targets):
            y_true = [sent2labels(s, idx + 1) for s in test_sentences]
            evaluate_and_plot(y_true, predictions[target], target, args.results_dir)
            
        # Display sample prediction
        print("\n--- Sample Prediction (First Sentence) ---")
        for i, word in enumerate(test_sentences[0]):
            word_text = word[0]
            pred_outer = predictions['OUTER'][0][i]
            pred_inner = predictions['INNER'][0][i]
            pred_clause = predictions['CLAUSE'][0][i]
            print(f"{word_text:15} | Outer: {pred_outer:10} | Inner: {pred_inner:10} | Clause: {pred_clause:10}")

    elif args.mode == "run":
        # Mevcut test dosyasını analiz et
        trainer.load_models(args.model_dir)
        test_sentences = load_conll(args.test_file)
        X_test = [sent2features(s) for s in test_sentences]
        predictions = trainer.predict(X_test)
        
        print("\n--- Örnek Cümle Analizi ---")
        for i, word in enumerate(test_sentences[0]):
            print(f"{word[0]:15} -> Dış: {predictions['OUTER'][0][i]}, İç: {predictions['INNER'][0][i]}, Tümce: {predictions['CLAUSE'][0][i]}")

    elif args.mode == "predict":
        # Kullanıcının verdiği cümleyi analiz et
        if not args.sentence:
            print("Hata: Lütfen analiz edilecek bir cümle girin. Örn: --sentence \"Bugün okula gittim.\"")
            return
            
        trainer.load_models(args.model_dir)
        # Basit bir şekilde boşluklara göre ayırıyoruz (Tokenization)
        import re
        tokens = re.findall(r"[\w']+|[.,!?;]", args.sentence)
        
        # Özellikleri çıkar (Sent2features tuple listesi bekler: (kelime, placeholder, placeholder, placeholder))
        sent_for_feat = [(t, 'O', '_', 'O') for t in tokens]
        X_test = [sent2features(sent_for_feat)]
        
        predictions = trainer.predict(X_test)
        
        print(f"\n--- '{args.sentence}' Analizi ---")
        print(f"{'Kelime':15} | {'Dış Öbek':10} | {'İç Öbek':10} | {'Tümce Yapısı'}")
        print("-" * 60)
        for i, token in enumerate(tokens):
            p_outer = predictions['OUTER'][0][i]
            p_inner = predictions['INNER'][0][i]
            p_clause = predictions['CLAUSE'][0][i]
            print(f"{token:15} | {p_outer:10} | {p_inner:10} | {p_clause}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Türkçe Chunking Projesi")
    parser.add_argument("--mode", choices=["train", "test", "run", "predict"], default="train", help="Çalıştırma modu")
    parser.add_argument("--sentence", type=str, help="Analiz edilecek cümle (predict modu için)")
    parser.add_argument("--train_file", default="data/train.conll", help="Eğitim verisi yolu")
    parser.add_argument("--test_file", default="data/test.conll", help="Test verisi yolu")
    parser.add_argument("--model_dir", default="models", help="Modellerin kaydedileceği dizin")
    parser.add_argument("--results_dir", default="results", help="Sonuçların kaydedileceği dizin")
    
    args = parser.parse_args()
    main(args)
