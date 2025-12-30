from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


MODELS_TO_TEST = [
    {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "description": "Baseline (small, fast)",
        "size": "~400MB",
    },
    {
        "name": "intfloat/multilingual-e5-base",
        "description": "E5 multilingual base",
        "size": "~1.1GB",
    },
    {
        "name": "vinai/phobert-large",
        "description": "PhoBERT large (Vietnamese)",
        "size": "~400MB",
    },
]

TEST_PAIRS = [
    ("toi mat dien thoai", "toi danh roi dien thoai"),
    ("khong tai duoc app", "khong cai dat duoc ung dung"),
    ("the bi khoa", "card bi khoa"),
    ("chuyen tien", "chuyen khoan"),
    ("quen mat khau", "mat mat khau"),
]


def main():
    results = []

    for model_info in MODELS_TO_TEST:
        model_name = model_info["name"]

        try:
            model = SentenceTransformer(model_name)

            pair_results = []
            for text1, text2 in TEST_PAIRS:
                emb1 = model.encode([text1])[0].reshape(1, -1)
                emb2 = model.encode([text2])[0].reshape(1, -1)
                similarity = cosine_similarity(emb1, emb2)[0][0]
                pair_results.append(similarity)

            avg_sim = sum(pair_results) / len(pair_results)

            results.append(
                {
                    "Model": model_info["name"].split("/")[-1][:30],
                    "Avg Similarity": avg_sim,
                    "Size": model_info["size"],
                    "Description": model_info["description"],
                }
            )
        except Exception as exc:
            results.append(
                {
                    "Model": model_info["name"].split("/")[-1][:30],
                    "Avg Similarity": 0.0,
                    "Size": model_info["size"],
                    "Description": f"Error: {str(exc)[:50]}",
                }
            )

    df_results = pd.DataFrame(results).sort_values("Avg Similarity", ascending=False)
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
