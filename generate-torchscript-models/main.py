import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
import os
from typing import List


def gen_test_sentences() -> List[str]:
    test_sentences = [
        "The weather is lovely today.",
        "It's so suuny outside!",
        "High Chancellor looks into the lantern that has no light in it and where the attendant wigs are all stuck in a fog-bank! This is the Court of Chancery, which has its decaying houses and its blighted lands in every shire, which has its worn-out lunatic in every madhouse and its dead in every churchyard, which has its ruined suitor with his slipshod heels and threadbare dress borrowing and begging through the round of every man’s acquaintance, which gives to monied might the means abundantly of wearying out the right, which so exhausts finances, patience, courage, hope, so overthrows the brain and breaks the heart, that there is not an honourable man among its practitioners who would not give—who does not often give—the warning, “Suffer any wrong that can be done you rather than come here!” Who happen to be in the Lord Chancellor’s court this murky afternoon besides the Lord Chancellor, the counsel in the cause, two or three counsel who are never in any cause, and the well of solicitors before mentioned? There is the registrar below the judge, in wig and gown; and there are two or three maces, or petty-bags, or privy purses, or whatever they may be, in legal court suits. These are all yawning, for no crumb of amusement",
        "ever falls from Jarndyce and Jarndyce (the cause in hand), which was squeezed dry years upon years ago. The short-hand writers, the reporters of the court, and the reporters of the newspapers invariably decamp with the rest of the regulars when Jarndyce and Jarndyce comes on. Their places are a blank. Standing on a seat at the side of the hall, the better to peer into the curtained sanctuary, is a little mad old woman in a squeezed bonnet who is always in court, from its sitting to its rising, and always expecting some incomprehensible judgment to be given in her favour. Some say she really is, or was, a party to a suit, but no one knows for certain because no one cares. She carries some small litter in a reticule which she calls her documents, principally consisting of paper matches and dry lavender. A sallow prisoner has come up, in custody, for the half-dozenth time to make a personal application “to purge himself of his contempt,” which, being a solitary surviving executor who has fallen into a state of conglomeration about accounts of which it is not pretended that he had ever any knowledge, he is not at all likely ever to do. In the meantime his prospects in life are ended. Another ruined suitor, who periodically",
        "appears from Shropshire and breaks out into efforts to address the Chancellor at the close of the day’s business and who can by no means be made to understand that the Chancellor is legally ignorant of his existence after making it desolate for a quarter of a century, plants himself in a good place and keeps an eye on the judge, ready to call out “My Lord!” in a voice of sonorous complaint on the instant of his rising. A few lawyers’ clerks and others who know this suitor by sight linger on the chance of his furnishing some fun and enlivening the dismal weather a little. Jarndyce and Jarndyce drones on. This scarecrow of a suit has, in course of time, become so complicated that no man alive knows what it means. The parties to it understand it least, but it has been observed that no two Chancery lawyers can talk about it for five minutes without coming to a total disagreement as to all the premises. Innumerable children have been born into the cause; innumerable young people have married into it; innumerable old people have died out of it. Scores of persons have deliriously found themselves made parties in Jarndyce and Jarndyce without knowing how or why; whole families have inherited legendary hatreds with the suit. The",
        "hello my name is bob how are you",
        "I ate eggs this morning and now I'm hungry again",
        "The weather is lovely today.",
        "It's so suuny outside!",
    ]

    return test_sentences


def gen_test_tokens(model: SentenceTransformer):
    return model.tokenize(gen_test_sentences())


def test_exported_model(
    test_tokens,
    original_embeddings,
    exported_model_path: str,
    eps: float,
):
    test_sentences = gen_test_sentences()
    # original_embeddings = original_model.encode(test_sentences)

    # tokens = original_model.tokenize(test_sentences)
    test_model = torch.jit.load(exported_model_path)
    test_embeddings = test_model(test_tokens)["sentence_embedding"]
    for sentence, original_embedding, test_embedding in zip(
        test_sentences, original_embeddings, test_embeddings
    ):
        test_embedding = test_embedding.detach().numpy()
        print("Sentence:", sentence[:100])
        diff = np.max(np.abs(original_embedding - test_embedding))
        print("Max diff between embedding and traced_embedding: ", diff)
        print("")
        print(np.sum(np.abs(original_embedding)), np.sum(np.abs(test_embedding)))

        if diff > eps:
            raise ValueError(f"Diff too great: {diff}")

    print(f"Test embeddings shape: {test_embeddings.shape}")


def export_to_torchscript(model_name: str, output_path: str):
    # dummy model
    # m = opensearch_py_ml.ml_models.SentenceTransformerModel(
    #     "sentence-transformers/all-MiniLM-L6-v2"
    # )

    # m.save_as_pt(model_name)

    model = SentenceTransformer(model_name)
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]
    test_tokens = gen_test_tokens(model)
    original_embeddings = model.encode(gen_test_sentences())

    # tokens = {k: v.to("mps") for (k, v) in tokens.items()}

    # strict = False is necessary to avoid some warnings about returning a dict from forward
    # traced_encode = torch.jit.trace_module(model, {"forward": tokens}, strict=False)
    # torch.jit.save(traced_encode, output_path)
    # traced_embeddings = (
    #     traced_encode(tokens)["sentence_embedding"].cpu().detach().numpy()
    # )

    # handle when model_max_length is unproperly defined in model's tokenizer (e.g. "intfloat/e5-small-v2")
    # (See PR #219 and https://github.com/huggingface/transformers/issues/14561 for more context)
    if model.tokenizer.model_max_length > model.get_max_seq_length():
        model.tokenizer.model_max_length = model.get_max_seq_length()
        print(
            f"The model_max_length is not properly defined in tokenizer_config.json. Setting it to be {model.tokenizer.model_max_length}"
        )
    # save tokenizer.json in save_json_folder_name
    # model.save(save_json_folder_path)
    # super()._fill_null_truncation_field(
    #     save_json_folder_path, model.tokenizer.model_max_length
    # )
    # convert to pt format will need to be in cpu,
    # set the device to cpu, convert its input_ids and attention_mask in cpu and save as .pt format
    device = torch.device("cpu")
    cpu_model = model.to(device)
    features = cpu_model.tokenizer(
        sentences, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    compiled_model = torch.jit.trace(
        cpu_model,
        (
            {
                "input_ids": features["input_ids"],
                "attention_mask": features["attention_mask"],
            }
        ),
        strict=False,
    )
    torch.jit.save(compiled_model, output_path)
    print("model file is saved to ", output_path)

    print("Testing exported model")
    test_exported_model(test_tokens, original_embeddings, output_path, 1e-6)
    print("Export successful!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert sentence transformer to TorchScript"
    )
    parser.add_argument(
        "model_name", help="Name or path of the sentence transformer model"
    )
    parser.add_argument(
        "output_path", help="Output path for the TorchScript model (.pt file)"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    export_to_torchscript(args.model_name, args.output_path)


if __name__ == "__main__":
    main()
