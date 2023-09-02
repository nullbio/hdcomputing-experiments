import torchhd
import torch

DIMS = 100000
VSA = "MAP"
PRG = torch.Generator(device="cpu")
PRG.manual_seed(2147483647)


def hd_encode(str=None):
    filebytes = bytes(str, encoding="ascii")

    ascii_char_vectors = torchhd.level(
        num_vectors=256,
        dimensions=DIMS,
        vsa=VSA,
        generator=PRG,
    )

    # file_index_vectors
    file_index_vectors = torchhd.random(
        len(filebytes), dimensions=DIMS, vsa=VSA, generator=PRG
    )

    binded_pos_vectors = []
    # bind the vector for the file byte index to the ascii position vector
    for idx in range(len(filebytes)):
        ascii_char_vec = ascii_char_vectors[filebytes[idx]]
        binded_pos_vectors.append(file_index_vectors[idx].bind(ascii_char_vec))

    # Convert to Tensor object
    binded_pos_vectors = torch.stack(binded_pos_vectors)

    # Compress into bundled hypervector
    compressed_vector = torchhd.multibundle(binded_pos_vectors)

    # Encoding should be finished now...
    # Attempt to decode.
    decoded = []
    for i in range(len(filebytes)):
        unbound_vector = compressed_vector.bind(file_index_vectors[i].inverse())

        top_idx = 0
        top_match = None
        for i in range(len(ascii_char_vectors)):
            x = torchhd.cosine_similarity(ascii_char_vectors[i], unbound_vector)
            if top_match == None or x > top_match:
                top_match = x
                top_idx = i

        decoded.append(chr(top_idx))

    print("".join(decoded))


if __name__ == "__main__":
    hd_encode(str="Let's see how far we can get with this.")
