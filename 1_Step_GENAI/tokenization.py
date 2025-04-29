import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")

print(f'Vocab Size {encoder.n_vocab}')

text = 'Hii I am Syed Faisal Abdul Rahman Zulfequar'
token = encoder.encode(text)
print(f'Generated Toekens {token}') 

Genrated_Tokens = [39, 3573, 357, 939, 25254, 295, 186326, 280, 93275, 29676, 2309, 124632, 2302, 351, 277]

token_decyption = encoder.decode(Genrated_Tokens)
print(f'Original Tokens are {token_decyption}')